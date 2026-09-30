import os
import json
import time
import hashlib
import argparse

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from pharos.model import PCAutoencoder, PD2Latent, ShapeNet
from pharos.data import make_dataset, split_indices, CACHE_VERSION
from pharos.losses import chamfer_distance


LOSS_MODES = ("mse", "mse_cd", "cd")


def precompute_latents(ae, dataset, batch_size: int, device, num_workers: int, use_norm: bool):
    """Latent target and optical input for every frame, in dataset order.

    `use_norm` picks the min-max normalised optical vector over the raw one.
    The raw readings span six orders of magnitude between channels (per-column
    std from 1 to 5.7e6), which makes the input covariance catastrophically
    ill-conditioned and is felt directly in how many steps the regressor needs.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optical_signals = []
    latent_vectors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_points, batch_optical, batch_optical_norm = batch
            batch_points = batch_points.to(device).float()
            z = ae.encoder(batch_points)
            latent_vectors.append(z.cpu())
            # clone(): a tensor handed over by a worker lives in shared memory
            # and holds a file descriptor for as long as it is referenced. This
            # list keeps every batch, so without the copy the run accumulates
            # one fd per batch -- 5512 of them over the training cache -- and
            # dies with "Too many open files" a few hundred batches in. The
            # latents above come back from the GPU into fresh storage already.
            optical_signals.append((batch_optical_norm if use_norm else batch_optical).clone())
            if (batch_idx + 1) % 200 == 0:
                print(f"Precomputing latents: {batch_idx + 1}/{len(loader)} batches", flush=True)
    return torch.cat(latent_vectors, dim=0), torch.cat(optical_signals, dim=0)


def _encoder_fingerprint(ae) -> str:
    """Hash of the encoder weights: what actually determines the latents."""
    h = hashlib.sha256()
    for k, v in sorted(ae.encoder.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _array_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _dataset_fingerprint(dataset, cache_root: str) -> dict:
    """What identifies the frames (and their optical scaling) behind a dataset.

    Not the path: the cache directory can move. What matters is which frames
    the dataset keeps, in what order, and the min/max the normalised optical
    vector is scaled by -- any of which changes if the cache is rebuilt, the
    bounding box or max_frame_id changes, or sessions are filtered.
    """
    info_path = os.path.join(cache_root, "cache_info.json")
    info_sha = None
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            info_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    return {
        "cache_info_sha": info_sha,
        "n_items": len(dataset),
        "valid_indices_sha": _array_sha(np.asarray(dataset.valid_indices)),
        "optical_minmax_sha": _array_sha(np.stack([np.asarray(dataset.col_min),
                                                   np.asarray(dataset.col_max)])),
    }


def _latent_cache_manifest(ae, dataset, cfg, use_norm, ae_run) -> dict:
    """Everything a cached latent file has to agree with to be reusable."""
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    return {
        "ae_run": ae_run,
        "encoder_sha": _encoder_fingerprint(ae),
        "dataset": _dataset_fingerprint(dataset, data_cfg["cache_dir"]),
        "stride": data_cfg.get("stride"),
        "max_frame_id": data_cfg.get("max_frame_id"),
        "sessions": data_cfg.get("sessions"),
        "input": "norm" if use_norm else "raw",
        "latent_dim": model_cfg["latent_dim"],
        "optical_dim": model_cfg["optical_dim"],
        "num_points": model_cfg["num_points"],
        "tnet1": model_cfg["tnet1"],
        "tnet2": model_cfg["tnet2"],
        "data_cache_version": CACHE_VERSION,
    }


def load_or_precompute_latents(ae, dataset, cfg, device, use_norm, ae_run, mode="use"):
    """Precomputed latents and optical inputs, cached next to the dataset.

    The cache lives inside the dataset's own cache directory and is keyed by
    the autoencoder run name, so a file can only ever be found by the dataset
    and the run it was built from. On top of that the manifest has to match
    exactly -- including a hash of the encoder weights, so retraining the
    autoencoder under the same run name invalidates it rather than silently
    feeding the regressor stale targets. Any mismatch recomputes.

    `mode`: "use" (read and write), "refresh" (ignore what is there and
    rewrite), "off" (do neither).
    """
    tr = cfg["train_optical"]

    def compute():
        return precompute_latents(ae, dataset, tr["batch_size"], device,
                                  tr["num_workers"], use_norm)

    cache_root = cfg["data"].get("cache_dir")
    if mode == "off" or not cache_root:
        return compute()

    cache_dir = os.path.join(cache_root, "latents", ae_run)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    want = _latent_cache_manifest(ae, dataset, cfg, use_norm, ae_run)

    if mode == "use" and os.path.exists(manifest_path):
        try:
            have = json.load(open(manifest_path))
        except Exception:
            have = None
        if have == want:
            Z = torch.from_numpy(np.load(os.path.join(cache_dir, "latents.npy")))
            X = torch.from_numpy(np.load(os.path.join(cache_dir, "optical.npy")))
            if (Z.shape == (len(dataset), cfg["model"]["latent_dim"])
                    and X.shape == (len(dataset), cfg["model"]["optical_dim"])):
                print(f"Loaded cached latents from {cache_dir}")
                return Z, X
            print(f"Latent cache at {cache_dir} has arrays of the wrong shape "
                  f"({tuple(Z.shape)}, {tuple(X.shape)}); recomputing")
        else:
            diff = [k for k in want if have is None or have.get(k) != want[k]]
            print(f"Latent cache at {cache_dir} does not match ({', '.join(diff)}); recomputing")

    Z, X = compute()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Manifest first out, last in: while the arrays are being rewritten
        # there is no manifest, so an interrupted write can never leave a
        # valid-looking manifest next to half-new arrays.
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        np.save(os.path.join(cache_dir, "latents.npy"), Z.numpy())
        np.save(os.path.join(cache_dir, "optical.npy"), X.numpy())
        # Written last: a manifest on disk means the arrays beside it are complete.
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(want, f, indent=2)
        print(f"Cached latents to {cache_dir}")
    except OSError as e:
        print(f"Could not write the latent cache ({e}); continuing without it")
    return Z, X


class OpticalLatentSet(Dataset):
    """(optical, latent) for a set of frames, plus ground-truth points on demand.

    The latents and optical vectors are precomputed once and indexed as
    tensors; only the Chamfer modes need the point clouds, and those are read
    back from the (memory-mapped) dataset per item, since the full training
    split is far too large to hold at 5929 points per frame.
    """

    def __init__(self, base, X: torch.Tensor, Z: torch.Tensor, positions, with_points: bool):
        self.base = base
        self.X = X
        self.Z = Z
        self.positions = np.asarray(positions)
        self.with_points = with_points

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i: int):
        p = int(self.positions[i])
        if not self.with_points:
            return self.X[p], self.Z[p]
        pts, _, _ = self.base[p]
        return self.X[p], self.Z[p], torch.from_numpy(np.asarray(pts, dtype=np.float32))


def compute_loss(model, decoder, batch, device, mode: str, lam: float, cd_chunk: int):
    """Loss and its components for one batch.

    In the Chamfer modes the prediction is decoded and compared with the
    ground-truth cloud. The decoder takes part in the forward and backward pass
    -- that is how the gradient reaches PD2Latent -- but its parameters are
    frozen (requires_grad False) and are not in the optimizer, so the shared
    ShapeDecoder stays exactly the one the autoencoder trained.
    """
    if mode == "mse":
        batch_x, batch_z = batch
        batch_x, batch_z = batch_x.to(device).float(), batch_z.to(device).float()
        pred_z = model(batch_x)
        mse = torch.mean((pred_z - batch_z) ** 2)
        return mse, mse.detach().item(), 0.0

    batch_x, batch_z, batch_pts = batch
    batch_x = batch_x.to(device, non_blocking=True).float()
    batch_z = batch_z.to(device, non_blocking=True).float()
    batch_pts = batch_pts.to(device, non_blocking=True).float()

    pred_z = model(batch_x)
    mse = torch.mean((pred_z - batch_z) ** 2)
    cd, _ = chamfer_distance(decoder(pred_z), batch_pts, chunk=cd_chunk)

    loss = cd if mode == "cd" else mse + lam * cd
    return loss, mse.detach().item(), cd.detach().item()


def train_epoch(model, decoder, loader, optimizer, device, mode, lam, cd_chunk):
    model.train()
    sums = np.zeros(3)   # loss, mse, cd -- weighted by batch size
    n = 0
    for batch in loader:
        bs = batch[0].size(0)
        optimizer.zero_grad()
        loss, mse, cd = compute_loss(model, decoder, batch, device, mode, lam, cd_chunk)
        loss.backward()
        optimizer.step()
        sums += np.array([loss.detach().item(), mse, cd]) * bs
        n += bs
    # Plain floats: these end up in a checkpoint, and torch.load defaults to
    # weights_only=True, which rejects pickled numpy scalars.
    return tuple(float(v) for v in sums / max(1, n))


@torch.no_grad()
def evaluate_epoch(model, decoder, loader, device, mode, lam, cd_chunk):
    model.eval()
    sums = np.zeros(3)
    n = 0
    for batch in loader:
        bs = batch[0].size(0)
        loss, mse, cd = compute_loss(model, decoder, batch, device, mode, lam, cd_chunk)
        sums += np.array([loss.detach().item(), mse, cd]) * bs
        n += bs
    return tuple(float(v) for v in sums / max(1, n))


def train(model, decoder, train_loader, val_loader, optimizer, sched, cfg: dict, device, ckpt_dir: str):
    tr = cfg["train_optical"]
    loss_cfg = tr["loss"]
    mode, lam, cd_chunk = loss_cfg["mode"], loss_cfg["lambda_cd"], loss_cfg.get("cd_chunk", 512)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Every epoch, not just the ones that improve. A checkpoint is only written
    # when the validation loss sets a new record, so the saved checkpoints are
    # a monotone sequence by construction and cannot answer whether validation
    # ever turned back up -- which is exactly the question worth asking when
    # the training loss keeps falling.
    log_path = os.path.join(os.path.dirname(ckpt_dir.rstrip("/")), "optical_record_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Config: {cfg['train_optical']}\n")
        f.write("epoch,lr,train_loss,train_mse,train_cd,val_loss,val_mse,val_cd,seconds\n")

    best_val = float("inf")
    best_path = None
    epochs_without_improve = 0

    for epoch in range(tr["epochs"]):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        train_stats = train_epoch(model, decoder, train_loader, optimizer, device, mode, lam, cd_chunk)
        val_stats = evaluate_epoch(model, decoder, val_loader, device, mode, lam, cd_chunk)
        sched.step()

        # Early stopping follows the objective actually being optimised, so the
        # three modes are each stopped on their own terms.
        val_loss = val_stats[0]
        extra = "" if mode == "mse" else (f", MSE: {val_stats[1]:.4f}, CD: {val_stats[2]:.1f}")
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/{tr['epochs']}, lr {cur_lr:.2e}, Train: {train_stats[0]:.6f}, "
              f"Val: {val_loss:.6f}{extra} (gap {val_loss / max(train_stats[0], 1e-12):.2f}x, "
              f"{elapsed:.1f}s)", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{cur_lr:.6e},{train_stats[0]:.6f},{train_stats[1]:.6f},"
                    f"{train_stats[2]:.4f},{val_loss:.6f},{val_stats[1]:.6f},{val_stats[2]:.4f},"
                    f"{elapsed:.1f}\n")

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            best_path = os.path.join(ckpt_dir, f"model_ep{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mse": val_stats[1],
                "val_cd": val_stats[2],
                "loss_mode": mode,
            }, best_path)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= tr["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    return best_val, best_path


def main(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    tr = cfg["train_optical"]
    loss_cfg = tr["loss"]
    out_cfg = cfg["output"]

    mode = loss_cfg["mode"]
    if mode not in LOSS_MODES:
        raise ValueError(f"train_optical.loss.mode must be one of {LOSS_MODES}, got {mode!r}")
    needs_points = mode != "mse"
    use_norm = tr.get("input", "norm") == "norm"

    # Output always goes into the current run directory
    run_dir = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    ckpt_dir = os.path.join(run_dir, "optical_checkpoints")
    combined_path = os.path.join(run_dir, "best_combined.pth")

    # AE checkpoint: use ae_run_name override if set, otherwise same run
    ae_run = tr.get("ae_run_name") or out_cfg["run_name"]
    ae_run_dir = os.path.join(out_cfg["base_dir"], ae_run)
    ae_ckpt_path = os.path.join(ae_run_dir, "checkpoints", "best_model.pth")

    # "cuda" (no ordinal) means the current device, so CUDA_VISIBLE_DEVICES
    # still selects the GPU on a multi-GPU box without hardcoding an index.
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    print(f"Using device {device}")
    print(f"[Loss] mode={mode}" + (f", lambda_cd={loss_cfg['lambda_cd']}" if mode == "mse_cd" else "") +
          f" | input={'optical_norm' if use_norm else 'raw optical'} | optimizer={tr.get('optimizer', 'adam')}")

    dataset = make_dataset(data_cfg)
    sample_point, sample_optical, _ = dataset[0]
    print(f"Sample point cloud shape: {sample_point.shape}")
    print(f"Sample optical signal shape: {sample_optical.shape}")

    ae = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    best_ae = torch.load(ae_ckpt_path, map_location=device)
    prefix = "module."
    ae_state = {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in best_ae["model_state_dict"].items()}
    ae.load_state_dict(ae_state)
    ae.eval()
    print(f"Autoencoder loaded from {ae_ckpt_path}")

    # The decoder is supervision, not a trainable part: eval() so its BatchNorm
    # keeps the running statistics the autoencoder ended on, and no gradients
    # on its weights. Only PD2Latent goes into the optimizer below.
    decoder = ae.decoder
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    print("Precomputing latent vectors...")
    latent_vectors, optical_signals = load_or_precompute_latents(
        ae, dataset, cfg, device, use_norm, ae_run, mode=cfg.get("latent_cache", "use"))
    print(f"Latent vectors shape: {latent_vectors.shape}")

    X = optical_signals.float()
    Z = latent_vectors.float()
    if needs_points:
        # Only the Chamfer modes run workers, and those would otherwise be sent
        # a private copy of both tensors each (~0.6 GiB per worker).
        X.share_memory_()
        Z.share_memory_()

    # Same split as the autoencoder stage, so no frame that the autoencoder
    # validated on is used to fit the PD-to-latent regressor. precompute_latents
    # iterates the dataset unshuffled, so positions line up one to one.
    train_idx, val_idx = split_indices(dataset, data_cfg)
    train_dataset = OpticalLatentSet(dataset, X, Z, train_idx, needs_points)
    val_dataset = OpticalLatentSet(dataset, X, Z, val_idx, needs_points)
    print(f"[Split] {len(train_idx)} train, {len(val_idx)} val")

    # Workers only earn their keep when point clouds are read per item; the
    # pure-MSE path is tensor indexing and is fastest in-process.
    workers = tr["num_workers"] if needs_points else 0
    train_loader = DataLoader(train_dataset, batch_size=tr["batch_size"], shuffle=True,
                              num_workers=workers, pin_memory=needs_points)
    val_loader = DataLoader(val_dataset, batch_size=tr["batch_size"], shuffle=False,
                            num_workers=workers, pin_memory=needs_points)

    model = PD2Latent(in_features=model_cfg["optical_dim"], out_features=model_cfg["latent_dim"]).to(device)
    if tr.get("optimizer", "adam").lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=tr["lr"], momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=tr["lr"])
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["epochs"])

    print("Starting training...")
    best_val, best_path = train(model, decoder, train_loader, val_loader, optimizer, sched,
                                cfg, device, ckpt_dir)
    print(f"Best model saved at {best_path} with Val Loss: {best_val:.6f}")

    combined_model = ShapeNet(
        pd_in_features=model_cfg["optical_dim"],
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    )
    pd2latent_ckpt = torch.load(best_path, map_location=device)
    combined_model.pd2latent.load_state_dict(pd2latent_ckpt["model_state_dict"])
    combined_model.decoder.load_state_dict(decoder.state_dict())

    torch.save({
        "epoch": pd2latent_ckpt.get("epoch"),
        "val_loss": best_val,
        "optical_dim": model_cfg["optical_dim"],
        "latent_dim": model_cfg["latent_dim"],
        "num_points": model_cfg["num_points"],
        "tnet1": model_cfg["tnet1"],
        "tnet2": model_cfg["tnet2"],
        "loss_mode": mode,
        # Whoever feeds this model has to reproduce the training-time input
        # transform. With input_norm the model expects (x - optical_min) /
        # (optical_max - optical_min), clipped to [0, 1], with THESE constants
        # -- the ones from the training cache, not statistics recomputed on
        # whatever set is being evaluated.
        "input_norm": use_norm,
        "optical_min": torch.from_numpy(np.asarray(dataset.col_min, dtype=np.float32)),
        "optical_max": torch.from_numpy(np.asarray(dataset.col_max, dtype=np.float32)),
        "pd2latent_state_dict": combined_model.pd2latent.state_dict(),
        "decoder_state_dict": combined_model.decoder.state_dict(),
    }, combined_path)
    print(f"Combined model saved to {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Override output.run_name from config")
    parser.add_argument("--ae-run-name", default=None, help="Load AE weights from a different run (overrides train_optical.ae_run_name)")
    parser.add_argument("--loss-mode", default=None, choices=LOSS_MODES,
                        help="Override train_optical.loss.mode")
    parser.add_argument("--lambda-cd", type=float, default=None,
                        help="Override train_optical.loss.lambda_cd")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override train_optical.epochs (also the cosine schedule's T_max)")
    parser.add_argument("--latent-cache", default="use", choices=("use", "refresh", "off"),
                        help="Reuse precomputed latents under <cache_dir>/latents/<ae_run>/")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:1, cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name
    if args.ae_run_name is not None:
        cfg["train_optical"]["ae_run_name"] = args.ae_run_name
    if args.loss_mode is not None:
        cfg["train_optical"]["loss"]["mode"] = args.loss_mode
    if args.lambda_cd is not None:
        cfg["train_optical"]["loss"]["lambda_cd"] = args.lambda_cd
    if args.epochs is not None:
        cfg["train_optical"]["epochs"] = args.epochs
    cfg["device"] = args.device
    cfg["latent_cache"] = args.latent_cache

    import random
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    main(cfg)
