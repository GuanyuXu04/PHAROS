import os
import argparse

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from pharos.model import PCAutoencoder
from pharos.data import make_dataset, split_indices
from pharos.losses import repulsion_loss
from pharos.viz import plot_fscore_vs_tau, plot_pc_pair


@torch.no_grad()
def nn_distances(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 1024):
    """Nearest-neighbour distances in mm, both directions, for a batch.

    Returns (d_pg, d_gp): d_pg[b, n] is the distance from predicted point n to
    the closest ground-truth point, d_gp[b, m] the other way round. Every
    metric below (Chamfer, F-score at any tau) is a reduction of these two
    vectors, so they are computed once per batch and reused -- the old script
    re-ran the model and a fresh cKDTree for every one of the 31 tau values.

    Centred on the ground-truth centroid and computed in float32 for the same
    reason as `pharos.losses.chamfer_distance`: on raw millimetre coordinates
    the |a|^2 + |b|^2 - 2<a,b> expansion cancels badly enough to flip near-tie
    assignments. Chunked along the predicted cloud so peak memory is
    (B, chunk, M) rather than (B, N, M).
    """
    B, N, _ = pred.shape
    M = gt.shape[1]

    mu = gt.float().mean(dim=1, keepdim=True)
    a_all = pred.float() - mu
    b = gt.float() - mu
    b_sq = b.pow(2).sum(-1).unsqueeze(1)                       # (B, 1, M)
    bt = b.transpose(1, 2)

    d_pg = a_all.new_empty((B, N))
    best_gp = a_all.new_full((B, M), float("inf"))
    for start in range(0, N, chunk):
        a = a_all[:, start:start + chunk]
        d2 = a.pow(2).sum(-1).unsqueeze(2) + b_sq - 2.0 * torch.bmm(a, bt)
        d2.clamp_(min=0.0)
        d_pg[:, start:start + chunk] = d2.min(dim=2).values
        best_gp = torch.minimum(best_gp, d2.min(dim=1).values)

    return d_pg.sqrt(), best_gp.sqrt()


def build_eval_set(cfg, split: str):
    """The frames to evaluate, plus a label and the session id of each frame."""
    data_cfg = cfg["data"]

    if split == "test":
        # A separate cache of held-out shapes (Circle, Finger, Square, Triangle,
        # U), not a slice of the training sessions.
        dataset = make_dataset(data_cfg, cache_dir=data_cfg["test_cache_dir"])
        positions = np.arange(len(dataset))
    else:
        dataset = make_dataset(data_cfg)
        train_idx, val_idx = split_indices(dataset, data_cfg)
        positions = np.asarray(train_idx if split == "train" else val_idx)

    return dataset, positions


def evenly_spaced(positions: np.ndarray, n: int) -> np.ndarray:
    """A fixed, evenly spaced subset -- every session in proportion.

    Deterministic on purpose: the old `np.random.choice` gave a different
    sample every run, so two evaluations of the same checkpoint disagreed.
    """
    if n <= 0 or n >= len(positions):
        return positions
    return positions[np.linspace(0, len(positions) - 1, n).astype(int)]


def main(cfg: dict, args):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]
    loss_cfg = cfg["train_ae"]["loss"]

    run_dir = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    model_path = os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pth")
    out_dir = os.path.join(run_dir, f"eval_ae_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    model = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # Checkpoints written under DDP before the unwrapping in save_checkpoint
    # still carry the "module." prefix.
    prefix = "module."
    state = {(k[len(prefix):] if k.startswith(prefix) else k): v
             for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    # BN running statistics are settled by train_AE before every checkpoint is
    # written, so eval() reads the same statistics the reported val loss saw.
    model.eval()
    print(f"Model loaded from {model_path} "
          f"(step {checkpoint.get('step', '?')}, val_loss {checkpoint.get('val_loss', float('nan')):.4f})")

    dataset, positions = build_eval_set(cfg, args.split)
    positions = evenly_spaced(positions, args.n_samples)
    eval_set = Subset(dataset, positions.tolist())
    print(f"[{args.split}] evaluating {len(eval_set)} of {len(dataset)} frames")

    sid = getattr(dataset, "session_id", None)
    if sid is not None:
        frame_session = sid[dataset.valid_indices][positions]
        session_names = dataset.session_names
    else:
        frame_session = None

    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    taus = torch.arange(0, 3.05, 0.1)
    cd_mm, cd_sq_sum, rep_all = [], [], []
    prec, rec = [], []

    torch.manual_seed(cfg["seed"])   # the repulsion term samples a point subset
    t = taus.to(device)
    with torch.no_grad():
        for pts, _, _ in loader:
            points = pts.to(device, non_blocking=True).float()
            pred = model(points)
            d_pg, d_gp = nn_distances(pred, points, chunk=args.chunk)

            # Reported Chamfer: mean over points in each direction, in mm. Same
            # definition as pharos.metrics.chamfer_distance_batched and as the
            # numbers in scripts/evaluate.py, so the AE floor and the full
            # pipeline are directly comparable.
            cd_mm.append((d_pg.mean(1) + d_gp.mean(1)).cpu())
            # Training loss uses squared distances summed over points; kept so
            # this script's output can be checked against record_log.txt.
            cd_sq_sum.append((d_pg.pow(2).sum(1) + d_gp.pow(2).sum(1)).cpu())
            rep_all.append(repulsion_loss(pred, k=loss_cfg["repulsion_k"], h=loss_cfg["repulsion_h"],
                                          subset=loss_cfg.get("repulsion_subset"),
                                          weight=loss_cfg.get("repulsion_weight", 0.0)).cpu())

            prec.append((d_pg.unsqueeze(2) < t).float().mean(1).cpu())   # (B, T)
            rec.append((d_gp.unsqueeze(2) < t).float().mean(1).cpu())

    cd_mm = torch.cat(cd_mm)
    cd_sq_sum = torch.cat(cd_sq_sum)
    rep_mean = float(torch.stack(rep_all).mean())
    prec = torch.cat(prec)
    rec = torch.cat(rec)
    f_per_sample = 2 * prec * rec / (prec + rec + 1e-8)             # (S, T)
    f_scores = f_per_sample.mean(0)                                 # (T,)

    tau_idx = int(torch.argmin((taus - args.tau_report).abs()))
    lines = []

    def emit(s: str):
        print(s)
        lines.append(s)

    emit("")
    emit(f"=== {args.split} set, {len(cd_mm)} frames ===")
    emit(f"Chamfer (mm, mean of both directions): mean {cd_mm.mean():.4f}  "
         f"std {cd_mm.std(unbiased=False):.4f}  median {cd_mm.median():.4f}  max {cd_mm.max():.4f}")
    emit(f"F-score @ {float(taus[tau_idx]):.1f} mm: {f_scores[tau_idx]:.4f}")
    emit(f"Training-form loss (summed squared CD + weighted repulsion): "
         f"{cd_sq_sum.mean() + rep_mean:.1f}  (CD {cd_sq_sum.mean():.1f} + Rep {rep_mean:.1f})")

    if frame_session is not None:
        emit("")
        emit(f"{'session':<26s} {'frames':>7s} {'CD (mm)':>9s} {f'F@{args.tau_report:.1f}':>8s}")
        for i, name in enumerate(session_names):
            m = torch.from_numpy(frame_session == i)
            if not m.any():
                continue
            emit(f"{name:<26s} {int(m.sum()):7d} {cd_mm[m].mean():9.4f} "
                 f"{f_per_sample[m, tau_idx].mean():8.4f}")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {model_path}\n")
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(out_dir, "F_score_data_record.txt"), "w", encoding="utf-8") as f:
        f.write("tau, average_fscore\n")
        for t, s in zip(taus.tolist(), f_scores.tolist()):
            f.write(f"{t:.2f}, {s:.4f}\n")
    plot_fscore_vs_tau(
        taus.numpy(), f_scores.numpy(),
        save_path=os.path.join(out_dir, "average_fscore_vs_tau.png"),
        title=f"AE F-score vs Tau ({args.split})",
    )

    np.savetxt(os.path.join(out_dir, "hist_data_chamfer.txt"), cd_mm.numpy(), fmt="%.6f")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cd_mm.numpy(), bins=60)
    ax.set_xlabel("Chamfer distance (mm)")
    ax.set_ylabel("Frames")
    ax.set_title(f"AE reconstruction Chamfer ({args.split}), mean {cd_mm.mean():.3f} mm")
    fig.savefig(os.path.join(out_dir, "hist_chamfer.png"), dpi=200)
    plt.close(fig)

    # Qualitative check: evenly spaced frames, so every session is represented.
    viz_positions = evenly_spaced(positions, args.n_figs)
    for frame in viz_positions:
        pc, _, _ = dataset[int(frame)]
        pc = np.asarray(pc, dtype=np.float32)
        pc_t = torch.tensor(pc).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(pc_t)
            d_pg, d_gp = nn_distances(pred, pc_t, chunk=args.chunk)
        cd_one = float(d_pg.mean() + d_gp.mean())
        tau = args.tau_report
        p = float((d_pg < tau).float().mean())
        r = float((d_gp < tau).float().mean())
        f_one = 2 * p * r / (p + r + 1e-8)
        print(f"Frame {frame}: CD = {cd_one:.4f} mm, F-score @ {tau:.1f} mm = {f_one:.4f}")
        plot_pc_pair(
            pc, pred.squeeze(0).cpu().numpy(),
            save_path=os.path.join(out_dir, f"reconstruction_vs_original_{frame}.png"),
            left_title=f"Original (frame {frame})",
            right_title=f"Reconstructed (frame {frame})\nCD {cd_one:.3f} mm, "
                        f"F-score @ {tau:.1f}mm: {f_one:.4f}",
        )

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-name", default=None, help="Override output.run_name")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"),
                        help="test: the held-out shape sessions; val/train: the split train_AE used")
    parser.add_argument("--ckpt", default="best_model", choices=("best_model", "last"))
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Evenly spaced subset of the split; 0 evaluates every frame")
    parser.add_argument("--n-figs", type=int, default=10)
    parser.add_argument("--tau-report", type=float, default=1.0,
                        help="Tau (mm) for the F-score printed in the summary")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=1024,
                        help="Predicted points per distance-matrix chunk; lower it if VRAM is tight")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    main(cfg, args)
