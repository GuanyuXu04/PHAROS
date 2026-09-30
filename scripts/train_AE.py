import os
import gc
import sys
import math
import random
import argparse
import datetime

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pharos.model import PCAutoencoder
from pharos.data import make_dataset, split_indices
from pharos.ddp import (
    is_dist, is_main, ddp_setup, ddp_cleanup, all_reduce_mean,
    select_free_gpus, apply_visible_devices,
)
from pharos.losses import chamfer_distance, repulsion_loss
from pharos.viz import plot_pc_pair


# -------------------- Console log --------------------
class _Tee:
    """Mirror a text stream to a file.

    Teeing the streams rather than wrapping each print() is deliberate: it also
    captures what library code prints (the dataset's cache summary, warnings)
    and the traceback of a run that dies, which is exactly the output you want
    to still have in the morning and exactly what a print() wrapper would miss.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        self._handle.write(data)
        # Flush every write: a run that is killed or crashes should leave a
        # console log that goes right up to the last thing it printed.
        self._handle.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_tee_handle = None


def start_console_log(run_dir: str, path: str = "console_log.txt"):
    """Send everything printed from here on to run_dir/console_log.txt as well.

    Idempotent, so the single-process path can start it early enough to catch
    the GPU-selection banner and train_loop can still start it on a rank that
    was spawned after that point.
    """
    global _tee_handle
    if _tee_handle is not None:
        return
    os.makedirs(run_dir, exist_ok=True)
    _tee_handle = open(os.path.join(run_dir, path), "a", encoding="utf-8", buffering=1)
    _tee_handle.write(f"\n{'=' * 78}\n")
    _tee_handle.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)}\n")
    _tee_handle.write(f"{'=' * 78}\n")
    sys.stdout = _Tee(sys.stdout, _tee_handle)
    sys.stderr = _Tee(sys.stderr, _tee_handle)


def get_lr(step: int, cfg: dict) -> float:
    """
    Learning rate for a training step: linear warmup, then cosine decay.
    """
    tr = cfg["train_ae"]
    base_lr, min_lr, warmup = tr["base_lr"], tr["min_lr"], tr["warmup_steps"]
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if tr["lr_schedule"] != "cosine":
        # plateau mode: ReduceLROnPlateau owns the rate once warmup is over.
        return base_lr
    progress = min(1.0, (step - warmup) / max(1, tr["max_steps"] - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_bn_momentum(step: int, cfg: dict) -> float:
    """
    BatchNorm momentum for a training step, decaying towards a floor.
    """
    tr = cfg["train_ae"]
    samples = step * tr["batch_size"]
    momentum = tr["bn_init_decay"] * (tr["bn_decay_rate"] ** (samples / tr["bn_decay_step"]))
    return max(tr["bn_momentum_min"], momentum)


# SyncBatchNorm derives from _BatchNorm, not from BatchNorm1d/2d, so matching
# the concrete classes would silently skip every BN layer under DDP.
_BN = nn.modules.batchnorm._BatchNorm


def _set_bn_momentum(module: nn.Module, momentum: float):
    if isinstance(module, _BN):
        module.momentum = momentum


@torch.no_grad()
def recompute_bn_stats(model, loader, device, max_batches: int):
    """
    Re-estimate BN running statistics as a plain average over training batches.
    """
    if max_batches <= 0:
        return
    base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    saved = []
    for m in base.modules():
        if isinstance(m, _BN):
            saved.append((m, m.momentum))
            m.reset_running_stats()
            m.momentum = None          # None -> cumulative moving average
    if not saved:
        return
    was_training = base.training
    base.train()
    for i, (pts, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        base(pts.to(device).float())   # SyncBatchNorm still reduces across ranks
    for m, momentum in saved:
        m.momentum = momentum
    base.train(was_training)


_fixed_inp = None
_fixed_tgt = None


@torch.no_grad()
def save_progress_plot(step: int, model, device, out_dir: str, prefix: str = "progress_fixed"):
    if not is_main():
        return
    global _fixed_inp, _fixed_tgt
    assert _fixed_inp is not None
    model.eval()
    pred = model(_fixed_inp).squeeze(0).detach().cpu().numpy()
    tgt  = _fixed_tgt.numpy()
    os.makedirs(out_dir, exist_ok=True)
    plot_pc_pair(
        tgt, pred,
        save_path=os.path.join(out_dir, f"{prefix}_s{step:07d}.png"),
        left_title=f"Original (step {step})",
        right_title="Reconstructed",
    )


# -------------------- Train / Eval --------------------
def train_loop(rank: int, cfg: dict):
    local_rank = rank

    if "LOCAL_RANK" in os.environ:
        local_rank = ddp_setup()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    tr = cfg["train_ae"]
    loss_cfg = tr["loss"]
    out_cfg = cfg["output"]

    run_dir = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    fig_dir = os.path.join(run_dir, "figs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    last_path = os.path.join(ckpt_dir, "last.pth")
    log_path = os.path.join(run_dir, "record_log.txt")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main():
        # No-op when __main__ already started it; the spawn path reaches here
        # in a fresh process where it has not.
        start_console_log(run_dir)

    # In DDP, let rank 0 build any missing cache files first, then barrier so
    # other ranks always load from cache (avoids redundant computation and
    # simultaneous writes to the same cache file).
    if is_dist():
        import torch.distributed as dist
        if local_rank != 0:
            dist.barrier()

    dataset = make_dataset(data_cfg, verbose=is_main())

    if is_dist():
        if local_rank == 0:
            dist.barrier()
    n_total = len(dataset)

    # Not shuffled: neighbouring frames of the same indentation are highly
    # correlated, so each side of the split is a contiguous block (per session
    # by default, see data.split_mode).
    train_idx, val_idx = split_indices(dataset, data_cfg)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    if is_main():
        mode = data_cfg.get("split_mode", "per_session")
        print(f"[Split] {mode}: {len(train_idx)} train / {len(val_idx)} val "
              f"of {n_total} valid samples")
        sid = getattr(dataset, "session_id", None)
        if sid is not None:
            names = dataset.session_names
            sid_v = sid[dataset.valid_indices]
            tr_counts = np.bincount(sid_v[train_idx], minlength=len(names))
            va_counts = np.bincount(sid_v[val_idx], minlength=len(names))
            for i, nm in enumerate(names):
                if tr_counts[i] or va_counts[i]:
                    print(f"          {nm:<26s} train {tr_counts[i]:6d}  val {va_counts[i]:6d}")

    def evenly_spaced(subset, n: int):
        """
        A fixed, evenly spaced subset -- every session in proportion.
        """
        if n <= 0 or n >= len(subset):
            return subset
        return torch.utils.data.Subset(subset, np.linspace(0, len(subset) - 1, n).astype(int).tolist())

    val_eval_set = evenly_spaced(val_set, tr["eval_max_batches"] * tr["batch_size"])
    bn_set = evenly_spaced(train_set, tr["bn_recompute_batches"] * tr["batch_size"])

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_eval_set, shuffle=False) if is_dist() else None
    bn_sampler = DistributedSampler(bn_set, shuffle=False) if is_dist() else None

    # drop_last: with a step budget the loop re-iterates the loader anyway, and
    # 282161 samples at batch 16 leaves a final batch of one, which is a
    # degenerate input for BatchNorm.
    train_loader = DataLoader(
        train_set, batch_size=tr["batch_size"], shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=tr["num_workers"], pin_memory=True, drop_last=True,
    )
    # num_workers=0 for both: unlike the epoch loop this replaces, the training
    # loader's worker pool now stays alive across evaluations (the batch stream
    # never runs out), so any workers here are a third pool on top of it. They
    # would buy nothing -- validation is entirely GPU-bound, measured at 13.1 s
    # with 4 workers against 13.3 s with none -- and cost a process each.
    val_loader = DataLoader(
        val_eval_set, batch_size=tr["batch_size"], shuffle=False,
        sampler=val_sampler, num_workers=0, pin_memory=True, drop_last=False,
    )
    bn_loader = DataLoader(
        bn_set, batch_size=tr["batch_size"], shuffle=False,
        sampler=bn_sampler, num_workers=0, pin_memory=True, drop_last=False,
    ) if tr["bn_recompute_batches"] > 0 else None

    if is_main():
        print(f"[Eval] every {tr['eval_every_steps']} steps on {len(val_eval_set)} of "
              f"{len(val_set)} val frames; BN statistics over {len(bn_set)} train frames")

    model = PCAutoencoder(
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=tr["base_lr"])
    # Only consulted when lr_schedule is "plateau"; cosine is computed from the
    # step directly and needs no scheduler object.
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=tr["scheduler"]["factor"], patience=tr["scheduler"]["patience"]
    )

    # Resume from the most recent step, not the best one: best_model.pth can be
    # many evaluations behind when training is interrupted.
    start_step = 0
    best_val = float("inf")
    evals_without_improve = 0
    resume_path = last_path if os.path.exists(last_path) else best_path
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        prefix = "module."
        state = {(k[len(prefix):] if k.startswith(prefix) else k): v
                 for k, v in ckpt["model_state_dict"].items()}
        # strict: a checkpoint from a different latent_dim/num_points must fail
        # loudly instead of silently loading a fraction of the weights.
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "step" in ckpt:
            start_step = ckpt["step"]
        else:
            # Written before the run was measured in steps, when one epoch was
            # one whole pass over the training set.
            start_step = (ckpt["epoch"] + 1) * max(1, len(train_loader))
            if is_main():
                print(f"WARNING: {os.path.basename(resume_path)} predates step-based training; "
                      f"treating its epoch {ckpt['epoch'] + 1} as step {start_step}. The learning "
                      f"rate resumes from that point on the cosine schedule.")
        # Fall back to val_loss for checkpoints written before best_val was
        # recorded -- for best_model.pth the two are the same thing.
        best_val = ckpt.get("best_val", ckpt["val_loss"])
        evals_without_improve = ckpt.get("evals_without_improve", 0)
        if is_main():
            print(f"Resumed from {os.path.basename(resume_path)} (through step {start_step}, "
                  f"val {ckpt['val_loss']:.4f}, best {best_val:.4f})")

    if is_dist():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, static_graph=False,
        )

    global _fixed_inp, _fixed_tgt
    if is_main():
        item = val_set[42]
        pts = item[0] if isinstance(item, (list, tuple)) else item
        _fixed_tgt = pts.detach().cpu() if isinstance(pts, torch.Tensor) else torch.tensor(pts, dtype=torch.float32)
        _fixed_inp = _fixed_tgt.unsqueeze(0).to(device).float()
    if is_dist():
        import torch.distributed as dist
        dist.barrier()

    if is_main():
        # Append when resuming, or the whole history is lost on every restart.
        with open(log_path, "a" if start_step > 0 else "w", encoding="utf-8") as f:
            f.write(f"Config: {cfg}\n")
            if start_step == 0:
                f.write("step,pass,lr,train_loss,val_loss\n")

    def train_batches():
        """
        Endless stream of training batches.
        """
        pass_idx = start_step // max(1, len(train_loader))
        while True:
            if is_dist():
                train_sampler.set_epoch(pass_idx)
            for batch in train_loader:
                yield batch
            pass_idx += 1

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> float:
        model.eval()
        loss_sum = 0.0
        num_batches = 0
        for pts, _, _ in loader:
            points = pts.to(device).float()
            pred = model(points)
            cd, _ = chamfer_distance(pred, points)
            rep = repulsion_loss(pred, k=loss_cfg["repulsion_k"], h=loss_cfg["repulsion_h"],
                                 subset=loss_cfg.get("repulsion_subset"),
                                 weight=loss_cfg.get("repulsion_weight", 0.0))
            loss_sum += float((cd + rep).item())
            num_batches += 1

        loss_tensor = torch.tensor([loss_sum, num_batches], dtype=torch.float32, device=device)
        all_reduce_mean(loss_tensor)
        return (loss_tensor[0] / torch.clamp_min(loss_tensor[1], 1.0)).item()

    def save_checkpoint(path: str, step: int, val_loss: float):
        state_dict = (model.module.state_dict()
                      if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                      else model.state_dict())
        torch.save({
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val": best_val,
            "evals_without_improve": evals_without_improve,
        }, path)

    steps_per_pass = max(1, len(train_loader))
    max_steps = tr["max_steps"]
    stream = train_batches()
    window_sum, window_n = 0.0, 0     # training loss since the last evaluation
    stop = False

    model.train()
    for step in range(start_step, max_steps):
        lr = get_lr(step, cfg)
        # In plateau mode the scheduler owns the rate once warmup is done, so
        # only write it while warmup is still shaping it.
        if tr["lr_schedule"] == "cosine" or step < tr["warmup_steps"]:
            for g in optimizer.param_groups:
                g["lr"] = lr

        bn_mom = get_bn_momentum(step, cfg)
        base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        base.apply(lambda m: _set_bn_momentum(m, bn_mom))

        pts, _, _ = next(stream)
        points = pts.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        pred = model(points)
        cd, _ = chamfer_distance(pred, points)
        rep = repulsion_loss(pred, k=loss_cfg["repulsion_k"], h=loss_cfg["repulsion_h"],
                             subset=loss_cfg.get("repulsion_subset"),
                             weight=loss_cfg.get("repulsion_weight", 0.0))
        loss = cd + rep
        loss.backward()
        optimizer.step()

        window_sum += float(loss.item())
        window_n += 1

        if is_main() and (step + 1) % tr["log_every_steps"] == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step + 1}/{max_steps} (pass {step // steps_per_pass + 1}) "
                  f"lr {cur_lr:.2e}  bn_mom {bn_mom:.3f}  CD {cd.item():.1f}  Rep {rep.item():.1f}")

        at_eval = (step + 1) % tr["eval_every_steps"] == 0 or (step + 1) == max_steps
        if not at_eval:
            continue

        # -------- evaluation point --------
        window = torch.tensor([window_sum, window_n], dtype=torch.float32, device=device)
        all_reduce_mean(window)
        train_loss = (window[0] / torch.clamp_min(window[1], 1.0)).item()
        window_sum, window_n = 0.0, 0

        # Settle the BN running statistics before they are read: the checkpoint
        # saved below is loaded with .eval() everywhere downstream, so these are
        # the statistics the reported validation loss and every later model see.
        recompute_bn_stats(model, bn_loader, device, tr["bn_recompute_batches"])
        val_loss = evaluate(val_loader)

        if is_main():
            peak = (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[eval] step {step + 1}/{max_steps} (pass {step // steps_per_pass + 1}), "
                  f"lr {cur_lr:.2e}, Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                  f"peak GPU: {peak:.2f} GiB")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step + 1},{step // steps_per_pass + 1},{cur_lr:.6e},"
                        f"{train_loss:.6f},{val_loss:.6f}\n")
            save_progress_plot(step + 1, model, device, fig_dir)

        if tr["lr_schedule"] != "cosine":
            sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            evals_without_improve = 0
            if is_main():
                save_checkpoint(best_path, step + 1, val_loss)
        else:
            evals_without_improve += 1

        # Written at every evaluation so an interrupted run resumes where it
        # stopped rather than falling back to the last step that improved.
        if is_main():
            save_checkpoint(last_path, step + 1, val_loss)

        if evals_without_improve >= tr["early_stop_patience"]:
            if is_main():
                print(f"Early stopping at step {step + 1} "
                      f"({evals_without_improve} evaluations without improvement)")
            stop = True

        model.train()
        if stop:
            break

    if is_main():
        print(f"Best validation loss: {best_val:.4f} (checkpoint: {best_path})")

    ddp_cleanup()


# -------------------- Spawn worker (used by multiprocessing.spawn) --------------------
def _spawn_worker(rank: int, world_size: int, cfg: dict):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    train_loop(rank, cfg)


# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Override output.run_name from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name

    # Start the console log before anything prints, so the GPU-selection banner
    # and any failure during setup land in it too. Non-zero ranks under
    # torchrun skip it: one writer per file keeps the log readable.
    if os.environ.get("RANK", "0") == "0":
        start_console_log(os.path.join(cfg["output"]["base_dir"], cfg["output"]["run_name"]))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    # Decide which GPUs to use BEFORE any torch.cuda.* call so we can still
    # narrow CUDA_VISIBLE_DEVICES (a CUDA context locks visibility).
    gpus_cfg = cfg["train_ae"].get("gpus", "auto")
    min_free_gib = float(cfg["train_ae"].get("min_free_gpu_mem_gib", 20))
    launched_by_torchrun = "LOCAL_RANK" in os.environ

    if gpus_cfg == "auto" and not launched_by_torchrun:
        selected, free = select_free_gpus(min_free_mib=int(min_free_gib * 1024))
        if free:
            free_str = ", ".join(f"cuda:{i}={free[i] / 1024:.1f}GiB" for i in range(len(free)))
            print(f"[GPU] Free memory per visible device: {free_str}")
        if selected and len(selected) < len(free):
            phys = apply_visible_devices(selected)
            print(f"[GPU] Selected physical GPU(s) {','.join(phys)} "
                  f"(>= {min_free_gib:.1f} GiB free); skipping the rest.")
        elif selected:
            print(f"[GPU] All {len(selected)} visible GPU(s) meet the {min_free_gib:.1f} GiB threshold.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_available = torch.cuda.device_count()
    n_gpus = n_available if gpus_cfg == "auto" else min(int(gpus_cfg), n_available)

    if "LOCAL_RANK" in os.environ:
        # Already launched by torchrun — let it manage ranks
        train_loop(rank=int(os.environ["LOCAL_RANK"]), cfg=cfg)
    elif n_gpus > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        torch.multiprocessing.spawn(_spawn_worker, args=(n_gpus, cfg), nprocs=n_gpus)
    else:
        train_loop(rank=0, cfg=cfg)
