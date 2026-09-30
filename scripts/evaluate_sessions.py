"""Evaluate a trained ShapeNet on independently collected test sessions.

Each session of the test cache is one acquisition (one indenter, one
location, one depth) and is scored separately, because the frames inside a
session are repeated measurements of the same configuration and are therefore
not independent samples. Reported per session:

  * frame counts (files found, valid after the bounding-box filter)
  * deformation magnitude dz = z_max - z_min of the ground-truth cloud
  * symmetric Chamfer distance d_CD (Equation 2 of the manuscript), which is
    the SUM of the two one-sided mean nearest-neighbour distances; both
    one-sided means are reported separately so the scale is unambiguous
  * max nearest-neighbour distance (nnd_max), as annotated in Figure 7C
  * F-score at the requested thresholds

Outputs, under {base_dir}/{run_name}/eval_sessions/:
  per_frame_{session}.csv    one row per frame
  summary.csv / summary.md   one row per session plus a pooled row
  samples/{session}_{tag}.npz  best / median / worst frames for figures

Usage:
  python scripts/evaluate_sessions.py --run-name indentation
  python scripts/evaluate_sessions.py --run-name indentation \
      --sessions white_test_U white_test_Circle --save-samples
"""
import os
import csv
import json
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pharos.data import make_optical_transform, CachedWaveguideDataset
from pharos.model import ShapeNet


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--run-name", default=None, help="Override output.run_name")
    p.add_argument("--cache", default=None,
                   help="Test cache dir (default: data.test_cache_dir from the config)")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Session names in the test cache to evaluate (default: all)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to best_combined.pth (default: inside the run dir)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Chamfer is memory-heavy: B x 4096 x 5929 floats per batch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--taus", nargs="*", type=float, default=[1.0, 2.0],
                   help="F-score thresholds in mm")
    p.add_argument("--save-samples", action="store_true",
                   help="Dump best/median/worst frames per session as .npz")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


# ----------------------------- Metrics -----------------------------

@torch.no_grad()
def per_frame_metrics(pred: torch.Tensor, gt: torch.Tensor, taus):
    """pred (B,N,3), gt (B,M,3) -> dict of (B,) tensors, all in mm."""
    D = torch.cdist(pred, gt, p=2)
    d_pg = D.min(dim=2).values          # (B, N) predicted -> ground truth
    d_gp = D.min(dim=1).values          # (B, M) ground truth -> predicted

    out = {
        "cd_pred_to_gt": d_pg.mean(dim=1),
        "cd_gt_to_pred": d_gp.mean(dim=1),
        "nnd_max": d_pg.max(dim=1).values,
    }
    out["cd"] = out["cd_pred_to_gt"] + out["cd_gt_to_pred"]
    for tau in taus:
        precision = (d_pg < tau).float().mean(dim=1)
        recall = (d_gp < tau).float().mean(dim=1)
        out[f"fscore@{tau:g}"] = 2 * precision * recall / (precision + recall + 1e-8)
    return out


def describe(v: np.ndarray) -> dict:
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "min": float(v.min()),
        "q1": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "q3": float(np.percentile(v, 75)),
        "max": float(v.max()),
    }


# ----------------------------- Evaluation -----------------------------

def evaluate_session(model, optical_in, ds, name, n_files, device, batch_size, taus,
                     save_samples, out_dir):
    n_valid = len(ds)
    if n_valid == 0:
        print(f"[{name}] no valid frames (of {n_files} files) -- skipped")
        return None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    cols = {}
    dz_list = []

    for pts_gt, optical, _ in loader:
        pts_gt = pts_gt.to(device).float()
        pred = model(optical_in(optical).to(device))
        m = per_frame_metrics(pred, pts_gt, taus)
        for k, v in m.items():
            cols.setdefault(k, []).append(v.cpu())
        z = pts_gt[:, :, 2]
        dz_list.append((z.max(dim=1).values - z.min(dim=1).values).cpu())
        done = sum(len(c) for c in cols["cd"])
        print(f"  [{name}] {done}/{n_valid}", end="\r", flush=True)

    cols = {k: torch.cat(v).numpy() for k, v in cols.items()}
    cols["dz"] = torch.cat(dz_list).numpy()
    cols["frame_id"] = np.asarray([ds.frame_ids[int(i)] for i in ds.valid_indices], dtype=np.int64)

    # per-frame CSV
    keys = ["frame_id", "dz", "cd", "cd_pred_to_gt", "cd_gt_to_pred", "nnd_max"] + \
           [f"fscore@{t:g}" for t in taus]
    with open(os.path.join(out_dir, f"per_frame_{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in range(n_valid):
            w.writerow([cols[k][r] for k in keys])

    if save_samples:
        sample_dir = os.path.join(out_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        order = np.argsort(cols["cd"])
        picks = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}
        for tag, idx in picks.items():
            pts_gt, optical, _ = ds[int(idx)]
            with torch.no_grad():
                pred = model(optical_in(optical).unsqueeze(0).to(device))
            np.savez_compressed(
                os.path.join(sample_dir, f"{name}_{tag}.npz"),
                gt=pts_gt, pred=pred.squeeze(0).cpu().numpy(),
                cd=cols["cd"][idx], dz=cols["dz"][idx],
                nnd_max=cols["nnd_max"][idx], frame_id=cols["frame_id"][idx],
            )

    row = {
        "session": name,
        "n_files": n_files,
        "n_valid": n_valid,
        "dz_mean": float(cols["dz"].mean()),
        "dz_std": float(cols["dz"].std(ddof=0)),
        "dz_max": float(cols["dz"].max()),
    }
    cd = describe(cols["cd"])
    row.update({f"cd_{k}": v for k, v in cd.items()})
    row["cd_pred_to_gt_mean"] = float(cols["cd_pred_to_gt"].mean())
    row["cd_gt_to_pred_mean"] = float(cols["cd_gt_to_pred"].mean())
    row["nnd_max_mean"] = float(cols["nnd_max"].mean())
    row["nnd_max_max"] = float(cols["nnd_max"].max())
    for t in taus:
        row[f"fscore@{t:g}"] = float(cols[f"fscore@{t:g}"].mean())

    print(f"  [{name}] n={n_valid:5d} (of {n_files})  dz={row['dz_mean']:5.2f}+-{row['dz_std']:.2f} mm"
          f"  CD mean={cd['mean']:.3f} median={cd['median']:.3f} std={cd['std']:.3f} mm")
    return row, cols["cd"]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name

    model_cfg, out_cfg = cfg["model"], cfg["output"]
    run_dir = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    ckpt_path = args.checkpoint or os.path.join(run_dir, "best_combined.pth")
    out_dir = args.out_dir or os.path.join(run_dir, "eval_sessions")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = ShapeNet(
        pd_in_features=model_cfg["optical_dim"],
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.pd2latent.load_state_dict(ckpt["pd2latent_state_dict"])
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()
    # The same input transform training used, taken from the checkpoint.
    optical_in = make_optical_transform(ckpt)
    print(f"Model: {ckpt_path}  (optical_dim={model_cfg['optical_dim']}, "
          f"latent_dim={model_cfg['latent_dim']}, num_points={model_cfg['num_points']}, "
          f"input={'optical_norm' if ckpt.get('input_norm') else 'raw optical'}, "
          f"loss_mode={ckpt.get('loss_mode', 'mse')})")
    print(f"Device: {device}\n")

    cache_dir = args.cache or cfg["data"].get("test_cache_dir")
    if not cache_dir:
        raise SystemExit("give --cache, or set data.test_cache_dir in the config")
    idx = np.load(os.path.join(cache_dir, "index.npz"), allow_pickle=False)
    all_names = [str(x) for x in idx["session_names"]]
    counts = {n: int((idx["session_id"] == i).sum()) for i, n in enumerate(all_names)}
    names = args.sessions or all_names
    print("Cache: %s\nSessions: %s\n" % (cache_dir, ", ".join(names)))

    def build(name):
        return CachedWaveguideDataset(cache_dir, sessions=[name], verbose=False), counts[name]

    rows, pooled = [], []
    for name in names:
        ds, n_files = build(name)
        if len(ds) == 0:
            print(f"[{name}] no valid frames (of {n_files}) -- skipped")
            continue
        res = evaluate_session(
            model, optical_in, ds, name, n_files,
            device, args.batch_size, args.taus, args.save_samples, out_dir,
        )
        if res is not None:
            rows.append(res[0])
            pooled.append(res[1])

    if not rows:
        print("No sessions evaluated.")
        return

    # Pooled row: every frame of every session weighted equally, plus the
    # unweighted mean over sessions, which is the fairer headline number
    # because session sizes differ and frames within a session are correlated.
    allcd = np.concatenate(pooled)
    pooled_row = {"session": "ALL (frame-weighted)", "n_files": sum(r["n_files"] for r in rows),
                  "n_valid": int(allcd.size)}
    pooled_row.update({f"cd_{k}": v for k, v in describe(allcd).items()})
    rows.append(pooled_row)

    sess_means = np.array([r["cd_mean"] for r in rows[:-1]])
    rows.append({"session": "ALL (session-weighted)", "n_valid": len(sess_means),
                 "cd_mean": float(sess_means.mean()), "cd_std": float(sess_means.std(ddof=1))})

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "session", k))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md_cols = ["session", "n_valid", "dz_mean", "dz_std", "cd_mean", "cd_median",
               "cd_std", "cd_min", "cd_q1", "cd_q3", "cd_max", "nnd_max_mean"] + \
              [f"fscore@{t:g}" for t in args.taus]
    lines = ["| " + " | ".join(md_cols) + " |",
             "|" + "|".join(["---"] * len(md_cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            for c in md_cols) + " |")
    md = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")

    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({"checkpoint": ckpt_path, "cache": args.cache or cfg["data"].get("test_cache_dir"),
                   "sessions": names, "stride": cfg["data"]["stride"],
                   "model": model_cfg, "taus": args.taus}, f, indent=2)

    print("\n" + md)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
