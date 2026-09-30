import os
import random
import argparse
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from pharos.model import ShapeNet
from pharos.data import make_dataset, make_optical_transform
from pharos.metrics import F_score_batched, chamfer_distance_batched, F_score
from pharos.viz import plot_fscore_vs_tau, plot_pc_pair
from torch.utils.data import DataLoader, Subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-name", default=None, help="Override output.run_name")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:1, cpu")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Evenly spaced frames from the test set; 0 evaluates all of them")
    parser.add_argument("--n-figs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name
    cfg["device"] = args.device

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg   = cfg["output"]

    run_dir    = os.path.join(out_cfg["base_dir"], out_cfg["run_name"])
    model_path = os.path.join(run_dir, "best_combined.pth")
    out_dir    = os.path.join(run_dir, "eval")
    fig_dir    = os.path.join(out_dir, "figures")
    fscore_dir = os.path.join(out_dir, "fscore")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(fscore_dir, exist_ok=True)

    # "cuda" (no ordinal) means the current device, so CUDA_VISIBLE_DEVICES
    # still selects the GPU on a multi-GPU box without hardcoding an index.
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    print(f"Using device {device}")
    model = ShapeNet(
        pd_in_features=model_cfg["optical_dim"],
        latent_dim=model_cfg["latent_dim"],
        num_points=model_cfg["num_points"],
        tnet1=model_cfg["tnet1"],
        tnet2=model_cfg["tnet2"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.pd2latent.load_state_dict(checkpoint["pd2latent_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()
    # The same input transform training used, taken from the checkpoint: the
    # normalisation constants belong to the training cache, and a test cache's
    # own optical_norm is normalised by different statistics.
    optical_in = make_optical_transform(checkpoint)
    print(f"Model loaded from {model_path} "
          f"(input={'optical_norm' if checkpoint.get('input_norm') else 'raw optical'}, "
          f"loss_mode={checkpoint.get('loss_mode', 'mse')})")

    # The held-out sessions, not a random slice of the training frames: the
    # per-frame .npy backend data_cfg["path"] points at no longer exists, and a
    # random split would put neighbouring frames of the same indentation on
    # both sides anyway.
    test_set = make_dataset(data_cfg, cache_dir=data_cfg["test_cache_dir"])
    n_eval = min(args.n_samples, len(test_set)) if args.n_samples > 0 else len(test_set)
    test_subset = Subset(test_set, np.linspace(0, len(test_set) - 1, n_eval).astype(int).tolist())
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Test set: {len(test_subset)} of {len(test_set)} frames")

    # ------------------------------
    # Inference on test set
    # ------------------------------
    all_gt, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            points_gt, batch_optical, _ = batch
            points_pred = model(optical_in(batch_optical).to(device)).cpu().numpy()
            all_gt.append(points_gt.numpy())
            all_pred.append(points_pred)

    all_gt   = np.concatenate(all_gt, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # ------------------------------
    # Visualize ground truth vs reconstructed
    # ------------------------------
    TAU_VIZ = 1.5
    sample_indices = random.sample(range(len(test_set)), min(args.n_figs, len(test_set)))
    for idx in sample_indices:
        origin_pc, optical, _ = test_set[idx]
        with torch.no_grad():
            pred_pc = model(optical_in(optical).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        f_score = F_score(origin_pc, pred_pc, tau=TAU_VIZ)
        print(f"Sample {idx}: F-score @ {TAU_VIZ} mm = {f_score:.4f}")
        plot_pc_pair(
            origin_pc, pred_pc,
            save_path=os.path.join(fig_dir, f"PR_vs_GT_{idx}.svg"),
            left_title=f"Ground Truth Point Cloud (Sample {idx})",
            right_title=f"Reconstructed Point Cloud (Sample {idx})\nF-score @ {TAU_VIZ:.1f}mm: {f_score:.4f}",
        )

    # ------------------------------
    # F-score vs tau
    # ------------------------------
    tau_range = np.arange(0, 3.1, 0.1)
    f_scores, results_lines = [], []
    with torch.no_grad():
        for tau in tau_range:
            per_item = []
            for points_gt, batch_optical, _ in test_loader:
                points_pred_t = model(optical_in(batch_optical).to(device))
                fs_b = F_score_batched(points_pred_t, points_gt.to(device).float(), tau)
                per_item.append(fs_b.detach().cpu())
            avg_f = float(torch.cat(per_item).numpy().mean())
            f_scores.append(avg_f)
            print(f"Tau: {tau:.2f} mm, Average F-score: {avg_f:.4f}")
            results_lines.append(f"{tau:.2f}, {avg_f:.4f}")

    with open(os.path.join(fscore_dir, "F_score_data_record.txt"), "w") as f:
        f.write("tau, average_fscore\n")
        f.writelines(line + "\n" for line in results_lines)

    plot_fscore_vs_tau(
        tau_range, f_scores,
        save_path=os.path.join(fscore_dir, "average_fscore_vs_tau.png"),
    )

    # ------------------------------
    # Chamfer distance
    # ------------------------------
    cd_list = []
    with torch.no_grad():
        for points_gt, batch_optical, _ in test_loader:
            points_pred = model(optical_in(batch_optical).to(device))
            cd_list.append(chamfer_distance_batched(points_pred, points_gt.to(device).float()).detach().cpu())

    cd_all = torch.cat(cd_list).numpy()
    print(f"[Chamfer] mean = {cd_all.mean():.6f}, std = {cd_all.std(ddof=0):.6f}")

    with open(os.path.join(out_dir, "hist_data_chamfer.txt"), "w") as f:
        f.writelines(f"{v:.6f}\n" for v in cd_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cd_all, bins=40)
    ax.set_title("Chamfer Distance Histogram (Euclidean)")
    ax.set_xlabel("Chamfer distance")
    ax.set_ylabel("Count")
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_chamfer.png"), dpi=200)
    plt.close(fig)
