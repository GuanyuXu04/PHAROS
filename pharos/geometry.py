import numpy as np
from pathlib import Path
from typing import List, Tuple
from scipy.ndimage import gaussian_filter

DEPTH_UNIT_Z_THRESHOLD_MM = 150.0

# Frame-id ranges whose depth was recorded in the wrong unit, used when mode="ranges".
# The default "auto" mode detects the same frames from the point cloud itself.
DEPTH_SCALE_FIX: Tuple[Tuple[int, int, float], ...] = ((25717, 157785, 2.0),)

# Bump when the point-cloud computation changes, so cached valid indices and
# optical min/max from an earlier version are not silently reused.
CACHE_VERSION = "v2"


def scale_fix_for(frame_id: int) -> float:
    for lo, hi, s in DEPTH_SCALE_FIX:
        if lo <= frame_id < hi:
            return s
    return 1.0


def apply_depth_unit_fix(pts: np.ndarray, frame_id: int, mode: str = "auto") -> np.ndarray:
    """Rescale a point cloud recorded with the wrong depth unit. See above."""
    if mode == "none" or pts.shape[0] == 0:
        return pts
    if mode == "auto":
        if float(np.median(pts[:, 2])) < DEPTH_UNIT_Z_THRESHOLD_MM:
            pts *= np.float32(2.0)
        return pts
    if mode == "ranges":
        s = scale_fix_for(frame_id)
        if s != 1.0:
            pts *= np.float32(s)
        return pts
    raise ValueError(f"unknown depth-unit fix mode: {mode!r}")


def contiguous_split(n_total: int, train_split: float) -> Tuple[List[int], List[int]]:
    """Split sample positions into two contiguous blocks.

    Frames are acquired at 30 Hz during continuous indentations, so neighbouring
    samples are strongly correlated. A contiguous split keeps whole stretches of
    the recording on one side of the boundary instead of interleaving them.
    """
    n_train = int(train_split * n_total)
    return list(range(n_train)), list(range(n_train, n_total))


def per_session_split(session_ids, train_split: float) -> Tuple[List[int], List[int]]:
    """Within each session, the first train_split fraction trains, the rest validates.

    Applying the fraction per session rather than once over the whole dataset
    keeps every session represented on both sides, instead of letting the tail
    of the recording become the entire validation set. Each side is still a
    contiguous block within its session, so correlated neighbouring frames are
    not interleaved across the two sets.
    """
    session_ids = np.asarray(session_ids)
    train_idx, val_idx = [], []
    for s in np.unique(session_ids):
        pos = np.nonzero(session_ids == s)[0]      # ascending, i.e. acquisition order
        k = int(train_split * len(pos))
        train_idx.extend(pos[:k].tolist())
        val_idx.extend(pos[k:].tolist())
    return sorted(train_idx), sorted(val_idx)


# ---------- helpers ----------
def _load_meta(meta_path: str):
    Z = np.load(meta_path, allow_pickle=True)
    roi_mask = Z["roi_mask"].astype(bool)
    scale_m  = float(Z["depth_scale_m_per_unit"])

    if "depth_K_roi" in Z:
        K = Z["depth_K_roi"].astype(float)
    else:
        if "depth_K_full" not in Z or "roi_origin_xy" not in Z:
            raise KeyError("Missing intrinsics in meta.npz")
        K_full = Z["depth_K_full"].astype(float)
        x0, y0 = Z["roi_origin_xy"]
        K = K_full.copy()
        K[0, 2] -= float(x0)
        K[1, 2] -= float(y0)
    return roi_mask, scale_m, K


def subsample_indices(h: int, w: int, stride):
    """Column/row indices of the fixed sampling grid (integer or float stride)."""
    if isinstance(stride, (float, np.floating)) and not float(stride).is_integer():
        # target size = round(original / stride)
        tw = int(round(w / float(stride)))
        th = int(round(h / float(stride)))
        # evenly spaced integer indices via linspace -> round (endpoint=False avoids duplication)
        u_idx = np.round(np.linspace(0, w - 1, num=tw, endpoint=False)).astype(np.int32)
        v_idx = np.round(np.linspace(0, h - 1, num=th, endpoint=False)).astype(np.int32)
    else:
        s = int(stride)
        u_idx = np.arange(0, w, s, dtype=np.int32)
        v_idx = np.arange(0, h, s, dtype=np.int32)
    return u_idx, v_idx


def points_from_subgrid(depth_sub, mask_sub, uu, vv, K, scale_m):
    """Project an already-subsampled depth patch to metric points.

    Both dataset backends call this, so a cached subsampled depth patch and the
    corresponding full-resolution frame produce bit-identical point clouds.
    """
    depth = np.asarray(depth_sub, dtype=np.float32)
    m = mask_sub

    # Gaussian smoothing
    ksize, sigma = 11, 1.2
    truncate = ((ksize - 1) / 2) / sigma  # solve for truncate so kernel matches size
    depth = gaussian_filter(depth, sigma=sigma, truncate=truncate)
    uu = uu[m]; vv = vv[m]; z_m = depth[m] * scale_m

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x_m = (uu - cx) * z_m / fx
    y_m = (vv - cy) * z_m / fy
    pts_mm = np.column_stack([x_m, y_m, z_m]).astype(np.float32) * 1000.0

    if pts_mm.shape[0] > 0:
        pts_mm[:, 0] -= pts_mm[0, 0]
        pts_mm[:, 1] -= pts_mm[0, 1]
    
    #print(pts_mm)
    return pts_mm


def _depth_to_points_mm_fixedgrid(depth_u16, mask, K, scale_m, stride=2):
    """Subsample a full-resolution depth crop, then project it."""
    h, w = depth_u16.shape
    u_idx, v_idx = subsample_indices(h, w, stride)
    uu, vv = np.meshgrid(u_idx.astype(np.float32), v_idx.astype(np.float32), indexing='xy')
    return points_from_subgrid(
        depth_u16[np.ix_(v_idx, u_idx)], mask[np.ix_(v_idx, u_idx)], uu, vv, K, scale_m
    )


