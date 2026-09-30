import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

from pharos.config import HARDWARE, FEATURES, HardwareConfig
from pharos.geometry import (
    DEPTH_UNIT_Z_THRESHOLD_MM, DEPTH_SCALE_FIX, scale_fix_for, apply_depth_unit_fix,
    contiguous_split, per_session_split, subsample_indices, points_from_subgrid,
    _load_meta, _depth_to_points_mm_fixedgrid,
)


def split_indices(dataset, data_cfg: dict):
    """Train/val positions for a dataset, using the config's split_mode.

    "per_session" (the default when the backend knows which session each frame
    came from) applies train_split inside every session; "contiguous" applies it
    once across the whole dataset.
    """
    mode = data_cfg.get("split_mode", "per_session")
    train_split = data_cfg["train_split"]
    sid = getattr(dataset, "session_id", None)
    if mode == "per_session":
        if sid is None:
            raise ValueError(
                "split_mode 'per_session' needs a cache backend that records session ids; "
                "use split_mode: contiguous with the per-frame .npy backend"
            )
        return per_session_split(sid[dataset.valid_indices], train_split)
    if mode == "contiguous":
        return contiguous_split(len(dataset), train_split)
    raise ValueError(f"unknown split_mode: {mode!r}")

# Bump when the point-cloud computation changes, so cached valid indices and
# optical min/max from an earlier version are not silently reused.
CACHE_VERSION = "v2"


# ---------- Dataset ----------
class WaveguideDataset(Dataset):
    def __init__(
        self,
        folder: str,
        stride: int = 1,
        *,
        leds: Optional[Sequence[int]] = None,
        pds: Optional[Sequence[int]] = None,
        hardware: Optional[HardwareConfig] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        max_frame_id: Optional[int] = None,
        depth_unit_fix: str = "auto",
        verbose: bool = True,
    ):
        self.folder = Path(folder)
        self.roi_mask, self.scale_m, self.K = _load_meta(self.folder / "meta.npz")

        self.hardware = hardware if hardware is not None else HARDWARE
        self.leds = list(leds) if leds is not None else list(FEATURES.leds)
        self.pds = list(pds) if pds is not None else list(FEATURES.pds)
        self._validate_selection()

        self.stride = stride
        self.x_range = x_range if x_range is not None else self.hardware.x_range
        self.y_range = y_range if y_range is not None else self.hardware.y_range
        self.z_range = z_range if z_range is not None else self.hardware.z_range
        self.max_frame_id = max_frame_id
        self.depth_unit_fix = depth_unit_fix
        self.verbose = verbose

        # Pair depth + optical
        self.depth_files = sorted(self.folder.glob("frame_*_depth.npy"))
        self.frame_ids = [int(re.search(r"frame_(\d+)_depth", f.name).group(1)) for f in self.depth_files]

        # Frame ids at or above max_frame_id are dropped entirely (e.g. to hold
        # out a trailing acquisition session).
        if self.max_frame_id is not None:
            keep = [k for k, fid in enumerate(self.frame_ids) if fid < self.max_frame_id]
            dropped = len(self.depth_files) - len(keep)
            self.depth_files = [self.depth_files[k] for k in keep]
            self.frame_ids = [self.frame_ids[k] for k in keep]
            self._log(f"Dropped {dropped} frames with id >= {self.max_frame_id}")

        self.optical_files = [self.folder / f.stem.replace("_depth","_optical.npy") for f in self.depth_files]

        # Optical rows: [timestamp, L0P0, L0P1, ..., L0P_M, L1P0, ..., L_N P_M]
        # where N = num_leds, M = num_pds. So index of (L_i, P_j) = 1 + i*(M+1) + j.
        pd_stride = self.hardware.num_pds + 1
        self._cols = np.array([1 + led * pd_stride + pd for led in self.leds for pd in self.pds], dtype=np.int64)

        # Cache paths. Valid indices depend on the stride and on the frame-id
        # cut-off, and the optical statistics depend on the selected channels,
        # so both are encoded in the file name.
        tag = f"{CACHE_VERSION}_s{self.stride}_max{self.max_frame_id if self.max_frame_id is not None else 'all'}"
        self.valid_idx_path = self.folder / f"valid_idx_{tag}.txt"
        self.optical_minmax_path = self.folder / f"optical_min_max_{tag}_L{len(self.leds)}P{len(self.pds)}.npy"

        self.valid_indices = self._load_or_build_valid_indices()
        self.col_min, self.col_max = self._load_or_build_minmax()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _validate_selection(self):
        for led in self.leds:
            if not 0 <= led <= self.hardware.num_leds:
                raise ValueError(f"LED index {led} out of range [0, {self.hardware.num_leds}]")
        for pd in self.pds:
            if not 0 <= pd <= self.hardware.num_pds:
                raise ValueError(f"PD index {pd} out of range [0, {self.hardware.num_pds}]")

    def _load_or_build_valid_indices(self):
        valid_indices = []
        if self.valid_idx_path.exists():
            # Use cached indices (one integer per line)
            try:
                cached = [int(line.strip()) for line in self.valid_idx_path.read_text().splitlines() if line.strip()]
            except Exception:
                cached = []

            self._log(f"Loaded {len(cached)} cached valid indices")

            # Keep only indices that still make sense (file exists, in-range)
            n = len(self.depth_files)
            for i in cached:
                if 0 <= i < n and self.optical_files[i].exists():
                    valid_indices.append(i)
            # If cache gave us something usable, stop here
            if len(valid_indices) > 0:
                return valid_indices

        # Otherwise, compute valid indices from scratch
        self._log("No cached valid indices found. Building valid indices from scratch...")
        for i, ofile in enumerate(self.optical_files):
            if not ofile.exists():
                continue
            depth = np.load(self.depth_files[i], mmap_mode="r")
            pts = _depth_to_points_mm_fixedgrid(depth, self.roi_mask, self.K, self.scale_m, stride=self.stride)
            ########################################################
            pts = apply_depth_unit_fix(pts, self.frame_ids[i], self.depth_unit_fix)
            ########################################################

            if pts.shape[0] == 0:
                continue
            x_valid = np.all((self.x_range[0] <= pts[:, 0]) & (pts[:, 0] <= self.x_range[1]))
            y_valid = np.all((self.y_range[0] <= pts[:, 1]) & (pts[:, 1] <= self.y_range[1]))
            z_valid = np.all((self.z_range[0] <= pts[:, 2]) & (pts[:, 2] <= self.z_range[1]))
            if x_valid and y_valid and z_valid:
                valid_indices.append(i)
        
        # Cache valid indices
        try:
            self.valid_idx_path.write_text("\n".join(str(i) for i in valid_indices))
        except Exception:
            pass
        return valid_indices
    
    def _load_or_build_minmax(self):
        if self.optical_minmax_path.exists():
            try:
                arr = np.load(self.optical_minmax_path)
                if isinstance(arr, np.ndarray) and arr.shape == (2, len(self._cols)):
                    self._log("Loaded cached optical min/max")
                    col_min, col_max = arr[0].astype(np.float32), arr[1].astype(np.float32)
                    return col_min, col_max
            except Exception:
                pass
        
        self._log("No cached optical min/max found. Building from scratch...")
        n_features = len(self._cols)
        if len(self.valid_indices) == 0:
            col_min = np.zeros((n_features,), dtype=np.float32)
            col_max = np.ones((n_features,), dtype=np.float32)
            return col_min, col_max

        col_min = np.full((n_features,), np.inf, dtype=np.float32)
        col_max = np.full((n_features,), -np.inf, dtype=np.float32)
        for i in self.valid_indices:
            x = np.load(self.optical_files[i]).astype(np.float32)
            vec = np.asarray(x, dtype=np.float32)[self._cols]

            m = np.isfinite(vec)
            if not np.all(m):
                vmin = np.where(m, vec, np.inf)
                vmax = np.where(m, vec, -np.inf)
            else:
                vmin, vmax = vec, vec
            
            col_min = np.minimum(col_min, vmin)
            col_max = np.maximum(col_max, vmax)
        
        col_min[~np.isfinite(col_min)] = 0.0
        col_max[~np.isfinite(col_max)] = 1.0

        col_min = col_min.astype(np.float32)
        col_max = col_max.astype(np.float32)

        # Cache min/max
        try:
            np.save(self.optical_minmax_path, np.stack([col_min, col_max], axis=0))
        except Exception:
            pass

        return col_min, col_max
    

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        i = self.valid_indices[idx]
        depth = np.load(self.depth_files[i], mmap_mode="r")
        optical_full = np.load(self.optical_files[i])
        optical = optical_full[self._cols].astype(np.float32)

        pts = _depth_to_points_mm_fixedgrid(depth, self.roi_mask, self.K, self.scale_m,stride=self.stride)
        ########################################################
        pts = apply_depth_unit_fix(pts, self.frame_ids[i], self.depth_unit_fix)
        ########################################################
        denom = (self.col_max - self.col_min).astype(np.float32)
        safe_denom = np.where(denom > 0.0, denom, 1.0).astype(np.float32)
        optical_norm = (optical - self.col_min.astype(np.float32)) / safe_denom
        optical_norm = np.clip(optical_norm, 0.0, 1.0)

        return pts.astype(np.float32), optical, optical_norm


# ---------- Cache-backed dataset ----------
class CachedWaveguideDataset(Dataset):
    """Dataset backed by a stride-specific cache built by tools/build_train_cache.py.

    The cache holds the depth already subsampled to the sampling grid, which is
    all the projection ever uses, so it is ~9x smaller than the full-resolution
    frames at stride 3 and can be memory-mapped. Only a handful of files exist
    on disk instead of one pair per frame, and the returned tensors are
    bit-identical to the file-backed WaveguideDataset.
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        leds: Optional[Sequence[int]] = None,
        pds: Optional[Sequence[int]] = None,
        hardware: Optional[HardwareConfig] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        max_frame_id: Optional[int] = None,
        sessions: Optional[Sequence[str]] = None,
        depth_unit_fix: str = "auto",
        verbose: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        idx_path = self.cache_dir / "index.npz"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"{idx_path} not found. Build it with tools/build_train_cache.py"
            )
        Z = np.load(idx_path, allow_pickle=False)

        self.hardware = hardware if hardware is not None else HARDWARE
        self.leds = list(leds) if leds is not None else list(FEATURES.leds)
        self.pds = list(pds) if pds is not None else list(FEATURES.pds)
        self._validate_selection()

        self.stride = float(Z["stride"])
        self.scale_m = float(Z["depth_scale_m_per_unit"])
        self.K = Z["depth_K_roi"].astype(float)
        self.mask_sub = Z["mask_sub"].astype(bool)
        self.uu = Z["uu"].astype(np.float32)
        self.vv = Z["vv"].astype(np.float32)
        self.frame_ids = Z["frame_ids"]
        self.session_id = Z["session_id"]
        self.session_names = [str(s) for s in Z["session_names"]]
        self.mtime_unix = Z["mtime_unix"]
        self.depth_unit_scale = Z["depth_unit_scale"]
        self.bbox = Z["bbox"]                      # (N, 6): xmin,xmax,ymin,ymax,zmin,zmax
        self.depth_unit_fix = depth_unit_fix
        self.verbose = verbose

        self.x_range = x_range if x_range is not None else self.hardware.x_range
        self.y_range = y_range if y_range is not None else self.hardware.y_range
        self.z_range = z_range if z_range is not None else self.hardware.z_range

        self._open_arrays()

        pd_stride = self.hardware.num_pds + 1
        self._cols = np.array(
            [1 + led * pd_stride + pd for led in self.leds for pd in self.pds], dtype=np.int64
        )

        # The bounding box of every frame is cached, so the validity filter is a
        # vector comparison instead of a pass over the whole dataset.
        keep = (
            (self.bbox[:, 0] >= self.x_range[0]) & (self.bbox[:, 1] <= self.x_range[1]) &
            (self.bbox[:, 2] >= self.y_range[0]) & (self.bbox[:, 3] <= self.y_range[1]) &
            (self.bbox[:, 4] >= self.z_range[0]) & (self.bbox[:, 5] <= self.z_range[1])
        )
        n_range = int((~keep).sum())
        if max_frame_id is not None:
            keep &= self.frame_ids < max_frame_id
        if sessions is not None:
            wanted = {self.session_names.index(s) for s in sessions}
            keep &= np.isin(self.session_id, list(wanted))

        self.valid_indices = np.nonzero(keep)[0]
        if self.verbose:
            print(f"Cache {self.cache_dir}: {len(self.frame_ids)} frames, "
                  f"{len(self.valid_indices)} kept "
                  f"({n_range} outside the bounding box, "
                  f"{len(self.frame_ids) - len(self.valid_indices) - n_range} filtered by "
                  f"max_frame_id/sessions)")

        self.col_min, self.col_max = self._load_or_build_minmax()

    _validate_selection = WaveguideDataset._validate_selection

    def _open_arrays(self):
        self.depth = np.load(self.cache_dir / "depth_sub.npy", mmap_mode="r")
        self.optical_all = np.load(self.cache_dir / "optical.npy", mmap_mode="r")

    def __getstate__(self):
        """Pickle without the memory maps, and reopen them on the other side.

        numpy serialises a memmap by value, so mmap_mode="r" survives only
        until the object crosses a process boundary. Under the forkserver start
        method every DataLoader worker is sent a pickle of the dataset, which
        made each worker materialise the 3.9 GiB depth cache as private
        anonymous memory -- four workers were enough to exhaust a 30 GiB
        machine. Reopening the maps in the worker keeps them shared, evictable
        page cache instead, and shrinks the pickle from 4.2 GiB to a few KiB.
        """
        state = self.__dict__.copy()
        state["depth"] = None
        state["optical_all"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_arrays()

    def _load_or_build_minmax(self):
        """Channel min/max over the kept frames, cached next to the arrays."""
        tag = f"L{len(self.leds)}P{len(self.pds)}_n{len(self.valid_indices)}"
        path = self.cache_dir / f"optical_min_max_{tag}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape == (2, len(self._cols)):
                return arr[0].astype(np.float32), arr[1].astype(np.float32)
        sel = np.asarray(self.optical_all[:, self._cols][self.valid_indices], dtype=np.float32)
        col_min = np.nanmin(sel, axis=0).astype(np.float32)
        col_max = np.nanmax(sel, axis=0).astype(np.float32)
        col_min[~np.isfinite(col_min)] = 0.0
        col_max[~np.isfinite(col_max)] = 1.0
        try:
            np.save(path, np.stack([col_min, col_max], axis=0))
        except Exception:
            pass
        return col_min, col_max

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        i = int(self.valid_indices[idx])
        pts = points_from_subgrid(
            self.depth[i], self.mask_sub, self.uu, self.vv, self.K, self.scale_m
        )
        ########################################################
        pts = apply_depth_unit_fix(pts, int(self.frame_ids[i]), self.depth_unit_fix)
        ########################################################
        optical = np.asarray(self.optical_all[i], dtype=np.float32)[self._cols]

        denom = (self.col_max - self.col_min).astype(np.float32)
        safe_denom = np.where(denom > 0.0, denom, 1.0).astype(np.float32)
        optical_norm = (optical - self.col_min.astype(np.float32)) / safe_denom
        optical_norm = np.clip(optical_norm, 0.0, 1.0)

        return pts.astype(np.float32), optical, optical_norm


def make_dataset(data_cfg: dict, *, verbose: bool = True, **overrides):
    """Build the dataset described by the ``data`` section of the config.

    ``cache_dir`` selects the cache backend (a few large files); ``path``
    selects the original per-frame .npy folder. Overrides are forwarded, so a
    caller can point at one test session without editing the config.
    """
    cache_dir = overrides.pop("cache_dir", data_cfg.get("cache_dir"))
    if cache_dir:
        return CachedWaveguideDataset(
            cache_dir,
            max_frame_id=overrides.pop("max_frame_id", data_cfg.get("max_frame_id")),
            sessions=overrides.pop("sessions", data_cfg.get("sessions")),
            verbose=verbose, **overrides,
        )
    return WaveguideDataset(
        overrides.pop("path", data_cfg["path"]),
        stride=overrides.pop("stride", data_cfg["stride"]),
        max_frame_id=overrides.pop("max_frame_id", data_cfg.get("max_frame_id")),
        verbose=verbose, **overrides,
    )


def make_optical_transform(ckpt: dict):
    """Reproduce the optical input transform a trained model was fitted with.

    best_combined.pth records whether train_optical.py fed the min-max
    normalised optical vector, together with the constants it used. Those
    constants have to travel with the model: every dataset computes its own
    min/max over the frames it happens to hold, so the `optical_norm` a test
    cache returns is normalised by test statistics and is *not* the quantity
    the model saw in training. Feeding it would be a silent distribution shift.

    Returns a callable mapping the raw optical vector (tensor or array, last
    axis = selected channels) to the model input. Checkpoints written before
    the flag existed were trained on raw readings, and get the identity.
    """
    if not ckpt.get("input_norm", False):
        return lambda optical: torch.as_tensor(optical, dtype=torch.float32)

    lo = torch.as_tensor(ckpt["optical_min"], dtype=torch.float32)
    hi = torch.as_tensor(ckpt["optical_max"], dtype=torch.float32)
    denom = torch.where(hi - lo > 0, hi - lo, torch.ones_like(hi))

    def transform(optical):
        x = torch.as_tensor(optical, dtype=torch.float32)
        return ((x - lo.to(x.device)) / denom.to(x.device)).clamp_(0.0, 1.0)

    return transform
