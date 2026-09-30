import re
import sys
import zipfile
import argparse
from pathlib import Path

import numpy as np
import numpy.lib.format as npformat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pharos.geometry import DEPTH_UNIT_Z_THRESHOLD_MM

_FRAME_RE = re.compile(r"frame_(\d+)_depth\.npy$", re.IGNORECASE)

# Keys of meta.npz that are copied as meta_<key>, and the dtype they are stored with.
META_KEYS = {
    "roi_origin_xy": np.int32,
    "roi_mask": np.uint8,
    "roi_polygon_xy": np.int32,
    "depth_scale_m_per_unit": np.float64,
    "depth_K_full": np.float64,
    "depth_size_full_wh": np.int32,
    "depth_K_roi": np.float64,
    "depth_size_roi_wh": np.int32,
    "depth_dist_coeffs": np.float64,
    "depth_dist_model": np.int32,
    "cfg_emitter": np.int32,
    "cfg_laser_power": np.float32,
    "cfg_manual_exposure": np.int32,
    "cfg_exposure_us": np.float32,
    "cfg_gain": np.float32,
    "cfg_global_time_enabled": np.int32,
    "cfg_depth_units_target": np.float32,
}


def find_pairs(session_dir: Path, expected_cols: int, roi_shape):
    """(frame id, depth path, optical path) for every usable frame, in order.

    A frame is usable when both files exist, the depth crop has the ROI shape
    and the optical row has the expected number of columns. Everything else is
    reported and left out.
    """
    found = []
    for p in session_dir.glob("frame_*_depth.npy"):
        m = _FRAME_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()

    pairs, skipped = [], []
    for fid, dpath in found:
        opath = dpath.with_name(dpath.name.replace("_depth.npy", "_optical.npy"))
        if not opath.exists():
            skipped.append((fid, "no optical file"))
            continue
        if np.load(dpath, mmap_mode="r").shape != tuple(roi_shape):
            skipped.append((fid, "depth shape differs from the ROI"))
            continue
        if np.load(opath, mmap_mode="r").shape != (expected_cols,):
            skipped.append((fid, f"optical row is not {expected_cols} values"))
            continue
        pairs.append((fid, dpath, opath))
    return pairs, skipped


def write_array(zf: zipfile.ZipFile, name: str, arr: np.ndarray):
    """Write a whole array as <name>.npy inside the archive."""
    with zf.open(name + ".npy", "w", force_zip64=True) as fh:
        npformat.write_array(fh, np.asarray(arr), version=(2, 0), allow_pickle=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session_dir", help="Directory written by collect_data.py (frame_*_*.npy and meta.npz)")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--compresslevel", type=int, default=6, help="zlib level, 1 (fast) to 9 (small)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    session_dir = Path(args.session_dir)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    meta_path = session_dir / "meta.npz"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found: a session needs the meta.npz written by collect_data.py")

    meta = np.load(meta_path, allow_pickle=True)
    missing = [k for k in list(META_KEYS) + ["labels"] if k not in meta.files]
    if missing:
        raise SystemExit(f"{meta_path} lacks {missing}")
    roi_mask = meta["roi_mask"].astype(np.uint8)
    labels = np.array([str(x) for x in meta["labels"]])
    scale_m = float(meta["depth_scale_m_per_unit"])
    # Raw-unit equivalent of the millimetre threshold, e.g. 150 mm / 0.5 mm = 300.
    raw_threshold = DEPTH_UNIT_Z_THRESHOLD_MM / (scale_m * 1000.0)

    pairs, skipped = find_pairs(session_dir, len(labels), roi_mask.shape)
    for fid, why in skipped:
        print(f"  skip frame {fid}: {why}")
    if not pairs:
        raise SystemExit(f"no usable frames in {session_dir}")
    n = len(pairs)
    H, W = roi_mask.shape
    print(f"{n} frames ({len(skipped)} skipped), depth {H}x{W}, {len(labels)} optical columns")

    out.parent.mkdir(parents=True, exist_ok=True)
    in_roi = roi_mask.astype(bool)
    optical = np.empty((n, len(labels)), np.int32)
    mtime = np.empty(n, np.float64)
    unit_scale = np.ones(n, np.float32)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=args.compresslevel,
                         allowZip64=True) as zf:
        # The depth stack is streamed frame by frame: it is by far the largest member.
        with zf.open("depth.npy", "w", force_zip64=True) as fh:
            npformat.write_array_header_2_0(fh, {
                "descr": npformat.dtype_to_descr(np.dtype(np.uint16)),
                "fortran_order": False, "shape": (n, H, W),
            })
            for i, (fid, dpath, opath) in enumerate(pairs):
                depth = np.load(dpath).astype(np.uint16)
                fh.write(np.ascontiguousarray(depth).tobytes())
                optical[i] = np.load(opath)
                mtime[i] = dpath.stat().st_mtime
                valid = depth[in_roi]
                valid = valid[valid > 0]
                if valid.size and float(np.median(valid)) < raw_threshold:
                    unit_scale[i] = 2.0
                if (i + 1) % 2000 == 0 or i + 1 == n:
                    print(f"  packed {i + 1}/{n}", end="\r", flush=True)
        print()

        write_array(zf, "optical", optical)
        write_array(zf, "frame_ids", np.arange(n, dtype=np.int64))
        write_array(zf, "mtime_unix", mtime)
        write_array(zf, "depth_unit_scale", unit_scale)
        for key, dtype in META_KEYS.items():
            write_array(zf, "meta_" + key, np.asarray(meta[key], dtype=dtype))
            if key == "depth_dist_model":
                write_array(zf, "meta_labels", labels)

    n_half = int((unit_scale == 2.0).sum())
    print(f"Wrote {out}  ({out.stat().st_size / 2**20:.1f} MiB), "
          f"{n_half} frame(s) flagged as half depth unit")


if __name__ == "__main__":
    main()
