"""Build a memory-mappable training cache from packed session .npz archives.

The projection only ever uses the depth values on the fixed sampling grid, so
the cache stores depth already subsampled at one stride. At stride 3 that is
77x77 uint16 = 11.9 kB per frame instead of 230x230 = 103 kB, which makes the
whole indentation training set about 4 GB: small enough to memory-map and let the
page cache hold it, and only three files on disk instead of one pair per frame.

A compressed .npz cannot be seeked into, so members are decompressed as a
stream, one frame at a time, and written straight into a memory-mapped output.
Memory stays constant no matter how large a session is.

Written to --out:
    depth_sub.npy   (N, h, w) uint16   depth on the sampling grid
    optical.npy     (N, C)    int32    raw optical rows
    index.npz                          frame ids, session ids, mtimes, per-frame
                                       bounding boxes, depth-unit scales, grid
                                       geometry and camera intrinsics

Usage:
    python tools/build_train_cache.py \
        --npz "data/indentation/white_train_S*.npz" \
        --out data/indentation/cache/train_stride3 --stride 3
    python tools/build_train_cache.py \
        --npz "data/indentation/white_test_*.npz" \
        --out data/indentation/cache/test_stride3 --stride 3
"""
import re
import glob
import json
import time
import zipfile
import argparse
from pathlib import Path

import numpy as np
import numpy.lib.format as npformat

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pharos.geometry import (
    subsample_indices, points_from_subgrid, DEPTH_UNIT_Z_THRESHOLD_MM,
)


# ----------------------------- streaming reads -----------------------------

def _member_header(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as fh:
        version = npformat.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = npformat.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = npformat.read_array_header_2_0(fh)
        else:
            raise ValueError(f"{member}: unsupported .npy version {version}")
        if fortran:
            raise ValueError(f"{member}: Fortran-ordered arrays are not supported")
        return shape, dtype


def _read_exact(fh, n: int) -> bytes:
    """ZipExtFile.read may return fewer bytes than asked for."""
    chunks, remaining = [], n
    while remaining > 0:
        b = fh.read(remaining)
        if not b:
            raise EOFError(f"stream ended {remaining} bytes early")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


def iter_member(zf: zipfile.ZipFile, member: str):
    """Yield rows of a stacked array member one at a time, in constant memory."""
    with zf.open(member) as fh:
        version = npformat.read_magic(fh)
        reader = npformat.read_array_header_1_0 if version == (1, 0) else npformat.read_array_header_2_0
        shape, fortran, dtype = reader(fh)
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        for _ in range(shape[0]):
            yield np.frombuffer(_read_exact(fh, row_bytes), dtype=dtype).reshape(row_shape)


def read_small(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as fh:
        return npformat.read_array(fh, allow_pickle=False)


# ----------------------------- build -----------------------------

def session_key(path: Path):
    """Sort S2 before S10, and keep other names alphabetical."""
    m = re.search(r"_S(\d+)$", path.stem)
    return (0, int(m.group(1)), "") if m else (1, 0, path.stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, nargs="+",
                   help="Glob(s) matching the packed session archives")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=float, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    paths = sorted({Path(q) for pat in args.npz for q in glob.glob(pat)}, key=session_key)
    if not paths:
        raise SystemExit(f"no archives matched {args.npz}")
    stride = int(args.stride) if float(args.stride).is_integer() else float(args.stride)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    depth_path, opt_path, idx_path = out / "depth_sub.npy", out / "optical.npy", out / "index.npz"
    if idx_path.exists() and not args.overwrite:
        raise SystemExit(f"{idx_path} exists (use --overwrite)")

    # Pass 1: headers and metadata only, nothing is decompressed.
    print(f"{len(paths)} archive(s), stride {stride}")
    totals, metas = [], []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            dshape, ddtype = _member_header(zf, "depth.npy")
            oshape, odtype = _member_header(zf, "optical.npy")
            if dshape[0] != oshape[0]:
                raise ValueError(f"{path.name}: {dshape[0]} depth vs {oshape[0]} optical frames")
            metas.append({
                "roi_mask": read_small(zf, "meta_roi_mask.npy").astype(bool),
                "K": read_small(zf, "meta_depth_K_roi.npy").astype(float),
                "scale_m": float(read_small(zf, "meta_depth_scale_m_per_unit.npy")),
                "labels": read_small(zf, "meta_labels.npy"),
            })
            totals.append((path, dshape, ddtype, oshape, odtype))
            print(f"  {path.name:34s} {dshape[0]:7d} frames  depth{dshape[1:]} optical{oshape[1:]}")

    ref = metas[0]
    for path_meta, (path, *_rest) in zip(metas[1:], totals[1:]):
        if not (np.array_equal(path_meta["roi_mask"], ref["roi_mask"])
                and np.allclose(path_meta["K"], ref["K"])
                and np.isclose(path_meta["scale_m"], ref["scale_m"])):
            raise ValueError(
                f"{path.name}: camera geometry differs from {totals[0][0].name}; "
                "sessions with different intrinsics need separate caches"
            )

    N = sum(t[1][0] for t in totals)
    H, W = totals[0][1][1], totals[0][1][2]
    C = totals[0][3][1]
    u_idx, v_idx = subsample_indices(H, W, stride)
    h, w = len(v_idx), len(u_idx)
    mask_sub = ref["roi_mask"][np.ix_(v_idx, u_idx)]
    uu, vv = np.meshgrid(u_idx.astype(np.float32), v_idx.astype(np.float32), indexing="xy")
    n_pts = int(mask_sub.sum())
    print(f"\n{N} frames total; grid {h}x{w}, {n_pts} points per cloud")
    print(f"depth cache: {N * h * w * 2 / 2**30:.2f} GiB, optical: {N * C * 4 / 2**30:.2f} GiB")

    depth_out = npformat.open_memmap(depth_path, mode="w+", dtype=np.uint16, shape=(N, h, w))
    opt_out = npformat.open_memmap(opt_path, mode="w+", dtype=totals[0][4], shape=(N, C))

    frame_ids = np.zeros(N, np.int64)
    session_id = np.zeros(N, np.int32)
    mtime_unix = np.zeros(N, np.float64)
    unit_scale = np.zeros(N, np.float32)
    bbox = np.zeros((N, 6), np.float32)
    session_names = [pp.stem for pp, *_ in totals]

    k, t0 = 0, time.time()
    for si, (path, dshape, ddtype, oshape, odtype) in enumerate(totals):
        with zipfile.ZipFile(path) as zf:
            fids = read_small(zf, "frame_ids.npy")
            mts = read_small(zf, "mtime_unix.npy")
            stored_scale = read_small(zf, "depth_unit_scale.npy")

            n = dshape[0]
            frame_ids[k:k + n] = fids
            session_id[k:k + n] = si
            mtime_unix[k:k + n] = mts

            mismatches = 0
            for j, frame in enumerate(iter_member(zf, "depth.npy")):
                sub = frame[np.ix_(v_idx, u_idx)]
                depth_out[k + j] = sub
                pts = points_from_subgrid(sub, mask_sub, uu, vv, ref["K"], ref["scale_m"])
                s = 2.0 if float(np.median(pts[:, 2])) < DEPTH_UNIT_Z_THRESHOLD_MM else 1.0
                if s != 1.0:
                    pts *= np.float32(s)
                unit_scale[k + j] = s
                mismatches += (s != float(stored_scale[j]))
                bbox[k + j] = (pts[:, 0].min(), pts[:, 0].max(),
                               pts[:, 1].min(), pts[:, 1].max(),
                               pts[:, 2].min(), pts[:, 2].max())
                if (j + 1) % 2000 == 0 or j + 1 == n:
                    el = time.time() - t0
                    print(f"    {path.stem} {j + 1}/{n}  total {k + j + 1}/{N}  "
                          f"{el / 60:5.1f} min  ({(k + j + 1) / max(el, 1e-9):.0f} fr/s)",
                          end="\r", flush=True)
            print()
            if mismatches:
                print(f"    WARNING: {mismatches} frames where the cached depth-unit scale "
                      f"disagrees with the archive")

            for j, row in enumerate(iter_member(zf, "optical.npy")):
                opt_out[k + j] = row
            k += n

    depth_out.flush(); opt_out.flush()
    del depth_out, opt_out

    np.savez(
        idx_path,
        frame_ids=frame_ids, session_id=session_id,
        session_names=np.array(session_names), mtime_unix=mtime_unix,
        depth_unit_scale=unit_scale, bbox=bbox,
        stride=np.float64(stride), mask_sub=mask_sub, uu=uu, vv=vv,
        u_idx=u_idx, v_idx=v_idx,
        depth_K_roi=ref["K"], depth_scale_m_per_unit=np.float64(ref["scale_m"]),
        labels=ref["labels"], full_shape=np.array([H, W], np.int32),
    )

    summary = {
        "n_frames": int(N), "n_points": n_pts, "stride": stride,
        "grid": [h, w], "sessions": session_names,
        "session_counts": {t[0].stem: int(t[1][0]) for t in totals},
        "n_half_unit_frames": int((unit_scale == 2.0).sum()),
        "built_from": [str(pp) for pp, *_ in totals],
    }
    (out / "cache_info.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}  ({time.time() - t0:.0f} s)")
    for f in (depth_path, opt_path, idx_path):
        print(f"  {f.name:18s} {f.stat().st_size / 2**30:7.2f} GiB")


if __name__ == "__main__":
    main()
