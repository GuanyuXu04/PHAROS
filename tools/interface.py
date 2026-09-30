import sys
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import serial
import torch
import open3d as o3d
from matplotlib import cm  # fast enough for per-frame colormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pharos.data import make_optical_transform
from pharos.model import ShapeNet

# ---------------- Config ----------------
PORT = "COM6"
BAUD = 2000000
TIMEOUT = 0.1
EXPECTED_COLUMNS = 218
MODEL_PATH = "best_combined.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERBOSE = False
COLUMNS = np.concatenate([np.arange(9 + 7 * i, 15 + 7 * i) for i in range(30)])

CONTOUR = False

# ---------------- Model -----------------
model = ShapeNet(
    pd_in_features=180, latent_dim=512, num_points=4096,  # must match the model section of config/config.yaml
    tnet1=False, tnet2=False
).to(DEVICE).eval()
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.pd2latent.load_state_dict(ckpt["pd2latent_state_dict"])
model.decoder.load_state_dict(ckpt["decoder_state_dict"])
# Reproduce the input normalisation the model was trained with (stored in the checkpoint).
optical_in = make_optical_transform(ckpt)
torch.backends.cudnn.benchmark = True  # helps if shapes are stable

# ---------------- Shared buffer ----------------
q_raw  = deque(maxlen=1)   # raw serial line (bytes or str)
q_pred = deque(maxlen=1)   # latest point cloud (N,3) float32
running = True

# ======== Serial thread: block reads + manual framing, so a partial line never yields a malformed frame ========
def serial_reader(ser):
    buf = bytearray()
    while running:
        try:
            chunk = ser.read(4096)   # more robust than readline(), which can split a line on timeout
            if not chunk:
                continue
            buf.extend(chunk)
            # Split out complete lines on '\n'
            while True:
                nl = buf.find(b'\n')
                if nl < 0:
                    break
                line = bytes(buf[:nl]).strip()  # drop '\r\n'
                del buf[:nl+1]
                if line:
                    print(line)
                    q_raw.append(line)
        except serial.SerialException:
            break

def infer_worker(model, device, COLUMNS, EXPECTED_COLUMNS=218):
    # Only allocate a pinned host tensor when running on CUDA
    use_cuda = (device.type == "cuda")
    host_in = torch.empty(180, dtype=torch.float32, pin_memory=True) if use_cuda else None
    dev_in  = torch.empty((1, 180), dtype=torch.float32, device=device)

    with torch.inference_mode():
        while running:
            if not q_raw:
                time.sleep(0.001)
                continue

            # Take the newest line and drop older ones (no queue build-up)
            last = None
            while q_raw:
                last = q_raw.pop()

            # ===== Timer #2: whole frame in infer_worker (parse -> copy -> inference -> post-process -> enqueue) =====
            t_frame0 = time.perf_counter()

            # Parse
            try:
                s = last.decode('utf-8', errors='replace')
                parts = s.split(',')
                if len(parts) != EXPECTED_COLUMNS:
                    continue
                vals = np.fromstring(s, sep=',', dtype=np.float32)
            except Exception:
                continue

            optical = vals[COLUMNS]  # select the 180 LED/PD channels
            if optical.shape[0] != 180 or not np.isfinite(optical).all():
                continue

            # numpy -> torch
            src = optical_in(optical)  # CPU float32 tensor, normalised as in training
            if use_cuda:
                host_in.copy_(src, non_blocking=True)        # host -> pinned host
                dev_in[0].copy_(host_in, non_blocking=True)  # host -> device
            else:
                dev_in[0].copy_(src)                          # CPU-only path

            # ===== Timer #1: the forward pass only (pred = model(dev_in)) =====
            t_model0 = time.perf_counter()
            pred = model(dev_in)        # [1,N,3] or [N,3]
            t_model1 = time.perf_counter()

            if pred.dim() == 3:
                pred = pred[0]

            # Give the visualiser thread its own buffer (the underlying storage may be reused)
            pts = pred.detach().cpu().numpy().astype(np.float32, copy=False).copy()
            q_pred.append(pts)          # keep only the latest

            t_frame1 = time.perf_counter()

            # Report both timings
            model_ms = (t_model1 - t_model0) * 1000.0
            frame_ms = (t_frame1 - t_frame0) * 1000.0

            if VERBOSE:
                z = pts[:, 2]
                print(
                    f"model forward: {model_ms:.2f} ms | "
                    f"infer_worker total: {frame_ms:.2f} ms | "
                    f"pred {pts.shape}; z:[{z.min():.2f},{z.max():.2f}]"
                )
            else:
                print(f"model forward: {model_ms:.2f} ms | infer_worker total: {frame_ms:.2f} ms")


def point_cloud_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Predicted Point Cloud", width=960, height=720, visible=True)

    pcd = o3d.geometry.PointCloud()
    inited = False
    saved_cam = None

    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.array([0, 0, 0])

    zmin, zmax = 0, 20
    alpha = 0.05
    viridis = cm.get_cmap("viridis")

    try:
        while running:
            # Drop stale frames, keep only the latest
            if len(q_pred) > 1:
                while len(q_pred) > 1:
                    q_pred.popleft()

            if q_pred:
                pts = q_pred.popleft()  # (N,3) float32
                if pts.size == 0 or not np.isfinite(pts).all():
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.005)
                    continue

                z = -pts[:, 2]
                zmin = (1 - alpha) * zmin + alpha * float(np.min(z))
                zmax = (1 - alpha) * zmax + alpha * float(np.max(z))
                if zmax <= zmin:
                    zmax = zmin + 1e-6
                nz = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
                col = viridis(nz)[:, :3]

                if not inited:
                    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))
                    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64, copy=False))
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    ctr = vis.get_view_control()
                    center = pts.mean(axis=0).tolist()        # look at the cloud center
                    ctr.set_lookat(center)

                    ctr.set_front([0.0, 0.0, -1.0])           # camera pointing along -Z
                    ctr.set_up([0.0, -1.0,  0.0])              # +Y is "up" on screen

                    ctr.set_zoom(0.8)                         # 0..1, smaller = farther
                    saved_cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                    inited = True
                else:
                    if saved_cam is not None:
                        vis.get_view_control().convert_from_pinhole_camera_parameters(saved_cam, allow_arbitrary=True)
                    # Update the geometry
                    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))
                    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64, copy=False))
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            if inited:
                saved_cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

            # Sleep once per loop iteration so the UI does not stall
            time.sleep(0.005)
    finally:
        vis.destroy_window()

def contour_visualizer():
    """
    Real-time top-view contour visualization (lines only).
    Uses global `CONTOUR_LEVELS` to control contour line density.
    No UI interactions; optimized for steady refresh.

    Expected globals:
      - q_pred: deque with latest predicted point cloud (N,3) float32
      - running: bool loop flag
      - CONTOUR_LEVELS: int, number of contour levels (optional; default=12)
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np

    levels_count = int(globals().get("CONTOUR_LEVELS", 12))

    plt.ion()
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=110)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # Smooth z-range to reduce flicker (EMA)
    zmin, zmax = -10.0, 30.0
    alpha = 0.05

    contour_set = None

    try:
        while running:
            # Keep only the latest frame
            if len(q_pred) > 1:
                while len(q_pred) > 1:
                    q_pred.popleft()

            if q_pred:
                pts = q_pred.popleft()
                if pts.size == 0 or not np.isfinite(pts).all():
                    plt.pause(0.004)
                    continue

                x = pts[:, 0]
                y = pts[:, 1]
                z = pts[:, 2]

                # Smooth the z range over time
                zmin = (1 - alpha) * zmin + alpha * float(np.min(z))
                zmax = (1 - alpha) * zmax + alpha * float(np.max(z))
                if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
                    zmin, zmax = -10.0, 30.0

                # Read the global contour density (it may be updated externally)
                lc = globals().get("CONTOUR_LEVELS", levels_count)
                try:
                    levels_count = int(lc)
                except Exception:
                    levels_count = 12
                levels_count = max(2, min(256, levels_count))

                # Rebuild the contours (redrawing every frame and removing the previous one is the simplest reliable way)
                tri = mtri.Triangulation(x, y)
                if contour_set is not None:
                    for coll in contour_set.collections:
                        coll.remove()

                levels = np.linspace(zmin, zmax, levels_count)
                contour_set = ax.tricontour(
                    tri, z, levels=levels,
                    colors="#E35959", linewidths=1.5, antialiased=False
                )

            fig.canvas.draw_idle()
            plt.pause(0.004)  # brief sleep so the UI can refresh
    finally:
        try:
            plt.ioff()
            plt.close(fig)
        except Exception:
            pass


def run_visualizer():
    if CONTOUR:
        contour_visualizer()
    else:
        point_cloud_visualizer()

# ---------------- Main ----------------
if __name__ == "__main__":
    ser = serial.Serial(PORT, BAUD, timeout=0)  # non-blocking
    ser.reset_input_buffer()

    t_rx  = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    t_inf = threading.Thread(target=infer_worker, args=(model, DEVICE, COLUMNS), daemon=True)
    t_rx.start(); t_inf.start()

    try:
        run_visualizer()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        try:
            ser.close()
        except:
            pass
        t_rx.join(timeout=1)
        t_inf.join(timeout=1)
