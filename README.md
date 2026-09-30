# Pharos

Pharos reconstructs the 3D shape of a soft optical waveguide membrane from photodiode readings alone. A 30-LED x 6-photodiode waveguide sensor produces 180 light-transmission channels per frame; a neural network maps them to a dense point cloud of the deformed surface (4096 points by default), so the membrane can be tracked without any camera at inference time.

The reconstruction uses two stages:

1. **Point-cloud autoencoder** (`scripts/train_AE.py`): learns a latent code for membrane shapes from depth-camera point clouds.
2. **Photodiode-to-latent regressor** (`scripts/train_optical.py`): maps the 180 optical channels to that latent code. The frozen autoencoder decoder then turns the predicted code into a point cloud.

This repository contains the model and training code, the data tooling, the real-time inference tools, and the sensor firmware (`firmware/`).

## Repository layout

```
pharos/       Python package: model, datasets, losses, metrics, plotting
scripts/      Training and evaluation entry points
tools/        Cache builder, real-time inference / video rendering, data acquisition
config/       config.yaml, the single configuration file used by all scripts
firmware/     STM32 (NUCLEO-L432KC) firmware for LED multiplexing and photodiode readout
data/         Datasets (not tracked by git, see "Data")
```

## Installation

Python 3.9 or newer and a CUDA-capable GPU are recommended (CPU works but is slow).

```bash
git clone <repository-url>
cd Pharos

# Install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -e .
```

The real-time tools and data acquisition scripts need extra packages (serial port, Open3D, OpenCV, RealSense):

```bash
pip install -e ".[hardware]"
```

## Data

The indentation dataset is distributed as packed session archives. Place them in `data/indentation/`:

```
data/indentation/
  white_train_S1.npz ... white_train_S7.npz    training sessions
  white_test_Circle.npz, _Poke, _Square, _Triangle, _U.npz    held-out indenter shapes
```

Each archive holds the depth frames (`depth`), the raw optical rows (`optical`, 218 columns: timestamp plus `(30 + 1) x (6 + 1)` LED/photodiode readings), the camera intrinsics and per-frame metadata. The test sessions use indenter shapes that never appear in training.

Build the memory-mappable caches once (about 4 GB for the training set):

```bash
python tools/build_train_cache.py --npz "data/indentation/white_train_S*.npz" \
    --out data/indentation/cache/train_stride3 --stride 3
python tools/build_train_cache.py --npz "data/indentation/white_test_*.npz" \
    --out data/indentation/cache/test_stride3 --stride 3
```

The cache stores the depth already sub-sampled to the sampling grid, so a frame is a 77 x 77 grid (5929 ground-truth points).

## Training

All scripts read `config/config.yaml` and write to `checkpoints/<output.run_name>/`. Pass `--config` to use another file and `--run-name` to override the run name.

```bash
# Stage 1: point-cloud autoencoder
python scripts/train_AE.py

# Stage 2: photodiode readings -> latent code, combined with the autoencoder decoder
python scripts/train_optical.py
```

`train_AE.py` resumes automatically from `checkpoints/<run>/checkpoints/last.pth`. `train_optical.py` writes the final model to `checkpoints/<run>/best_combined.pth`; that file also records the input normalisation constants, which every evaluation and inference script applies automatically.

Useful options of `train_optical.py`: `--loss-mode {mse,mse_cd,cd}`, `--lambda-cd`, `--epochs`, `--ae-run-name`, `--device`.

Multi-GPU training of the autoencoder is supported through `torchrun` (Linux) or by setting `train_ae.gpus: auto`.

## Evaluation

```bash
# Full pipeline (photodiodes -> point cloud) on the held-out test sessions:
# F-score vs. threshold, Chamfer distance, example reconstructions
python scripts/evaluate.py

# Per-session metrics (Chamfer distance, max nearest-neighbour distance, F-score at 1 and 2 mm)
python scripts/evaluate_sessions.py --save-samples

# Autoencoder only: the reconstruction floor of the decoder, independent of the optical input
python scripts/evaluate_AE.py --split test
```

Results are written under `checkpoints/<run>/eval/`, `eval_sessions/` and `eval_ae_<split>/`.

## Configuration

`config/config.yaml` is commented in place. The main fields:

| Section | Purpose |
| --- | --- |
| `data` | Cache locations, sampling stride, train/validation split |
| `model` | Latent size and number of output points (`optical_dim` must stay 180) |
| `train_ae` | Autoencoder optimiser, schedule, loss and batch-norm settings |
| `train_optical` | Regressor optimiser, epochs and loss mode |
| `output` | Checkpoint directory and run name |

Sensor-specific constants (number of LEDs and photodiodes, valid depth bounding box) are in `pharos/config.py`.

## Real-time inference and data acquisition

The scripts in `tools/` need the optional `hardware` dependencies and the sensor connected over a serial port.

| Script | Purpose |
| --- | --- |
| `tools/interface.py` | Live 3D (or contour) visualisation of the predicted shape from the serial stream |
| `tools/d2v.py` | Render recorded optical frames as a point-cloud video |
| `tools/collect_data.py` | Record synchronised optical and RealSense depth frames |
| `tools/collect_optical.py` | Record optical frames only |

`interface.py` and `d2v.py` load `best_combined.pth` from the working directory and build the network with the values hard-coded at the top of the file (`latent_dim=512`, `num_points=4096`); keep them in sync with `config/config.yaml`. Set the serial port (`PORT` / `SERIAL_PORT`) at the top of each script.

## Firmware

The sensor firmware, previously published as [Optical_Tomography](https://github.com/XuGuaaaanyu/Optical_Tomography), now lives in [`firmware/`](firmware/). It targets a NUCLEO-L432KC board that reads the ADPD2211 photodiodes through an AD7175-8 ADC and time-multiplexes the SK9822 LEDs. See [`firmware/README.md`](firmware/README.md) for the hardware overview, peripheral configuration and build notes.
