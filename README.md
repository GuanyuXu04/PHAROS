# Pharos

Pharos reconstructs the 3D shape of a soft optical waveguide membrane from photodiode readings alone.

The reconstruction uses two stages:

1. **Point-cloud autoencoder** (`scripts/train_AE.py`): learns a latent code for membrane shapes from depth-camera point clouds.
2. **Photodiode-to-latent regressor** (`scripts/train_optical.py`): maps the optical channels to that latent code. The frozen autoencoder decoder then turns the predicted code into a point cloud.

This repository contains the model and training code, the data tooling, the real-time inference tools, and the sensor firmware (`firmware/`).

## Repository layout

```
pharos/       Model, datasets, losses, metrics, plotting
scripts/      Training and evaluation entry points
tools/        Cache builder, real-time inference / video rendering, data acquisition
config/       config.yaml, the single configuration file used by all scripts
firmware/     STM32 (NUCLEO-L432KC) firmware for LED multiplexing and photodiode readout
data/         Datasets, downloaded separately from Hugging Face (see "Data")
```

## Installation

Python 3.9 or newer and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/GuanyuXu04/PHAROS.git
cd PHAROS

# Install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -e .
```

The real-time tools and data acquisition scripts need extra packages (serial port, Open3D, OpenCV, RealSense):

```bash
pip install -e ".[hardware]"
```

## Data

The datasets are hosted on Hugging Face: [xuguanyu04/PHAROS-data](https://huggingface.co/datasets/xuguanyu04/PHAROS-data). It contains the datasets (`indentation/`, `bending/` and `stretch/`) and the trained model (`checkpoints/`). Download the datasets into `data/`:

```bash
pip install -U huggingface_hub
hf download xuguanyu04/PHAROS-data --repo-type dataset --exclude "checkpoints/*" --local-dir data
```

The trained model used in the paper is in the same repository. Download it into the repository root, where it lands in `checkpoints/final-run/`:

```bash
hf download xuguanyu04/PHAROS-data --repo-type dataset --include "checkpoints/*" --local-dir .
```

The training and evaluation scripts use the indentation data, which is distributed as packed session archives in `data/indentation/`:

```
data/indentation/
  white_train_S1.npz ... white_train_S7.npz    training sessions
  white_test_Circle.npz, _Finger, _Square, _Triangle, _U.npz    held-out indenter shapes
```

`bending/` and `stretch/` hold additional recordings that the scripts in this repository do not read. `bending/` contains one CSV log per recording (`TIME`, `EXT` and the photodiode readings of LEDs 1 to 3), named after the bending angle and whether the LEDs were on (`haslight`) or off (`nolight`). `stretch/` contains a single archive, `stretch.npz`, with the arrays `depth`, `optical`, `depth_frames` and `optical_frames` (the last two give the frame number of every row).

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


## Evaluation

The commands below use `output.run_name` from the config. To evaluate the downloaded model, add `--run-name final-run` to each of them.

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
| `tools/pack_session.py` | Pack a `collect_data.py` session into the `.npz` format used for training |
| `tools/collect_optical.py` | Record optical frames only |

`interface.py` and `d2v.py` load `best_combined.pth` from the working directory and read the network architecture (latent size, number of points) from the checkpoint, so nothing needs to be kept in sync by hand. Set the serial port (`PORT` / `SERIAL_PORT`) at the top of each script.

### Training on your own recordings

`collect_data.py` saves one depth file and one optical file per frame, plus a `meta.npz` with the camera settings. The training pipeline expects the packed format of the released dataset, so convert each recorded session before building a cache:

```bash
python tools/collect_data.py        # records into data/<SESSION_NAME>/
python tools/pack_session.py data/<SESSION_NAME> --out data/mine/my_session.npz
python tools/build_train_cache.py --npz "data/mine/my_session.npz" \
    --out data/mine/cache/train_stride3 --stride 3
```

Then point `data.cache_dir` in `config/config.yaml` at the new cache. A held-out test cache is built the same way from separate sessions. Frames with a missing or malformed file are skipped and reported, and `frame_ids` are renumbered from 0 within each archive.

## Firmware

The sensor firmware lives in [`firmware/`](firmware/). It targets a NUCLEO-L432KC board that reads the ADPD2211 photodiodes through an AD7175-8 ADC and time-multiplexes the SK9822 LEDs. See [`firmware/README.md`](firmware/README.md) for the hardware overview, peripheral configuration and build notes.
