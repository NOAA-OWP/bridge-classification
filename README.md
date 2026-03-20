# USGS Lidar Bridge Classification

A comprehensive pipeline for processing bridge lidar data organized by Hydrologic Unit Code (HUC) regions. This project downloads lidar point cloud data, applies weak supervision rules for labeling, normalizes coordinates, and prepares data for machine learning. It includes **model training** (sparse 3D U-Net) and **inference** for bridge point cloud classification with multiple output modes (masked, raw, or both); **scaling with AWS Batch** for parallel inference on SPOT instances is [supported](docs/aws-batch-inference.md).

## Table of Contents

- [Documentation](#documentation)
- [Project Overview](#project-overview)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)
- [Data Download](#data-download)
- [Classification Labels for Training](#classification-labels-for-training)
- [Output Structure](#output-structure)
- [Notebooks](#notebooks)
- [Visualizing training metrics](#visualizing-training-metrics)

## Documentation

For detailed design documentation, see the `docs/` directory (or build the docs site with MkDocs):

- **[Architecture](docs/architecture.md)** — System design, classification schema, algorithm details.
- **[Conda environments](docs/envs.md)** — Full vs data-processing-only envs; use on GPU vs CPU-only machines.
- **[Data Pipeline](docs/data-pipeline.md)** — Step-by-step data flow walkthrough with data shapes at each stage.
- **[AWS Batch Inference](docs/aws-batch-inference.md)** — Terraform infrastructure, job submission, SPOT instance handling, inference modes, post-run audit, and configuration reference for scaling inference with AWS Batch array jobs.
- **[Module Reference](docs/module-reference.md)** — Summary of every module's public API and CLI arguments.
- **[Design Decisions](docs/decisions.md)** — Rationale for key architectural choices.

### Building the docs locally

The docs are readable as plain Markdown on GitHub. To build a searchable site with navigation and Mermaid diagrams:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open <http://localhost:8000> to preview. Changes to `docs/` are hot-reloaded.

### Deploying to GitHub Pages

```bash
mkdocs gh-deploy
```

## Project Overview

This project provides tools for:

- **Data Download & Weak Supervision**: Downloads lidar data from USGS Entwine sources and applies automated labeling rules to identify bridge decks, ground, water, and obstacles
- **Data Normalization**: Normalizes point cloud coordinates and remaps classification labels for model training
- **Model Training**: Prepares normalized data for training sparse 3D U-Net models for bridge point cloud classification

The pipeline processes bridge geometries from OpenStreetMap, finds intersecting lidar sources, applies ground filtering (SMRF), performs quality checks (RANSAC plane fitting, linearity validation), and generates labeled training data.

## Pipeline Overview

Data flows from OSM bridge geometries and USGS LiDAR sources through download and weak supervision, then normalization, train/val/test split, optional class-weight computation, and finally model training. The pipeline is designed to scale to hundreds of thousands of bridges; ensure sufficient disk space for silver_training and normalized outputs.

```mermaid
flowchart LR;
  DataDownload[Download and Weak Supervision]
  Preprocess[Preprocess and Normalize]
  Split[Split Train/Val/Test]
  CalculateWeights[Calculate Weights]
  Train[Train Model]
  DataDownload --> Preprocess;
  Preprocess --> Split;
  Split --> CalculateWeights;
  CalculateWeights --> Train;
  Split --> Train;
```

## Installation

### Prerequisites

- Python 3.11
- Conda or Mamba package manager
- **GPU Requirement:** An NVIDIA GPU is required for `spconv` and full model training.

### Setup

#### Option 1: Docker (Recommended)

We use Docker to manage the complex geospatial (GDAL/PDAL) and GPU (CUDA) dependencies. This ensures the environment works consistently across different machines.

**Build the image:**

```bash
# If on Linux/Windows (Standard)
docker compose build

# If on Mac (Build only - cannot run GPU training locally)
docker build --platform linux/amd64 -t bridge-classifier .
```

**Before running with Docker:**

- **Environment file:** Copy `.env.example` to `.env` and edit if needed. `DATA_DIR` is used by docker-compose to mount the ML data directory (e.g. `/data/ml-data` or an absolute path on the host).

  ```bash
  cp .env.example .env
  ```

- **Experiments directory (before training):** Create the experiments directory at repo root and make it writable so the container can write logs and checkpoints (default `./experiments`). Required when using Docker; without it, Step 4 (Train Model) may fail with a permission error.

  ```bash
  mkdir -p experiments && chmod 777 experiments
  ```

**Run the Pipeline**:

```bash
# Step 1: Download & Weak Supervision
# --skip-existing skips already processed outputs and bridges that previously had no lidar points (count==0).
docker compose run --rm bridge-classifier python src/download-and-weak-supervise-hucs.py --source-dir ./data/ml-data/source --silver-dir ./data/ml-data/silver_training --hucs-dir ./data/osm/hucs --lidar-resources ./data/usgs_entwine/lidar_resources.geojson --worker 12 --skip-existing


# Step 2: Preprocess & Normalization
# --skip-existing skips already processed outputs.
docker compose run --rm bridge-classifier python src/preprocess_bridges.py --input-dir ./data/ml-data/silver_training --output-dir ./data/ml-data/silver_training_normalized

# Step 3: Split data (train/val/test)
docker compose run --rm bridge-classifier python utils/split_data.py --laz-dir ./data/ml-data/silver_training --npy-dir ./data/ml-data/silver_training_normalized --output-dir ./data/ml-data --holdout-test-ids ./data/ml-data/holdout_test.txt --train-ratio 0.8 --val-ratio 0.2 --symlink

# Step 3a: Compute class weights (optional). Use output in training with --class-weights ./data/ml-data/class_weights.json
docker compose run --rm bridge-classifier python utils/calculate_weights.py --data-dir ./data/ml-data/training --output ./data/ml-data/class_weights.json

# Step 4: Train Model (Requires NVIDIA GPU). Ensure the experiments directory exists and is writable (see setup above).
# Pass class weights: add --class-weights ./data/ml-data/class_weights.json if you ran Step 3a.
# if gpu has headroom: batch_size -> 32
# num_workers: For 550K files, 4–8 can help; increase if CPU/disk are the bottleneck.

# used for g5.2xlarge ec2
# change rules
# --batch-size 4 --accumulate-grad-batches 4 → effective 16, ~half the steps per epoch.
# --batch-size 8 --accumulate-grad-batches 2 → effective 16, ~quarter of the steps (if it doesn’t OOM).

docker compose run --rm \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bridge-classifier python src/train.py \
  --train --augment \
  --val-dir='/data/ml-data/validation' \
  --train-dir='/data/ml-data/training' \
  --epochs 10 \
  --voxel-size 0.1 \
  --batch-size 4 \
  --accumulate-grad-batches 4 \
  --exp-name bridge-base-all-data-v0 \
  --class-weights /data/ml-data/class_weights.json \
  --num-workers 4 \
  --early-stopping \
  --early-stopping-patience 6 \
  --max-voxels 100000


# used for g5.4xlarge ec2
# change rules
# --batch-size 4 --accumulate-grad-batches 4 → effective 16, ~half the steps per epoch.
# --batch-size 8 --accumulate-grad-batches 2 → effective 16, ~quarter of the steps (if it doesn’t OOM).

docker compose run --rm \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bridge-classifier python src/train.py \
  --train --augment \
  --val-dir='/data/ml-data/validation' \
  --train-dir='/data/ml-data/training' \
  --epochs 25 \
  --ckpt-path /app/experiments/bridge-base-all-data-v0/version_0/checkpoints/last.ckpt \
  --voxel-size 0.1 \
  --batch-size 16 \
  --accumulate-grad-batches 1 \
  --exp-name bridge-base-all-data-v0 \
  --class-weights /data/ml-data/class_weights.json \
  --num-workers 10 \
  --early-stopping \
  --early-stopping-patience 12 \
  --max-voxels 100000
```


**Logging training output to a file:** For long runs, you can capture stdout/stderr so the experiment dir is self-contained. Create the experiment directory first (so the log file path exists), then use `tee` to write to a log file while still showing output in the terminal:

```bash
mkdir -p experiments/bridge-base-all-data-v1
docker compose run --rm \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bridge-classifier python src/train.py \
  --train --augment \
  --val-dir=/data/ml-data/validation \
  --train-dir=/data/ml-data/training \
  --epochs 25 \
  --ckpt-path /app/experiments/bridge-base-all-data-v0/version_0/checkpoints/last.ckpt \
  --exp-name bridge-base-all-data-v1 \
  --voxel-size 0.1 \
  --batch-size 4 \
  --accumulate-grad-batches 4 \
  --class-weights /data/ml-data/class_weights.json \
  --num-workers 4 \
  --early-stopping \
  --early-stopping-patience 6 \
  --max-voxels 100000 \
  2>&1 | tee experiments/bridge-base-all-data-v1/training_console.log
```

Use the same `--exp-name` as in your train command so the log lives next to `version_0/` (e.g. `experiments/bridge-base-all-data-v1/training_console.log`). The directory must exist before the run because `tee` does not create parent directories.

**Resume training** (continue from a saved checkpoint to more epochs, e.g. 10 → 25): use `--ckpt-path` and a new `--exp-name` so the resumed run writes to a separate experiment directory. Checkpoints are saved under `experiments/<exp_name>/version_0/checkpoints/` (e.g. `last.ckpt`).

```bash
docker compose run --rm \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bridge-classifier python src/train.py \
  --train --augment \
  --val-dir=/data/ml-data/validation \
  --train-dir=/data/ml-data/training \
  --epochs 25 \
  --ckpt-path /app/experiments/bridge-base-all-data-v0/version_0/checkpoints/last.ckpt \
  --exp-name bridge-base-all-data-v1 \
  --voxel-size 0.1 \
  --batch-size 4 \
  --accumulate-grad-batches 4 \
  --class-weights /data/ml-data/class_weights.json \
  --num-workers 4 \
  --early-stopping \
  --early-stopping-patience 6 \
  --max-voxels 100000
```

(Adjust `--ckpt-path` if your experiments dir is mounted elsewhere; e.g. if `experiments` is at `/app/experiments`, use `/app/experiments/bridge-base-all-data-v0/version_0/checkpoints/last.ckpt`.)

**Training options** (for `src/train.py`):

- **`--monitor`**: Metric used for best-model checkpointing and early stopping (default: `val_deck_iou`). Use `val_deck_iou` to optimize for deck IoU, or `val_loss` for validation loss. When no validation data is used, `train_loss` is used instead.
- **`--early-stopping`**: Stop training when the monitored metric does not improve.
- **`--early-stopping-patience`**: Number of epochs to wait with no improvement before stopping (default: 10). Used only when `--early-stopping` is set.
- **`--ckpt-path`**: Path to a checkpoint to resume training (e.g. `.../checkpoints/last.ckpt`). Use a new `--exp-name` for the resumed run so logs and checkpoints go to a separate experiment directory; the original run is left unchanged.

Example: train with early stopping on deck IoU (default), or on validation loss for comparison:

```bash
python src/train.py --train --augment --val-dir=./data/ml-data/validation --epochs 50 --early-stopping --early-stopping-patience 10
python src/train.py --train --augment --val-dir=./data/ml-data/validation --epochs 50 --monitor val_loss --early-stopping --early-stopping-patience 10
```

See [Troubleshooting](#troubleshooting) for permission and other issues.

**Development Mode**:

This opens an interactive shell inside the container where you can run scripts manually or debug.

```bash
docker compose run --rm bridge-classifier
# or manually:
# docker run --gpus all -v "$(pwd):/app" -it bridge-classifier
```

Once inside the container:

```bash
python src/download-and-weak-supervise-hucs.py ...
python src/preprocess_bridges.py ...
python src/train.py ...
# Type 'exit' to leave the container
```

#### Option 2: Local Conda Install

If you prefer to run locally without Docker, this project includes an `environment.yaml` file that handles all dependencies, including geospatial libraries and CUDA-accelerated ML tools.

```bash
# 1. Create the environment from file
mamba env create -f environment.yaml

# 2. Activate the environment
mamba activate bridge-classify

# 3. Verify GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

See [Troubleshooting](#troubleshooting).

**Data-processing only (no GPU):** For running only data-processing utils on a CPU-only machine, use the lighter env that omits PyTorch/CUDA: `conda env create -f environment-data.yaml` then `conda activate bridge-classify-data`. See [Conda environments](docs/envs.md) for details.

#### Option 3: Manual Installation

If the YAML installation fails or you need to build the environment step-by-step, follow these commands:

```bash
# Create mamba/ conda environment
mamba create -n bridge-classify python=3.11
mamba activate bridge-classify

# Install core dependencies
mamba install -c conda-forge python-pdal gdal entwine matplotlib geopandas tqdm seaborn
# if needed interactive shell
mamba install ipython

# Install PyTorch & Lightning
# Note: Using --index-url to find CUDA 12.6 specific wheels
# adjust cuda version as needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install lightning tensorboard

# Install Spconv (Sparse Convolution)
# for CPU (only for forward pass and network check); won't full train
# pip install spconv
# needs gpu for full training
# Adjust CUDA version as needed (https://github.com/traveller59/spconv)
pip install spconv-cu120

# Pin NumPy to avoid the Floating point exception (core dumped) error ([spconv #725](https://github.com/traveller59/spconv/issues/725))
mamba install numpy=1.26.4

# Optional: for saving graph
# pip install torchview graphviz
# on linux, you also need an OS-level graphviz package
# sudo apt-get install graphviz

# later when running script if getting an error
# Error: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
#mamba install -c conda-forge libstdcxx-ng
# if above doesn't work; run this in terminal before starting the script
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

See [Troubleshooting](#troubleshooting) for libstdc++ and other issues.

## Troubleshooting

- **Permission denied**: When running the pipeline (e.g. writing to `data/`), ensure permissions: `chmod -R 777 <folder>`. If training fails with permission errors on the experiments directory, ensure `experiments` exists and is writable: `mkdir -p experiments && chmod 777 experiments`.
- **libstdc++ / CXXABI_1.3.15**: Common on Linux. Try `mamba install -c conda-forge libstdcxx-ng`. If that fails, run before scripts: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
- **NumPy and spconv**: Pin NumPy to avoid "Floating point exception (core dumped)" ([spconv #725](https://github.com/traveller59/spconv/issues/725)): `mamba install numpy=1.26.4`.
- **No JSON files / no class distribution**: If `calculate_weights.py` reports no files or no distribution, run the split step (Step 3) first and use `--data-dir ./data/ml-data/training`.

## Data Download
- Make a folder named `data/` in the same level as `src/`
- Make a subfolder `usgs_entwine/` and `osm/hucs/` inside `data/` folder.
- Download the [USGS lidar resources](https://raw.githubusercontent.com/hobuinc/usgs-lidar/refs/heads/master/boundaries/resources.geojson): `wget https://raw.githubusercontent.com/hobuinc/usgs-lidar/refs/heads/master/boundaries/resources.geojson -O data/usgs_entwine/lidar_resources.geojson`
- HUC-level OSM data: `s3://fimc-data/bridge-classification/osm/hucs/` (organized by huc_id folder level)
- The pipeline creates `data/ml-data/` and its subfolders when you run Steps 1–4; see [Output directory (data/ml-data)](#output-directory-dataml-data) for the full layout.

Download and organize it to match this structure.

```text
data/
├── usgs_entwine/
│   └── lidar_resources.geojson
├── osm/
│   └── hucs/
│       ├── 02050206/
│       │   └── osm_bridges_lidar_subset__02050206.gpkg
│       ├── 03070101/
│       │   └── osm_bridges_lidar_subset__03070101.gpkg
│       ├── 11010009/
│       │   └── osm_bridges_lidar_subset__11010009.gpkg
│       └── ... (other huc_id folders)
└── ml-data/                 # Created by pipeline (Steps 1–4); see Output Structure
    └── ...
```

## Classification Labels for Training

The pipeline uses the following classification scheme:

- **0**: Background/Unclassified (Piers/Pylons, Trees, Low Noise, Birds)
- **1**: Ground/ Water (Non-Bridge Surface, River Banks, Water Surface)
- **2**: Bridge Deck (Primary Target)
- **3**: Obstacles (Cars, Light Poles, High Noise)

## Output Structure

### Output directory (data/ml-data)

The directory `data/ml-data/` is the default base for pipeline outputs: download (Step 1), preprocess (Step 2), split (Step 3), optional class weights (Step 3a), and training (Step 4).

```text
data/ml-data/
├── source/                    # Step 1: raw LAZ per HUC
├── silver_training/           # Step 1: weak-supervised LAZ per HUC
├── silver_training_normalized/ # Step 2: .npy and .json per HUC
├── training/                  # Step 3: train split (.npy, .json)
├── validation/                # Step 3: validation split
├── testing/                   # Step 3: test split (+ .laz if present)
├── split_manifest.json        # Step 3: split manifest
├── split_train_ids.txt        # Step 3: train IDs
├── split_val_ids.txt          # Step 3: validation IDs
├── split_test_ids.txt         # Step 3: test IDs
├── class_weights.json         # Step 3a (optional): from calculate_weights
└── holdout_test.txt           # Optional: fixed test IDs for split_data
```

### Notebooks

The **notebooks/** folder contains reproducible Jupyter notebooks:

- **`notebooks/dataset_overview.ipynb`** — Downloads ml-data artifacts from S3 (split ID files, `class_weights.json`, optional `osm_bridge_counts.json`), computes unique HUCs in the split, train/val/test line counts, total points from class weights, and OSM bridge counts (when the counts file is on S3). Produces a **class distribution** horizontal bar chart and, if `SILVER_NORMALIZED_DIR` is set to a local path, a **per-bridge point count histogram**. Downloads HUC8 boundaries from S3 and produces a map of which HUC8s appear in the dataset split.

- **`notebooks/training_plots.ipynb`** — Plots training curves from experiment metrics (compare/merge runs, optional best-epoch annotation). Configure `EXPERIMENTS_ROOT`, `EXPERIMENT_NAMES`, and `ANNOTATE_BEST_METRIC` in the notebook. Can be extended later with validation/test metrics, confusion matrix, etc.

Run the dataset overview notebook after configuring the S3 bucket/prefix (and optional AWS profile) in the Config cell or via environment variables (`BRIDGE_S3_BUCKET`, `BRIDGE_S3_ML_PREFIX`, `AWS_PROFILE`). HUC8 boundaries are read from S3 (`BRIDGE_S3_HUC8_KEY`); see the notebook for details.

### File Naming Conventions

**Download & Weak Supervision**:
- Source files: `bridge_{osmid}_{source_name}.laz`
- Silver files: `bridge_{osmid}_{source_name}.laz`

**Normalization**:
- Data files: `bridge_{osmid}_{source_name}.npy`
- Metadata files: `bridge_{osmid}_{source_name}.json`

### Metadata JSON Structure

The normalization script generates JSON metadata files with the following structure:

```json
{
    "original_file": "bridge_5069009_USGS_LPC_PA_South_Central_B2_2017_LAS_2019.laz",
    "original_path": "data/ml-data/silver_training/02050206/bridge_5069009_USGS_LPC_PA_South_Central_B2_2017_LAS_2019.laz",
    "offsets": {
        "x_center": -8548539.406538319,
        "y_center": 4995360.8107266445,
        "z_min": 128.93
    },
    "stats": {
        "point_count": 27166,
        "bridge_points": 13030,
        "ground_water_points": 5910,
        "obstacle_points": 6112,
        "background_points": 2114
    },
    "class_distribution": {
        "0": 2114,
        "1": 5910,
        "2": 13030,
        "3": 6112
    }
}
```

## Visualizing training metrics

### Viewing TensorBoard while training is running

You can run TensorBoard in a separate terminal (same Docker image) to watch metrics live. Use `-p 6006:6006` to map the container port to the host so you can open `http://localhost:6006` in the browser. Use `--bind_all` so TensorBoard listens on all interfaces (needed when running inside Docker).

```bash
docker compose run -p 6006:6006 --rm bridge-classifier tensorboard --logdir=experiments/bridge-base-all-data-v0/version_0/ --bind_all
```

The `--logdir` path should match your experiment name and version (e.g. `experiments/<exp_name>/version_<N>/`). If you use a different `--experiments-dir` when training, use that base path for `--logdir`. TensorBoard logs are written by Lightning's TensorBoardLogger to the same directory as the CSV metrics.

### CSV metrics and static plots

Training (Step 4) writes metrics via Lightning CSVLogger to `./experiments/<exp_name>/version_<N>/metrics.csv`. The script `utils/visualize_metrics.py` plots epoch-level train/validation curves (loss, deck IoU, precision, recall, overall accuracy) and saves `training_curves.png` in the same directory. Diagnostic metrics such as Num Voxels and Max Sample Voxels are excluded. In compare mode, plots use distinct colors for train vs validation and short labels suitable for presentation slides.

**Examples:**

- Default (loads `./experiments/bridge_classify_base/version_0/metrics.csv`):

  ```bash
  python utils/visualize_metrics.py
  ```

- Specific experiment and version:

  ```bash
  python utils/visualize_metrics.py --exp bridge_classify_base --ver 0
  # optionally: --root ./experiments
  ```

- Direct path to a metrics file:

  ```bash
  python utils/visualize_metrics.py --csv ./experiments/bridge_classify_base/version_0/metrics.csv
  ```

- With Docker:

  ```bash
  docker compose run --rm bridge-classifier python utils/visualize_metrics.py
  ```

**Compare mode**

Use `--compare` with comma-separated experiment names (under `--root`) to plot multiple experiments on the same axes. Optionally set `--compare-versions` (e.g. `0,0,0`); if omitted, version 0 is used for all. Use `--out` to set the output path (default: `{root}/compare_training_curves.png`).

  ```bash
  python utils/visualize_metrics.py --root experiments_copy --compare bridge-base-all-data-v0,bridge-base-all-data-v1 --out experiments_copy/compare_v0_v1.png
  ```

**Merge-resumed (one continuous curve)**

When comparing exactly two experiments, add `--merge-resumed` to merge them into one timeline (first run’s epochs, then second run’s later epochs) and plot a single train/val series with a short legend (e.g. "Train", "Val") and run info in the figure suptitle (e.g. "v0 → v1").

  ```bash
  python utils/visualize_metrics.py --root experiments_copy --compare bridge-base-all-data-v0,bridge-base-all-data-v1 --merge-resumed --out experiments_copy/merged_v0_v1.png
  ```

**Reference**

For full CLI options: `python utils/visualize_metrics.py --help`
