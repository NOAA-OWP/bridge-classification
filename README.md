# USGS Lidar Bridge Classification

A comprehensive pipeline for processing bridge lidar data organized by Hydrologic Unit Code (HUC) regions. This project downloads lidar point cloud data, applies weak supervision rules for labeling, normalizes coordinates, and prepares data for machine learning model training.

## Project Overview

This project provides tools for:

- **Data Download & Weak Supervision**: Downloads lidar data from USGS Entwine sources and applies automated labeling rules to identify bridge decks, ground, water, and obstacles
- **Data Normalization**: Normalizes point cloud coordinates and remaps classification labels for model training
- **Model Training**: Prepares normalized data for training sparse 3D U-Net models for bridge point cloud classification

The pipeline processes bridge geometries from OpenStreetMap, finds intersecting lidar sources, applies ground filtering (SMRF), performs quality checks (RANSAC plane fitting, linearity validation), and generates labeled training data.

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

**Run the Pipeline**:

```bash
# Step 1: Download & Weak Supervision
docker compose run --rm bridge-classifier python src/download-and-weak-supervise-hucs.py

# Step 2: Preprocess & Normalization
docker compose run --rm bridge-classifier python src/preprocess_bridges.py

# Step 3: Split data (train/val/test)
docker compose run --rm bridge-classifier python utils/split_data.py --symlink
# docker compose run --rm bridge-classifier python utils/split_data.py --train-ratio 0.75 --test-ratio 0.1

# Step 4: Train Model (Requires NVIDIA GPU)
docker compose run --rm bridge-classifier python src/train.py --train --augment --val-dir='./data/ml-data/validation' --train-dir='./data/ml-data/training' --epochs 50 --batch-size 16 --exp-name bridge-base-v0
```

Optional: To compute class weights from the training set (e.g. for imbalanced loss), run `docker compose run --rm bridge-classifier utils/calculate_weights.py --data-dir ./data/ml-data/training` after the split step.

Note: If you run into permission denied error, make sure proper permission is provided by running `chmod -R 777 <folder>`.

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

**Troubleshooting**: If you encounter `libstdc++` errors (common on Linux) when running scripts:

```bash
# Try installing the system library
mamba install -c conda-forge libstdcxx-ng

# OR export the library path before running your script
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

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

# Pin NumPy to avoid the Floating point exception (core dumped) error
# More info here: https://github.com/traveller59/spconv/issues/725
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

### Data Download
- Make a folder named `data/` in the same level as `src/`
- Make a subfolder `usgs_entwine/` and `osm/hucs/` inside `data/` folder.
- Download the usgs lidar resources as `wget https://raw.githubusercontent.com/hobuinc/usgs-lidar/refs/heads/master/boundaries/resources.geojson -O data/usgs_entwine/lidar_resources.geojson`
- HUCS level osm data can be found at `s3://fimc-data/bridge-classification/osm/hucs/` (organized by huc_id folder level)

Download and organize it to match this structure.

```text
data/
├── usgs_entwine/
│   └── lidar_resources.geojson
└── osm/
    └── hucs/
        ├── 02050206/
        │   └── osm_bridges_lidar_subset__02050206.gpkg
        ├── 03070101/
        │   └── osm_bridges_lidar_subset__03070101.gpkg
        ├── 11010009/
        │   └── osm_bridges_lidar_subset__11010009.gpkg
        └── ... (other huc_id folders)
```

## Classification Labels for Training

The pipeline uses the following classification scheme:

- **0**: Background/Unclassified (Piers/Pylons, Trees, Low Noise, Birds)
- **1**: Ground/ Water (Non-Bridge Surface, River Banks, Water Surface)
- **2**: Bridge Deck (Primary Target)
- **3**: Obstacles (Cars, Light Poles, High Noise)

## Output Structure

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

### Visualizing training metrics

Training (Step 4) writes metrics via Lightning CSVLogger to `./experiments/<exp_name>/version_<N>/metrics.csv`. The script `utils/visualize_metrics.py` plots loss and deck/overall accuracy curves and saves `training_curves.png` in the same directory.

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
