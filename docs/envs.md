# Conda environments

This project provides two conda environment files:


| File                    | Env name               | Use case                                                                                                      |
| ----------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `environment.yaml`      | `bridge-classify`      | Full stack: training, inference, and data processing. Requires an NVIDIA GPU for training (PyTorch + spconv). |
| `environment-data.yaml` | `bridge-classify-data` | Data processing only. No GPU/training deps. Safe for CPU-only machines.                                       |


## When to use which

- `**environment.yaml**` — Use for model training, inference, and any workflow that needs PyTorch, Lightning, or spconv. Install with:
  ```bash
  conda env create -f environment.yaml
  conda activate bridge-classify
  ```
- `**environment-data.yaml**` — Use for data-processing scripts only (download, split, verify, weights, visualize). Same conda stack (geopandas, pdal, boto3, pandas, matplotlib, seaborn, tqdm) but no pip-installed CUDA/PyTorch/spconv. Install with:
  ```bash
  conda env create -f environment-data.yaml
  conda activate bridge-classify-data
  ```

