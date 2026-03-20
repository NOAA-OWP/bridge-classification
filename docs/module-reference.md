# Module Reference

Summary of every module's public API and CLI arguments.

---

## Source Modules (`src/`)

### `src/download-and-weak-supervise-hucs.py`

Full HUC-based pipeline for downloading USGS LiDAR and generating weakly-supervised silver training data. This is the primary data acquisition script.

**Key classes:**


| Class                     | Description                                                                                                                                                    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BridgeProcessingConfig`  | Dataclass containing all pipeline parameters (PDAL settings, RANSAC thresholds, classification Z-ranges). Pass `--buffer` and other CLI args to customize.     |
| `LidarSourceFinder`       | Loads `lidar_resources.geojson` and builds a spatial index. `find_intersecting_sources(geometry, buffer)` returns a list of `{"url": ..., "name": ...}` dicts. |
| `WeakSupervisionPipeline` | Orchestrates download → SMRF → RANSAC → QC → classification for a single bridge. Constructed from a `BridgeProcessingConfig`.                                  |


**Key functions:**


| Function                            | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `process_bridge_worker(task_tuple)` | Multiprocessing worker: processes one bridge and writes source + silver LAZ files. |
| `process_huc(huc_id, ...)`          | Processes all bridges in a single HUC directory.                                   |
| `main()`                            | CLI entry point.                                                                   |


**CLI arguments:**


| Argument            | Default                                       | Description                              |
| ------------------- | --------------------------------------------- | ---------------------------------------- |
| `--hucs-dir`        | `./data/osm/hucs`                             | Directory containing per-HUC GPKG files  |
| `--source-dir`      | `./data/ml-data/source`                       | Output directory for raw LAZ downloads   |
| `--silver-dir`      | `./data/ml-data/silver_training`              | Output directory for weak-supervised LAZ |
| `--lidar-resources` | `./data/usgs_entwine/lidar_resources.geojson` | USGS EPT source index                    |
| `--hucs`            | all                                           | Space-separated HUC IDs to process       |
| `--osm-ids`         | all                                           | Space-separated OSM IDs to process       |
| `--buffer`          | 10.0                                          | Bridge geometry buffer in meters         |
| `--workers`         | CPU count                                     | Parallel worker processes                |
| `--skip-existing`   | False                                         | Skip bridges already processed           |
| `--log-dir`         | `./logs`                                      | Directory for processing logs            |
| `--no-progress`     | False                                         | Disable tqdm progress bars               |


---

### `src/download-and-weak-supervise-demo.py`

Simplified single-dataset demo for testing the weak supervision algorithm on a small set of known bridge OSM IDs. Configuration is via constants at the top of the file (`LIDAR_DATASET`, `TARGET_OSMIDS`, `BUFFER_METERS`).

**Key functions:**


| Function                                           | Description                                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `check_bridge_linearity(xy_points, z_points, ...)` | Slices bridge into bins, fits a skeleton line, returns `(is_curved, max_deviation)`. |
| `run_weak_supervision_pipeline()`                  | Main pipeline: download → SMRF → RANSAC → classify → write LAZ.                      |


No CLI arguments. Edit constants at top of file to configure.

---

### `src/preprocess_bridges.py`

Normalizes LAZ files from `silver_training/` into `.npy` + `.json` pairs for model training.

**Key constants:**


| Constant           | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `LAS_TO_MODEL_MAP` | `{2: 1, 9: 1, 17: 2, 18: 3}` — ASPRS code → model class. All others → 0. |


**Key functions:**


| Function                                                | Description                                                                                               |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `normalize_intensity(array)`                            | Divides by `max`; returns 0–1 range.                                                                      |
| `process_laz_file(filepath, output_dir, skip_existing)` | Processes one LAZ file: read → remap → normalize → save `.npy` + `.json`. Returns `(success, error_msg)`. |
| `process_huc_folder(huc_dir, output_base_dir, ...)`     | Processes all LAZ files in a HUC folder, with optional multiprocessing.                                   |


**CLI arguments:**


| Argument          | Default                                     | Description                                  |
| ----------------- | ------------------------------------------- | -------------------------------------------- |
| `--input-dir`     | `./data/ml-data/silver_training`            | Input directory with HUC-organized LAZ files |
| `--output-dir`    | `./data/ml-data/silver_training_normalized` | Output directory for `.npy` and `.json`      |
| `--skip-existing` | False                                       | Skip if `.npy` + `.json` already exist       |
| `--hucs`          | all                                         | Specific HUC IDs to process                  |
| `--workers`       | CPU count                                   | Parallel workers per HUC                     |
| `--no-progress`   | False                                       | Disable progress bars                        |


---

### `src/model.py`

Sparse 3D U-Net architecture using SpConv. No CLI — imported by `train.py` and `inference.py`.

**Classes:**


| Class                                                           | Description                                                                                                                 |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `ResidualBlock(in_ch, out_ch, norm_fn, indice_key)`             | Two `SubMConv3d` layers + shortcut. Maintains sparsity pattern.                                                             |
| `SparseUNet(input_channels=1, num_classes=4, base_channels=16)` | Full U-Net: 4-level encoder → bottleneck → 3-level decoder → classification head. Returns `(N_voxels, num_classes)` logits. |


---

### `src/train.py`

Training pipeline with voxelization, data loading, and PyTorch Lightning integration.

**Key classes:**


| Class                                                      | Description                                                                       |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `BridgeDataset(data_dir, voxel_size, augment, max_voxels)` | Dataset with on-the-fly voxelization. Recursively finds `.npy` files.             |
| `BridgeLightningModule(...)`                               | Lightning module. Weighted CrossEntropyLoss, logs deck IoU/precision/recall.      |
| `BridgeDataModule(train_dir, val_dir, ...)`                | Lightning data module. Supports explicit `val_dir` or `val_split` on `train_dir`. |


**Key functions:**


| Function                                                                  | Description                                                                                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `aggregate_voxel_points(xyz, features, labels, voxel_coords, voxel_size)` | Vectorized voxel aggregation: mean intensity + majority-vote labels. Returns `(agg_xyz, agg_features, agg_labels)`. |
| `sparse_collate_fn(batch)`                                                | Custom collate: prepends batch ID to coordinates → `[batch_id, x, y, z]`. Returns dict.                             |
| `visualize_voxelization(data_dir, sample_idx, voxel_size)`                | Matplotlib 3D plot: original vs voxelized point cloud side by side.                                                 |


**CLI arguments:**


| Argument                    | Default                   | Description                                                    |
| --------------------------- | ------------------------- | -------------------------------------------------------------- |
| `--train-dir`               | `./data/ml-data/training` | Training data directory                                        |
| `--val-dir`                 | None                      | Validation directory; if unset, uses `--val-split`             |
| `--val-split`               | 0.0                       | Fraction of training data to use as validation (0 = none)      |
| `--voxel-size`              | 0.1                       | Voxel size in meters                                           |
| `--max-voxels`              | None                      | Max voxels per sample; subsampled if exceeded (OOM prevention) |
| `--batch-size`              | 16                        | Batch size                                                     |
| `--augment`                 | False                     | Enable random Z-rotation + jitter                              |
| `--train`                   | False                     | Enable training mode                                           |
| `--epochs`                  | 50                        | Training epochs                                                |
| `--learning-rate`           | 0.001                     | AdamW learning rate                                            |
| `--weight-decay`            | 0.01                      | AdamW weight decay                                             |
| `--base-channels`           | 16                        | U-Net base channel count                                       |
| `--num-workers`             | 4                         | DataLoader workers                                             |
| `--exp-name`                | `bridge_classify_base`    | Experiment name for logs/checkpoints                           |
| `--experiments-dir`         | `./experiments`           | Base directory for experiments                                 |
| `--class-weights`           | None                      | Path to `class_weights.json` from `calculate_weights.py`       |
| `--gpus`                    | auto                      | Number of GPUs (`None`=auto-detect). GPU required.             |
| `--early-stopping`          | False                     | Stop when monitored metric stops improving                     |
| `--early-stopping-patience` | 10                        | Epochs to wait before early stopping                           |
| `--monitor`                 | `val_deck_iou`            | Metric for checkpointing + early stopping                      |
| `--accumulate-grad-batches` | 1                         | Gradient accumulation steps                                    |
| `--ckpt-path`               | None                      | Checkpoint path to resume training from                        |
| `--visualize`               | False                     | Visualize voxelization for a sample                            |
| `--sample-idx`              | 0                         | Sample index to visualize                                      |


---

### `src/inference.py`

Loads a trained checkpoint, classifies a raw LAS/LAZ file, and writes a classified output with ASPRS codes. Supports single-file and batch modes, with per-bridge timeout handling.

**Key constants:**


| Constant                  | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `MODEL_TO_LAS_MAP`        | `{0: 1, 1: 2, 2: 17, 3: 18}` — model class → ASPRS output code |
| `MIN_POINT_COUNT`         | `100` — skip files with fewer points                           |
| `SPATIAL_SHAPE_PADDING`   | `10` — padding added to voxel grid spatial shape               |
| `BRIDGE_DECK_MODEL_CLASS` | `2` — model class for bridge deck                              |
| `BRIDGE_DECK_ASPRS_CODE`  | `17` — ASPRS code for bridge deck                              |
| `OBSTACLES_MODEL_CLASS`   | `3` — model class for obstacles                                |
| `OBSTACLES_ASPRS_CODE`    | `18` — ASPRS code for obstacles                                |


**Key functions:**


| Function                                                         | Description                                                            |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `load_las(filepath)`                                             | PDAL read → returns `(points, intensities, metadata, original_arrays)` |
| `save_las(output_path, original_arrays, labels, metadata)`       | Updates `Classification` field, writes via PDAL                        |
| `load_model(checkpoint_path, device)`                            | Loads SparseUNet from Lightning or raw checkpoint                      |
| `run_inference(model, input_path, output_path, ...)`             | Classify a single file. Supports `mode` parameter                      |
| `apply_bridge_mask(original_classification, point_labels_model)` | Bridge deck only mask: model class 2 → ASPRS 17 overlaid on original   |
| `run_batch_inference(model, pairs, ...)`                         | Process multiple files with per-bridge timeout via SIGALRM             |
| `parse_pairs_file(filepath)`                                     | Parse TSV file of input/output path pairs                              |


**CLI arguments:**


| Argument           | Default      | Description                                          |
| ------------------ | ------------ | ---------------------------------------------------- |
| `--input`          | None         | Input LAS/LAZ file path (single-file mode)           |
| `--output`         | None         | Output LAS/LAZ file path (single-file mode)          |
| `--pairs-file`     | None         | TSV file with input/output pairs (batch mode)        |
| `--model`          | *(required)* | Path to `.ckpt` checkpoint                           |
| `--voxel-size`     | 0.1          | Voxel size (must match training)                     |
| `--bridge-timeout` | 150          | Seconds before a hung bridge is skipped (batch mode) |
| `--mode`           | `masked`     | Output mode: `masked`, `raw`, or `both`              |


**Modes:**

- `masked` — bridge deck only (class 2 → ASPRS 17) overlaid on original classification
- `raw` — all model classes replace original classification via `MODEL_TO_LAS_MAP`
- `both` — saves `_predicted` (raw) and `_bridge_masked` (masked) files

**Usage examples:**

```bash
# Single file, masked mode (default)
python src/inference.py \
    --model ./experiments/bridge-base-all-data-v0/version_0/checkpoints/epoch=35.ckpt \
    --input ./data/ml-data/testing/02050206/bridge_10598181_USGS_LPC_PA_SouthCentral_B2_2017.laz \
    --output ./data/ml-data/predictions/bridge_10598181_bridge_masked.laz

# Batch mode with pairs file (model loaded once, processes all pairs)
python src/inference.py \
    --pairs-file ./pairs.tsv \
    --model ./experiments/bridge-base-all-data-v0/version_0/checkpoints/epoch=35.ckpt \
    --mode masked --bridge-timeout 150

# Both mode (saves _predicted and _bridge_masked side by side)
python src/inference.py \
    --model ./experiments/bridge-base-all-data-v0/version_0/checkpoints/epoch=35.ckpt \
    --input ./bridge.laz --output ./bridge_predicted.laz \
    --mode both
```

---

### `src/s3.py`

Shared S3 utilities used by all batch scripts. Generic S3 operations and bridge-specific path conventions.

**Constants:**


| Constant           | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `PROBE_EXTENSIONS` | `['.laz', '.las']` — extensions to try when manifest line has no extension |


**Functions:**


| Function                                                            | Description                                                                                    |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `parse_s3_uri(uri)`                                                 | Split `s3://bucket/key` into `(bucket, key)` tuple                                             |
| `object_exists(s3_client, bucket, key)`                             | Check if S3 object exists via `head_object` (returns bool)                                     |
| `download_file(s3_client, bucket, key, local_path)`                 | Download S3 object, creating parent dirs as needed                                             |
| `upload_file(s3_client, local_path, bucket, key)`                   | Upload local file to S3                                                                        |
| `stream_manifest_lines(s3_client, manifest_uri)`                    | Generator yielding non-empty stripped lines from an S3 manifest                                |
| `resolve_input_key(s3_client, bucket, input_prefix, manifest_line)` | Resolve manifest line to full S3 key, probing extensions if needed                             |
| `resolve_extension(s3_client, bucket, input_prefix, manifest_line)` | Determine file extension by probing S3 (falls back to `.laz`)                                  |
| `resolve_output_keys(output_prefix, manifest_line, ext, mode)`      | Compute expected output S3 key(s) by mode. Returns dict with `primary` and optionally `masked` |


---

## Utility Modules (`utils/`)

### `utils/split_data.py`

Splits bridge data into train/validation/test by HUC.

**Key functions:**


| Function                             | Description                                                       |
| ------------------------------------ | ----------------------------------------------------------------- |
| `discover_bridges_by_huc(input_dir)` | Returns `{huc_id: [Path, ...]}` dict of NPY files grouped by HUC. |


**CLI arguments:**


| Argument             | Default                                     | Description                                              |
| -------------------- | ------------------------------------------- | -------------------------------------------------------- |
| `--laz-dir`          | `./data/ml-data/silver_training`            | Source LAZ directory (for test split)                    |
| `--npy-dir`          | `./data/ml-data/silver_training_normalized` | Normalized NPY directory                                 |
| `--output-dir`       | `./data/ml-data`                            | Output base directory                                    |
| `--holdout-test-ids` | None                                        | File with fixed test IDs (`huc_id/bridge_stem` per line) |
| `--train-ratio`      | 0.8                                         | Training split fraction                                  |
| `--val-ratio`        | 0.2                                         | Validation split fraction                                |
| `--symlink`          | False                                       | Create symlinks instead of copies                        |
| `--seed`             | 27                                          | Random seed for reproducibility                          |


---

### `utils/calculate_weights.py`

Computes inverse-frequency class weights from preprocessed `.json` metadata.

**Key functions:**


| Function                           | Description                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `_load_one_json(path)`             | Worker: loads `class_distribution` from one JSON. Returns `(counts_dict, None)` on success. |
| `load_class_counts(data_dir, ...)` | Aggregates counts across all JSON files with optional multiprocessing.                      |


**CLI arguments:**


| Argument     | Default                   | Description                                 |
| ------------ | ------------------------- | ------------------------------------------- |
| `--data-dir` | `./data/ml-data/training` | Directory containing `.json` metadata files |
| `--output`   | `class_weights.json`      | Output JSON file path                       |


---

### `utils/visualize_metrics.py`

Plots training curves from PyTorch Lightning CSVLogger output.

**Key functions:**


| Function                                          | Description                                                                   |
| ------------------------------------------------- | ----------------------------------------------------------------------------- |
| `get_experiment_dir(base_dir, exp_name, version)` | Resolves experiment directory; auto-selects latest version if `version=None`. |


**CLI arguments:**


| Argument             | Default                | Description                                        |
| -------------------- | ---------------------- | -------------------------------------------------- |
| `--csv`              | None                   | Direct path to `metrics.csv`                       |
| `--root`             | `./experiments`        | Experiments root directory                         |
| `--exp`              | `bridge_classify_base` | Experiment name                                    |
| `--ver`              | latest                 | Version number                                     |
| `--compare`          | None                   | Comma-separated experiment names to plot together  |
| `--compare-versions` | `0,0,...`              | Comma-separated versions for compare mode          |
| `--merge-resumed`    | False                  | Merge two experiments into one continuous timeline |
| `--out`              | auto                   | Output PNG file path                               |


---

### `utils/download-osm-hucs.py`

Downloads OSM bridge GeoPackages by HUC from S3.

**Key classes:**


| Class         | Description                                                                         |
| ------------- | ----------------------------------------------------------------------------------- |
| `BridgeStats` | TypedDict tracking processing stats (processed, total_bridges, lidar_bridges, etc.) |
| `HUCResult`   | TypedDict for per-HUC worker result                                                 |


**Modes:**

- **Info mode** (omit `--dir`): Lists HUC IDs and bridge counts without saving
- **Download mode** (provide `--dir`): Saves organized HUC folders with filtered subsets

**CLI arguments:**


| Argument    | Default | Description                                    |
| ----------- | ------- | ---------------------------------------------- |
| `--profile` | default | AWS profile name                               |
| `--dir`     | None    | Local output directory (enables download mode) |
| `--all`     | False   | Download all HUCs (otherwise uses `--limit`)   |
| `--limit`   | 10      | Number of HUCs to process in info mode         |
| `--hucs`    | None    | Specific HUC IDs to download                   |


---

### `utils/verify_ransac_parity.py`

Verifies RANSAC determinism across platforms using synthetic bridge data.

**Key functions:**


| Function                                    | Description                                                                                             |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `generate_synthetic_bridge(seed, n_points)` | Creates a flat plane with Gaussian noise and 30% gross outliers. Returns `(XY, Z)`.                     |
| `run_test()`                                | Runs RANSAC with `random_state=27`, prints inlier count and coefficients for cross-platform comparison. |


No CLI arguments. Run directly: `python utils/verify_ransac_parity.py`

---

## Scripts (`scripts/`)

### `scripts/batch_entrypoint.py`

AWS Batch entrypoint — per-bridge processing loop with SPOT handling. Imports `run_inference()` directly (model loaded once). See [AWS Batch Inference](aws-batch-inference.md) for full documentation.

**Required environment variables:** `S3_BUCKET`, `S3_INPUT_PREFIX`, `S3_MANIFEST_URI`, `S3_MODEL_URI`, `S3_OUTPUT_PREFIX`

**Optional environment variables:** `INFERENCE_MODE` (default `masked`), `BRIDGE_TIMEOUT` (default `150`)

**Key features:**

- Per-bridge download → infer → upload → cleanup (immediate upload, O(1 bridge) disk)
- SIGTERM handler for SPOT interruptions (finishes current bridge before exit)
- Skip-if-exists resumability via S3 `head_object`
- Mixed .laz/.las support via S3 extension probing
- Structured CloudWatch logging with per-bridge timing

---

### `scripts/submit_batch_job.py`

Submit single or array Batch jobs. Reads terraform outputs for AWS config, counts manifest lines from S3, and computes array size.

**CLI arguments:**


| Argument         | Default            | Description                                    |
| ---------------- | ------------------ | ---------------------------------------------- |
| `--manifest`     | None               | S3 URI of manifest file                        |
| `--total`        | None               | Total file count (skip manifest download)      |
| `--single`       | False              | Submit single job (no array)                   |
| `--dry-run`      | False              | Print config without submitting                |
| `--validate`     | False              | Validate manifest format                       |
| `--chunk-target` | 60                 | Target files per array child                   |
| `--env`          | []                 | Environment override as KEY=VALUE (repeatable) |
| `--job-name`     | `bridge-inference` | Job name prefix                                |
| `--profile`      | None               | AWS profile for S3 manifest access             |


---

### `scripts/audit_outputs.py`

Post-run verification — checks all expected outputs exist in S3. Uses parallel `head_object` checks (200 threads by default).

**CLI arguments:**


| Argument          | Default      | Description                                       |
| ----------------- | ------------ | ------------------------------------------------- |
| `--manifest`      | *(required)* | S3 URI of manifest file                           |
| `--bucket`        | *(required)* | S3 bucket for outputs                             |
| `--input-prefix`  | `""`         | S3 prefix for input files (for extension probing) |
| `--output-prefix` | *(required)* | S3 prefix for output files                        |
| `--mode`          | `masked`     | Inference mode (determines expected filenames)    |
| `--write-missing` | None         | Write missing manifest lines to this file         |
| `--workers`       | 200          | Parallel S3 check workers                         |
| `--profile`       | None         | AWS profile                                       |


