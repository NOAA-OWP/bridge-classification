# Data Pipeline Walkthrough

This document traces a single bridge from OSM geometry to a classified output file, showing exact data shapes and transformations at each stage.

---

## Input Data

Before running any pipeline steps, two inputs are required:

| Input | Location | Description |
|-------|----------|-------------|
| OSM bridge GeoPackages | `data/osm/hucs/{huc_id}/osm_bridges_lidar_subset__{huc_id}.gpkg` | Bridge geometries (LineStrings/Polygons) with `osmid` column, CRS EPSG:4326 |
| USGS lidar resources | `data/usgs_entwine/lidar_resources.geojson` | Spatial index of all USGS 3DEP EPT sources |

The pipeline projects all geometries to **EPSG:3857** (Web Mercator) for metric buffering.

---

## Step 1: Download & Weak Supervise

**Script**: `src/download_and_weak_supervise_hucs.py`

```
OSM bridge geometry (LineString)
  → buffer 10 m
  → PDAL readers.ept (polygon-clipped download)
  → SMRF ground filter
  → lexsort + seeded shuffle (deterministic ordering)
  → RANSAC plane fit
  → convex hull mask
  → QC checks (RMSE + linearity)
  → heuristic Z-distance labeling
```

**Outputs per bridge** (inside HUC subdirectory):

| File | Location | Contents |
|------|----------|----------|
| Source LAZ | `data/ml-data/source/{huc_id}/bridge_{osmid}_{source}.laz` | Raw downloaded point cloud, ASPRS original labels |
| Silver LAZ | `data/ml-data/silver_training/{huc_id}/bridge_{osmid}_{source}.laz` | Weak-supervised labels (ASPRS 17 = deck, 18 = obstacles) |

**Data format**: PDAL structured array — fields include `X`, `Y`, `Z`, `Intensity`, `Classification`, `ReturnNumber`, `NumberOfReturns`, `GpsTime`, etc.

**Rejection**: Bridges failing RMSE > 0.30 m or linearity deviation > 0.35 m are skipped (no silver LAZ produced). Curved/arched bridges are excluded by design.

---

## Step 2: Preprocess & Normalize

**Script**: `src/preprocess_bridges.py`

```
Silver LAZ
  → PDAL readers.las
  → ASPRS code remap (LAS_TO_MODEL_MAP)
  → XY centering + Z flooring
  → intensity 0-1 normalization
  → stack into (N, 5) float32
```

### Transformations

```python
# Coordinate normalization
X_norm = X - mean(X)         # center at 0
Y_norm = Y - mean(Y)         # center at 0
Z_norm = Z - min(Z)          # floor at 0

# Intensity normalization
I_norm = Intensity / max(Intensity)  # 0-1 range

# Class remapping (LAS_TO_MODEL_MAP)
# 2  → 1  (Ground → Ground/Water)
# 9  → 1  (Water  → Ground/Water)
# 17 → 2  (Bridge Deck → Bridge Deck)
# 18 → 3  (High Noise  → Obstacles)
# *  → 0  (all others → Background)
```

**Outputs per bridge**:

| File | Shape | dtype | Columns |
|------|-------|-------|---------|
| `.npy` | (N, 5) | float32 | `[x_centered, y_centered, z_floored, intensity_norm, model_label]` |
| `.json` | — | — | Offsets, point count, class distribution |

**Example `.json` metadata:**

```json
{
  "original_file": "bridge_5069009_USGS_LPC_PA_...laz",
  "offsets": {
    "x_center": -8548539.41,
    "y_center": 4995360.81,
    "z_min": 128.93
  },
  "stats": {
    "point_count": 27166,
    "bridge_points": 13030,
    "ground_water_points": 5910,
    "obstacle_points": 6112,
    "background_points": 2114
  },
  "class_distribution": {"0": 2114, "1": 5910, "2": 13030, "3": 6112}
}
```

---

## Step 3: Split Train / Val / Test

**Script**: `utils/split_data.py`

```
silver_training_normalized/{huc_id}/*.npy + *.json
  → group bridges by HUC
  → stratified random split (seed=27)
  → symlinks into training/ validation/ testing/
```

**Strategy**: All bridges within a single HUC land in the same split — preventing spatial data leakage between nearby bridges.

**Outputs**:

```
data/ml-data/
├── training/{huc_id}/*.npy, *.json     (80% default)
├── validation/{huc_id}/*.npy, *.json   (20% default)
├── testing/{huc_id}/*.npy, *.json, *.laz  (holdout, + LAZ for gold labeling)
├── split_manifest.json
├── split_train_ids.txt
├── split_val_ids.txt
└── split_test_ids.txt
```

### Step 3a: Class Weights

**Script**: `utils/calculate_weights.py`

```
training/**/*.json
  → aggregate class_distribution counts
  → W_c = Total / (4 × Count_c)
  → class_weights.json
```

**Output**: `data/ml-data/class_weights.json`

```json
{"weights": [6.22, 1.49, 0.36, 2.34]}
```

Higher weight = rarer class (Background is rare near bridges; Deck is common).

---

## Step 4: Training

**Script**: `src/train.py`

```
training/**/*.npy
  → BridgeDataset.__getitem__
    → load (N, 5) npy
    → shift xyz min to 0
    → [optional augmentation]
    → quantize: floor(xyz / voxel_size) → int32 coords
    → aggregate_voxel_points: mean intensity + majority-vote labels
    → [optional max_voxels subsample]
  → sparse_collate_fn
    → prepend batch_id → [batch_id, x, y, z]
  → SparseConvTensor
  → SparseUNet forward
  → WeightedCrossEntropyLoss
```

### Data Shape Transformations During Training

| Stage | Shape | dtype | Notes |
|-------|-------|-------|-------|
| Raw `.npy` | (N, 5) | float32 | N = raw point count per bridge |
| After `xyz -= min` | (N, 3) | float32 | Shifted to non-negative space |
| `discrete_coords` | (N, 3) | int32 | Quantized voxel grid indices |
| After voxel aggregation | (M, 3) coords + (M, 1) features + (M,) labels | int32 / float32 / int64 | M << N (typically 10–50× compression) |
| After collate (batch of B) | (ΣM, 4) coords + (ΣM, 1) features + (ΣM,) labels | int32 / float32 / int64 | First col = batch_id |
| `SparseConvTensor` input | (ΣM, 1) features + (ΣM, 4) indices | — | SpConv internal format |
| Model output | (ΣM, 4) | float32 | Per-voxel logits for 4 classes |
| After `argmax` | (ΣM,) | int64 | Predicted class per voxel |

**Voxel size**: 0.1 m (10 cm) default used in practice for larger datasets.

---

## Step 5: Inference

**Script**: `src/inference.py`

```
Raw LAZ (unprocessed)
  → PDAL readers.las
  → normalize intensity (0-1)
  → shift xyz min to 0
  → quantize → discrete_coords (N, 3)
  → np.unique(..., return_inverse=True)
    → unique_coords (M, 3)          ← fed to model
    → unique_inverse_indices (N,)   ← maps each point back to its voxel
  → mean intensity per voxel (M, 1)
  → SparseConvTensor (batch_size=1)
  → model.forward() [no_grad]
  → argmax → voxel_preds (M,)
  → voxel_preds[unique_inverse_indices] → point_labels (N,)
  → MODEL_TO_LAS_MAP: {0→1, 1→2, 2→17, 3→18}
  → PDAL writers.las (classified LAZ)
```

**Output**: `{bridge_stem}_predicted.laz` — same structure as input, with `Classification` field updated to ASPRS codes.

---

## Complete Data Shape Table

| Step | File/Object | Shape | dtype | Description |
|------|-------------|-------|-------|-------------|
| Step 1 output | Silver LAZ | Structured array, N rows | mixed | ASPRS-labeled point cloud |
| Step 2 output | `.npy` | (N, 5) | float32 | Normalized [x, y, z, intensity, label] |
| Step 2 output | `.json` | — | — | Offsets + class distribution |
| Step 3 output | Symlinks | — | — | Organized into train/val/test dirs |
| Step 3a output | `class_weights.json` | [4] | float | Inverse-frequency weights |
| Step 4 – voxel coords | `discrete_coords` | (M, 3) | int32 | 5 cm grid indices |
| Step 4 – voxel features | `aggregated_features` | (M, 1) | float32 | Mean intensity per voxel |
| Step 4 – batch coords | `coordinates` | (ΣM, 4) | int32 | [batch_id, x, y, z] |
| Step 4 – model output | logits | (ΣM, 4) | float32 | Per-voxel class scores |
| Step 5 – point labels | `point_labels_las` | (N,) | uint8 | ASPRS codes per original point |
| Step 5 output | Predicted LAZ | Structured array, N rows | mixed | ASPRS Classification field updated |

---

## Extended Workflows

### Gold Annotation Candidate Discovery

Finding new bridges to send for human annotation:

1. **Discover candidates**: Run `python utils/find_new_source_candidates.py --proven-linear false --sample-size 0` to find all bridge+source combinations not already in train/val/test splits.
2. **Run weak supervision**: Process candidates through `src/download_and_weak_supervise_hucs.py` using the output HUC/OSM ID lists. The pipeline automatically rejects complex/curved bridges via the RANSAC linearity check.
3. **Extract successes**: Use `archive/scripts/extract_successful_bridges.py` to parse pipeline logs and identify bridges that passed QC.
4. **Select final set**: Choose 1 bridge per HUC (for geographic diversity) and push silver `.laz` files to S3 `testing/` for annotators.
5. **Merge annotations**: After annotation returns, merge corrected files into `gold-data/`, re-normalize with `preprocess_bridges.py`, and re-split with `split_data.py`.

Use `--proven-linear true` as a fallback when you need guaranteed-linear bridges (same bridge geometry, different lidar source).
