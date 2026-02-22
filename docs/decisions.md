# Design Decisions

Log of non-obvious architectural choices and the reasoning behind them.

---

## 1. SpConv over MinkowskiEngine / TorchSparse

**Choice**: `spconv-cu120` (SpConv v2)

**Alternatives considered**: MinkowskiEngine, TorchSparse

**Reasoning**:

- **MinkowskiEngine**: Stale CUDA support, frequent compatibility issues with newer PyTorch and CUDA versions. Build process is fragile.
- **TorchSparse**: Less mature; smaller community; fewer architectural primitives (no `SparseInverseConv3d` equivalent at evaluation time).
- **SpConv**: Actively maintained, strong community, supports `SubMConv3d` (sparsity-preserving), `SparseConv3d` (downsampling), and `SparseInverseConv3d` (upsampling). Ships pre-built wheels for specific CUDA versions, making installation reproducible.

**Trade-off**: Must pin to a specific CUDA-matching wheel (e.g., `spconv-cu120` for CUDA 12.0). NumPy must be pinned to `1.26.x` to avoid a floating-point exception on some platforms ([spconv #725](https://github.com/traveller59/spconv/issues/725)).

---

## 2. Weak Supervision via RANSAC for Scale

**Choice**: Automated labeling using RANSAC plane fitting + heuristic Z-distance rules

**Reasoning**: There are 550K+ bridges in the continental US. Human labeling at that scale is infeasible (would take years). RANSAC reliably identifies flat bridge deck planes from LiDAR data, and Z-distance thresholds from the fitted plane cleanly separate the deck from above-deck obstacles and below-deck structure. Quality gates (RMSE < 0.30 m, linearity check) reject bad fits before they produce mislabeled training data.

**Trade-off**: Silver labels are noisier than gold human labels. Arched, curved, and structurally complex bridges are rejected by the linearity check — the pipeline skips these rather than producing bad training data. Estimated label accuracy on accepted bridges is high (deck region is tightly bounded by the RANSAC plane).

---

## 3. Per-HUC Data Organization

**Choice**: All data files organized by HUC (Hydrologic Unit Code) folder hierarchy

**Reasoning**: Matches the USGS data organization structure, making it natural to work with USGS datasets. Critically, per-HUC organization enables **spatially-stratified splitting**: all bridges in a HUC go to the same train/val/test split, preventing data leakage from spatially proximate bridges appearing in both training and test sets. Nearby bridges share similar terrain, land cover, and LiDAR acquisition parameters — splitting on HUC rather than randomly prevents overfitting to local conditions.

**Source**: `utils/split_data.py`

---

## 4. Deterministic RANSAC Ordering

**Choice**: `np.lexsort((Z, Y, X))` followed by a seeded shuffle (`seed=27`) before RANSAC

**Reasoning**: PDAL's EPT reader streams data in chunks from S3, and the chunk order varies between operating systems, network conditions, and PDAL versions. Even with a fixed `random_state` in `RANSACRegressor`, the `min_samples` selected for the initial plane estimate depend on the order of points in the input array. Without sorting first, RANSAC produces different inlier masks on macOS vs. Linux, breaking reproducibility.

The fix: sort by (X, Y, Z) to get a canonical order, then apply a seeded shuffle to break spatial clusters (which would bias RANSAC sample selection). This produces bitwise-identical results across platforms.

**Verification**: `utils/verify_ransac_parity.py` tests this on synthetic data.

---

## 5. Voxel-Level Majority Vote for Labels

**Choice**: When multiple points map to the same voxel, assign the voxel the **majority-vote** label

**Reasoning**: Averaging labels would produce fractional values meaningless for categorical classification. Majority vote is the natural aggregation: the voxel takes the class that most of its constituent points belong to. This is semantically correct — if 60% of points in a voxel are Bridge Deck and 40% are Background (edge voxels), calling it a Deck voxel is the right choice.

Mean aggregation is used for the intensity feature (continuous value), which is appropriate.

**Source**: `src/train.py` — `aggregate_voxel_points` function

---

## 6. `max_voxels` Cap for OOM Prevention

**Choice**: Optional random subsampling of voxels per sample when exceeding a cap

**Reasoning**: Bridge size varies enormously — a small footbridge has ~5K voxels while a major highway bridge can have 300K+. Without a cap, a single outlier bridge in a batch can OOM the GPU. `max_voxels` (e.g., 100,000) randomly subsamples the voxels for that sample, keeping the batch size manageable. This is simpler and more effective than dynamic batching or gradient checkpointing.

Additionally, the training loop catches `torch.cuda.OutOfMemoryError` per batch, clears the cache, and continues — providing a second layer of defense for the rare case where a batch still exceeds memory.

**Source**: `src/train.py` — lines 235–241 (subsample), lines 442–448 (OOM handler)

---

## 7. Intensity-Only Features (XYZ via Voxel Grid)

**Choice**: Input to the model is 1 channel (intensity). XYZ spatial information is encoded implicitly via the voxel grid structure.

**Reasoning**: In a sparse convolution framework, the voxel grid coordinates *are* the spatial information — SpConv's convolutions operate on the spatial layout of active voxels. Explicitly passing XYZ as input features would be redundant and would require the network to learn to separate the role of "where am I" (already encoded in the grid) from "what am I" (what the feature channel should answer).

Intensity encodes surface reflectance — useful for material discrimination (asphalt bridge deck vs. water vs. vegetation vs. metal guard rails). It is the only physically meaningful per-point scalar available consistently across USGS 3DEP datasets.

**Source**: `src/train.py` — line 195 (`feat = data[:, 3:4]`), `src/model.py` line 93 (`input_channels=1`)
