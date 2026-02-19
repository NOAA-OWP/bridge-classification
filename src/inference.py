"""
Bridge Classification Inference Script

Loads a trained Sparse U-Net model, processes a raw LAS/LAZ file,
and outputs a classified LAS/LAZ file with ASPRS standard codes.

Workflow:
1. Load LAS file.
2. Voxelize points (keep track of which point belongs to which voxel).
3. Run Model Inference.
4. Map Voxel Labels -> Original Points.
5. Save LAS file.

Usage:
    python src/inference.py \
        --input ./data/ml-data/testing/02050206/bridge_10598181_....laz \
        --output ./data/ml-data/prediction.laz \
        --model ./experiments/bridge-base-v0/.../checkpoints/....ckpt \
        --gpu
"""

import argparse
import json
import torch
import numpy as np
import pdal
import sys
import os

# Ensure we can import the model structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    import spconv.pytorch as spconv
    from src.model import SparseUNet
except ImportError:
    # Fallback if running directly from src
    from model import SparseUNet
    import spconv.pytorch as spconv

# --- CONFIGURATION ---
# Map Model Classes (0-3) back to ASPRS LAS Codes
MODEL_TO_LAS_MAP = {
    0: 1,   # Background -> Unclassified
    1: 2,   # Ground/Water -> Ground
    2: 17,  # Bridge Deck -> Bridge Deck
    3: 18   # Obstacles -> High Noise
}

def load_las(filepath):
    """Read LAS file and return XYZ + Intensity."""
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(filepath)
            }
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    arrays = pipeline.arrays[0]

    # Extract data
    X = arrays['X']
    Y = arrays['Y']
    Z = arrays['Z']
    Intensity = arrays['Intensity']

    # Stack (N, 3)
    points = np.stack([X, Y, Z], axis=1).astype(np.float32)
    intensities = Intensity.astype(np.float32)

    # Normalize Intensity (0-1) like in training
    if intensities.max() > 0:
        intensities /= intensities.max()

    return points, intensities, pipeline.metadata, arrays

def save_las(output_path, original_arrays, labels, metadata):
    """Save the classified point cloud."""
    # Ensure labels are uint8
    classification = labels.astype(np.uint8)

    # Update classification in the original array to preserve all other fields (GPS, etc.)
    original_arrays['Classification'] = classification

    writer_stage = {
        "type": "writers.las",
        "filename": str(output_path),
        "extra_dims": "all",
        "a_srs": "EPSG:3857",
        "forward": "all",
    }

    # Execute writer pipeline
    pipeline_json = json.dumps({"pipeline": [writer_stage]})
    pipeline = pdal.Pipeline(pipeline_json, arrays=[original_arrays])
    pipeline.execute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input LAS/LAZ file')
    parser.add_argument('--output', type=str, required=True, help='Output LAS/LAZ file')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth/.ckpt checkpoint')
    parser.add_argument('--voxel-size', type=float, default=0.05, help='Voxel size (must match training)')
    parser.add_argument('--gpu', action='store_true', help='Force use of GPU')
    args = parser.parse_args()

    # Device handling
    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # 1. LOAD MODEL
    print(f"Loading model from {args.model}...")
    model = SparseUNet(input_channels=1, num_classes=4, base_channels=16)

    # Load weights
    checkpoint = torch.load(args.model, map_location=device)

    # Handle Lightning Checkpoint vs Raw State Dict
    # Filter Lightning keys (remove 'class_weights', 'criterion', etc.)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                name = k.replace("model.", "")
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        # Fallback to raw state dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 2. LOAD DATA
    print(f"Loading data: {args.input}")
    try:
        raw_xyz, raw_intensity, meta, original_arrays = load_las(args.input)
    except RuntimeError as e:
        print(f"Failed to load LAS: {e}")
        return

    if len(raw_xyz) < 100:
        print("File empty or too small. Skipping.")
        return

    # 3. PREPROCESS (Normalize & Voxelize)
    # --- NORMALIZATION ON THE FLY ---
    # Shift to local coordinates (min=0) to match training distribution
    xyz_min = raw_xyz.min(axis=0)
    xyz_centered = raw_xyz - xyz_min

    # Calculate stats
    # x_mean = raw_xyz[:, 0].mean()
    # y_mean = raw_xyz[:, 1].mean()
    # z_min = raw_xyz[:, 2].min()

    # # Apply shifts
    # xyz_centered = raw_xyz.copy()
    # xyz_centered[:, 0] -= x_mean
    # xyz_centered[:, 1] -= y_mean
    # xyz_centered[:, 2] -= z_min

    # Quantize
    discrete_coords = np.floor(xyz_centered / args.voxel_size).astype(np.int32)

    # Unique Voxel Logic
    # unique_coords: The voxels fed to the network (M, 3)
    # unique_inverse_indices: Mapping from Original Points (N) -> Voxel Index (M)
    unique_coords, unique_inverse_indices = np.unique(discrete_coords, axis=0, return_inverse=True)

    # Feature Aggregation (Mean Intensity per Voxel)
    print("Aggregating features...")
    flat_intensity = raw_intensity.ravel()

    # Sum of intensity per voxel index
    sum_features = np.bincount(unique_inverse_indices, weights=flat_intensity)
    count_features = np.bincount(unique_inverse_indices)

    # Mean intensity per voxel
    voxel_features = (sum_features / count_features).reshape(-1, 1)

    print(f"Voxelization: {len(raw_xyz)} points -> {len(unique_coords)} voxels")

    # Prepare Tensor for SpConv
    # Add Batch Dimension (0) to coordinates -> [BatchIdx, X, Y, Z]
    batch_coords = np.pad(unique_coords, ((0,0), (1,0)), mode='constant', constant_values=0)

    # Dynamic Spatial Shape (max coord + padding)
    spatial_shape = (unique_coords.max(0) + 10).tolist()

    input_tensor = spconv.SparseConvTensor(
        features=torch.as_tensor(voxel_features, dtype=torch.float32, device=device),
        indices=torch.as_tensor(batch_coords, dtype=torch.int32, device=device),
        spatial_shape=spatial_shape,
        batch_size=1
    )

    # 4. INFERENCE
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        # output is dense features tensor (N_voxels, Num_Classes)
        voxel_logits = output.cpu().numpy()
        voxel_preds = np.argmax(voxel_logits, axis=1)

    # 5. MAP PREDICTIONS BACK TO POINTS
    # Assign every point the label of the voxel it falls into
    point_labels_model = voxel_preds[unique_inverse_indices]

    # Map Model Classes -> LAS Codes
    point_labels_las = np.zeros_like(point_labels_model, dtype=np.uint8)
    for model_class, las_code in MODEL_TO_LAS_MAP.items():
        point_labels_las[point_labels_model == model_class] = las_code

    # 6. SAVE
    print(f"Saving to {args.output}...")
    save_las(args.output, original_arrays, point_labels_las, meta)
    print("Done.")

if __name__ == "__main__":
    main()
