"""
Bridge Classification Inference Script

Loads a trained Sparse U-Net model, processes raw LAS/LAZ file(s),
and outputs classified LAS/LAZ file(s) with ASPRS standard codes.

Workflow:
1. Load Model (once).
2. For each input file:
   a. Load LAS file.
   b. Voxelize points (keep track of which point belongs to which voxel).
   c. Run Model Inference.
   d. Map Voxel Labels -> Original Points.
   e. Save LAS file.

Usage (single file):
    python src/inference.py \
        --input ./data/ml-data/testing/02050206/bridge_10598181_....laz \
        --output ./data/ml-data/prediction.laz \
        --model ./experiments/bridge-base-v0/.../checkpoints/....ckpt \
        --gpu

Usage (batch via pairs file):
    python src/inference.py \
        --pairs-file ./pairs.tsv \
        --model ./experiments/bridge-base-v0/.../checkpoints/....ckpt \
        --gpu

    Where pairs.tsv has one tab-separated line per file:
        /path/to/input1.laz\t/path/to/output1.laz
        /path/to/input2.laz\t/path/to/output2.laz
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


def load_model(checkpoint_path, device):
    """Load a trained SparseUNet model from a checkpoint file.

    Handles both Lightning checkpoints (strips 'model.' prefix from keys)
    and raw state dict checkpoints.

    Args:
        checkpoint_path: Path to .ckpt or .pth checkpoint file.
        device: Target torch device (cuda or cpu).

    Returns:
        SparseUNet model in eval mode on the specified device.
    """
    print(f"Loading model from {checkpoint_path}...")
    model = SparseUNet(input_channels=1, num_classes=4, base_channels=16)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    return model


def run_inference(model, input_path, output_path, voxel_size=0.05, device=torch.device("cpu")):
    """Run inference on a single LAS/LAZ file and save the classified result.

    Args:
        model: Pre-loaded SparseUNet model (already on device, in eval mode).
        input_path: Path to input LAS/LAZ file.
        output_path: Path to write classified LAS/LAZ file.
        voxel_size: Voxel size in meters (must match training). Default: 0.05.
        device: Device the model is on. Default: cpu.

    Returns:
        True if inference succeeded, False if the file was skipped or failed.
    """
    try:
        # 1. LOAD DATA
        print(f"Loading data: {input_path}")
        raw_xyz, raw_intensity, meta, original_arrays = load_las(input_path)

        if len(raw_xyz) < 100:
            print(f"WARN: {input_path} has < 100 points, skipping.")
            return False

        # 2. PREPROCESS (Normalize & Voxelize)
        # Shift to local coordinates (min=0) to match training distribution
        xyz_min = raw_xyz.min(axis=0)
        xyz_centered = raw_xyz - xyz_min

        # Quantize
        discrete_coords = np.floor(xyz_centered / voxel_size).astype(np.int32)

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

        # 3. INFERENCE
        print("Running inference...")
        with torch.no_grad():
            output = model(input_tensor)
            # output is dense features tensor (N_voxels, Num_Classes)
            voxel_logits = output.cpu().numpy()
            voxel_preds = np.argmax(voxel_logits, axis=1)

        # 4. MAP PREDICTIONS BACK TO POINTS
        # Assign every point the label of the voxel it falls into
        point_labels_model = voxel_preds[unique_inverse_indices]

        # Map Model Classes -> LAS Codes
        point_labels_las = np.zeros_like(point_labels_model, dtype=np.uint8)
        for model_class, las_code in MODEL_TO_LAS_MAP.items():
            point_labels_las[point_labels_model == model_class] = las_code

        # 5. SAVE
        print(f"Saving to {output_path}...")
        save_las(output_path, original_arrays, point_labels_las, meta)
        print(f"Done: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR: failed processing {input_path}: {e}")
        return False


def parse_pairs_file(filepath):
    """Parse a TSV file of input/output path pairs.

    Each line: input_path<TAB>output_path

    Returns:
        List of (input_path, output_path) tuples.
    """
    pairs = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_num} in {filepath}: expected 2 tab-separated fields, got {len(parts)}"
                )
            pairs.append((parts[0], parts[1]))
    return pairs


def run_batch_inference(model, pairs, voxel_size=0.05, device=torch.device("cpu")):
    """Run inference on multiple input/output file pairs.

    Processes each pair sequentially, continuing on failure so one bad file
    does not prevent the rest from being processed.

    Args:
        model: Pre-loaded SparseUNet model (already on device, in eval mode).
        pairs: List of (input_path, output_path) tuples.
        voxel_size: Voxel size in meters. Default: 0.05.
        device: Device the model is on. Default: cpu.

    Returns:
        Tuple of (succeeded_count, failed_count).
    """
    succeeded = 0
    failed = 0
    total = len(pairs)
    for i, (input_path, output_path) in enumerate(pairs, 1):
        print(f"\n[{i}/{total}] {input_path} -> {output_path}")
        ok = run_inference(model, input_path, output_path, voxel_size, device)
        if ok:
            succeeded += 1
        else:
            failed += 1
    print(f"\nBatch complete: {succeeded} succeeded, {failed} failed out of {total}")
    return succeeded, failed


def main():
    parser = argparse.ArgumentParser(description="Bridge Classification Inference")
    parser.add_argument('--input', type=str, default=None, help='Input LAS/LAZ file (single-file mode)')
    parser.add_argument('--output', type=str, default=None, help='Output LAS/LAZ file (single-file mode)')
    parser.add_argument('--pairs-file', type=str, default=None,
                        help='TSV file with input<TAB>output pairs (batch mode)')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth/.ckpt checkpoint')
    parser.add_argument('--voxel-size', type=float, default=0.05, help='Voxel size (must match training)')
    parser.add_argument('--gpu', action='store_true', help='Force use of GPU')
    args = parser.parse_args()

    # Validate: either single-file mode or batch mode, not both
    if args.pairs_file and (args.input or args.output):
        parser.error("Cannot use --pairs-file with --input/--output. Choose one mode.")
    if args.pairs_file is None and (args.input is None or args.output is None):
        parser.error("Provide either --input and --output, or --pairs-file.")

    # Device handling
    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load model ONCE
    model = load_model(args.model, device)

    # Dispatch
    if args.pairs_file:
        pairs = parse_pairs_file(args.pairs_file)
        succeeded, failed = run_batch_inference(model, pairs, args.voxel_size, device)
        if failed > 0:
            sys.exit(1)
    else:
        ok = run_inference(model, args.input, args.output, args.voxel_size, device)
        if not ok:
            sys.exit(1)

if __name__ == "__main__":
    main()
