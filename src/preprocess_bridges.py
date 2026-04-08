"""
Bridge Normalization Preprocessing Script

Processes bridge lidar data from HUC-organized silver_training directory,
normalizes coordinates and classifications per bridge, and saves normalized
outputs while preserving HUC folder structure.

Usage:
    python src/preprocess_bridges.py
    python src/preprocess_bridges.py --input-dir ./data/ml-data/silver_training --output-dir ./data/ml-data/silver_training_normalized
    python src/preprocess_bridges.py --skip-existing --workers 4
"""

import os
import json
import argparse
import multiprocessing
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.constants import LAS_TO_MODEL_MAP
from src.las_io import read_las, normalize_intensity

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Progress bars disabled.")


def process_laz_file(filepath: Path, output_dir: Path, skip_existing: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Process a single LAZ file with normalization and class remapping.

    Args:
        filepath: Path to input LAZ file
        output_dir: Output directory for .npy and .json files
        skip_existing: If True, skip files that already have output files

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    filename = filepath.name
    file_id = filepath.stem

    # Check if already processed
    if skip_existing:
        npy_path = output_dir / f"{file_id}.npy"
        json_path = output_dir / f"{file_id}.json"
        if npy_path.exists() and json_path.exists():
            return True, None

    try:
        # 1. READ DATA
        arrays, _ = read_las(filepath)

        X = arrays['X']
        Y = arrays['Y']
        Z = arrays['Z']
        Intensity = arrays['Intensity']
        Classification = arrays['Classification']

        # 2. FILTER & REMAP LABELS
        # Create a new label array filled with 0 (Background)
        labels = np.zeros_like(Classification, dtype=np.uint8)

        # Apply mapping
        for las_class, model_label in LAS_TO_MODEL_MAP.items():
            mask = (Classification == las_class)
            labels[mask] = model_label

        # 3. NORMALIZATION
        # Center X, Y at mean (bridge centered at 0,0 in XY plane)
        x_center = np.mean(X)
        y_center = np.mean(Y)
        X_norm = X - x_center
        Y_norm = Y - y_center

        # Zero-floor Z (relative to lowest point)
        z_min = np.min(Z)
        Z_norm = Z - z_min

        # Normalize Intensity (0-1)
        I_norm = normalize_intensity(Intensity)

        # 4. CONSTRUCT FINAL TENSOR
        # Shape: (N, 5) -> [x, y, z, intensity, label]
        data_block = np.stack([X_norm, Y_norm, Z_norm, I_norm, labels], axis=1).astype(np.float32)

        # 5. SAVE ARTIFACTS
        npy_path = output_dir / f"{file_id}.npy"
        json_path = output_dir / f"{file_id}.json"

        np.save(npy_path, data_block)

        # Save Reconstruction Metadata
        metadata = {
            "original_file": filename,
            "original_path": str(filepath),
            "offsets": {
                "x_center": float(x_center),
                "y_center": float(y_center),
                "z_min": float(z_min)
            },
            "stats": {
                "point_count": int(len(X)),
                # Class 2 is Bridge Deck
                "bridge_points": int(np.sum(labels == 2)),
                # Class 1 is Ground + Water
                "ground_water_points": int(np.sum(labels == 1)),
                # Class 3 is Obstacles (Noise)
                "obstacle_points": int(np.sum(labels == 3)),
                # Class 0 is Background
                "background_points": int(np.sum(labels == 0)),
            },
            "class_distribution": {
                "0": int(np.sum(labels == 0)),
                "1": int(np.sum(labels == 1)),
                "2": int(np.sum(labels == 2)),
                "3": int(np.sum(labels == 3))
            }
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        return True, None

    except Exception as e:
        error_msg = f"Failed to process {filename}: {str(e)}"
        return False, error_msg


def process_laz_file_worker(args: Tuple[Path, Path, bool]) -> Tuple[bool, Optional[str], Path, bool]:
    """
    Worker function for multiprocessing that processes a single LAZ file.

    Args:
        args: Tuple of (filepath, output_dir, skip_existing)

    Returns:
        Tuple of (success: bool, error_message: Optional[str], filepath: Path, was_skipped: bool)
    """
    filepath, output_dir, skip_existing = args

    # Check if already processed (before processing)
    was_skipped = False
    if skip_existing:
        file_id = filepath.stem
        npy_path = output_dir / f"{file_id}.npy"
        json_path = output_dir / f"{file_id}.json"
        if npy_path.exists() and json_path.exists():
            was_skipped = True
            return True, None, filepath, was_skipped

    success, error = process_laz_file(filepath, output_dir, skip_existing)
    return success, error, filepath, was_skipped


def process_huc_folder(huc_dir: Path, output_base_dir: Path, skip_existing: bool = False,
                       show_progress: bool = True, num_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Process all LAZ files in a HUC folder.

    Args:
        huc_dir: Input HUC directory containing LAZ files
        output_base_dir: Base output directory (will create huc_id subfolder)
        skip_existing: If True, skip files that already have output files
        show_progress: If True, show progress bar (if tqdm available)
        num_workers: Number of parallel workers (None = use CPU count, 1 = sequential)

    Returns:
        Dictionary with processing statistics
    """
    huc_id = huc_dir.name
    output_huc_dir = output_base_dir / huc_id

    # Find all .laz files
    laz_files = list(huc_dir.glob("*.laz")) + list(huc_dir.glob("*.las"))

    if not laz_files:
        return {
            "huc_id": huc_id,
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }

    results = {
        "huc_id": huc_id,
        "total": len(laz_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }

    output_huc_dir.mkdir(parents=True, exist_ok=True) # create output dir once safely
    # Prepare tasks for multiprocessing
    tasks = [(laz_file, output_huc_dir, skip_existing) for laz_file in laz_files]

    # Process files
    if num_workers is None or num_workers > 1:
        # Use multiprocessing
        num_workers = num_workers or multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_workers) as pool:
            if show_progress and HAS_TQDM:
                processed_results = list(tqdm(
                    pool.imap(process_laz_file_worker, tasks),
                    total=len(tasks),
                    desc=f"Processing {huc_id}",
                    leave=False
                ))
            else:
                processed_results = pool.map(process_laz_file_worker, tasks)
    else:
        # Sequential processing
        iterator = tasks
        if show_progress and HAS_TQDM:
            iterator = tqdm(tasks, desc=f"Processing {huc_id}", leave=False)
        processed_results = [process_laz_file_worker(task) for task in iterator]

    # Aggregate results
    for success, error, laz_file, was_skipped in processed_results:
        if error is None and success:
            if was_skipped:
                results["skipped"] += 1
            else:
                results["success"] += 1
        else:
            results["failed"] += 1
            if error:
                results["errors"].append(error)

    return results


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Normalize bridge lidar data per bridge with class remapping and coordinate normalization'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='./data/ml-data/silver_training',
        help='Input directory containing HUC-organized LAZ files (default: ./data/ml-data/silver_training)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/ml-data/silver_training_normalized',
        help='Output directory for normalized .npy and .json files (default: ./data/ml-data/silver_training_normalized)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already have output files (resume capability)'
    )

    parser.add_argument(
        '--hucs',
        nargs='+',
        help='List of specific HUC IDs to process (default: all HUCs)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: CPU count = {multiprocessing.cpu_count()})'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find HUC directories
    huc_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if args.hucs:
        # Filter to specified HUCs
        huc_set = set(args.hucs)
        huc_dirs = [d for d in huc_dirs if d.name in huc_set]
        if len(huc_dirs) < len(huc_set):
            found_hucs = {d.name for d in huc_dirs}
            missing = huc_set - found_hucs
            print(f"Warning: Some specified HUCs not found: {missing}")

    if not huc_dirs:
        print(f"No HUC directories found in {input_dir}")
        return

    print(f"Found {len(huc_dirs)} HUC directory(ies) to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if args.skip_existing:
        print("Skipping already processed files")
    print("=" * 60)

    # Process each HUC folder
    all_results = []
    iterator = huc_dirs
    if not args.no_progress and HAS_TQDM:
        iterator = tqdm(huc_dirs, desc="Processing HUCs")

    # Determine number of workers
    num_workers = args.workers if args.workers is not None else multiprocessing.cpu_count()
    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")

    for huc_dir in iterator:
        results = process_huc_folder(
            huc_dir,
            output_dir,
            skip_existing=args.skip_existing,
            show_progress=not args.no_progress,
            num_workers=num_workers
        )
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)

    total_files = sum(r["total"] for r in all_results)
    total_success = sum(r["success"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_skipped = sum(r["skipped"] for r in all_results)

    print(f"Total HUCs processed: {len(all_results)}")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Skipped (already existed): {total_skipped}")

    # Print per-HUC summary
    if len(all_results) > 1:
        print("\nPer-HUC Summary:")
        for r in all_results:
            print(f"  {r['huc_id']}: {r['success']} success, {r['failed']} failed, {r['skipped']} skipped")

    # Print errors if any
    all_errors = []
    for r in all_results:
        all_errors.extend(r["errors"])

    if all_errors:
        print(f"\nErrors encountered ({len(all_errors)}):")
        for error in all_errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")

    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
