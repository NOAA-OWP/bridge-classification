"""
Split bridge data into fixed 70/15/15 train/validation/test by HUC.

Distributes files based on the target consumer:
- Training/Validation: Receives Preprocessed .npy/.json files (Optimized for Training).
- Testing: Receives EVERYTHING (.laz + .npy + .json).
           - .laz for Human Annotation/Gold Data generation.
           - .npy/.json for Model Inference/Evaluation.

After running, training/ and validation/ are ready for train.py, e.g.:
  train.py --train-dir <output-dir>/training [--val-dir <output-dir>/validation]

Usage:
    # Basic usage (Symlinking is recommended to save space)
    # Defaults to seed=27
    python utils/split_data.py --symlink

    # Custom paths
    python utils/split_data.py \
        --laz-dir ./data/ml-data/silver_training \
        --npy-dir ./data/ml-data/silver_training_normalized \
        --output-dir ./data/ml-data \
        --symlink \
        --seed 27

    # With holdout test IDs (gold/test set fixed; excluded from train/val)
    python utils/split_data.py --holdout-test-ids holdout_test_ids.txt ...
    # File format: one line per bridge, huc_id/bridge_stem (same as split_test_ids.txt)
"""

import argparse
import json
import multiprocessing
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


SPLIT_NAMES = ("training", "validation", "testing")


def discover_bridges_by_huc(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover all .laz bridge files grouped by huc_id.
    This serves as the "Source of Truth" for which bridges exist.
    """
    huc_bridges: Dict[str, List[Path]] = {}
    # Sort ensures deterministic order before shuffling
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        huc_id = child.name
        laz_files = list(child.glob("*.laz")) + list(child.glob("*.las"))
        if not laz_files:
            continue
        # Sort ensures deterministic order before shuffling
        huc_bridges[huc_id] = sorted(laz_files)
    return huc_bridges


def _load_holdout_test_ids(path: Path) -> Tuple[List[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Load holdout test IDs from file (one line per bridge: huc_id/bridge_stem).

    Returns:
        (list in file order, deduped; set for fast lookup).
    """
    if not path.exists():
        raise FileNotFoundError(f"Holdout test IDs file not found: {path}")
    pairs: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid holdout line (expected huc_id/bridge_stem): {line!r}")
            pairs.append((parts[0], parts[1]))
    # Dedupe preserving order (first occurrence wins)
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return (deduped, seen)


def _filter_holdout_from_pool(
    huc_bridges: Dict[str, List[Path]],
    holdout_set: Set[Tuple[str, str]],
) -> Dict[str, List[Path]]:
    """
    Exclude holdout (huc_id, bridge_stem) from the pool. Omit HUCs that end up empty.
    """
    filtered: Dict[str, List[Path]] = {}
    for huc_id, laz_paths in huc_bridges.items():
        kept = [p for p in laz_paths if (huc_id, p.stem) not in holdout_set]
        if kept:
            filtered[huc_id] = kept
    return filtered


def assign_splits(
    huc_bridges: Dict[str, List[Path]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    For each HUC, shuffle bridges with seed and assign train_ratio/val_ratio/test_ratio.
    Returns lists of (huc_id, bridge_stem).
    """
    rng = random.Random(seed)
    train_ids: List[Tuple[str, str]] = []
    val_ids: List[Tuple[str, str]] = []
    test_ids: List[Tuple[str, str]] = []

    for huc_id, laz_paths in huc_bridges.items():
        bridges = [p.stem for p in laz_paths]
        # This shuffle is reproducible ONLY if the input list 'bridges'
        # is sorted identically every time (which we enforce above).
        rng.shuffle(bridges)

        n = len(bridges)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val
        if test_ratio == 0:
            n_test = 0
            n_val = n - n_train

        i = 0
        for _ in range(n_train):
            train_ids.append((huc_id, bridges[i]))
            i += 1
        for _ in range(n_val):
            val_ids.append((huc_id, bridges[i]))
            i += 1
        for _ in range(n_test):
            test_ids.append((huc_id, bridges[i]))
            i += 1

    return train_ids, val_ids, test_ids


def transfer_file(src: Path, dst: Path, use_symlink: bool):
    """Helper to copy or symlink a file."""
    if not src.exists():
        # Do not crash if missing, just warn (useful if some preprocess failed)
        print(f"Warning: Source file missing: {src}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
        except OSError as e:
            # Fallback to copy if symlink fails (e.g., Windows without admin)
            print(f"Symlink failed ({e}), copying instead: {src.name}")
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def _transfer_task(args: Tuple[str, str, bool]) -> None:
    """
    Picklable worker for multiprocessing: performs one copy/symlink.

    Args:
        args: (src_str, dst_str, use_symlink) with paths as strings for pickling.
    """
    src_str, dst_str, use_symlink = args
    transfer_file(Path(src_str), Path(dst_str), use_symlink)


def _build_transfer_tasks(
    laz_dir: Path,
    npy_dir: Path,
    output_dir: Path,
    train_ids: List[Tuple[str, str]],
    val_ids: List[Tuple[str, str]],
    test_ids: List[Tuple[str, str]],
    use_symlink: bool,
) -> List[Tuple[str, str, bool]]:
    """
    Build a flat list of (src_str, dst_str, use_symlink) for all file transfers.
    """
    task_defs = [
        # Split Name | ID List | Source Dir | Extensions
        ("training", train_ids, npy_dir, [".npy", ".json"]),
        ("validation", val_ids, npy_dir, [".npy", ".json"]),
        ("testing", test_ids, laz_dir, [".laz", ".las"]),     # For Humans
        ("testing", test_ids, npy_dir, [".npy", ".json"]),   # For Model
    ]
    tasks: List[Tuple[str, str, bool]] = []
    for split_name, id_list, source_root, extensions in task_defs:
        split_base = output_dir / split_name
        for huc_id, bridge_stem in id_list:
            for ext in extensions:
                src = source_root / huc_id / f"{bridge_stem}{ext}"
                dst = split_base / huc_id / f"{bridge_stem}{ext}"
                tasks.append((str(src), str(dst), use_symlink))
    return tasks


def write_split_dirs(
    laz_dir: Path,
    npy_dir: Path,
    output_dir: Path,
    train_ids: List[Tuple[str, str]],
    val_ids: List[Tuple[str, str]],
    test_ids: List[Tuple[str, str]],
    use_symlink: bool,
    num_workers: Optional[int] = None,
) -> None:
    """
    Create training/, validation/, testing/ directories.
    - Training/Validation: Get .npy and .json from npy_dir.
    - Testing: Gets .laz (from laz_dir) AND .npy/.json (from npy_dir).

    Args:
        num_workers: Number of parallel workers (None = CPU count, 1 = sequential).
    """
    tasks = _build_transfer_tasks(
        laz_dir, npy_dir, output_dir, train_ids, val_ids, test_ids, use_symlink
    )
    n_tasks = len(tasks)
    workers = num_workers if num_workers is not None else multiprocessing.cpu_count()

    if workers <= 1:
        iterator = tasks
        if HAS_TQDM:
            iterator = tqdm(tasks, desc="Transferring files", total=n_tasks)
        for task in iterator:
            _transfer_task(task)
    else:
        print(f"Transferring {n_tasks} files with {workers} workers...")
        with multiprocessing.Pool(processes=workers) as pool:
            if HAS_TQDM:
                list(tqdm(
                    pool.imap_unordered(_transfer_task, tasks),
                    total=n_tasks,
                    desc="Transferring files",
                ))
            else:
                pool.map(_transfer_task, tasks)


def write_manifests(
    output_dir: Path,
    train_ids: List[Tuple[str, str]],
    val_ids: List[Tuple[str, str]],
    test_ids: List[Tuple[str, str]],
) -> None:
    """Write split IDs to text files and JSON manifest."""
    for name, id_list in (
        ("split_train_ids.txt", train_ids),
        ("split_val_ids.txt", val_ids),
        ("split_test_ids.txt", test_ids),
    ):
        path = output_dir / name
        with open(path, "w") as f:
            for huc_id, bridge_stem in id_list:
                f.write(f"{huc_id}/{bridge_stem}\n")

    manifest = {
        "train": [{"huc_id": h, "bridge_stem": b} for h, b in train_ids],
        "val": [{"huc_id": h, "bridge_stem": b} for h, b in val_ids],
        "test": [{"huc_id": h, "bridge_stem": b} for h, b in test_ids],
    }
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split bridge data. Distributes NPY/JSON for Train/Val and ALL formats for Test."
    )
    parser.add_argument(
        "--laz-dir",
        type=Path,
        default=Path("./data/ml-data/silver_training"),
        help="Input: Directory containing raw .laz files",
    )
    parser.add_argument(
        "--npy-dir",
        type=Path,
        default=Path("./data/ml-data/silver_training_normalized"),
        help="Input: Directory containing preprocessed .npy/.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/ml-data"),
        help="Output base directory; training/ and validation/ under it are ready for train.py --train-dir <output-dir>/training --val-dir <output-dir>/validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Random seed for reproducible split (default: 27)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying (Recommended)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers for file transfer (default: CPU count). Use 1 for sequential.",
    )
    # Ratios (default=None so we can detect "not provided"; if none given, use 0.70/0.15/0.15)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Training fraction; if only two of train/val/test are provided and sum to 1, the third is 0 (two-way split); if sum < 1, third is computed",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation fraction; see --train-ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Test fraction; see --train-ratio. If none provided, defaults 0.70/0.15/0.15 are used.",
    )
    parser.add_argument(
        "--holdout-test-ids",
        type=Path,
        default=None,
        help="Path to file of holdout test IDs (one per line: huc_id/bridge_stem). These bridges are always placed in testing/ and excluded from train/val.",
    )

    args = parser.parse_args()

    # Resolve train/val/test ratios (default=None means "not provided")
    train_r = args.train_ratio
    val_r = args.val_ratio
    test_r = args.test_ratio
    n_given = sum(1 for x in (train_r, val_r, test_r) if x is not None)

    if n_given == 0:
        train_r, val_r, test_r = 0.70, 0.15, 0.15
    elif n_given == 1:
        the_one = next(x for x in (train_r, val_r, test_r) if x is not None)
        remainder = 1.0 - the_one
        if remainder < 0:
            raise SystemExit("Error: The provided ratio must be ≤ 1.")
        half = remainder / 2.0
        if train_r is None:
            train_r = half
        if val_r is None:
            val_r = half
        if test_r is None:
            test_r = half
    elif n_given == 2:
        given_vals = [x for x in (train_r, val_r, test_r) if x is not None]
        sum_two = sum(given_vals)
        if sum_two > 1:
            raise SystemExit(
                "Error: The two provided ratios must sum to 1 or less "
                f"(got {given_vals[0]} + {given_vals[1]} = {sum_two})."
            )
        if sum_two == 1:
            # Two-way split: missing ratio = 0
            if train_r is None:
                train_r = 0.0
            if val_r is None:
                val_r = 0.0
            if test_r is None:
                test_r = 0.0
        else:
            # Three-way: missing = 1 - sum_two
            missing = 1.0 - sum_two
            if not (0 <= missing <= 1):
                raise SystemExit(f"Error: Computed third ratio {missing} is not in [0, 1].")
            if train_r is None:
                train_r = missing
            elif val_r is None:
                val_r = missing
            else:
                test_r = missing
    else:
        # n_given == 3
        total = train_r + val_r + test_r
        if abs(total - 1.0) > 1e-6:
            raise SystemExit(
                f"Error: Ratios must sum to 1 (got {train_r} + {val_r} + {test_r} = {total})."
            )

    # Final validation
    for name, r in (("train", train_r), ("val", val_r), ("test", test_r)):
        if not (0 <= r <= 1):
            raise SystemExit(f"Error: {name}_ratio must be in [0, 1], got {r}.")
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise SystemExit("Error: Resolved ratios must sum to 1.")

    # 1. Validation
    laz_dir = args.laz_dir.resolve()
    npy_dir = args.npy_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not laz_dir.exists():
        raise SystemExit(f"Error: LAZ directory not found: {laz_dir}")
    if not npy_dir.exists():
        raise SystemExit(
            f"Error: NPY directory not found: {npy_dir}\n"
            "Training and validation splits require preprocessed .npy/.json files. Run preprocess_bridges.py first."
        )

    # 2. Discover Bridges (Source of Truth is LAZ)
    huc_bridges = discover_bridges_by_huc(laz_dir)
    total_bridges = sum(len(x) for x in huc_bridges.values())
    print(f"Found {total_bridges} bridges in {len(huc_bridges)} HUCs.")

    # 3. Calculate Split (optionally exclude holdout from pool, then merge into test)
    if args.holdout_test_ids is not None:
        holdout_path = args.holdout_test_ids.resolve()
        try:
            holdout_list, holdout_set = _load_holdout_test_ids(holdout_path)
        except FileNotFoundError as e:
            raise SystemExit(f"Error: {e}") from e
        except ValueError as e:
            raise SystemExit(f"Error: {e}") from e
        full_bridge_set: Set[Tuple[str, str]] = set()
        for huc_id, paths in huc_bridges.items():
            for p in paths:
                full_bridge_set.add((huc_id, p.stem))
        holdout_present = [(h, b) for (h, b) in holdout_list if (h, b) in full_bridge_set]
        for (h, b) in holdout_list:
            if (h, b) not in full_bridge_set:
                print(f"Warning: Holdout ID not found in data: {h}/{b}")
        huc_bridges_pool = _filter_holdout_from_pool(huc_bridges, holdout_set)
        train_ids, val_ids, split_test_ids = assign_splits(
            huc_bridges_pool, args.seed, train_r, val_r, test_r
        )
        test_ids = holdout_present + split_test_ids
    else:
        train_ids, val_ids, test_ids = assign_splits(
            huc_bridges, args.seed, train_r, val_r, test_r
        )

    # 4. Clean existing output dirs
    for s in SPLIT_NAMES:
        path = output_dir / s
        if path.exists():
            # Safety check: Don't delete if it's the source directory itself
            if path == laz_dir or path == npy_dir:
                continue
            shutil.rmtree(path)

    # 5. Distribute Files
    write_split_dirs(
        laz_dir, npy_dir, output_dir,
        train_ids, val_ids, test_ids,
        args.symlink,
        num_workers=args.workers,
    )

    # 6. Write Manifests
    write_manifests(output_dir, train_ids, val_ids, test_ids)

    print("\n" + "="*50)
    print("Split Complete")
    print("="*50)
    print(f"Ratios:     train={train_r:.2f} val={val_r:.2f} test={test_r:.2f}")
    print(f"Seed used:  {args.seed}")
    print(f"Training:   {len(train_ids)} bridges (.npy/.json)")
    print(f"Validation: {len(val_ids)} bridges (.npy/.json)")
    print(f"Testing:    {len(test_ids)} bridges (.laz AND .npy/.json)")
    print(f"Manifests saved to {output_dir}")

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    main()
