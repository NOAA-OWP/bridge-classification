"""
Calculate class weights for imbalanced segmentation from preprocessed JSON metadata.

Reads all *.json files under the given data directory. Each JSON must have a
"class_distribution" object with string keys ("0", "1", ...) and integer counts
per class. Aggregates counts across files and computes inverse-frequency weights:
W_c = Total_Points / (N_classes * Count_c).

Usage:
    python utils/calculate_weights.py [--data-dir PATH] [--output weights.json]
    Then copy the printed list into train.py (argument --class-weights).

    After running utils/split_data.py, use the training folder so weights reflect
    the actual training set: --data-dir ./data/ml-data/training
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_class_counts(data_dir: Path) -> dict[int, int]:
    """
    Load and aggregate class counts from all JSON metadata files under data_dir.

    Expects each JSON to have "class_distribution" with string keys ("0", "1", ...)
    and integer values. Keys are converted to int; counts are summed across files.

    Args:
        data_dir: Root path (e.g. silver_training_normalized) to search for *.json.

    Returns:
        Mapping from class_id (int) to total point count across all files.
        Empty dict if no JSON files found.
    """
    json_files = list(data_dir.rglob("*.json"))
    total_counts: dict[int, int] = defaultdict(int)

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
            for class_id, count in meta.get("class_distribution", {}).items():
                total_counts[int(class_id)] += count
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    return dict(total_counts)


def compute_weights(total_counts: dict[int, int], n_classes: int) -> list[float]:
    """
    Compute inverse-frequency weight for each class id 0..max_class_id.

    Formula: W_c = total_points / (n_classes * count_c). Classes with no samples
    get weight 0.0.

    Args:
        total_counts: Class id -> total point count.
        n_classes: Number of classes (len of present classes).

    Returns:
        List of weights, index = class id; length = max_class_id + 1.
    """
    if not total_counts:
        return []
    total_points = sum(total_counts.values())
    max_class_id = max(total_counts.keys())
    weights_list: list[float] = []

    for c in range(max_class_id + 1):
        if c not in total_counts or total_counts[c] == 0:
            weights_list.append(0.0)
            continue
        raw_weight = total_points / (n_classes * total_counts[c])
        weights_list.append(raw_weight)

    return weights_list


def print_report(
    total_counts: dict[int, int],
    weights_list: list[float],
    verbose: bool = False,
) -> None:
    """
    Print dataset distribution and recommended class weights to stdout.

    Args:
        total_counts: Class id -> total point count.
        weights_list: Weight per class (index = class id).
        verbose: If True, also print raw total_counts and weights_list.
    """
    total_points = sum(total_counts.values())
    sorted_classes = sorted(total_counts.keys())

    print("\n" + "=" * 40)
    print("DATASET DISTRIBUTION")
    print("=" * 40)
    print(f"Total Points: {total_points:,}")

    for c in sorted_classes:
        count = total_counts[c]
        pct = (count / total_points) * 100
        print(f"Class {c}: {count:>15,} ({pct:>6.2f}%)")

    n_classes = len(sorted_classes)
    max_class_id = max(sorted_classes) if sorted_classes else -1

    print("\n" + "=" * 40)
    print("RECOMMENDED CLASS WEIGHTS")
    print("Formula: N_total / (N_classes * N_class)")
    print("=" * 40)

    for c in range(max_class_id + 1):
        if c not in total_counts or total_counts[c] == 0:
            print(f"Class {c}: 0.00 (No samples found)")
            continue
        w = weights_list[c] if c < len(weights_list) else 0.0
        print(f"Class {c}: {w:.4f}")

    print("\nCopy this list into train.py (argument --class-weights):")
    print(str(weights_list))

    if verbose:
        print(total_counts)
        print(weights_list)


def print_imbalance_warning(total_counts: dict[int, int]) -> None:
    """
    Print Ground/Bridge (class 1 vs 2) imbalance ratio if both classes exist.

    Warns when ratio > 100 and suggests manually increasing bridge weight.
    """
    if (
        1 not in total_counts
        or 2 not in total_counts
        or total_counts[2] == 0
    ):
        return
    ratio = total_counts[1] / total_counts[2]
    print(f"\nImbalance Factor (Ground/Bridge): {ratio:.1f}x")
    if ratio > 100:
        print(
            "(!) High imbalance. Consider increasing Bridge weight manually "
            "(e.g., 2x calculated value)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Class Weights from Preprocessed JSON Metadata"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/ml-data/silver_training_normalized"),
        help="Path to 'silver_training_normalized' directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON with 'weights' and 'total_counts'",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw total_counts and weights_list",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    json_files = list(data_dir.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} metadata files. Aggregating stats...")

    total_counts = load_class_counts(data_dir)
    if not total_counts:
        print("No class distribution data found in any JSON.")
        return

    n_classes = len(total_counts)
    weights_list = compute_weights(total_counts, n_classes)

    print_report(total_counts, weights_list, verbose=args.verbose)
    print_imbalance_warning(total_counts)

    if args.output is not None:
        out = {
            "weights": weights_list,
            "total_counts": total_counts,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote weights and counts to {args.output}")


if __name__ == "__main__":
    main()
