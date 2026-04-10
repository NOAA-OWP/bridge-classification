"""Compare model evaluation metrics from the model registry.

Reads registry.json (local or S3) and prints a sorted comparison table.

Usage:
    # Local registry (default):
    python utils/compare_experiments.py

    # Specific eval set:
    python utils/compare_experiments.py --eval-set gold-35

    # Sort by recall:
    python utils/compare_experiments.py --sort bridge_deck_recall

    # From S3:
    python utils/compare_experiments.py --from-s3 --profile data
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRIC_COLUMNS = {
    "bridge_deck_iou": "Deck IoU",
    "bridge_deck_precision": "Precision",
    "bridge_deck_recall": "Recall",
    "bridge_deck_f1": "F1",
    "overall_accuracy": "Acc",
    "mean_iou": "Mean IoU",
}


def build_comparison_table(registry, eval_set, sort_by):
    """Build a DataFrame comparing models on a given eval set."""
    rows = []
    for name, entry in registry["models"].items():
        evals = entry.get("evaluation", {})
        if eval_set not in evals:
            continue
        metrics = evals[eval_set]
        parent = entry.get("parent_model")
        # Shorten parent name for display
        if parent:
            parent = parent.replace("bridge-base-all-data-", "")
        row = {
            "Model": name,
            "Stage": entry.get("stage", ""),
            "Parent": parent or "—",
        }
        for key, label in METRIC_COLUMNS.items():
            row[label] = metrics.get(key, "")
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    sort_label = METRIC_COLUMNS.get(sort_by, sort_by)
    if sort_label in df.columns:
        df = df.sort_values(sort_label, ascending=False, ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare model evaluation metrics from the registry"
    )
    parser.add_argument("--registry", type=Path,
                        default=Path("data/models/registry.json"),
                        help="Local registry path (default: data/models/registry.json)")
    parser.add_argument("--eval-set", default=None,
                        help="Evaluation set to compare (e.g. gold-134, gold-35). If not set, shows all eval sets.")
    parser.add_argument("--sort", default="bridge_deck_iou",
                        help="Metric to sort by, descending (default: bridge_deck_iou)")
    parser.add_argument("--from-s3", action="store_true",
                        help="Load registry from S3 instead of local file")
    parser.add_argument("--bucket", default="fimc-data",
                        help="S3 bucket (default: fimc-data)")
    parser.add_argument("--prefix", default="bridge-classification/models",
                        help="S3 prefix (default: bridge-classification/models)")
    parser.add_argument("--profile", default=None,
                        help="AWS profile name")
    args = parser.parse_args()

    if args.from_s3:
        from register_model import load_registry
        from src.s3 import create_s3_client
        s3_client = create_s3_client(profile=args.profile)
        registry_key = f"{args.prefix}/registry.json"
        registry = load_registry(s3_client, args.bucket, registry_key)
    else:
        registry_path = args.registry.resolve()
        if not registry_path.exists():
            raise SystemExit(f"Error: registry not found: {registry_path}")
        with open(registry_path) as f:
            registry = json.load(f)

    # Discover all eval sets in the registry
    all_eval_sets = set()
    for entry in registry["models"].values():
        all_eval_sets.update(entry.get("evaluation", {}).keys())

    if not all_eval_sets:
        print("No evaluations found in registry.")
        return

    eval_sets = sorted(all_eval_sets) if args.eval_set is None else [args.eval_set]
    sort_label = METRIC_COLUMNS.get(args.sort, args.sort)
    prod = registry.get("production")
    prod_info = f"  production: {prod['model_name']}" if prod else ""

    for i, eval_set in enumerate(eval_sets):
        df = build_comparison_table(registry, eval_set, args.sort)
        if df is None:
            print(f"No models have evaluation set '{eval_set}'.")
            continue
        if i > 0:
            print()
        print(f"Comparison: {eval_set} (sorted by {sort_label} desc)\n")
        print(df.to_string(index=False))
        print(f"\n{len(df)} models, eval set: {eval_set}{prod_info}")


if __name__ == "__main__":
    main()
