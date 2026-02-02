"""
Plot training curves from PyTorch Lightning CSVLogger metrics.

By default loads metrics from ./experiments/bridge_classify_base/version_0/metrics.csv
and saves training_curves.png in the same directory. Use --csv to specify a different
metrics file path, or --root/--exp/--ver to pick an experiment/version.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def get_experiment_dir(base_dir, exp_name, version=None):
    exp_dir = Path(base_dir) / exp_name

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    if version is None:
        # Find latest version
        versions = sorted([d for d in exp_dir.glob("version_*") if d.is_dir()],
                         key=lambda x: int(x.name.split('_')[1]))
        if not versions:
            raise FileNotFoundError(f"No versions found in {exp_dir}")
        target_dir = versions[-1]
        print(f"Auto-selected latest version: {target_dir.name}")
    else:
        target_dir = exp_dir / f"version_{version}"
        if not target_dir.exists():
            raise FileNotFoundError(f"Version {version} not found in {exp_dir}")

    return target_dir


def discover_metric_pairs(df: pd.DataFrame) -> list[tuple[str, str | None, str]]:
    """Discover (title, train_col, val_col) from CSV columns. train_col may be None if only val is present."""
    cols = set(df.columns)
    skip = {"epoch", "step"}
    pairs: list[tuple[str, str | None, str]] = []

    for col in df.columns:
        if col in skip:
            continue
        if not col.startswith("val_"):
            continue
        name = col[4:]  # strip "val_"
        val_key = col
        train_key = None
        if f"train_{name}_epoch" in cols:
            train_key = f"train_{name}_epoch"
        elif f"train_{name}" in cols:
            train_key = f"train_{name}"
        title = name.replace("_", " ").title()
        pairs.append((title, train_key, val_key))

    # Loss first, then alphabetical by metric name
    def sort_key(item: tuple[str, str | None, str]) -> tuple[int, str]:
        title, _, _ = item
        name_lower = title.replace(" ", "_").lower()
        return (0 if name_lower == "loss" else 1, name_lower)

    pairs.sort(key=sort_key)
    return pairs


def plot_metrics(csv_path, output_dir):
    # Load metrics
    df = pd.read_csv(csv_path)

    metrics_to_plot = discover_metric_pairs(df)
    if not metrics_to_plot:
        print("No train/val metric columns found in CSV.")
        return

    # Setup plots style
    sns.set_theme(style="whitegrid")

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    # Pandas trick: PyTorch Lightning CSVLogger writes sparse rows (NaNs).
    # We group by epoch to combine them into one row per epoch.
    epoch_df = df.groupby("epoch").max()

    for i, (title, train_key, val_key) in enumerate(metrics_to_plot):
        ax = axes[i]

        has_train = train_key is not None and train_key in epoch_df.columns
        has_val = val_key in epoch_df.columns

        if has_train:
            sns.lineplot(data=epoch_df, x=epoch_df.index, y=train_key, label="Train", marker="o", ax=ax)
        if has_val:
            sns.lineplot(data=epoch_df, x=epoch_df.index, y=val_key, label="Validation", marker="o", linestyle="--", ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"Curves saved to: {save_path}")
    # plt.show() # Uncomment if running locally with display

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training curves from Lightning CSVLogger metrics. "
        "Default: ./experiments/bridge_classify_base/version_0/metrics.csv. Use --csv to override."
    )
    parser.add_argument("--root", type=str, default="./experiments", help="Root experiments folder")
    parser.add_argument(
        "--exp",
        type=str,
        default="bridge_classify_base",
        help="Experiment name (default: bridge_classify_base, i.e. version_0 under root)",
    )
    parser.add_argument(
        "--ver",
        type=int,
        default=0,
        help="Version number (default: 0). Ignored if --csv is set.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to metrics.csv; if set, use this file and save plot in its directory (ignores --root/--exp/--ver)",
    )

    args = parser.parse_args()

    try:
        if args.csv is not None:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                print(f"Error: metrics file not found: {csv_path}")
                return
            target_dir = csv_path.parent
        else:
            target_dir = get_experiment_dir(args.root, args.exp, args.ver)
            csv_path = target_dir / "metrics.csv"
            if not csv_path.exists():
                print(f"Error: metrics.csv not found in {target_dir}")
                return

        print(f"Visualizing metrics from: {csv_path}")
        plot_metrics(csv_path, target_dir)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
