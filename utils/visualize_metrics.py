"""
Plot training curves from PyTorch Lightning CSVLogger metrics.

By default loads metrics from ./experiments/bridge_classify_base/version_0/metrics.csv
and saves training_curves.png in the same directory. Use --csv to specify a different
metrics file path, or --root/--exp/--ver to pick an experiment/version.
Use --compare exp1,exp2,... to plot multiple experiments on the same axes.
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


# Metrics we never plot (diagnostic/sampling only)
EXCLUDED_METRIC_TITLES = {"Max Sample Voxels", "Num Voxels"}

# Presentation styling: distinct colors for train vs validation
TRAIN_COLOR = "#1f77b4"   # blue
VAL_COLOR = "#d62728"     # red
# Optional: larger fonts for slides (set in plot_metrics_compare)
PRESENTATION_FONT = {"title": 14, "labels": 12, "legend": 11, "suptitle": 16}


def _shorten_run_label(label: str) -> str:
    """Shorten experiment label for legend (e.g. bridge-base-all-data-v0 -> v0)."""
    if "-v" in label:
        return "v" + label.split("-v")[-1].strip()
    if label.startswith("bridge-"):
        return label.replace("bridge-base-all-data-", "").strip() or label
    return label[:20] + "…" if len(label) > 20 else label


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
    metrics_to_plot = [
        (title, train_key, val_key)
        for (title, train_key, val_key) in metrics_to_plot
        if title not in EXCLUDED_METRIC_TITLES and (train_key is None or "epoch" in train_key)
    ]

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


def plot_metrics_compare(
    metrics_paths_with_labels: list[tuple[Path, str]],
    output_path: Path,
    merge_resumed: bool = False,
) -> None:
    """
    Plot metrics from multiple experiments on the same axes for comparison.

    Args:
        metrics_paths_with_labels: List of (path to metrics.csv, legend label).
        output_path: Where to save the figure.
        merge_resumed: If True and exactly two experiments, merge into one continuous
            timeline (first run's epochs then second run's later epochs) and plot as one series.
    """
    if not metrics_paths_with_labels:
        print("No metrics paths provided for comparison.")
        return

    # Load each CSV and aggregate by epoch
    epoch_dfs_with_labels: list[tuple[pd.DataFrame, str]] = []
    for csv_path, label in metrics_paths_with_labels:
        df = pd.read_csv(csv_path)
        epoch_df = df.groupby("epoch").max()
        epoch_dfs_with_labels.append((epoch_df, label))

    is_single_series = False
    run_caption = None
    if merge_resumed and len(epoch_dfs_with_labels) == 2:
        df0, label0 = epoch_dfs_with_labels[0]
        df1, label1 = epoch_dfs_with_labels[1]
        max_epoch_0 = df0.index.max()
        merged_df = pd.concat([
            df0[df0.index <= max_epoch_0],
            df1[df1.index > max_epoch_0],
        ]).sort_index()
        merged_label = f"{label0}→{label1}"
        epoch_dfs_with_labels = [(merged_df, merged_label)]
        is_single_series = True
        run_caption = f"{_shorten_run_label(label0)} → {_shorten_run_label(label1)}"

    first_df = pd.read_csv(metrics_paths_with_labels[0][0])
    metrics_to_plot = discover_metric_pairs(first_df)
    metrics_to_plot = [
        (title, train_key, val_key)
        for (title, train_key, val_key) in metrics_to_plot
        if title not in EXCLUDED_METRIC_TITLES and (train_key is None or "epoch" in train_key)
    ]

    if not metrics_to_plot:
        print("No train/val metric columns found in CSV.")
        return

    if not is_single_series and len(epoch_dfs_with_labels) == 1:
        is_single_series = True
        run_caption = _shorten_run_label(epoch_dfs_with_labels[0][1])

    sns.set_theme(style="whitegrid")
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    # Distinct colors for train vs val; single series uses short legend
    linestyles = ["-", "--", "-.", ":"]
    if is_single_series:
        epoch_df, _ = epoch_dfs_with_labels[0]
        for i, (title, train_key, val_key) in enumerate(metrics_to_plot):
            ax = axes[i]
            has_train = train_key is not None and train_key in epoch_df.columns
            has_val = val_key in epoch_df.columns
            if has_train:
                ax.plot(
                    epoch_df.index,
                    epoch_df[train_key],
                    label="Train",
                    marker="o",
                    markersize=4,
                    linestyle="-",
                    color=TRAIN_COLOR,
                    linewidth=2,
                )
            if has_val:
                ax.plot(
                    epoch_df.index,
                    epoch_df[val_key],
                    label="Val",
                    marker="s",
                    markersize=4,
                    linestyle="--",
                    color=VAL_COLOR,
                    linewidth=2,
                )
            ax.set_title(title, fontsize=PRESENTATION_FONT["title"])
            ax.set_xlabel("Epoch", fontsize=PRESENTATION_FONT["labels"])
            ax.set_ylabel(title, fontsize=PRESENTATION_FONT["labels"])
            ax.tick_params(axis="both", labelsize=PRESENTATION_FONT["labels"])
            ax.legend(loc="best", fontsize=PRESENTATION_FONT["legend"])
    else:
        for i, (title, train_key, val_key) in enumerate(metrics_to_plot):
            ax = axes[i]
            for idx, (epoch_df, label) in enumerate(epoch_dfs_with_labels):
                short = _shorten_run_label(label)
                ls = linestyles[idx % len(linestyles)]
                has_train = train_key is not None and train_key in epoch_df.columns
                has_val = val_key in epoch_df.columns
                if has_train:
                    ax.plot(
                        epoch_df.index,
                        epoch_df[train_key],
                        label=f"{short} (train)",
                        marker="o",
                        markersize=3,
                        linestyle=ls,
                        color=TRAIN_COLOR,
                        linewidth=1.5,
                    )
                if has_val:
                    ax.plot(
                        epoch_df.index,
                        epoch_df[val_key],
                        label=f"{short} (val)",
                        marker="s",
                        markersize=3,
                        linestyle=ls,
                        color=VAL_COLOR,
                        linewidth=1.5,
                    )
            ax.set_title(title, fontsize=PRESENTATION_FONT["title"])
            ax.set_xlabel("Epoch", fontsize=PRESENTATION_FONT["labels"])
            ax.set_ylabel(title, fontsize=PRESENTATION_FONT["labels"])
            ax.tick_params(axis="both", labelsize=PRESENTATION_FONT["labels"])
            ax.legend(loc="best", fontsize=PRESENTATION_FONT["legend"])

    if run_caption:
        fig.suptitle(run_caption, fontsize=PRESENTATION_FONT["suptitle"], y=1.02)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison curves saved to: {output_path}")


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
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Comma-separated experiment names under --root to compare on the same axes (e.g. exp1,exp2,exp3). Ignores --exp/--ver.",
    )
    parser.add_argument(
        "--compare-versions",
        type=str,
        default=None,
        help="Comma-separated version numbers, one per experiment in --compare (e.g. 0,0,0). If omitted, version 0 is used for all.",
    )
    parser.add_argument(
        "--merge-resumed",
        action="store_true",
        help="With --compare and exactly two experiments, merge into one continuous run (v0 epochs then v1 later epochs) and plot as a single series.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the comparison figure when using --compare. Default: {root}/compare_training_curves.png",
    )

    args = parser.parse_args()

    try:
        if args.compare is not None:
            # Compare mode: resolve (csv_path, label) for each experiment
            exp_names = [s.strip() for s in args.compare.split(",") if s.strip()]
            if not exp_names:
                print("Error: --compare must list at least one experiment name.")
                return
            if args.compare_versions is not None:
                version_strs = [s.strip() for s in args.compare_versions.split(",")]
                if len(version_strs) != len(exp_names):
                    print(
                        f"Error: --compare-versions must have the same number of values as --compare "
                        f"({len(exp_names)} experiments)."
                    )
                    return
                versions = [int(v) for v in version_strs]
            else:
                versions = [0] * len(exp_names)

            root = Path(args.root)
            metrics_paths_with_labels: list[tuple[Path, str]] = []
            for exp_name, ver in zip(exp_names, versions):
                metrics_path = root / exp_name / f"version_{ver}" / "metrics.csv"
                if not metrics_path.exists():
                    print(f"Error: metrics file not found: {metrics_path}")
                    return
                metrics_paths_with_labels.append((metrics_path, exp_name))

            output_path = Path(args.out) if args.out else root / "compare_training_curves.png"
            print(f"Comparing {len(metrics_paths_with_labels)} experiments.")
            plot_metrics_compare(
                metrics_paths_with_labels, output_path, merge_resumed=args.merge_resumed
            )
        elif args.csv is not None:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                print(f"Error: metrics file not found: {csv_path}")
                return
            target_dir = csv_path.parent
            print(f"Visualizing metrics from: {csv_path}")
            plot_metrics(csv_path, target_dir)
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
