"""
Visualize 3D LiDAR point clouds for poster/presentation figures.

Two modes:
  1. gold-vs-model: Human annotations (gold NPY) vs model predictions (inference LAZ)
  2. source-vs-silver: Raw elevation (source LAZ) vs weak supervision labels (silver LAZ)

Usage:
  python utils/visualize_pointcloud.py gold-vs-model \
      --gold data/ml-data/gold-data-normalized/03070101/bridge_40787878_GA_Central_1_2018.npy \
      --model data/ml-data/evaluation_results/v5-gold-134/inference_output/03070101/bridge_40787878_GA_Central_1_2018.laz \
      --title "Bridge 40787878 (GA) — Deck IoU: 79.3%" \
      -o notebooks/outputs/poster_gold_vs_model.png

  python utils/visualize_pointcloud.py source-vs-silver \
      --source data/ml-data/source/01010005/bridge_1090522653_ME_Eastern_TL_2017.laz \
      --silver data/ml-data/silver_training/01010005/bridge_1090522653_ME_Eastern_TL_2017.laz \
      -o notebooks/outputs/poster_source_vs_silver.png
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.las_io import read_las
from src.constants import CLASS_NAMES, CLASS_COLORS_HEX, LAS_TO_MODEL_MAP

DEFAULT_ELEV = 30
DEFAULT_AZIM = 225
DEFAULT_POINT_SIZE = 8
DEFAULT_MAX_POINTS = 50_000
DEFAULT_DPI = 300
DEFAULT_FONT_SCALE = 1.0
DEFAULT_BG_COLOR = "white"
DEFAULT_BORDER_COLOR = "#cccccc"
DEFAULT_BORDER_WIDTH = 0


def load_laz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, classification) arrays from a LAZ/LAS file."""
    arr, _ = read_las(path)
    xyz = np.column_stack([arr["X"], arr["Y"], arr["Z"]]).astype(np.float64)
    classification = np.asarray(arr["Classification"], dtype=np.int32)
    return xyz, classification


def load_gold_npy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, labels) from a gold-normalized NPY file (N,5)."""
    data = np.load(path)
    xyz = data[:, :3]
    labels = data[:, 4].astype(int)
    return xyz, labels


def asprs_to_model_labels(asprs_labels: np.ndarray) -> np.ndarray:
    return np.array([LAS_TO_MODEL_MAP.get(c, 0) for c in asprs_labels])


def apply_local_offset(xyz: np.ndarray) -> np.ndarray:
    return xyz - xyz.min(axis=0)


def downsample(
    xyz: np.ndarray, labels: np.ndarray | None, max_pts: int | None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_pts is None or len(xyz) <= max_pts:
        return xyz, labels
    idx = np.random.RandomState(seed).choice(len(xyz), max_pts, replace=False)
    return xyz[idx], labels[idx] if labels is not None else None


def set_equal_aspect_3d(ax: plt.Axes) -> None:
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    mr = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    for fn, lim in [(ax.set_xlim, xlim), (ax.set_ylim, ylim), (ax.set_zlim, zlim)]:
        c = (lim[0] + lim[1]) / 2
        fn([c - mr / 2, c + mr / 2])


def style_3d_ax(ax: plt.Axes, elev: float = DEFAULT_ELEV, azim: float = DEFAULT_AZIM, font_scale: float = DEFAULT_FONT_SCALE) -> None:
    ax.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 1.0))
    ax.yaxis.pane.set_facecolor((0.93, 0.93, 0.93, 1.0))
    ax.zaxis.pane.set_facecolor((0.90, 0.90, 0.90, 1.0))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 0.5), "linewidth": 0.5})
    ax.tick_params(axis="both", labelsize=9 * font_scale, pad=1)
    ax.set_xlabel("X (m)", fontsize=11 * font_scale, labelpad=5)
    ax.set_ylabel("Y (m)", fontsize=11 * font_scale, labelpad=5)
    ax.set_zlabel("Z (m)", fontsize=11 * font_scale, labelpad=5)
    ax.view_init(elev=elev, azim=azim)


def scatter_by_class(ax: plt.Axes, xyz: np.ndarray, labels: np.ndarray, point_size: float = DEFAULT_POINT_SIZE) -> None:
    # Render deck last so it draws on top
    for lbl in [0, 1, 3, 2]:
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(
            xyz[mask, 0], xyz[mask, 1], xyz[mask, 2],
            c=CLASS_COLORS_HEX[lbl], label=CLASS_NAMES[lbl],
            s=point_size, alpha=0.75, linewidths=0, rasterized=True,
            zorder=4 if lbl == 2 else 3,
        )


def add_legend(ax: plt.Axes, font_scale: float = DEFAULT_FONT_SCALE) -> None:
    leg = ax.legend(
        loc="upper left", fontsize=10 * font_scale, frameon=True,
        fancybox=True, framealpha=0.92, edgecolor="#cccccc",
        markerscale=4, handletextpad=0.5,
    )
    leg.get_frame().set_facecolor("white")


def plot_gold_vs_model(
    gold_path: Path, model_path: Path, output_path: Path,
    title: str | None = None, elev: float = DEFAULT_ELEV, azim: float = DEFAULT_AZIM,
    point_size: float = DEFAULT_POINT_SIZE, max_points: int = DEFAULT_MAX_POINTS,
    dpi: int = DEFAULT_DPI, font_scale: float = DEFAULT_FONT_SCALE,
    bg_color: str = DEFAULT_BG_COLOR, border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = DEFAULT_BORDER_WIDTH,
) -> None:
    """Side-by-side: human gold annotations vs model inference predictions."""
    gold_xyz, gold_labels = load_gold_npy(gold_path)
    model_xyz, model_asprs = load_laz(model_path)
    model_labels = asprs_to_model_labels(model_asprs)

    gold_xyz = apply_local_offset(gold_xyz)
    model_xyz = apply_local_offset(model_xyz)
    gold_xyz, gold_labels = downsample(gold_xyz, gold_labels, max_points, 42)
    model_xyz, model_labels = downsample(model_xyz, model_labels, max_points, 43)

    fig = plt.figure(figsize=(18, 8), facecolor=bg_color)
    if border_width > 0:
        fig.patch.set_edgecolor(border_color)
        fig.patch.set_linewidth(border_width)
    if title:
        fig.suptitle(title, fontsize=15 * font_scale, fontweight="bold", y=1.0, color="#1a365d")

    for idx, (xyz, labels, panel_title) in enumerate([
        (gold_xyz, gold_labels, "Human Annotations (Gold)"),
        (model_xyz, model_labels, "v5 Model Predictions"),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        scatter_by_class(ax, xyz, labels, point_size)
        ax.set_title(panel_title, fontsize=14 * font_scale, fontweight="bold", pad=16 * font_scale, color="#1a365d")
        style_3d_ax(ax, elev, azim, font_scale)
        set_equal_aspect_3d(ax)
        add_legend(ax, font_scale)

    plt.subplots_adjust(wspace=0.05, top=0.88)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color, edgecolor=border_color if border_width > 0 else "none")
    plt.close()
    print(f"Saved: {output_path}")


def plot_source_vs_silver(
    source_path: Path, silver_path: Path, output_path: Path,
    title: str | None = None, elev: float = DEFAULT_ELEV, azim: float = DEFAULT_AZIM,
    point_size: float = DEFAULT_POINT_SIZE, max_points: int = DEFAULT_MAX_POINTS,
    dpi: int = DEFAULT_DPI, font_scale: float = DEFAULT_FONT_SCALE,
    bg_color: str = DEFAULT_BG_COLOR, border_color: str = DEFAULT_BORDER_COLOR,
    border_width: float = DEFAULT_BORDER_WIDTH,
) -> None:
    """Side-by-side: raw LiDAR elevation vs RANSAC weak supervision labels."""
    src_xyz, _ = load_laz(source_path)
    sil_xyz, sil_asprs = load_laz(silver_path)
    sil_labels = asprs_to_model_labels(sil_asprs)

    src_xyz = apply_local_offset(src_xyz)
    sil_xyz = apply_local_offset(sil_xyz)
    src_xyz, _ = downsample(src_xyz, None, max_points, 42)
    sil_xyz, sil_labels = downsample(sil_xyz, sil_labels, max_points, 43)

    fig = plt.figure(figsize=(18, 8), facecolor=bg_color)
    if border_width > 0:
        fig.patch.set_edgecolor(border_color)
        fig.patch.set_linewidth(border_width)
    if title:
        fig.suptitle(title, fontsize=15 * font_scale, fontweight="bold", y=0.96, color="#1a365d")

    # Left: elevation
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(
        src_xyz[:, 0], src_xyz[:, 1], src_xyz[:, 2],
        c=src_xyz[:, 2], cmap="viridis", s=point_size, alpha=0.75,
        linewidths=0, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax1, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label("Elevation (m)", fontsize=10 * font_scale)
    cbar.ax.tick_params(labelsize=9 * font_scale)
    ax1.set_title("Raw LiDAR (colored by elevation)", fontsize=14 * font_scale, fontweight="bold", pad=12, color="#1a365d")
    style_3d_ax(ax1, elev, azim, font_scale)
    set_equal_aspect_3d(ax1)

    # Right: silver classification
    ax2 = fig.add_subplot(122, projection="3d")
    scatter_by_class(ax2, sil_xyz, sil_labels, point_size)
    add_legend(ax2, font_scale)
    ax2.set_title("Weak Supervision Labels (RANSAC)", fontsize=14 * font_scale, fontweight="bold", pad=12, color="#1a365d")
    style_3d_ax(ax2, elev, azim, font_scale)
    set_equal_aspect_3d(ax2)

    plt.subplots_adjust(wspace=0.05, top=0.90)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color, edgecolor=border_color if border_width > 0 else "none")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    gm = sub.add_parser("gold-vs-model", help="Gold annotations vs model predictions")
    gm.add_argument("--gold", required=True, help="Path to gold-normalized .npy file")
    gm.add_argument("--model", required=True, help="Path to model inference output .laz file")
    gm.add_argument("-o", "--output", default="notebooks/outputs/gold_vs_model.png")
    gm.add_argument("--title", default=None)
    gm.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    gm.add_argument("--azim", type=float, default=DEFAULT_AZIM)
    gm.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    gm.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    gm.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    gm.add_argument("--font-scale", type=float, default=DEFAULT_FONT_SCALE, help="Multiply all font sizes (1.5-1.8 for posters)")
    gm.add_argument("--bg-color", default=DEFAULT_BG_COLOR, help="Figure background color (e.g. '#F0F0F0')")
    gm.add_argument("--border-color", default=DEFAULT_BORDER_COLOR, help="Border color")
    gm.add_argument("--border-width", type=float, default=DEFAULT_BORDER_WIDTH, help="Border width in points (0=no border)")

    ss = sub.add_parser("source-vs-silver", help="Raw elevation vs silver labels")
    ss.add_argument("--source", required=True, help="Path to source .laz file")
    ss.add_argument("--silver", required=True, help="Path to silver_training .laz file")
    ss.add_argument("-o", "--output", default="notebooks/outputs/source_vs_silver.png")
    ss.add_argument("--title", default=None)
    ss.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    ss.add_argument("--azim", type=float, default=DEFAULT_AZIM)
    ss.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    ss.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    ss.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ss.add_argument("--font-scale", type=float, default=DEFAULT_FONT_SCALE, help="Multiply all font sizes (1.5-1.8 for posters)")
    ss.add_argument("--bg-color", default=DEFAULT_BG_COLOR, help="Figure background color (e.g. '#F0F0F0')")
    ss.add_argument("--border-color", default=DEFAULT_BORDER_COLOR, help="Border color")
    ss.add_argument("--border-width", type=float, default=DEFAULT_BORDER_WIDTH, help="Border width in points (0=no border)")

    args = parser.parse_args()

    if args.mode == "gold-vs-model":
        plot_gold_vs_model(
            Path(args.gold), Path(args.model), Path(args.output),
            title=args.title, elev=args.elev, azim=args.azim,
            point_size=args.point_size, max_points=args.max_points,
            dpi=args.dpi, font_scale=args.font_scale,
            bg_color=args.bg_color, border_color=args.border_color,
            border_width=args.border_width,
        )
    elif args.mode == "source-vs-silver":
        plot_source_vs_silver(
            Path(args.source), Path(args.silver), Path(args.output),
            title=args.title, elev=args.elev, azim=args.azim,
            point_size=args.point_size, max_points=args.max_points,
            dpi=args.dpi, font_scale=args.font_scale,
            bg_color=args.bg_color, border_color=args.border_color,
            border_width=args.border_width,
        )


if __name__ == "__main__":
    main()
