"""
Bridge Classification Model Evaluation Script

Evaluates a trained bridge classification model against human-annotated (gold) data.
Reports metrics for both model predictions and silver (auto-labeled) baseline,
allowing comparison of model performance vs. the weak supervision labels.

Usage:
    # Full evaluation (runs inference + computes silver baseline)
    python utils/evaluate_model.py \
        --model ./experiments/.../checkpoints/best.ckpt \
        --gold-dir ./data/ml-data/gold-data \
        --test-dir ./data/ml-data/testing \
        --output-dir ./evaluation_results

    # Pre-computed inference (skip re-running model)
    python utils/evaluate_model.py \
        --gold-dir ./data/ml-data/gold-data \
        --test-dir ./data/ml-data/testing \
        --inference-dir ./evaluation_results/inference_output \
        --output-dir ./evaluation_results
"""

import argparse
import json
import os
import shlex
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Allow importing from src/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.constants import (
    LAS_TO_MODEL_MAP, NUM_CLASSES, BRIDGE_DECK_MODEL_CLASS as BRIDGE_DECK_CLASS,
    CLASS_NAMES, BridgeTimeout, _timeout_handler,
)
from src.las_io import read_las

try:
    import torch
except ImportError:
    torch = None

try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Install it for full metrics.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def discover_gold_bridges(gold_dir: Path) -> list:
    """Scan gold directory for all LAS/LAZ files.

    Returns:
        Sorted list of (huc_id, bridge_stem, gold_path) tuples.
    """
    bridges = []
    for huc_dir in sorted(gold_dir.iterdir()):
        if not huc_dir.is_dir():
            continue
        huc_id = huc_dir.name
        for f in sorted(huc_dir.iterdir()):
            if f.suffix.lower() in (".las", ".laz"):
                bridges.append((huc_id, f.stem, f))
    return bridges


def find_matching_file(directory: Path, huc_id: str, stem: str,
                       extensions: Optional[list] = None) -> Optional[Path]:
    """Find a file at directory/huc_id/stem{ext}, trying each extension in order."""
    if extensions is None:
        extensions = [".laz", ".las"]
    for ext in extensions:
        candidate = directory / huc_id / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------

def load_classifications(filepath: Path) -> tuple:
    """Load a LAS/LAZ file and return (xyz, model_labels).

    Maps ASPRS Classification codes to model classes via LAS_TO_MODEL_MAP.
    All unmapped ASPRS codes become class 0 (Background).

    Returns:
        xyz: float32 array of shape (N, 3)
        model_labels: int32 array of shape (N,) with values 0-3
    """
    arrays, _ = read_las(filepath)

    xyz = np.stack([
        arrays["X"].astype(np.float32),
        arrays["Y"].astype(np.float32),
        arrays["Z"].astype(np.float32),
    ], axis=1)

    asprs_codes = arrays["Classification"].astype(np.int32)
    model_labels = np.zeros(len(asprs_codes), dtype=np.int32)
    for asprs_code, model_class in LAS_TO_MODEL_MAP.items():
        model_labels[asprs_codes == asprs_code] = model_class

    return xyz, model_labels


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _iou_from_cm(cm: np.ndarray) -> np.ndarray:
    """Compute per-class IoU from a (C, C) confusion matrix."""
    iou = np.zeros(cm.shape[0])
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou[c] = float(tp / denom) if denom > 0 else 0.0
    return iou


def _pr_from_cm(cm: np.ndarray) -> tuple:
    """Compute per-class precision, recall, f1, support from a confusion matrix."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    support = cm.sum(axis=1).astype(int)  # true positives per class (row sums)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = precision[c], recall[c]
        f1[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return precision, recall, f1, support


def evaluate_bridge(gold_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Compute evaluation metrics for a single bridge.

    Returns:
        Dict with confusion_matrix, per_class metrics, overall_accuracy, mean_iou.
    """
    labels = list(range(NUM_CLASSES))
    cm = sk_confusion_matrix(gold_labels, pred_labels, labels=labels)
    iou = _iou_from_cm(cm)
    precision, recall, f1, support = _pr_from_cm(cm)

    overall_acc = float((gold_labels == pred_labels).mean() * 100)
    present_classes = [c for c in labels if cm[c, :].sum() > 0]
    mean_iou = float(iou[present_classes].mean() * 100) if present_classes else 0.0

    per_class = {}
    for c in labels:
        per_class[c] = {
            "precision": float(precision[c] * 100),
            "recall": float(recall[c] * 100),
            "f1": float(f1[c] * 100),
            "iou": float(iou[c] * 100),
            "support": int(support[c]),
        }

    # Binary (bridge vs non-bridge) metrics
    binary_gold = (gold_labels == BRIDGE_DECK_CLASS).astype(np.int32)
    binary_pred = (pred_labels == BRIDGE_DECK_CLASS).astype(np.int32)
    binary_cm = sk_confusion_matrix(binary_gold, binary_pred, labels=[0, 1])
    b_iou = _iou_from_cm(binary_cm)
    b_prec, b_rec, b_f1, _ = _pr_from_cm(binary_cm)
    binary = {
        "precision": float(b_prec[1] * 100),
        "recall": float(b_rec[1] * 100),
        "f1": float(b_f1[1] * 100),
        "iou": float(b_iou[1] * 100),
        "confusion_matrix": binary_cm,
    }

    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "overall_accuracy": overall_acc,
        "mean_iou": mean_iou,
        "num_points": int(len(gold_labels)),
        "binary": binary,
    }


def aggregate_metrics(bridge_results: list) -> dict:
    """Aggregate per-bridge results into micro-averaged metrics.

    Computed from the global summed confusion matrix (point-weighted).
    """
    if not bridge_results:
        return {}

    global_cm = sum(r["confusion_matrix"] for r in bridge_results)
    iou = _iou_from_cm(global_cm)
    precision, recall, f1, support = _pr_from_cm(global_cm)

    total_points = int(global_cm.sum())
    overall_acc = float(np.diag(global_cm).sum() / total_points * 100) if total_points > 0 else 0.0
    present_classes = [c for c in range(NUM_CLASSES) if global_cm[c, :].sum() > 0]
    mean_iou = float(iou[present_classes].mean() * 100) if present_classes else 0.0

    per_class = {}
    for c in range(NUM_CLASSES):
        per_class[c] = {
            "precision": float(precision[c] * 100),
            "recall": float(recall[c] * 100),
            "f1": float(f1[c] * 100),
            "iou": float(iou[c] * 100),
            "support": int(support[c]),
        }

    # Aggregate binary metrics
    global_binary_cm = sum(r["binary"]["confusion_matrix"] for r in bridge_results)
    b_iou = _iou_from_cm(global_binary_cm)
    b_prec, b_rec, b_f1, _ = _pr_from_cm(global_binary_cm)
    binary = {
        "precision": float(b_prec[1] * 100),
        "recall": float(b_rec[1] * 100),
        "f1": float(b_f1[1] * 100),
        "iou": float(b_iou[1] * 100),
    }
    binary_nonbridge = {
        "precision": float(b_prec[0] * 100),
        "recall": float(b_rec[0] * 100),
        "f1": float(b_f1[0] * 100),
        "iou": float(b_iou[0] * 100),
    }
    b_tp = global_binary_cm[1, 1]
    b_tn = global_binary_cm[0, 0]
    binary_total = int(global_binary_cm.sum())
    binary_accuracy = float((b_tp + b_tn) / binary_total * 100) if binary_total > 0 else 0.0

    return {
        "total_bridges": len(bridge_results),
        "total_points": total_points,
        "overall_accuracy": overall_acc,
        "mean_iou": mean_iou,
        "per_class": per_class,
        "confusion_matrix": global_cm.tolist(),
        "binary": binary,
        "binary_nonbridge": binary_nonbridge,
        "binary_accuracy": binary_accuracy,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(aggregate: dict, label: str = "Model Predictions"):
    """Print a formatted summary table to console."""
    print()
    print("=" * 72)
    print(f" Bridge Classification Evaluation — {label}")
    print("=" * 72)
    print(f" Bridges: {aggregate['total_bridges']}   |   "
          f"Total points: {aggregate['total_points']:,}")
    print(f" {'Class':<26} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7} {'Support':>12}")
    print(f" {'-' * 70}")
    for c in range(NUM_CLASSES):
        m = aggregate["per_class"][c]
        name = f"{c}: {CLASS_NAMES[c]}"
        marker = " *" if c == BRIDGE_DECK_CLASS else "  "
        print(
            f" {name:<26}{marker}"
            f" {m['precision']:>6.1f}%"
            f" {m['recall']:>6.1f}%"
            f" {m['f1']:>6.1f}%"
            f" {m['iou']:>6.1f}%"
            f" {m['support']:>12,}"
        )
    print(f" {'-' * 70}")
    print(f" Overall Accuracy: {aggregate['overall_accuracy']:.1f}%   |   "
          f"Mean IoU: {aggregate['mean_iou']:.1f}%")
    print(f" Bridge Deck IoU: {aggregate['per_class'][BRIDGE_DECK_CLASS]['iou']:.1f}%  "
          f"  Bridge Deck Recall: {aggregate['per_class'][BRIDGE_DECK_CLASS]['recall']:.1f}%")
    print("=" * 72)


def print_comparison_table(model_agg: dict, silver_agg: dict):
    """Print a side-by-side comparison table with delta (model - silver)."""
    if not model_agg or not silver_agg:
        return
    W = 84
    print()
    print("=" * W)
    print(" Model vs Silver Baseline — Comparison (Delta = Model - Silver)")
    print("=" * W)
    print(f" {'Class':<26}  {'Model IoU':>10} {'Silver IoU':>11} {'Delta':>8}")
    print(f" {'-' * 82}")
    for c in range(NUM_CLASSES):
        m_iou = model_agg["per_class"][c]["iou"]
        s_iou = silver_agg["per_class"][c]["iou"]
        delta = m_iou - s_iou
        sign = "+" if delta >= 0 else ""
        name = f"{c}: {CLASS_NAMES[c]}"
        marker = " *" if c == BRIDGE_DECK_CLASS else "  "
        print(f" {name:<26}{marker}  {m_iou:>9.1f}%  {s_iou:>10.1f}%  {sign}{delta:>6.1f}%")
    print(f" {'-' * 82}")
    m_oa = model_agg["overall_accuracy"]
    s_oa = silver_agg["overall_accuracy"]
    d_oa = m_oa - s_oa
    m_mi = model_agg["mean_iou"]
    s_mi = silver_agg["mean_iou"]
    d_mi = m_mi - s_mi
    print(f" {'Overall Accuracy':<30}  {m_oa:>9.1f}%  {s_oa:>10.1f}%  {'+' if d_oa >= 0 else ''}{d_oa:>6.1f}%")
    print(f" {'Mean IoU':<30}  {m_mi:>9.1f}%  {s_mi:>10.1f}%  {'+' if d_mi >= 0 else ''}{d_mi:>6.1f}%")
    print("=" * W)


def print_binary_table(model_agg: dict, silver_agg: dict):
    """Print binary (bridge vs non-bridge) metrics for model and silver.

    Shows both bridge and non-bridge class metrics plus binary overall accuracy.
    The non-bridge metrics and binary accuracy are the key value here — they
    show a cleaner picture than 4-class overall accuracy for presentation.
    """
    W = 84
    print()
    print("=" * W)
    print(" Binary Metrics — Bridge vs Non-Bridge")
    print("=" * W)
    print(f" {'Source':<20} {'Class':<14} {'Prec':>9} {'Rec':>9} {'F1':>9} {'IoU':>9}")
    print(f" {'-' * 82}")
    for label, agg in [("Model", model_agg), ("Silver Baseline", silver_agg)]:
        if not agg or "binary" not in agg:
            continue
        b = agg["binary"]
        nb = agg["binary_nonbridge"]
        print(f" {label:<20} {'Bridge':<14} {b['precision']:>8.1f}% {b['recall']:>8.1f}% {b['f1']:>8.1f}% {b['iou']:>8.1f}%")
        print(f" {'':<20} {'Non-Bridge':<14} {nb['precision']:>8.1f}% {nb['recall']:>8.1f}% {nb['f1']:>8.1f}% {nb['iou']:>8.1f}%")
        print(f" {'':<20} {'Binary Acc':<14} {agg['binary_accuracy']:>8.1f}%")
        print(f" {'-' * 82}")
    if model_agg and silver_agg and "binary" in model_agg and "binary" in silver_agg:
        mb = model_agg["binary"]
        sb = silver_agg["binary"]
        d_iou = mb["iou"] - sb["iou"]
        d_acc = model_agg["binary_accuracy"] - silver_agg["binary_accuracy"]
        mnb = model_agg["binary_nonbridge"]
        snb = silver_agg["binary_nonbridge"]
        d_nb_iou = mnb["iou"] - snb["iou"]
        sign_i = "+" if d_iou >= 0 else ""
        sign_a = "+" if d_acc >= 0 else ""
        sign_nb = "+" if d_nb_iou >= 0 else ""
        print(f" Delta: Bridge IoU {sign_i}{d_iou:.1f}%  |  Non-Bridge IoU {sign_nb}{d_nb_iou:.1f}%"
              f"  |  Binary Acc {sign_a}{d_acc:.1f}%")
    print("=" * W)


def _to_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def save_outputs(model_agg: dict, silver_agg: dict, bridge_results: list,
                 output_dir: Path, model_path: str, no_plot: bool):
    """Save per-bridge CSV, summary JSON, and confusion matrix PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-bridge CSV
    if HAS_PANDAS:
        rows = []
        for r in bridge_results:
            for source in ("model", "silver"):
                result = r.get(f"{source}_result")
                if result is None:
                    continue
                row = {
                    "huc_id": r["huc_id"],
                    "bridge_stem": r["bridge_stem"],
                    "source": source,
                    "status": r.get(f"{source}_status", "ok"),
                    "num_points": result["num_points"],
                    "overall_accuracy": result["overall_accuracy"],
                    "mean_iou": result["mean_iou"],
                }
                for c in range(NUM_CLASSES):
                    pc = result["per_class"].get(c, {})
                    for metric in ("precision", "recall", "f1", "iou", "support"):
                        row[f"class_{c}_{metric}"] = pc.get(metric)
                # Binary metrics
                binary = result.get("binary", {})
                for metric in ("precision", "recall", "f1", "iou"):
                    row[f"binary_{metric}"] = binary.get(metric)
                rows.append(row)
        csv_path = output_dir / "per_bridge_metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nSaved per-bridge CSV:   {csv_path}")
    else:
        print("\nWarning: pandas not installed, skipping CSV output.")

    # Summary JSON
    out = {
        "model_checkpoint": model_path,
        "model": model_agg,
        "silver_baseline": silver_agg,
    }
    json_path = output_dir / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(_to_serializable(out), f, indent=2)
    print(f"Saved metrics JSON:     {json_path}")

    # Confusion matrix PNG
    if no_plot or not HAS_PLOT:
        if not no_plot and not HAS_PLOT:
            print("Warning: matplotlib/seaborn not installed, skipping plot.")
        return

    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    class_labels = [f"{c}: {CLASS_NAMES[c]}" for c in range(NUM_CLASSES)]

    for ax, agg, title in [
        (axes[0], model_agg, "Model Predictions"),
        (axes[1], silver_agg, "Silver Baseline"),
    ]:
        if not agg:
            ax.set_visible(False)
            continue
        cm = np.array(agg.get("confusion_matrix", np.zeros((NUM_CLASSES, NUM_CLASSES))))
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums * 100, 0.0)
        sns.heatmap(
            cm_norm, ax=ax, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            vmin=0, vmax=100, cbar_kws={"label": "Recall (%)"},
        )
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")

    # row-normalized confusion matrix
    plt.suptitle("Bridge Classification — Confusion Matrices (row-normalized)", fontsize=13)
    plt.tight_layout()
    png_path = output_dir / "confusion_matrix.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {png_path}")

    # Binary confusion matrix (bridge vs non-bridge)
    binary_labels = ["Non-Bridge", "Bridge"]
    _, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, agg, title in [
        (axes2[0], model_agg, "Model Predictions"),
        (axes2[1], silver_agg, "Silver Baseline"),
    ]:
        if not agg:
            ax.set_visible(False)
            continue
        # Collapse 4-class CM to binary: class 2 = Bridge, rest = Non-Bridge
        cm4 = np.array(agg.get("confusion_matrix", np.zeros((NUM_CLASSES, NUM_CLASSES))))
        tp = cm4[BRIDGE_DECK_CLASS, BRIDGE_DECK_CLASS]
        fn = cm4[BRIDGE_DECK_CLASS, :].sum() - tp
        fp = cm4[:, BRIDGE_DECK_CLASS].sum() - tp
        tn = cm4.sum() - tp - fn - fp
        bcm = np.array([[tn, fp], [fn, tp]])
        row_sums = bcm.sum(axis=1, keepdims=True)
        bcm_norm = np.where(row_sums > 0, bcm / row_sums * 100, 0.0)
        sns.heatmap(
            bcm_norm, ax=ax, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=binary_labels, yticklabels=binary_labels,
            vmin=0, vmax=100, cbar_kws={"label": "Recall (%)"},
        )
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    plt.suptitle("Binary (Bridge vs Non-Bridge) — Confusion Matrices (row-normalized)", fontsize=13)
    plt.tight_layout()
    bin_png_path = output_dir / "confusion_matrix_binary.png"
    plt.savefig(bin_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved binary CM:        {bin_png_path}")


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def _register_evaluation(model_name, eval_name, model_metrics, output_dir,
                         bucket, prefix, profile):
    """Update registry.json with evaluation metrics and upload artifacts to S3."""
    # Lazy imports — only needed when --register is used
    sys.path.insert(0, os.path.dirname(__file__))
    from register_model import load_registry, upload_registry
    from src.s3 import create_s3_client, upload_file

    registry_key = f"{prefix}/registry.json"
    s3_client = create_s3_client(profile=profile)
    registry = load_registry(s3_client, bucket, registry_key)

    if model_name not in registry["models"]:
        print(f"ERROR: model '{model_name}' not found in registry")
        print(f"  Available: {', '.join(sorted(registry['models'].keys()))}")
        sys.exit(1)

    deck = model_metrics["per_class"][BRIDGE_DECK_CLASS]
    eval_entry = {
        "bridge_deck_iou": round(deck["iou"], 1),
        "bridge_deck_precision": round(deck["precision"], 1),
        "bridge_deck_recall": round(deck["recall"], 1),
        "bridge_deck_f1": round(deck["f1"], 1),
        "overall_accuracy": round(model_metrics["overall_accuracy"], 1),
        "mean_iou": round(model_metrics["mean_iou"], 1),
        "eval_bridges": model_metrics["total_bridges"],
    }

    entry = registry["models"][model_name]
    entry["evaluation"][eval_name] = eval_entry
    if entry.get("stage") == "experimental":
        entry["stage"] = "evaluated"

    # Upload eval artifacts to S3
    s3_eval_prefix = f"{prefix}/{model_name}/evaluation/{eval_name}"
    output_dir = Path(output_dir)
    for filename in ["evaluation_metrics.json", "confusion_matrix.png",
                     "confusion_matrix_binary.png", "per_bridge_metrics.csv",
                     "per_bridge_metrics.xlsx"]:
        local_path = output_dir / filename
        if local_path.exists():
            s3_key = f"{s3_eval_prefix}/{filename}"
            print(f"Uploading {filename} -> s3://{bucket}/{s3_key}")
            upload_file(s3_client, str(local_path), bucket, s3_key)

    upload_registry(s3_client, bucket, registry_key, registry)

    print(f"\nRegistry updated: {model_name} [{eval_name}]")
    print(f"  bridge_deck_iou: {eval_entry['bridge_deck_iou']}")
    print(f"  stage: {entry['stage']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_evaluate(pred_path: Path, gold_labels: np.ndarray,
                       bridge_id: str) -> tuple:
    """Load predicted labels from file and evaluate against gold.

    Returns (status, result) where status is 'ok', 'point_count_mismatch',
    or 'load_error', and result is the evaluate_bridge dict or None.
    """
    try:
        _, pred_labels = load_classifications(pred_path)
        if len(pred_labels) != len(gold_labels):
            print(f"WARN (bridge={bridge_id}): Model/gold point count mismatch "
                  f"({len(pred_labels)} vs {len(gold_labels)})")
            return "point_count_mismatch", None
        return "ok", evaluate_bridge(gold_labels, pred_labels)
    except Exception as e:
        print(f"WARN (bridge={bridge_id}): Failed to load inference output: {e}")
        return "load_error", None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bridge classification model against gold-labeled data"
    )
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Gold (human-annotated) data directory (HUC-organized LAS/LAZ)")
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Test data directory with silver LAZ files")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .ckpt checkpoint (required if --inference-dir not provided)")
    parser.add_argument("--inference-dir", type=Path, default=None,
                        help="Pre-computed inference output directory (skips running model)")
    parser.add_argument("--output-dir", type=Path, default=Path("./evaluation_results"),
                        help="Output directory for results (default: ./evaluation_results)")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                        help="Device for inference (default: auto)")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size in meters, must match training (default: 0.1)")
    parser.add_argument("--bridge-timeout", type=float, default=150,
                        help="Per-bridge inference timeout in seconds (default: 150)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip confusion matrix PNG")
    # Registry integration
    parser.add_argument("--register", action="store_true",
                        help="Update model registry with evaluation metrics")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Registry model name (required with --register)")
    parser.add_argument("--eval-name", type=str, default=None,
                        help="Evaluation set name for registry key (default: gold-{N})")
    parser.add_argument("--bucket", type=str, default="fimc-data",
                        help="S3 bucket (default: fimc-data)")
    parser.add_argument("--prefix", type=str, default="bridge-classification/models",
                        help="S3 prefix (default: bridge-classification/models)")
    parser.add_argument("--profile", type=str, default=None,
                        help="AWS profile name")
    args = parser.parse_args()

    if args.inference_dir is None and args.model is None:
        parser.error("--model is required when --inference-dir is not provided")
    if args.register and not args.model_name:
        parser.error("--model-name is required when --register is set")
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn is required. Install it with: pip install scikit-learn")
        sys.exit(1)

    # Device
    if torch is None:
        device = None
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Save run config for reproducibility
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)
    with open(output_dir / "run_command.txt", "w") as f:
        f.write(sys.executable + " " + " ".join(shlex.quote(a) for a in sys.argv) + "\n")

    # Discover gold bridges
    gold_dir = args.gold_dir.resolve()
    if not gold_dir.exists():
        print(f"ERROR: --gold-dir does not exist: {gold_dir}")
        sys.exit(1)

    gold_bridges = discover_gold_bridges(gold_dir)
    if not gold_bridges:
        print(f"ERROR: No LAS/LAZ files found in {gold_dir}")
        sys.exit(1)
    print(f"Found {len(gold_bridges)} gold bridges across "
          f"{len(set(huc for huc, _, _ in gold_bridges))} HUCs")

    # Load model (only when not using pre-computed inference)
    # Lazy-import inference functions here so spconv is not required at module level.
    model = None
    run_inference_fn = None
    inference_output_dir = args.output_dir / "inference_output"
    if args.inference_dir is None:
        # Add src/ to path so inference.py's `from model import SparseUNet` fallback works
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        from src.inference import load_model, run_inference as _run_inference
        run_inference_fn = _run_inference
        model = load_model(args.model, device)
        inference_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup SIGALRM timeout (Unix only, only when running inference)
    use_timeout = sys.platform != "win32" and model is not None
    if use_timeout:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    bridge_results = []
    skipped = []

    iterator = gold_bridges
    if HAS_TQDM:
        iterator = tqdm(gold_bridges, desc="Evaluating bridges")

    try:
        for huc_id, stem, gold_path in iterator:
            bridge_id = f"{huc_id}/{stem}"

            # Find matching test LAZ (needed for silver baseline and as inference input)
            test_laz = find_matching_file(args.test_dir, huc_id, stem)
            if test_laz is None:
                print(f"WARN (bridge={bridge_id}): No test LAZ found, skipping")
                skipped.append((bridge_id, "no_test_file"))
                continue

            # Load gold ground truth
            try:
                _, gold_labels = load_classifications(gold_path)
            except Exception as e:
                print(f"WARN (bridge={bridge_id}): Failed to load gold file: {e}")
                skipped.append((bridge_id, "gold_load_error"))
                continue

            if len(gold_labels) < 100:
                skipped.append((bridge_id, "too_few_points"))
                continue

            entry = {"huc_id": huc_id, "bridge_stem": stem}

            # --- Silver baseline ---
            try:
                _, silver_labels = load_classifications(test_laz)
                if len(silver_labels) != len(gold_labels):
                    print(f"WARN (bridge={bridge_id}): Silver/gold point count mismatch "
                          f"({len(silver_labels)} vs {len(gold_labels)})")
                    entry["silver_status"] = "point_count_mismatch"
                    entry["silver_result"] = None
                else:
                    entry["silver_result"] = evaluate_bridge(gold_labels, silver_labels)
                    entry["silver_status"] = "ok"
            except Exception as e:
                print(f"WARN (bridge={bridge_id}): Silver evaluation failed: {e}")
                entry["silver_status"] = "error"
                entry["silver_result"] = None

            # --- Model predictions ---
            if args.inference_dir is not None:
                pred_path = find_matching_file(args.inference_dir, huc_id, stem)
                if pred_path is None:
                    print(f"WARN (bridge={bridge_id}): No inference output found")
                    entry["model_status"] = "no_inference_file"
                    entry["model_result"] = None
                else:
                    entry["model_status"], entry["model_result"] = \
                        _load_and_evaluate(pred_path, gold_labels, bridge_id)
            else:
                pred_path = inference_output_dir / huc_id / f"{stem}.laz"
                pred_path.parent.mkdir(parents=True, exist_ok=True)

                if use_timeout:
                    signal.setitimer(signal.ITIMER_REAL, args.bridge_timeout)
                try:
                    ok = run_inference_fn(model, test_laz, pred_path, args.voxel_size, device)
                except BridgeTimeout:
                    print(f"TIMEOUT (bridge={bridge_id}): exceeded {args.bridge_timeout}s")
                    entry["model_status"] = "timeout"
                    entry["model_result"] = None
                    bridge_results.append(entry)
                    continue
                except Exception as e:
                    print(f"WARN (bridge={bridge_id}): Inference failed: {e}")
                    entry["model_status"] = "inference_error"
                    entry["model_result"] = None
                    ok = False
                finally:
                    if use_timeout:
                        signal.setitimer(signal.ITIMER_REAL, 0)

                if ok:
                    entry["model_status"], entry["model_result"] = \
                        _load_and_evaluate(pred_path, gold_labels, bridge_id)
                elif "model_status" not in entry:
                    entry["model_status"] = "inference_error"
                    entry["model_result"] = None

            bridge_results.append(entry)

    finally:
        if use_timeout:
            signal.signal(signal.SIGALRM, old_handler)

    # Aggregate
    model_results = [r["model_result"] for r in bridge_results if r.get("model_result")]
    silver_results = [r["silver_result"] for r in bridge_results if r.get("silver_result")]

    if not model_results and not silver_results:
        print("\nERROR: No bridges were successfully evaluated.")
        sys.exit(1)

    model_agg = aggregate_metrics(model_results) if model_results else {}
    silver_agg = aggregate_metrics(silver_results) if silver_results else {}

    if model_agg:
        print_summary_table(model_agg, label="Model Predictions")
    if silver_agg:
        print_summary_table(silver_agg, label="Silver Baseline")

    # Comparison & binary tables
    if model_agg and silver_agg:
        print_comparison_table(model_agg, silver_agg)
        print_binary_table(model_agg, silver_agg)

    # Skipped summary
    if skipped:
        print(f"\nSkipped {len(skipped)} bridge(s):")
        for bridge_id, reason in skipped:
            print(f"  {bridge_id}: {reason}")

    model_ok = sum(1 for r in bridge_results if r.get("model_status") == "ok")
    silver_ok = sum(1 for r in bridge_results if r.get("silver_status") == "ok")
    print(f"\nModel:  {model_ok} bridges evaluated")
    print(f"Silver: {silver_ok} bridges evaluated")

    save_outputs(model_agg, silver_agg, bridge_results, args.output_dir,
                 args.model or "pre-computed", args.no_plot)

    if args.inference_dir is None:
        print(f"\nInference outputs saved to: {inference_output_dir}")
        print(f"Tip: Re-run with --inference-dir {inference_output_dir} to skip inference.")

    if args.register and model_agg:
        eval_name = args.eval_name or f"gold-{model_agg['total_bridges']}"
        _register_evaluation(
            model_name=args.model_name,
            eval_name=eval_name,
            model_metrics=model_agg,
            output_dir=args.output_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
