"""Register a trained model to the S3 model registry.

Uploads the best checkpoint + config files from a Lightning experiment
directory to S3, creates lineage.json if fine-tuned, and adds an entry
to registry.json.

Usage:
    python utils/register_model.py \
      --exp-dir ./experiments/bridge-base-all-data-v3/version_0 \
      --name bridge-base-all-data-v3 \
      --description "Silver-only training, 477K bridges, base_channels=16, 0.1m voxel" \
      --bucket fimc-data --prefix bridge-classification/models

    # Fine-tuned model with parent lineage:
    python utils/register_model.py \
      --exp-dir ./experiments/ft-gold-optA-v0/version_0 \
      --name ft-gold-optA-v0 \
      --description "Fine-tuned from v3, frozen encoder, 80 gold bridges" \
      --parent bridge-base-all-data-v3 \
      --bucket fimc-data --prefix bridge-classification/models
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.s3 import create_s3_client, upload_file, download_file, object_exists


def find_best_checkpoint(checkpoints_dir):
    """Find the best checkpoint by metric value in the filename.

    Expects filenames like: bridge-unet-epoch=XX-val_deck_iou=YY.YYYY.ckpt
    For iou/acc metrics, higher is better. For loss metrics, lower is better.
    """
    ckpts = [
        p for p in Path(checkpoints_dir).glob("bridge-unet-*.ckpt")
        if p.name != "last.ckpt"
    ]
    if not ckpts:
        return None

    # Extract metric name and value from the last key=value pair in filename
    pattern = re.compile(r'-(\w+)=([\d.]+)\.ckpt$')
    scored = []
    for ckpt in ckpts:
        match = pattern.search(ckpt.name)
        if match:
            metric_name = match.group(1)
            metric_value = float(match.group(2))
            scored.append((ckpt, metric_name, metric_value))

    if not scored:
        return ckpts[0]

    # Higher is better for iou/acc, lower is better for loss
    metric_name = scored[0][1]
    reverse = "iou" in metric_name or "acc" in metric_name
    scored.sort(key=lambda x: x[2], reverse=reverse)
    return scored[0][0]


def load_registry(s3_client, bucket, registry_key):
    """Download registry.json from S3, or return a skeleton if it doesn't exist."""
    if object_exists(s3_client, bucket, registry_key):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            download_file(s3_client, bucket, registry_key, tmp_path)
            with open(tmp_path) as f:
                return json.load(f)
        finally:
            os.unlink(tmp_path)
    return {"schema_version": 1, "models": {}, "production": None}


def upload_registry(s3_client, bucket, registry_key, registry):
    """Write registry.json to a temp file and upload to S3."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(registry, tmp, indent=2)
        tmp_path = tmp.name
    try:
        upload_file(s3_client, tmp_path, bucket, registry_key)
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Register a trained model to the S3 model registry")
    parser.add_argument("--exp-dir", required=True, help="Path to Lightning experiment version directory")
    parser.add_argument("--checkpoint", default="best", help="'best' (default), 'last', or a specific filename")
    parser.add_argument("--name", required=True, help="Model name for the registry")
    parser.add_argument("--description", required=True, help="Human-readable description")
    parser.add_argument("--parent", default=None, help="Parent model name (for fine-tuned experiments)")
    parser.add_argument("--bucket", required=True, help="S3 bucket")
    parser.add_argument("--prefix", required=True, help="S3 prefix (e.g. bridge-classification/models)")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoints_dir = exp_dir / "checkpoints"

    # Validate experiment directory
    if not exp_dir.is_dir():
        sys.exit(f"Error: experiment directory not found: {exp_dir}")
    if not checkpoints_dir.is_dir():
        sys.exit(f"Error: checkpoints directory not found: {checkpoints_dir}")

    # Find checkpoint
    if args.checkpoint == "best":
        ckpt_path = find_best_checkpoint(checkpoints_dir)
        if ckpt_path is None:
            sys.exit(f"Error: no bridge-unet-*.ckpt files found in {checkpoints_dir}")
    elif args.checkpoint == "last":
        ckpt_path = checkpoints_dir / "last.ckpt"
        if not ckpt_path.exists():
            sys.exit(f"Error: last.ckpt not found in {checkpoints_dir}")
    else:
        ckpt_path = checkpoints_dir / args.checkpoint
        if not ckpt_path.exists():
            sys.exit(f"Error: checkpoint not found: {ckpt_path}")

    print(f"Selected checkpoint: {ckpt_path.name}")

    # Load config files
    hparams_path = exp_dir / "hparams.yaml"
    if not hparams_path.exists():
        sys.exit(f"Error: hparams.yaml not found in {exp_dir}")
    with open(hparams_path) as f:
        hparams = yaml.safe_load(f)

    train_config_path = exp_dir / "train_config.json"
    train_config = {}
    if train_config_path.exists():
        with open(train_config_path) as f:
            train_config = json.load(f)
    else:
        print(f"Warning: train_config.json not found in {exp_dir}")

    # Build S3 paths
    s3_prefix = f"{args.prefix}/{args.name}"
    s3_ckpt_key = f"{s3_prefix}/checkpoints/{ckpt_path.name}"
    registry_key = f"{args.prefix}/registry.json"

    s3_client = create_s3_client(profile=args.profile)

    # Check for name collision
    registry = load_registry(s3_client, args.bucket, registry_key)
    if args.name in registry["models"]:
        sys.exit(f"Error: model '{args.name}' already exists in registry. Choose a different name.")

    # Upload checkpoint
    print(f"Uploading checkpoint to s3://{args.bucket}/{s3_ckpt_key}")
    upload_file(s3_client, str(ckpt_path), args.bucket, s3_ckpt_key)

    # Upload config files
    config_files = ["hparams.yaml", "train_config.json", "run_command.txt", "class_weights.json"]
    for fname in config_files:
        local_path = exp_dir / fname
        if local_path.exists():
            s3_key = f"{s3_prefix}/config/{fname}"
            print(f"Uploading {fname}")
            upload_file(s3_client, str(local_path), args.bucket, s3_key)
        else:
            print(f"Skipping {fname} (not found)")

    # Create and upload lineage.json if parent specified
    if args.parent:
        parent_info = registry["models"].get(args.parent, {})
        lineage = {
            "parent_model": args.parent,
            "parent_checkpoint": parent_info.get("s3_checkpoint_uri", "unknown"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(lineage, tmp, indent=2)
            tmp_path = tmp.name
        try:
            lineage_key = f"{s3_prefix}/lineage.json"
            print(f"Uploading lineage.json (parent: {args.parent})")
            upload_file(s3_client, tmp_path, args.bucket, lineage_key)
        finally:
            os.unlink(tmp_path)
        if not parent_info:
            print(f"Warning: parent '{args.parent}' not found in registry")

    # Build registry entry
    entry = {
        "description": args.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt_path.name,
        "s3_checkpoint_uri": f"s3://{args.bucket}/{s3_ckpt_key}",
        "model_config": {
            "base_channels": hparams.get("base_channels", 16),
            "num_classes": hparams.get("num_classes", 4),
            "input_channels": hparams.get("input_channels", 1),
            "voxel_size": train_config.get("voxel_size", 0.1),
        },
        "evaluation": {},
        "stage": "experimental",
        "parent_model": args.parent,
        "git_commit": train_config.get("git_commit", "unknown"),
    }

    registry["models"][args.name] = entry
    upload_registry(s3_client, args.bucket, registry_key, registry)

    print(f"\nRegistered '{args.name}' (stage=experimental)")
    print(f"  checkpoint: {ckpt_path.name}")
    print(f"  s3_checkpoint_uri: s3://{args.bucket}/{s3_ckpt_key}")


if __name__ == "__main__":
    main()
