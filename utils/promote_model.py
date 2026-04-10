"""Promote a model to production in the S3 model registry.

Demotes the current production model (if any), promotes the specified model,
updates registry.json in S3, and prints the s3_checkpoint_uri for Terraform.

Usage:
    # Dry run (no S3 changes):
    python utils/promote_model.py \
      --name bridge-base-all-data-v3 --dry-run \
      --bucket fimc-data --prefix bridge-classification/models --profile data

    # Promote for real:
    python utils/promote_model.py \
      --name bridge-base-all-data-v3 \
      --bucket fimc-data --prefix bridge-classification/models --profile data
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from register_model import load_registry, upload_registry
from src.s3 import create_s3_client


def main():
    parser = argparse.ArgumentParser(
        description="Promote a model to production in the S3 model registry"
    )
    parser.add_argument("--name", required=True,
                        help="Model name to promote (must exist in registry)")
    parser.add_argument("--bucket", default="fimc-data",
                        help="S3 bucket (default: fimc-data)")
    parser.add_argument("--prefix", default="bridge-classification/models",
                        help="S3 prefix for model registry (default: bridge-classification/models)")
    parser.add_argument("--profile", default=None,
                        help="AWS profile name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying S3")
    args = parser.parse_args()

    registry_key = f"{args.prefix}/registry.json"
    s3_client = create_s3_client(profile=args.profile)
    registry = load_registry(s3_client, args.bucket, registry_key)

    # Validate model exists
    if args.name not in registry["models"]:
        available = ", ".join(sorted(registry["models"].keys()))
        raise SystemExit(f"Error: model '{args.name}' not found in registry.\nAvailable: {available}")

    model_entry = registry["models"][args.name]

    # Validate model has been evaluated
    if not model_entry.get("evaluation"):
        raise SystemExit(f"Error: model '{args.name}' has no evaluations. Evaluate before promoting.")

    # Check if already production
    current_prod = registry.get("production")
    if current_prod and current_prod["model_name"] == args.name:
        print(f"'{args.name}' is already the production model. Nothing to do.")
        return

    # Demote current production model
    if current_prod:
        old_name = current_prod["model_name"]
        if old_name in registry["models"]:
            registry["models"][old_name]["stage"] = "evaluated"
        print(f"Demoted: {old_name} to evaluated")

    # Promote new model
    registry["models"][args.name]["stage"] = "production"
    registry["production"] = {
        "model_name": args.name,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }

    # Upload or dry-run
    if args.dry_run:
        print("\n[DRY RUN] No changes written to S3.")
    else:
        upload_registry(s3_client, args.bucket, registry_key, registry)
        print(f"\nRegistry updated: s3://{args.bucket}/{registry_key}")

    # Print output for Terraform
    uri = model_entry["s3_checkpoint_uri"]
    print(f"\nPromoted: {args.name} to production")
    print(f"Checkpoint: {uri}")
    print(f"\nUpdate terraform.tfvars:")
    print(f'  s3_model_uri = "{uri}"')


if __name__ == "__main__":
    main()
