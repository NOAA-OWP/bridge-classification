"""
Bridge Classification — Batch Job Submission Script

AWS_PROFILE: the account where Batch infra lives. Used for job submission.
--profile:   S3 data access override. Only needed when the manifest/data
             bucket is in a different account than Batch. Falls back to
             AWS_PROFILE if not set.

Usage:
    # Array job from S3 manifest
    python scripts/submit_batch_job.py --manifest s3://bucket/path/manifest.txt

    # Explicit count (skip manifest download)
    python scripts/submit_batch_job.py --total 1500000

    # Dry run (prints config, does not submit)
    python scripts/submit_batch_job.py --manifest s3://bucket/manifest.txt --dry-run

    # Validate manifest (check for empty lines, duplicates, format)
    python scripts/submit_batch_job.py --manifest s3://bucket/manifest.txt --validate

    # Pass env overrides
    python scripts/submit_batch_job.py --manifest s3://... --env INFERENCE_MODE=both --env BRIDGE_TIMEOUT=200

    # Single job (no array)
    python scripts/submit_batch_job.py --single

    # Run config (_run_config.json) is saved automatically from terraform outputs.
    # Override output location via --env:
    python scripts/submit_batch_job.py --manifest s3://bucket/manifest.txt \
        --env S3_OUTPUT_PREFIX=other/prefix --env S3_BUCKET=other-bucket
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.s3_client import create_s3_client, stream_manifest_lines, upload_json
from src.terraform import get_terraform_outputs

MAX_ARRAY_SIZE = 10_000  # AWS Batch hard limit
DEFAULT_CHUNK_TARGET = 60
SPOT_PRICE_PER_HOUR = 0.234  # g4dn.xlarge spot estimate (fluctuates) - check https://aws.amazon.com/ec2/spot/pricing/


def count_manifest_lines(s3_client: Any, manifest_uri: str) -> int:
    """Stream an S3 manifest and count non-empty lines."""
    return sum(1 for _ in stream_manifest_lines(s3_client, manifest_uri))


def validate_manifest(s3_client: Any, manifest_uri: str) -> tuple:
    """Check manifest for common issues. Returns (line_count, issues)."""
    issues = []
    seen = {}
    count = 0

    for i, line in enumerate(stream_manifest_lines(s3_client, manifest_uri), 1):
        count = i
        if '/' not in line:
            issues.append(f"Line {i}: no '/' separator (expected huc_id/bridge_stem): {line[:80]}")
        if line in seen:
            issues.append(f"Line {i}: duplicate of line {seen[line]}: {line[:80]}")
        else:
            seen[line] = i

    return count, issues


def compute_array_size(total: int, chunk_target: int, max_array_size: int = MAX_ARRAY_SIZE) -> int:
    """Compute array size, capped at AWS Batch limit."""
    if total <= chunk_target:
        return 1
    size = math.ceil(total / chunk_target)
    return min(size, max_array_size)


def build_container_overrides(array_size: int, env_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Build the containerOverrides JSON for Batch submission."""
    env_items = [{'name': 'ARRAY_SIZE', 'value': str(array_size)}]

    if env_overrides:
        for key, value in env_overrides.items():
            env_items.append({'name': key, 'value': str(value)})

    return {'environment': env_items}


def main() -> None:
    parser = argparse.ArgumentParser(description='Bridge Classification — Submit Batch Job')
    parser.add_argument('--manifest', type=str, help='S3 URI of manifest file')
    parser.add_argument('--total', type=int, help='Total file count (skip manifest download)')
    parser.add_argument('--single', action='store_true', help='Submit single job (no array)')
    parser.add_argument('--dry-run', action='store_true', help='Print config without submitting')
    parser.add_argument('--validate', action='store_true', help='Validate manifest format')
    parser.add_argument('--chunk-target', type=int, default=DEFAULT_CHUNK_TARGET,
                        help=f'Target files per array child (default: {DEFAULT_CHUNK_TARGET})')
    parser.add_argument('--env', action='append', default=[],
                        help='Container env override as KEY=VALUE (can repeat). '
                             'Common: S3_OUTPUT_PREFIX, S3_BUCKET, INFERENCE_MODE, BRIDGE_TIMEOUT')
    parser.add_argument('--job-name', type=str, default='bridge-inference',
                        help='Job name prefix (default: bridge-inference)')
    parser.add_argument('--profile', type=str, help='AWS profile override for S3 access')
    args = parser.parse_args()

    # --- Validate flag combinations ---
    if args.total is not None and args.total <= 0:
        parser.error("--total must be positive")
    if args.chunk_target < 1:
        parser.error("--chunk-target must be >= 1")
    if args.validate and not args.manifest:
        parser.error("--validate requires --manifest")
    if args.single and (args.manifest or args.total is not None):
        parser.error("--single cannot be combined with --manifest or --total")

    # --- Read terraform config ---
    tf = get_terraform_outputs()
    aws_region = os.environ.get('AWS_REGION') or tf.get('aws_region')
    aws_profile = os.environ.get('AWS_PROFILE')
    job_def_name = os.environ.get('JOB_DEF_NAME') or tf.get('job_definition_name')
    job_queue = os.environ.get('JOB_QUEUE') or tf.get('job_queue_name')

    missing = []
    if not aws_region: missing.append('AWS_REGION')
    if not aws_profile: missing.append('AWS_PROFILE')
    if not job_def_name: missing.append('JOB_DEF_NAME')
    if not job_queue: missing.append('JOB_QUEUE')
    if missing:
        print(f"ERROR: Missing config: {' '.join(missing)}")
        print("Run 'cd infra/terraform/app && terraform init && terraform apply' or set env vars.")
        sys.exit(1)

    # Fall back to s3_manifest_uri from terraform outputs when --manifest not provided
    manifest = args.manifest or tf.get('s3_manifest_uri')
    if manifest and not args.manifest:
        print(f"Using manifest from terraform output: {manifest}")

    s3_profile = args.profile or os.environ.get('S3_PROFILE') or aws_profile
    s3 = create_s3_client(s3_profile)

    # --- Validate manifest ---
    if args.validate:
        print(f"Validating manifest: {manifest}")
        line_count, issues = validate_manifest(s3, manifest)
        print(f"Lines: {line_count}")
        if issues:
            print(f"\nFound {len(issues)} issue(s):")
            for issue in issues[:50]:  # cap output
                print(f"  {issue}")
            if len(issues) > 50:
                print(f"  ... and {len(issues) - 50} more")
            sys.exit(1)
        else:
            print("Manifest OK — no issues found.")
        return

    # --- Determine total files ---
    if args.single:
        array_size = 1
        total_files = 1
        print("Mode: single job (no array)")
    elif args.total is not None:
        total_files = args.total
        array_size = compute_array_size(total_files, args.chunk_target)
        print(f"Total files (provided): {total_files}")
    elif manifest:
        print(f"Counting lines from S3: {manifest}")
        total_files = count_manifest_lines(s3, manifest)
        if total_files == 0:
            print("ERROR: manifest is empty")
            sys.exit(1)
        array_size = compute_array_size(total_files, args.chunk_target)
        print(f"Total files in manifest: {total_files}")
    else:
        parser.error("Provide --manifest, --total, or --single (or set s3_manifest_uri in terraform.tfvars)")

    actual_chunk = math.ceil(total_files / array_size) if array_size > 0 else total_files
    ideal_array = math.ceil(total_files / args.chunk_target) if args.chunk_target > 0 else 1
    if ideal_array > MAX_ARRAY_SIZE:
        print(f"Array size: {array_size} (capped from ideal {ideal_array}; "
              f"chunk_target={args.chunk_target} requested → actual ~{actual_chunk} per child)")
    else:
        print(f"Array size: {array_size}, ~{actual_chunk} files per child")

    # --- Parse env overrides ---
    env_overrides = {}
    if manifest:
        env_overrides['S3_MANIFEST_URI'] = manifest
    for item in args.env:
        if '=' not in item:
            print(f"ERROR: --env value must be KEY=VALUE, got: {item}")
            sys.exit(1)
        key, value = item.split('=', 1)
        env_overrides[key] = value

    container_overrides = build_container_overrides(array_size, env_overrides)

    # --- Cost estimate ---
    est_seconds_per_bridge = 90  # ~90s average including download/upload
    est_hours_per_child = (actual_chunk * est_seconds_per_bridge) / 3600
    est_total_cost = array_size * est_hours_per_child * SPOT_PRICE_PER_HOUR

    print(f"\nEstimated runtime: ~{est_hours_per_child:.1f} hours per child")
    print(f"Estimated cost: ~${est_total_cost:,.0f} (SPOT @ ${SPOT_PRICE_PER_HOUR}/hr)")

    # --- Dry run ---
    if args.dry_run:
        print("\n--- DRY RUN (not submitting) ---")
        print(f"Job definition: {job_def_name}")
        print(f"Job queue: {job_queue}")
        print(f"Region: {aws_region}")
        print(f"Container overrides: {json.dumps(container_overrides, indent=2)}")
        return

    # --- Submit ---
    batch_session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
    batch = batch_session.client('batch')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    full_job_name = f"{args.job_name}-{timestamp}"

    submit_kwargs = {
        'jobName': full_job_name,
        'jobDefinition': job_def_name,
        'jobQueue': job_queue,
        'containerOverrides': container_overrides,
    }

    if array_size > 1:
        submit_kwargs['arrayProperties'] = {'size': array_size}
        print(f"\nSubmitting array job: {full_job_name} (size={array_size})")
    else:
        print(f"\nSubmitting single job: {full_job_name}")

    response = batch.submit_job(**submit_kwargs)

    job_id = response['jobId']
    print(f"\nJob submitted: {full_job_name}")
    print(f"Job ID: {job_id}")
    print(f"Monitor at: https://console.aws.amazon.com/batch/home?region={aws_region}#jobs")

    # --- Save run config to S3 ---
    run_bucket = env_overrides.get('S3_BUCKET') or tf.get('s3_bucket')
    run_output_prefix = (
        env_overrides.get('S3_OUTPUT_PREFIX')
        or tf.get('s3_output_prefix')
    )

    if run_bucket and run_output_prefix:
        git_commit = "unknown"
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], text=True
            ).strip()
        except Exception:
            pass

        run_config = {
            "job_id": job_id,
            "job_name": full_job_name,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "manifest_uri": manifest,
            "s3_bucket": run_bucket,
            "s3_output_prefix": run_output_prefix,
            "array_size": array_size,
            "chunk_target": args.chunk_target,
            "total_bridges": total_files,
            "spot_rate_usd": SPOT_PRICE_PER_HOUR,
            "git_commit": git_commit,
            "env_overrides": env_overrides,
        }

        config_key = f"{run_output_prefix}/_run_config.json"
        upload_json(s3, run_config, run_bucket, config_key)
        print(f"Run config saved: s3://{run_bucket}/{config_key}")
    else:
        print("\nWARN: Could not determine s3_bucket/s3_output_prefix - run config not saved.")
        print("Set them in terraform.tfvars or pass --env S3_BUCKET=... --env S3_OUTPUT_PREFIX=...")


if __name__ == '__main__':
    main()
