"""
Bridge Classification — Post-Run Output Audit

Verifies that all expected outputs exist in S3 after a batch run.
Optionally writes missing entries to a new manifest for re-submission.

Uses a thread pool for parallel S3 head_object checks.

Usage:
    # Check all outputs exist
    python scripts/audit_outputs.py \
        --manifest s3://bucket/manifest.txt \
        --bucket my-bucket \
        --output-prefix predictions/v3 \
        --mode masked

    # Write missing entries to a file for re-submission
    python scripts/audit_outputs.py \
        --manifest s3://bucket/manifest.txt \
        --bucket my-bucket \
        --output-prefix predictions/v3 \
        --mode masked \
        --write-missing missing.txt

    # Tune concurrency (default: 200)
    python scripts/audit_outputs.py ... --workers 100

    # Use a specific AWS profile
    python scripts/audit_outputs.py ... --profile Data
"""

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config as BotoConfig

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.s3 import (
    object_exists, resolve_extension, resolve_output_keys,
    stream_manifest_lines,
)

DEFAULT_WORKERS = 200
AWS_MAX_RETRIES = 3


def _make_session(profile):
    """Create a boto3 session (called once per thread via threading.local)."""
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client('s3', config=BotoConfig(retries={'max_attempts': AWS_MAX_RETRIES, 'mode': 'adaptive'}))


def check_line(thread_local, profile, bucket, input_prefix, output_prefix, mode, line):
    """Check whether all expected outputs exist for a single manifest line.

    Creates a per-thread S3 client on first use (boto3 clients are not thread-safe).

    Returns:
        (line, all_exist) tuple.
    """
    if not hasattr(thread_local, 's3'):
        thread_local.s3 = _make_session(profile)
    s3 = thread_local.s3

    ext = resolve_extension(s3, bucket, input_prefix, line) if input_prefix else '.laz'
    output_keys = resolve_output_keys(output_prefix, line, ext, mode)
    all_exist = all(object_exists(s3, bucket, k) for k in output_keys.values())
    return line, all_exist


def main():
    parser = argparse.ArgumentParser(description='Audit bridge classification outputs in S3')
    parser.add_argument('--manifest', type=str, required=True, help='S3 URI of manifest file')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket for outputs')
    parser.add_argument('--input-prefix', type=str, default='', help='S3 prefix for input files (for extension probing)')
    parser.add_argument('--output-prefix', type=str, required=True, help='S3 prefix for output files')
    parser.add_argument('--mode', type=str, default='masked', choices=['raw', 'masked', 'both'],
                        help='Inference mode (determines expected output filenames)')
    parser.add_argument('--write-missing', type=str, help='Write missing manifest lines to this file')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Parallel S3 check workers (default: {DEFAULT_WORKERS})')
    parser.add_argument('--profile', type=str, help='AWS profile')
    args = parser.parse_args()

    # Use a single session only for reading the manifest (single-threaded)
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3_main = session.client('s3', config=BotoConfig(retries={'max_attempts': 3, 'mode': 'adaptive'}))

    # Read manifest
    lines = list(stream_manifest_lines(s3_main, args.manifest))
    total = len(lines)
    print(f"Manifest: {total} entries")
    print(f"Checking outputs in s3://{args.bucket}/{args.output_prefix}/ "
          f"(mode={args.mode}, workers={args.workers})")

    thread_local = threading.local()
    found = 0
    missing_lines = []
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                check_line, thread_local, args.profile,
                args.bucket, args.input_prefix, args.output_prefix, args.mode, line
            ): line
            for line in lines
        }

        for future in as_completed(futures):
            line, all_exist = future.result()  # re-raises any exception from the thread
            with lock:
                completed += 1
                if all_exist:
                    found += 1
                else:
                    missing_lines.append(line)
                if completed % 10000 == 0:
                    print(f"  Checked {completed}/{total} — {found} found, {len(missing_lines)} missing")

    missing = len(missing_lines)
    print(f"\nResults: {found} found, {missing} missing out of {total} total")

    if args.write_missing and missing_lines:
        with open(args.write_missing, 'w') as f:
            for line in missing_lines:
                f.write(line + '\n')
        print(f"Missing manifest written to: {args.write_missing}")
        print(f"Re-submit with: python scripts/submit_batch_job.py --manifest <upload-this-file>")

    if missing > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
