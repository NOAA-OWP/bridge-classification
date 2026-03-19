"""
Bridge Classification — Post-Run Output Audit

Verifies that all expected outputs exist in S3 after a batch run.
Optionally writes missing entries to a new manifest for re-submission.

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

    # Use a specific AWS profile
    python scripts/audit_outputs.py ... --profile Data
"""

import argparse
import sys
from pathlib import PurePosixPath

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# Must match the extensions probed in batch_entrypoint.py
PROBE_EXTENSIONS = ['.laz', '.las']


def parse_s3_uri(uri):
    """Split s3://bucket/key into (bucket, key)."""
    path = uri[5:]
    bucket, _, key = path.partition('/')
    return bucket, key


def object_exists(s3_client, bucket, key):
    """Check if an S3 object exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def resolve_extension(s3_client, bucket, input_prefix, manifest_line):
    """Determine the actual extension for a manifest line by probing S3."""
    p = PurePosixPath(manifest_line)
    if p.suffix in ('.laz', '.las'):
        return p.suffix

    for ext in PROBE_EXTENSIONS:
        key = f"{input_prefix}/{manifest_line}{ext}"
        if object_exists(s3_client, bucket, key):
            return ext

    return '.laz'  # default fallback


def expected_output_keys(output_prefix, manifest_line, ext, mode):
    """Return list of expected S3 output keys for a manifest line."""
    p = PurePosixPath(manifest_line)
    huc_id = str(p.parent)
    stem = p.stem

    keys = []
    if mode == 'masked':
        keys.append(f"{output_prefix}/{huc_id}/{stem}_bridge_masked{ext}")
    elif mode == 'raw':
        keys.append(f"{output_prefix}/{huc_id}/{stem}_predicted{ext}")
    elif mode == 'both':
        keys.append(f"{output_prefix}/{huc_id}/{stem}_predicted{ext}")
        keys.append(f"{output_prefix}/{huc_id}/{stem}_bridge_masked{ext}")

    return keys


def main():
    parser = argparse.ArgumentParser(description='Audit bridge classification outputs in S3')
    parser.add_argument('--manifest', type=str, required=True, help='S3 URI of manifest file')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket for outputs')
    parser.add_argument('--input-prefix', type=str, default='', help='S3 prefix for input files (for extension probing)')
    parser.add_argument('--output-prefix', type=str, required=True, help='S3 prefix for output files')
    parser.add_argument('--mode', type=str, default='masked', choices=['raw', 'masked', 'both'],
                        help='Inference mode (determines expected output filenames)')
    parser.add_argument('--write-missing', type=str, help='Write missing manifest lines to this file')
    parser.add_argument('--profile', type=str, help='AWS profile')
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3 = session.client('s3', config=BotoConfig(retries={'max_attempts': 3, 'mode': 'adaptive'}))

    # Read manifest
    manifest_bucket, manifest_key = parse_s3_uri(args.manifest)
    response = s3.get_object(Bucket=manifest_bucket, Key=manifest_key)

    lines = []
    for raw_line in response['Body'].iter_lines():
        line = raw_line.decode('utf-8').strip() if isinstance(raw_line, bytes) else raw_line.strip()
        if line:
            lines.append(line)

    print(f"Manifest: {len(lines)} entries")
    print(f"Checking outputs in s3://{args.bucket}/{args.output_prefix}/ (mode={args.mode})")

    found = 0
    missing_lines = []

    for i, line in enumerate(lines, 1):
        ext = resolve_extension(s3, args.bucket, args.input_prefix, line) if args.input_prefix else '.laz'
        keys = expected_output_keys(args.output_prefix, line, ext, args.mode)

        all_exist = all(object_exists(s3, args.bucket, k) for k in keys)
        if all_exist:
            found += 1
        else:
            missing_lines.append(line)

        if i % 10000 == 0:
            print(f"  Checked {i}/{len(lines)} — {found} found, {len(missing_lines)} missing")

    missing = len(missing_lines)
    print(f"\nResults: {found} found, {missing} missing out of {len(lines)} total")

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
