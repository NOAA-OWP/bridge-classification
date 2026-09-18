"""
Bridge Classification — Post-Run Output Audit

Verifies that all expected outputs exist in S3 after a batch run.
Optionally writes missing entries to a new manifest for re-submission.

Uses a thread pool for parallel S3 head_object checks.

--profile: S3 data access. Falls back to AWS_PROFILE if not set.

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

    # Use a specific AWS profile for S3 access
    python scripts/audit_outputs.py ... --profile my-profile

    # Save audit results to S3
    python scripts/audit_outputs.py \
        --manifest s3://bucket/manifest.txt \
        --bucket my-bucket \
        --output-prefix predictions/v3 \
        --mode masked \
        --save-to-s3
"""

import argparse
import os
import sys
from datetime import datetime, timezone

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.constants import InferenceMode
from src.s3_audit import DEFAULT_AUDIT_WORKERS, audit_s3_outputs
from src.s3_client import create_s3_client, stream_manifest_lines, upload_json

MAX_MISSING_ENTRIES = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit bridge classification outputs in S3')
    parser.add_argument('--manifest', type=str, required=True, help='S3 URI of manifest file')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket for outputs')
    parser.add_argument('--input-prefix', type=str, default='', help='S3 prefix for input files (for extension probing)')
    parser.add_argument('--output-prefix', type=str, required=True, help='S3 prefix for output files')
    parser.add_argument('--mode', type=InferenceMode, default=InferenceMode.MASKED,
                        help='Inference mode: masked (default), raw, or both')
    parser.add_argument('--write-missing', type=str, help='Write missing manifest lines to this file')
    parser.add_argument('--workers', type=int, default=DEFAULT_AUDIT_WORKERS,
                        help=f'Parallel S3 check workers (default: {DEFAULT_AUDIT_WORKERS})')
    parser.add_argument('--profile', type=str, help='AWS profile')
    parser.add_argument('--save-to-s3', action='store_true',
                        help='Upload audit summary JSON to S3 at {output-prefix}/_audit_results.json')
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    s3_main = create_s3_client(args.profile)

    lines = list(stream_manifest_lines(s3_main, args.manifest))
    total = len(lines)
    if total == 0:
        print("ERROR: manifest is empty")
        sys.exit(1)
    print(f"Manifest: {total} entries")
    print(f"Checking outputs in s3://{args.bucket}/{args.output_prefix}/ "
          f"(mode={args.mode}, workers={args.workers})")

    found, missing_lines = audit_s3_outputs(
        profile=args.profile,
        bucket=args.bucket,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        mode=args.mode,
        manifest_lines=lines,
        workers=args.workers,
        progress_interval=10000,
    )

    missing = len(missing_lines)
    print(f"\nResults: {found} found, {missing} missing out of {total} total")

    if args.write_missing:
        if missing_lines:
            with open(args.write_missing, 'w') as f:
                for line in missing_lines:
                    f.write(line + '\n')
            print(f"Missing manifest written to: {args.write_missing}")
            print(f"Re-submit with: python scripts/submit_batch_job.py --manifest <upload-this-file>")
        else:
            print("No missing entries to write.")

    if args.save_to_s3:
        audit_result = {
            'audited_at': datetime.now(timezone.utc).isoformat(),
            'manifest_uri': args.manifest,
            'total': total,
            'found': found,
            'missing': missing,
        }
        if missing_lines:
            audit_result['missing_entries'] = missing_lines[:MAX_MISSING_ENTRIES]
            if len(missing_lines) > MAX_MISSING_ENTRIES:
                audit_result['missing_entries_truncated'] = True

        audit_key = f"{args.output_prefix}/_audit_results.json"
        upload_json(s3_main, audit_result, args.bucket, audit_key)
        print(f"Audit results saved: s3://{args.bucket}/{audit_key}")

    if missing > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
