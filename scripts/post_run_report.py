"""
Bridge Classification - Post-Run Report

Generates a report after an AWS Batch inference run completes.
Reads _run_config.json (saved at submission), audits S3 outputs, queries
CloudWatch logs for per-child summaries and per-bridge timing, and saves
_run_report.json to the output prefix.

--profile:       S3 data access. Falls back to AWS_PROFILE if not set.
--batch-profile: Batch/CloudWatch access. Only needed when Batch infra
                 is in a different account than the S3 data bucket.

Usage:
    # Auto-discovers --bucket, --output-prefix, --region from terraform outputs
    python scripts/post_run_report.py --mode masked --profile my-profile

    # Explicit overrides
    python scripts/post_run_report.py \
        --bucket my-bucket \
        --output-prefix bridge-classification/runs/my-run/predictions \
        --mode masked \
        --profile my-profile

    # Cross-account: S3 data and Batch/CloudWatch on different profiles
    python scripts/post_run_report.py \
        --bucket my-bucket \
        --output-prefix bridge-classification/runs/my-run/predictions \
        --input-prefix bridge-classification/ml-data/source \
        --mode masked \
        --profile data-profile \
        --batch-profile infra-profile
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.constants import InferenceMode
from src.s3_audit import DEFAULT_AUDIT_WORKERS, audit_s3_outputs
from src.s3_client import (
    create_s3_client, download_json, stream_manifest_lines,
    upload_json, upload_text,
)
from src.terraform import get_terraform_outputs


SUMMARY_PATTERN = re.compile(
    r'SUMMARY succeeded=(\d+) failed=(\d+) skipped_exists=(\d+) '
    r'skipped_too_few_points=(\d+) download_failed=(\d+) total=(\d+) '
    r'wall_clock_seconds=([\d.]+) wall_clock_hours=([\d.]+)'
)

BRIDGE_TIME_PATTERN = re.compile(r'bridge_seconds=([\d.]+)s')

LOG_GROUP = '/aws/batch/bridge-classifier'
CLOUDWATCH_TIME_PADDING_MS = 60_000
MAX_MISSING_ENTRIES = 1000
MAX_MISSING_REASONS_DISPLAY = 10


def describe_batch_job(batch_client: Any, job_id: str) -> Dict[str, Any]:
    """Get job status and timing from AWS Batch.

    For array jobs, the parent has createdAt but not startedAt/stoppedAt
    (those are on the children). Falls back to createdAt for time window.
    """
    response = batch_client.describe_jobs(jobs=[job_id])
    if not response['jobs']:
        return {'status': 'NOT_FOUND'}
    job = response['jobs'][0]
    result = {
        'status': job.get('status', 'UNKNOWN'),
        'status_reason': job.get('statusReason', ''),
    }
    if 'createdAt' in job:
        result['created_at'] = datetime.fromtimestamp(job['createdAt'] / 1000, tz=timezone.utc).isoformat()
        result['created_at_ms'] = job['createdAt']
    if 'startedAt' in job:
        result['started_at'] = datetime.fromtimestamp(job['startedAt'] / 1000, tz=timezone.utc).isoformat()
        result['started_at_ms'] = job['startedAt']
    if 'stoppedAt' in job:
        result['stopped_at'] = datetime.fromtimestamp(job['stoppedAt'] / 1000, tz=timezone.utc).isoformat()
        result['stopped_at_ms'] = job['stoppedAt']
        start = job.get('startedAt', job.get('createdAt', 0))
        result['wall_clock_hours'] = round((job['stoppedAt'] - start) / 1000 / 3600, 2)
    if 'arrayProperties' in job:
        result['array_size'] = job['arrayProperties'].get('size', 0)
    return result


def _query_cloudwatch(
    logs_client: Any,
    filter_pattern: str,
    start_ms: int,
    end_ms: int,
    processor: Callable[[str], None],
) -> None:
    """Paginate CloudWatch filter_log_events calls, passing each message to processor."""
    kwargs = {
        'logGroupName': LOG_GROUP,
        'startTime': start_ms,
        'endTime': end_ms + CLOUDWATCH_TIME_PADDING_MS,
        'filterPattern': filter_pattern,
    }
    while True:
        response = logs_client.filter_log_events(**kwargs)
        for event in response.get('events', []):
            processor(event['message'])
        if 'nextToken' not in response:
            break
        kwargs['nextToken'] = response['nextToken']


def query_cloudwatch_summaries(
    logs_client: Any, start_ms: int, end_ms: int
) -> Tuple[Dict[str, int], List[float]]:
    """Query CloudWatch for SUMMARY lines, return aggregated totals and per-child wall clock seconds."""
    totals = {
        'succeeded': 0, 'failed': 0, 'skipped_exists': 0,
        'skipped_too_few_points': 0, 'download_failed': 0, 'total': 0,
    }
    child_seconds: List[float] = []

    def process(msg: str) -> None:
        match = SUMMARY_PATTERN.search(msg)
        if match:
            totals['succeeded'] += int(match.group(1))
            totals['failed'] += int(match.group(2))
            totals['skipped_exists'] += int(match.group(3))
            totals['skipped_too_few_points'] += int(match.group(4))
            totals['download_failed'] += int(match.group(5))
            totals['total'] += int(match.group(6))
            child_seconds.append(float(match.group(7)))

    _query_cloudwatch(logs_client, 'SUMMARY', start_ms, end_ms, process)
    return totals, child_seconds


def query_cloudwatch_bridge_times(
    logs_client: Any, start_ms: int, end_ms: int
) -> List[float]:
    """Query CloudWatch for INFER_OK lines, extract per-bridge seconds."""
    times: List[float] = []

    def process(msg: str) -> None:
        match = BRIDGE_TIME_PATTERN.search(msg)
        if match:
            times.append(float(match.group(1)))

    _query_cloudwatch(logs_client, 'INFER_OK', start_ms, end_ms, process)
    return times


def query_cloudwatch_missing_reasons(
    logs_client: Any, start_ms: int, end_ms: int, missing: List[str],
) -> Dict[str, str]:
    """Query CloudWatch for failure/skip reasons on missing bridges."""
    reasons: Dict[str, str] = {}
    fail_pattern = re.compile(r'INFER_FAILED reason=(\S+)')
    skip_pattern = re.compile(r'SKIP_SMALL_FILE')

    for bridge_line in missing:
        bridge_stem = bridge_line.split('/')[-1] if '/' in bridge_line else bridge_line
        quoted_stem = f'"{bridge_stem}"'

        def _make_capture(bl: str) -> Callable[[str], None]:
            def _capture(msg: str) -> None:
                fm = fail_pattern.search(msg)
                if fm:
                    reasons[bl] = fm.group(1)
                elif skip_pattern.search(msg):
                    reasons[bl] = 'too_few_points'
            return _capture

        _query_cloudwatch(logs_client, quoted_stem, start_ms, end_ms, _make_capture(bridge_line))

    return reasons


def compute_percentile(values: List[float], pct: float) -> float:
    """Compute percentile from sorted values using nearest-rank method."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(len(values_sorted) * pct / 100)
    idx = min(idx, len(values_sorted) - 1)
    return values_sorted[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description='Bridge Classification - Post-Run Report')
    parser.add_argument('--bucket', type=str, help='S3 bucket (default: from terraform output s3_bucket)')
    parser.add_argument('--output-prefix', type=str, help='S3 output prefix (default: from terraform output s3_output_prefix)')
    parser.add_argument('--input-prefix', type=str, default='', help='S3 input prefix (for extension probing during audit)')
    parser.add_argument('--mode', type=InferenceMode, default=InferenceMode.MASKED,
                        help='Inference mode: masked (default), raw, or both')
    parser.add_argument('--audit-workers', type=int, default=DEFAULT_AUDIT_WORKERS,
                        help=f'Parallel S3 audit workers (default: {DEFAULT_AUDIT_WORKERS})')
    parser.add_argument('--skip-timing', action='store_true',
                        help='Skip per-bridge timing extraction (faster, fewer CloudWatch queries)')
    parser.add_argument('--region', type=str, help='AWS region for Batch and CloudWatch (default: from terraform or us-east-1)')
    parser.add_argument('--profile', type=str, help='AWS profile for S3 operations')
    parser.add_argument('--batch-profile', type=str,
                        help='AWS profile for Batch and CloudWatch (defaults to --profile)')
    args = parser.parse_args()

    # --- Resolve config from terraform outputs ---
    tf = get_terraform_outputs()
    bucket = args.bucket or tf.get('s3_bucket')
    output_prefix = args.output_prefix or tf.get('s3_output_prefix')
    region = args.region or os.environ.get('AWS_REGION') or tf.get('aws_region') or 'us-east-1'

    missing_config = []
    if not bucket:
        missing_config.append('--bucket (or set s3_bucket in terraform.tfvars)')
    if not output_prefix:
        missing_config.append('--output-prefix (or set s3_output_prefix in terraform.tfvars)')
    if missing_config:
        print(f"ERROR: Missing config: {', '.join(missing_config)}")
        sys.exit(1)

    batch_profile = args.batch_profile or args.profile or os.environ.get('AWS_PROFILE')
    s3 = create_s3_client(args.profile)

    # --- 1. Load run config ---
    print("Loading run config from S3...")
    try:
        config_key = f"{output_prefix}/_run_config.json"
        run_config = download_json(s3, bucket, config_key)
    except Exception as e:
        print(f"ERROR: Could not load s3://{bucket}/{output_prefix}/_run_config.json: {e}")
        print("Was S3_OUTPUT_PREFIX set correctly when submit_batch_job.py ran?")
        sys.exit(1)

    job_id = run_config.get('job_id')
    if not job_id:
        print("ERROR: _run_config.json missing job_id")
        sys.exit(1)
    manifest_uri = run_config.get('manifest_uri')
    expected_array_size = run_config.get('array_size', 0)
    print(f"Job: {run_config.get('job_name')} (ID: {job_id})")
    print(f"Manifest: {manifest_uri}")
    print(f"Total bridges: {run_config.get('total_bridges')}")

    # --- 2. Check job status ---
    print("\nChecking job status...")
    session = boto3.Session(profile_name=batch_profile)
    batch_client = session.client('batch', region_name=region)
    job_info = describe_batch_job(batch_client, job_id)
    print(f"Status: {job_info['status']}")

    # --- 3. Audit outputs ---
    print("\nAuditing S3 outputs...")
    manifest_lines = list(stream_manifest_lines(s3, manifest_uri))
    print(f"Manifest lines: {len(manifest_lines)}")

    found, missing = audit_s3_outputs(
        profile=args.profile,
        bucket=bucket,
        input_prefix=args.input_prefix,
        output_prefix=output_prefix,
        mode=args.mode,
        manifest_lines=manifest_lines,
        workers=args.audit_workers,
    )
    print(f"Found: {found}, Missing: {len(missing)}")

    # --- 4. Query CloudWatch ---
    logs_client = session.client('logs', region_name=region)

    totals = {}
    child_seconds: List[float] = []
    timing = {}
    missing_reasons: Dict[str, str] = {}

    start_ms = job_info.get('started_at_ms') or job_info.get('created_at_ms')
    end_ms = job_info.get('stopped_at_ms') or int(datetime.now(timezone.utc).timestamp() * 1000)

    if start_ms:
        print("\nQuerying CloudWatch for SUMMARY lines...")
        totals, child_seconds = query_cloudwatch_summaries(logs_client, start_ms, end_ms)
        summary_count = len(child_seconds)
        print(f"Found {summary_count} child summaries (expected {expected_array_size})")
        if summary_count < expected_array_size:
            print(f"WARNING: {expected_array_size - summary_count} children did not log SUMMARY (possible SPOT interruption)")

        if not args.skip_timing:
            print("\nQuerying CloudWatch for per-bridge timing (this may take some time)...")
            bridge_times = query_cloudwatch_bridge_times(logs_client, start_ms, end_ms)
            if bridge_times:
                timing = {
                    'bridges_timed': len(bridge_times),
                    'avg_seconds': round(sum(bridge_times) / len(bridge_times), 1),
                    'p50_seconds': round(compute_percentile(bridge_times, 50), 1),
                    'p95_seconds': round(compute_percentile(bridge_times, 95), 1),
                    'min_seconds': round(min(bridge_times), 1),
                    'max_seconds': round(max(bridge_times), 1),
                }
        else:
            print("\nSkipping per-bridge timing (--skip-timing)")

        if missing:
            print(f"\nQuerying CloudWatch for failure reasons on {len(missing)} missing bridges...")
            missing_reasons = query_cloudwatch_missing_reasons(logs_client, start_ms, end_ms, missing)
            if missing_reasons:
                print(f"Found reasons for {len(missing_reasons)} of {len(missing)} bridges:")
                for bridge, reason in list(missing_reasons.items())[:MAX_MISSING_REASONS_DISPLAY]:
                    print(f"  {bridge}: {reason}")
    else:
        print("\nWARNING: Job has no timestamps - skipping CloudWatch queries")

    # --- 5. Build report ---
    job_info_clean = {k: v for k, v in job_info.items() if not k.endswith('_ms')}
    total_child_hours = round(sum(s / 3600 for s in child_seconds), 4) if child_seconds else None
    spot_rate = run_config.get('spot_rate_usd', 0)

    report = {
        'reported_at': datetime.now(timezone.utc).isoformat(),
        'job': job_info_clean,
        'audit': {
            'total': len(manifest_lines),
            'found': found,
            'missing': len(missing),
        },
        'aggregated_results': totals,
        'child_timing': {
            'children_reported': len(child_seconds),
            'children_expected': expected_array_size,
            'min_seconds': round(min(child_seconds), 1) if child_seconds else None,
            'max_seconds': round(max(child_seconds), 1) if child_seconds else None,
            'avg_seconds': round(sum(child_seconds) / len(child_seconds), 1) if child_seconds else None,
            'total_child_hours': total_child_hours,
        },
        'bridge_timing': timing,
        'cost_estimate': {
            'total_child_hours': total_child_hours,
            'spot_rate_usd': spot_rate,
            'estimated_compute_usd': round(total_child_hours * spot_rate, 2),
            'note': 'Based on sum of child wall-clock hours x spot rate; actual billing may differ',
        } if total_child_hours and spot_rate else None,
        'run_config': run_config,
    }

    if missing:
        report['missing_entries'] = missing[:MAX_MISSING_ENTRIES]
        if len(missing) > MAX_MISSING_ENTRIES:
            report['missing_entries_truncated'] = True
            report['total_missing'] = len(missing)
        if missing_reasons:
            capped_reasons = {k: v for k, v in missing_reasons.items() if k in set(missing[:MAX_MISSING_ENTRIES])}
            report['missing_reasons'] = capped_reasons

    # --- 6. Save report to S3 ---
    report_key = f"{output_prefix}/_run_report.json"
    upload_json(s3, report, bucket, report_key)
    print(f"\nReport saved: s3://{bucket}/{report_key}")

    # --- 7. Save missing manifest ---
    if missing:
        missing_key = f"{output_prefix}/_missing.txt"
        missing_text = "\n".join(missing) + "\n"
        upload_text(s3, missing_text, bucket, missing_key)
        print(f"Missing manifest: s3://{bucket}/{missing_key} ({len(missing)} entries)")

    # --- 8. Print summary ---
    print(f"\n{'='*60}")
    print(f"POST-RUN REPORT")
    print(f"{'='*60}")
    print(f"Job:        {run_config.get('job_name')}")
    print(f"Status:     {job_info['status']}")
    if child_seconds:
        print(f"Children:   {len(child_seconds)} reported, "
              f"max={round(max(child_seconds), 1)}s, avg={round(sum(child_seconds)/len(child_seconds), 1)}s")
    print(f"Audit:      {found} found, {len(missing)} missing out of {len(manifest_lines)}")
    if totals:
        print(f"Results:    {totals['succeeded']} succeeded, {totals['failed']} failed, "
              f"{totals['skipped_too_few_points']} skipped")
    if timing:
        print(f"Timing:     avg={timing['avg_seconds']}s, p50={timing['p50_seconds']}s, p95={timing['p95_seconds']}s")
    if report.get('cost_estimate'):
        ce = report['cost_estimate']
        print(f"Cost est:   ${ce['estimated_compute_usd']:.2f} "
              f"({ce['total_child_hours']:.4f} hrs x ${spot_rate}/hr)")
    print(f"Report:     s3://{bucket}/{report_key}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
