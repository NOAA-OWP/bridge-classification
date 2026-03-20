"""
Bridge Classification — Batch Entrypoint for AWS Batch

Required env vars (set by Batch job definition, managed by Terraform):
  AWS_BATCH_JOB_ARRAY_INDEX  - child index (0-based), set automatically by Batch
  For other cloud providers, AWS_BATCH_JOB_ARRAY_INDEX becomes BATCH_TASK_INDEX (GCP Batch) or AZ_BATCH_TASK_ID (Azure Batch)

  ARRAY_SIZE                 - total number of array children
  S3_BUCKET                  - S3 bucket for input/output data
  S3_INPUT_PREFIX            - S3 prefix for source LAS/LAZ files
  S3_MANIFEST_URI            - full S3 URI of the manifest file
  S3_MODEL_URI               - full S3 URI of the model checkpoint
  S3_OUTPUT_PREFIX           - S3 prefix for prediction outputs

Optional env vars:
  INFERENCE_MODE    - "masked" (default), "raw", or "both"
  BRIDGE_TIMEOUT    - per-bridge timeout in seconds (default: 150)
"""

import os
import signal
import sys
import time
from pathlib import Path, PurePosixPath

import boto3
import torch
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

AWS_MAX_RETRIES = 3

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.inference import BridgeTimeout, _timeout_handler, load_model, run_inference
from src.s3 import (
    download_file, object_exists, parse_s3_uri, resolve_input_key,
    resolve_output_keys, upload_file,
)


def log(msg, child_index=None, bridge_id=None):
    """Structured log line for logging to CloudWatch."""
    prefix = f"[Child {child_index}]" if child_index is not None else "[Entrypoint]"
    if bridge_id:
        prefix += f" [bridge={bridge_id}]"
    print(f"{prefix} {msg}", flush=True)


def parse_config():
    """Read and validate environment variables."""
    required = ['S3_BUCKET', 'S3_INPUT_PREFIX', 'S3_MANIFEST_URI', 'S3_MODEL_URI', 'S3_OUTPUT_PREFIX']
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: required environment variables not set: {' '.join(missing)}", flush=True)
        print("These should be set in the Batch job definition (managed by Terraform).", flush=True)
        sys.exit(1)

    return {
        's3_bucket': os.environ['S3_BUCKET'],
        's3_input_prefix': os.environ['S3_INPUT_PREFIX'],
        's3_manifest_uri': os.environ['S3_MANIFEST_URI'],
        's3_model_uri': os.environ['S3_MODEL_URI'],
        's3_output_prefix': os.environ['S3_OUTPUT_PREFIX'],
        'job_index': int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', '0')),
        'array_size': int(os.environ.get('ARRAY_SIZE', '1')),
        'inference_mode': os.environ.get('INFERENCE_MODE', 'masked'),
        'bridge_timeout': float(os.environ.get('BRIDGE_TIMEOUT', '150')),
    }


def compute_chunk(job_index, array_size, total_lines):
    """Compute this child's slice of manifest lines (0-based indices).

    Returns (start, end) where start is inclusive and end is exclusive.
    """
    chunk_size = (total_lines + array_size - 1) // array_size
    start = job_index * chunk_size
    end = min(start + chunk_size, total_lines)
    return start, end


def cleanup(*paths):
    """Remove local files if they exist."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def main():
    job_start = time.time()

    # --- Config ---
    cfg = parse_config()
    idx = cfg['job_index']

    s3 = boto3.client('s3', config=BotoConfig(
        retries={'max_attempts': AWS_MAX_RETRIES, 'mode': 'adaptive'}
    ))

    # --- SIGTERM handler for SPOT interruptions ---
    shutdown_requested = False

    def sigterm_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        log("SIGTERM received — finishing current bridge then exiting", child_index=idx)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # --- Work directories ---
    work_dir = Path('/tmp/batch')
    input_dir = work_dir / 'inputs'
    output_dir = work_dir / 'outputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Download manifest + model ---
    manifest_bucket, manifest_key = parse_s3_uri(cfg['s3_manifest_uri'])
    manifest_path = work_dir / 'manifest.txt'
    log(f"Downloading manifest: {cfg['s3_manifest_uri']}", child_index=idx)
    download_file(s3, manifest_bucket, manifest_key, str(manifest_path))

    model_bucket, model_key = parse_s3_uri(cfg['s3_model_uri'])
    model_path = work_dir / 'model.ckpt'
    log(f"Downloading model: {cfg['s3_model_uri']}", child_index=idx)
    download_file(s3, model_bucket, model_key, str(model_path))

    # --- 2. Read manifest and compute chunk ---
    with open(manifest_path) as f:
        all_lines = [line.strip() for line in f if line.strip()]

    total_lines = len(all_lines)
    start, end = compute_chunk(idx, cfg['array_size'], total_lines)
    chunk_lines = all_lines[start:end]
    chunk_size = len(chunk_lines)

    huc_ids = sorted(set(line.split('/')[0] for line in chunk_lines))
    log(f"Processing lines {start+1}-{end} of {total_lines} "
        f"(chunk_size={chunk_size}, huc_ids=[{','.join(huc_ids)}])", child_index=idx)

    # --- 3. Load model — GPU required (spconv-cu120) ---
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available. GPU is required for inference.", child_index=idx)
        sys.exit(1)
    device = torch.device('cuda')
    log(f"Using device: {device}", child_index=idx)
    model = load_model(str(model_path), device)

    # --- 4. Per-bridge processing loop ---
    mode = cfg['inference_mode']
    bridge_timeout = cfg['bridge_timeout']
    bucket = cfg['s3_bucket']
    succeeded = 0
    failed = 0
    skipped = 0
    download_failed = 0

    old_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        for i, manifest_line in enumerate(chunk_lines, 1):
            bridge_start = time.time()
            global_line = start + i  # 1-based global manifest position

            # Check for SPOT shutdown
            if shutdown_requested:
                log("SPOT_SHUTDOWN stopping before next bridge", child_index=idx)
                break

            bridge_id = PurePosixPath(manifest_line).stem
            huc_id = manifest_line.split('/')[0]

            # 4a. Resolve S3 paths
            try:
                input_key = resolve_input_key(s3, bucket, cfg['s3_input_prefix'], manifest_line)
            except FileNotFoundError:
                log(f"INPUT_NOT_FOUND manifest_line=\"{manifest_line}\"",
                    child_index=idx, bridge_id=bridge_id)
                download_failed += 1
                continue

            input_ext = PurePosixPath(input_key).suffix
            output_keys = resolve_output_keys(cfg['s3_output_prefix'], manifest_line, input_ext, mode)

            # 4b. Skip if output already exists in S3 (resumability)
            primary_exists = object_exists(s3, bucket, output_keys['primary'])
            if mode == 'both':
                masked_exists = object_exists(s3, bucket, output_keys.get('masked', ''))
                all_exist = primary_exists and masked_exists
            else:
                all_exist = primary_exists

            if all_exist:
                log(f"SKIP_EXISTS ({i}/{chunk_size}) manifest_line={global_line} huc={huc_id}",
                    child_index=idx, bridge_id=bridge_id)
                skipped += 1
                continue

            # 4c. Download input
            input_filename = PurePosixPath(input_key).name
            local_input = str(input_dir / input_filename)
            try:
                download_file(s3, bucket, input_key, local_input)
            except ClientError as e:
                log(f"DOWNLOAD_FAILED error=\"{e}\" huc={huc_id} manifest_line={global_line}",
                    child_index=idx, bridge_id=bridge_id)
                download_failed += 1
                continue

            # 4d. Prepare local output path
            input_p = PurePosixPath(input_key)
            ext = input_p.suffix
            stem = input_p.stem

            if mode == 'masked':
                output_name = f"{stem}_bridge_masked{ext}"
            else:
                output_name = f"{stem}_predicted{ext}"

            local_output_dir = output_dir / huc_id
            local_output_dir.mkdir(parents=True, exist_ok=True)
            local_output = str(local_output_dir / output_name)

            # 4e. Run inference with SIGALRM timeout
            log(f"INFER_START ({i}/{chunk_size}) mode={mode} huc={huc_id} manifest_line={global_line}",
                child_index=idx, bridge_id=bridge_id)
            signal.setitimer(signal.ITIMER_REAL, bridge_timeout)
            try:
                ok = run_inference(model, local_input, local_output, voxel_size=0.1,
                                   device=device, mode=mode)
            except BridgeTimeout:
                log(f"TIMEOUT bridge_timeout={bridge_timeout}s huc={huc_id}",
                    child_index=idx, bridge_id=bridge_id)
                ok = False
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)  # cancel alarm before upload

            if not ok:
                log(f"INFER_FAILED huc={huc_id} manifest_line={global_line}",
                    child_index=idx, bridge_id=bridge_id)
                failed += 1
                cleanup(local_input, local_output)
                continue

            # 4f. Upload output(s) immediately
            try:
                upload_file(s3, local_output, bucket, output_keys['primary'])
                log(f"UPLOADED s3://{bucket}/{output_keys['primary']}",
                    child_index=idx, bridge_id=bridge_id)

                # mode=both: also upload the masked file that run_inference wrote
                if mode == 'both' and 'masked' in output_keys:
                    masked_local = str(Path(local_output).with_name(
                        f"{stem}_bridge_masked{ext}"))
                    if os.path.isfile(masked_local):
                        upload_file(s3, masked_local, bucket, output_keys['masked'])
                        log(f"UPLOADED s3://{bucket}/{output_keys['masked']}",
                            child_index=idx, bridge_id=bridge_id)
                        cleanup(masked_local)

                bridge_seconds = time.time() - bridge_start
                log(f"INFER_OK bridge_seconds={bridge_seconds:.1f}s ({i}/{chunk_size}) huc={huc_id}",
                    child_index=idx, bridge_id=bridge_id)
                succeeded += 1
            except ClientError as e:
                log(f"UPLOAD_FAILED error=\"{e}\" huc={huc_id}",
                    child_index=idx, bridge_id=bridge_id)
                failed += 1

            # 4g. Cleanup local files
            cleanup(local_input, local_output)

            # 4h. Free GPU memory
            torch.cuda.empty_cache()

    finally:
        signal.signal(signal.SIGALRM, old_alarm_handler)

    # --- 5. Summary ---
    job_seconds = time.time() - job_start
    job_hours = job_seconds / 3600
    log(f"SUMMARY succeeded={succeeded} failed={failed} skipped={skipped} "
        f"download_failed={download_failed} total={chunk_size} "
        f"wall_clock_seconds={job_seconds:.0f} wall_clock_hours={job_hours:.4f}",
        child_index=idx)

    # Cleanup work directory
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    # Exit non-zero if any failures so Batch marks this child as failed
    if failed > 0 or download_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
