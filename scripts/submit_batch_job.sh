#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Bridge Classification — Batch Job Submission Script
#
# The job definition (managed by Terraform) already contains all S3 config.
# This script just submits the job. For array jobs, it needs the manifest
# line count to compute ARRAY_SIZE — provide via --manifest or --total.
#
# Usage:
#   # Single job (no array) — no manifest count needed
#   ./scripts/submit_batch_job.sh --single
#
#   # Array job — provide S3 manifest URI or explicit count
#   S3_PROFILE=Data ./scripts/submit_batch_job.sh --manifest s3://fimc-data/bridge-classification/ml-data/split_test_ids.txt --profile Data
#   ./scripts/submit_batch_job.sh --total 600000
#
#   # --manifest streams the S3 file to count lines and passes the same URI as
#   # S3_MANIFEST_URI container override — containers use the exact same file.
#   # --total skips line counting; containers use S3_MANIFEST_URI from the job definition.
# ---------------------------------------------------------------------------

# --- Configuration (override via environment variables) ---
AWS_PROFILE=${AWS_PROFILE:-test-se}
S3_PROFILE=${S3_PROFILE:-$AWS_PROFILE}
AWS_REGION=${AWS_REGION:-us-east-1}

JOB_NAME=${JOB_NAME:-bridge-inference}

# Try to read job definition and queue names from terraform outputs; fall back to defaults
if [ -d "terraform" ] && command -v terraform &>/dev/null; then
  JOB_DEF_NAME=${JOB_DEF_NAME:-$(cd terraform && terraform output -raw job_definition_name 2>/dev/null || echo "")}
  JOB_QUEUE=${JOB_QUEUE:-$(cd terraform && terraform output -raw job_queue_name 2>/dev/null || echo "")}
fi
JOB_DEF_NAME=${JOB_DEF_NAME:-bridge-classifier-inference}
JOB_QUEUE=${JOB_QUEUE:-bridge-classifier-inference-queue}

# Target number of files each array child processes
CHUNK_TARGET=${CHUNK_TARGET:-60}

# Max array size (AWS Batch hard limit is 10,000)
MAX_ARRAY_SIZE=10000

# --- Parse flags ---
SINGLE_MODE=false
MANIFEST_URI=""
TOTAL_FILES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --single)
      SINGLE_MODE=true
      shift ;;
    --manifest)
      MANIFEST_URI="$2"
      shift 2 ;;
    --total)
      TOTAL_FILES="$2"
      shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--single] [--manifest <s3-uri>] [--total <N>]"
      exit 1 ;;
  esac
done

# --- Determine total files (only needed for array jobs) ---
if [ "$SINGLE_MODE" = true ]; then
  ARRAY_SIZE=1
  echo "Mode: single job (no array)"
else
  # Need total file count to compute array size
  if [ -n "$TOTAL_FILES" ]; then
    echo "Total files (provided): $TOTAL_FILES"
  elif [ -n "$MANIFEST_URI" ]; then
    echo "Counting files from S3: $MANIFEST_URI"
    TOTAL_FILES=$(aws s3 cp "$MANIFEST_URI" - --profile "$S3_PROFILE" | wc -l | tr -d ' ') || {
      echo "ERROR: cannot read S3 manifest: $MANIFEST_URI"
      echo "Check that your S3 profile ($S3_PROFILE) has s3:GetObject access."
      exit 1
    }
    echo "Total files in manifest: $TOTAL_FILES"
    S3_MANIFEST_URI="$MANIFEST_URI"
  else
    echo "ERROR: cannot determine manifest or file count."
    echo ""
    echo "Alternatives:"
    echo "  $0 --manifest <s3-uri>    # stream S3 manifest to count lines (also sets S3_MANIFEST_URI override)"
    echo "  $0 --total <N>            # provide the count directly (uses S3_MANIFEST_URI from job definition)"
    exit 1
  fi

  if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "ERROR: manifest is empty"
    exit 1
  fi

  # Compute array size
  if [ "$TOTAL_FILES" -le "$CHUNK_TARGET" ]; then
    ARRAY_SIZE=1
    echo "Mode: single job (no array) — $TOTAL_FILES files fits in one container"
  else
    ARRAY_SIZE=$(( (TOTAL_FILES + CHUNK_TARGET - 1) / CHUNK_TARGET ))
    if [ "$ARRAY_SIZE" -gt "$MAX_ARRAY_SIZE" ]; then
      ARRAY_SIZE=$MAX_ARRAY_SIZE
    fi
    ACTUAL_CHUNK=$(( (TOTAL_FILES + ARRAY_SIZE - 1) / ARRAY_SIZE ))
    echo "Mode: array job — $ARRAY_SIZE children, ~$ACTUAL_CHUNK files each"
  fi
fi

# --- Build environment overrides ---
# ARRAY_SIZE is always needed. S3 overrides only if explicitly set by caller
# (otherwise the job definition defaults from Terraform are used).
ENV_ITEMS='{"name":"ARRAY_SIZE","value":"'"$ARRAY_SIZE"'"}'

[ -n "$S3_MANIFEST_URI" ] && ENV_ITEMS="$ENV_ITEMS"',{"name":"S3_MANIFEST_URI","value":"'"$S3_MANIFEST_URI"'"}'
[ -n "$S3_MODEL_URI" ] && ENV_ITEMS="$ENV_ITEMS"',{"name":"S3_MODEL_URI","value":"'"$S3_MODEL_URI"'"}'
[ -n "$S3_BUCKET" ] && ENV_ITEMS="$ENV_ITEMS"',{"name":"S3_BUCKET","value":"'"$S3_BUCKET"'"}'
[ -n "$S3_INPUT_PREFIX" ] && ENV_ITEMS="$ENV_ITEMS"',{"name":"S3_INPUT_PREFIX","value":"'"$S3_INPUT_PREFIX"'"}'
[ -n "$S3_OUTPUT_PREFIX" ] && ENV_ITEMS="$ENV_ITEMS"',{"name":"S3_OUTPUT_PREFIX","value":"'"$S3_OUTPUT_PREFIX"'"}'

CONTAINER_OVERRIDES='{"environment":['"$ENV_ITEMS"']}'

# --- Submit ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
FULL_JOB_NAME="${JOB_NAME}-${TIMESTAMP}"

if [ "$ARRAY_SIZE" -eq 1 ]; then
  echo "Submitting single job: $FULL_JOB_NAME"
  aws batch submit-job \
    --job-name "$FULL_JOB_NAME" \
    --job-definition "$JOB_DEF_NAME" \
    --job-queue "$JOB_QUEUE" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --container-overrides "$CONTAINER_OVERRIDES"
else
  echo "Submitting array job: $FULL_JOB_NAME (size=$ARRAY_SIZE)"
  aws batch submit-job \
    --job-name "$FULL_JOB_NAME" \
    --job-definition "$JOB_DEF_NAME" \
    --job-queue "$JOB_QUEUE" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --array-properties "size=$ARRAY_SIZE" \
    --container-overrides "$CONTAINER_OVERRIDES"
fi

echo ""
echo "Job submitted: $FULL_JOB_NAME"
echo "Monitor at: https://console.aws.amazon.com/batch/home?region=${AWS_REGION}#jobs"
