#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Chunked array-job entrypoint for AWS Batch
#
# Each array child processes a CHUNK of manifest lines (not just one).
# For eg. with 600K files and an array size of 10,000, each child handles ~60 files.
#
# The model is loaded ONCE per child — the entrypoint downloads all input
# files, builds a pairs.tsv, and calls inference.py once in batch mode.
#
# Required env (set by Batch or job definition):
#   AWS_BATCH_JOB_ARRAY_INDEX  – child index (0-based), set automatically
#     GCP Cloud Batch equivalent: BATCH_TASK_INDEX
#     Azure Batch equivalent:     AZ_BATCH_TASK_ID
#   ARRAY_SIZE                 – total number of array children (must match
#                                the --array-size you submit with)
# ---------------------------------------------------------------------------

JOB_START=$(date +%s)

# Job index: Batch sets AWS_BATCH_JOB_ARRAY_INDEX for array jobs; default 0 for single-job test
JOB_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}
ARRAY_SIZE=${ARRAY_SIZE:-1}

# Validate required env vars — must be set in the Batch job definition (managed by Terraform).
# can be overridden at submit time
REQUIRED_VARS=(S3_BUCKET S3_INPUT_PREFIX S3_MANIFEST_URI S3_MODEL_URI S3_OUTPUT_PREFIX)
MISSING=()
for VAR in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!VAR}" ]; then
    MISSING+=("$VAR")
  fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: required environment variables not set: ${MISSING[*]}"
  echo "These should be set in the Batch job definition (managed by Terraform)."
  exit 1
fi
USE_GPU=${USE_GPU:-true}

# Retry transient S3 errors automatically (covers all aws s3 cp calls in this script)
export AWS_MAX_ATTEMPTS=3
export AWS_RETRY_MODE=adaptive

# Use /tmp for downloads and inference (ephemeral storage; keeps /app read-only)
WORK_DIR=/tmp/batch
INPUT_DIR="$WORK_DIR/inputs"
OUTPUT_DIR="$WORK_DIR/outputs"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
cd "$WORK_DIR"

# 1. Download manifest and model
aws s3 cp "$S3_MANIFEST_URI" ./manifest.txt
aws s3 cp "$S3_MODEL_URI" ./model.ckpt

# 2. Compute this child's chunk of manifest lines
TOTAL_FILES=$(wc -l < manifest.txt)
CHUNK_SIZE=$(( (TOTAL_FILES + ARRAY_SIZE - 1) / ARRAY_SIZE ))
START=$(( JOB_INDEX * CHUNK_SIZE + 1 ))
END=$(( START + CHUNK_SIZE - 1 ))
if [ "$END" -gt "$TOTAL_FILES" ]; then END=$TOTAL_FILES; fi

echo "Child $JOB_INDEX: processing lines $START-$END of $TOTAL_FILES (chunk size $CHUNK_SIZE)"
HUC_IDS=$(sed -n "${START},${END}p" manifest.txt | cut -d'/' -f1 | sort -u | tr '\n' ',' | sed 's/,$//')
echo "Child $JOB_INDEX: huc_ids=[$HUC_IDS], bridge_count=$((END - START + 1))"

# 3. Pre-compute GPU flag once
GPU_ARG=""
if [[ "$USE_GPU" == "true" ]]; then
  GPU_ARG="--gpu"
fi

# 4. Download all input files and build pairs file
PAIRS_FILE="$WORK_DIR/pairs.tsv"
> "$PAIRS_FILE"

DOWNLOAD_FAILED=0
declare -a S3_OUTPUT_PATHS=()
declare -a LOCAL_OUTPUT_PATHS=()
declare -a BRIDGE_IDS=()

# Extract this child's chunk once (avoids re-scanning manifest per line)
while IFS= read -r MANIFEST_LINE || [[ -n "$MANIFEST_LINE" ]]; do
  MANIFEST_LINE="${MANIFEST_LINE%%$'\r'}"

  # Resolve S3 input path: full URI (s3://...) or relative (build from bucket + prefix)
  if [[ -z "$MANIFEST_LINE" ]]; then
    echo "WARN: empty manifest line, skipping"
    continue
  fi
  BRIDGE_ID=$(basename "$MANIFEST_LINE" .laz)
  if [[ "$MANIFEST_LINE" == s3://* ]]; then
    S3_INPUT_PATH="$MANIFEST_LINE"
  else
    if [[ "$MANIFEST_LINE" == *.laz ]]; then
      S3_INPUT_PATH="s3://${S3_BUCKET}/${S3_INPUT_PREFIX}/${MANIFEST_LINE}"
    else
      S3_INPUT_PATH="s3://${S3_BUCKET}/${S3_INPUT_PREFIX}/${MANIFEST_LINE}.laz"
    fi
  fi

  FILENAME=$(basename "$S3_INPUT_PATH")
  OUTPUT_BASENAME="${FILENAME%.*}"
  LOCAL_INPUT="$INPUT_DIR/${FILENAME}"
  LOCAL_OUTPUT="$OUTPUT_DIR/${OUTPUT_BASENAME}_predicted.laz"
  S3_OUTPUT_PATH="s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/${OUTPUT_BASENAME}_predicted.laz"

  # Download input file
  if aws s3 cp "$S3_INPUT_PATH" "$LOCAL_INPUT"; then
    printf '%s\t%s\n' "$LOCAL_INPUT" "$LOCAL_OUTPUT" >> "$PAIRS_FILE"
    S3_OUTPUT_PATHS+=("$S3_OUTPUT_PATH")
    LOCAL_OUTPUT_PATHS+=("$LOCAL_OUTPUT")
    BRIDGE_IDS+=("$BRIDGE_ID")
  else
    echo "ERROR: failed to download $S3_INPUT_PATH (bridge=$BRIDGE_ID)"
    DOWNLOAD_FAILED=$((DOWNLOAD_FAILED + 1))
  fi
done < <(sed -n "${START},${END}p" manifest.txt)

# 5. Run inference ONCE for all files (model loaded once, not per-file)
echo "Running batch inference..."
INFERENCE_START=$(date +%s)
INFERENCE_EXIT=0
python /app/src/inference.py \
  --pairs-file "$PAIRS_FILE" \
  --model "$WORK_DIR/model.ckpt" \
  $GPU_ARG \
  || INFERENCE_EXIT=$?
INFERENCE_END=$(date +%s)
INFERENCE_SECONDS=$(( INFERENCE_END - INFERENCE_START ))
echo "INFERENCE_WALL_CLOCK_SECONDS=$INFERENCE_SECONDS"
if [ "$INFERENCE_EXIT" -ne 0 ]; then
  echo "ERROR: inference process exited with code $INFERENCE_EXIT"
fi

# 6. Upload all output files
UPLOAD_SUCCESS=0
INFERENCE_FAILED=0
UPLOAD_FAILED=0

for i in "${!LOCAL_OUTPUT_PATHS[@]}"; do
  LOCAL_OUT="${LOCAL_OUTPUT_PATHS[$i]}"
  S3_OUT="${S3_OUTPUT_PATHS[$i]}"
  BRIDGE="${BRIDGE_IDS[$i]}"
  if [[ -f "$LOCAL_OUT" ]]; then
    if aws s3 cp "$LOCAL_OUT" "$S3_OUT"; then
      echo "Uploaded: $S3_OUT (bridge=$BRIDGE)"
      UPLOAD_SUCCESS=$((UPLOAD_SUCCESS + 1))
    else
      echo "ERROR: failed to upload $LOCAL_OUT to $S3_OUT (bridge=$BRIDGE)"
      UPLOAD_FAILED=$((UPLOAD_FAILED + 1))
    fi
  else
    echo "ERROR: inference failed for (bridge=$BRIDGE)"
    INFERENCE_FAILED=$((INFERENCE_FAILED + 1))
  fi
done

# 7. Summary and cleanup
TOTAL_ATTEMPTED=$((END - START + 1))
echo "Child $JOB_INDEX complete: $UPLOAD_SUCCESS uploaded, $INFERENCE_FAILED inference failures, $UPLOAD_FAILED upload failures, $DOWNLOAD_FAILED download failures out of $TOTAL_ATTEMPTED"

JOB_END=$(date +%s)
JOB_SECONDS=$(( JOB_END - JOB_START ))
JOB_HOURS=$(echo "$JOB_SECONDS" | awk '{printf "%.4f", $1/3600}')
echo "JOB_WALL_CLOCK_SECONDS=$JOB_SECONDS JOB_WALL_CLOCK_HOURS=$JOB_HOURS (billable instance time)"

rm -rf "$WORK_DIR"

# Exit non-zero if ANY failures so Batch marks this child as failed
if [ "$UPLOAD_FAILED" -gt 0 ] || [ "$DOWNLOAD_FAILED" -gt 0 ] || [ "$INFERENCE_FAILED" -gt 0 ]; then
  exit 1
fi
