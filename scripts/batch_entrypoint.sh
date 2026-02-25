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
#   ARRAY_SIZE                 – total number of array children (must match
#                                the --array-size you submit with)
# ---------------------------------------------------------------------------

# Job index: Batch sets AWS_BATCH_JOB_ARRAY_INDEX for array jobs; default 0 for single-job test
JOB_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}
ARRAY_SIZE=${ARRAY_SIZE:-1}

# Env-based config (defaults match current hardcoded values; override in Batch job definition)
S3_BUCKET=${S3_BUCKET:-fimc-data}
S3_INPUT_PREFIX=${S3_INPUT_PREFIX:-bridge-classification/ml-data/source}
S3_MANIFEST_URI=${S3_MANIFEST_URI:-s3://fimc-data/bridge-classification/ml-data/split_test_ids.txt}
S3_MODEL_URI=${S3_MODEL_URI:-s3://fimc-data/scratch/biplov.bhandari/bridge-classification-test/experiments/bridge-base-all-data-v3/version_0/checkpoints/bridge-unet-epoch=48-val_deck_iou=83.4327.ckpt}
S3_OUTPUT_PREFIX=${S3_OUTPUT_PREFIX:-scratch/biplov.bhandari/bridge-classification-test/predictions}
USE_GPU=${USE_GPU:-true}

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
echo "Child $JOB_INDEX: huc_ids=[$HUC_IDS]"

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

# Extract this child's chunk once (avoids re-scanning manifest per line)
while IFS= read -r MANIFEST_LINE || [[ -n "$MANIFEST_LINE" ]]; do
  MANIFEST_LINE="${MANIFEST_LINE%%$'\r'}"

  # Resolve S3 input path: full URI (s3://...) or relative (build from bucket + prefix)
  if [[ -z "$MANIFEST_LINE" ]]; then
    echo "WARN: empty manifest line, skipping"
    continue
  fi
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
  else
    echo "ERROR: failed to download $S3_INPUT_PATH"
    DOWNLOAD_FAILED=$((DOWNLOAD_FAILED + 1))
  fi
done < <(sed -n "${START},${END}p" manifest.txt)

# 5. Run inference ONCE for all files (model loaded once, not per-file)
echo "Running batch inference..."
python /app/src/inference.py \
  --pairs-file "$PAIRS_FILE" \
  --model "$WORK_DIR/model.ckpt" \
  $GPU_ARG \
  || true  # Don't fail here; per-file errors are tracked by checking outputs

# 6. Upload all output files
UPLOAD_SUCCESS=0
UPLOAD_FAILED=0

for i in "${!LOCAL_OUTPUT_PATHS[@]}"; do
  LOCAL_OUT="${LOCAL_OUTPUT_PATHS[$i]}"
  S3_OUT="${S3_OUTPUT_PATHS[$i]}"
  if [[ -f "$LOCAL_OUT" ]]; then
    if aws s3 cp "$LOCAL_OUT" "$S3_OUT"; then
      echo "Uploaded: $S3_OUT"
      UPLOAD_SUCCESS=$((UPLOAD_SUCCESS + 1))
    else
      echo "ERROR: failed to upload $LOCAL_OUT to $S3_OUT"
      UPLOAD_FAILED=$((UPLOAD_FAILED + 1))
    fi
  else
    echo "WARN: output not found (inference failed?): $LOCAL_OUT"
    UPLOAD_FAILED=$((UPLOAD_FAILED + 1))
  fi
done

# 7. Summary and cleanup
TOTAL_ATTEMPTED=$((END - START + 1))
echo "Child $JOB_INDEX complete: $UPLOAD_SUCCESS uploaded, $UPLOAD_FAILED failed, $DOWNLOAD_FAILED download failures out of $TOTAL_ATTEMPTED"

rm -rf "$WORK_DIR"

# Exit non-zero if ANY failures so Batch marks this child as failed
if [ "$UPLOAD_FAILED" -gt 0 ] || [ "$DOWNLOAD_FAILED" -gt 0 ]; then
  exit 1
fi
