#!/bin/bash
# set -e
# set -o pipefail

# Job index: Batch sets AWS_BATCH_JOB_ARRAY_INDEX for array jobs; default 0 for single-job test
JOB_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}

# Env-based config (defaults match current hardcoded values; override in Batch job definition)
S3_BUCKET=${S3_BUCKET:-fimc-data}
S3_INPUT_PREFIX=${S3_INPUT_PREFIX:-bridge-classification/ml-data/source}
S3_MANIFEST_URI=${S3_MANIFEST_URI:-s3://fimc-data/bridge-classification/ml-data/split_test_ids.txt}
S3_MODEL_URI=${S3_MODEL_URI:-s3://fimc-data/scratch/biplov.bhandari/bridge-classification-test/experiments/bridge-base-v0/version_0/checkpoints/bridge-unet-epoch=46-val_loss=0.9439.ckpt}
S3_OUTPUT_PREFIX=${S3_OUTPUT_PREFIX:-scratch/biplov.bhandari/bridge-classification-test/predictions}
USE_GPU=${USE_GPU:-true}

# Use /tmp for downloads and inference (ephemeral storage; keeps /app read-only)
WORK_DIR=/tmp/batch
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 1. Download manifest and model
aws s3 cp "$S3_MANIFEST_URI" ./manifest.txt
aws s3 cp "$S3_MODEL_URI" ./model.ckpt

# 2. Read the line for this job index (1-based line number)
MANIFEST_LINE=$(sed -n "$((JOB_INDEX + 1))p" manifest.txt | tr -d '\r\n')

# 3. Resolve S3 input path: full URI (s3://...) or relative (build from bucket + prefix)
if [[ -z "$MANIFEST_LINE" ]]; then
  echo "No manifest line for index $JOB_INDEX"
  exit 1
fi
if [[ "$MANIFEST_LINE" == s3://* ]]; then
  S3_INPUT_PATH="$MANIFEST_LINE"
else
  # Relative line (e.g. huc_id/bridge_stem); append .laz if not present
  if [[ "$MANIFEST_LINE" == *.laz ]]; then
    S3_INPUT_PATH="s3://${S3_BUCKET}/${S3_INPUT_PREFIX}/${MANIFEST_LINE}"
  else
    S3_INPUT_PATH="s3://${S3_BUCKET}/${S3_INPUT_PREFIX}/${MANIFEST_LINE}.laz"
  fi
fi

# 4. Output path (same bucket; output prefix + basename_predicted.laz)
FILENAME=$(basename "$S3_INPUT_PATH")
OUTPUT_BASENAME="${FILENAME%.*}"
S3_OUTPUT_PATH="s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/${OUTPUT_BASENAME}_predicted.laz"

# 5. Download input
aws s3 cp "$S3_INPUT_PATH" ./input.laz

# 6. Run inference (from /app so src imports resolve)
GPU_ARG=""
if [[ "$USE_GPU" == "true" ]]; then
  GPU_ARG="--gpu"
fi
python /app/src/inference.py \
  --input "$WORK_DIR/input.laz" \
  --output "$WORK_DIR/output.laz" \
  --model "$WORK_DIR/model.ckpt" \
  $GPU_ARG

# 7. Upload result
aws s3 cp ./output.laz "$S3_OUTPUT_PATH"
echo "Done: $S3_OUTPUT_PATH"
