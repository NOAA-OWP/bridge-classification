#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Bridge Classification — Build Docker image and push to ECR
#
# Usage:
#   ./scripts/build_and_push.sh
#
# Gets ECR URL from terraform output. Override with ECR_REPO env var.
# ---------------------------------------------------------------------------

AWS_PROFILE=${AWS_PROFILE:-test-se}
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-591210920133}

# Get ECR repo URL from terraform or env var
if [ -n "$ECR_REPO" ]; then
  echo "Using ECR_REPO from environment: $ECR_REPO"
elif [ -d "terraform" ] && command -v terraform &>/dev/null; then
  ECR_REPO=$(cd terraform && terraform output -raw ecr_repository_url 2>/dev/null) || true
fi

if [ -z "$ECR_REPO" ]; then
  # Fallback: construct from account ID and project name
  PROJECT_NAME=${PROJECT_NAME:-bridge-classifier}
  ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"
  echo "Using constructed ECR URL: $ECR_REPO"
else
  echo "Using ECR URL: $ECR_REPO"
fi

# 1. Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 2. Build
echo "Building Docker image (linux/amd64)..."
docker build --platform linux/amd64 -t bridge-classifier .

# 3. Tag
docker tag bridge-classifier:latest "${ECR_REPO}:latest"

# 4. Push
echo "Pushing to ECR..."
docker push "${ECR_REPO}:latest"

echo ""
echo "Done. Image pushed to: ${ECR_REPO}:latest"
