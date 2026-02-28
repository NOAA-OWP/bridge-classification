#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Bridge Classification — Build Docker image and push to ECR
#
# Usage:
#   ./scripts/build_and_push.sh
#
# Reads config from Terraform outputs first, then falls back to environment
# variables. Exits early if required values are missing — no hardcoded defaults.
# ---------------------------------------------------------------------------

# Read from terraform outputs first, then env vars
if [ -d "terraform" ] && command -v terraform &>/dev/null; then
  _tf_region=$(cd terraform && terraform output -raw aws_region 2>/dev/null) || true
  _tf_profile=$(cd terraform && terraform output -raw aws_profile 2>/dev/null) || true
  _tf_ecr=$(cd terraform && terraform output -raw ecr_repository_url 2>/dev/null) || true
  [ -n "$_tf_region" ]  && AWS_REGION="$_tf_region"
  [ -n "$_tf_profile" ] && AWS_PROFILE="$_tf_profile"
  [ -n "$_tf_ecr" ]     && ECR_REPO="${ECR_REPO:-$_tf_ecr}"
fi

missing=""
[ -z "$AWS_REGION" ]  && missing="${missing}AWS_REGION "
[ -z "$AWS_PROFILE" ] && missing="${missing}AWS_PROFILE "
[ -z "$ECR_REPO" ]    && missing="${missing}ECR_REPO "
if [ -n "$missing" ]; then
  echo "ERROR: Missing: $missing" >&2
  echo "Run 'cd terraform && terraform init && terraform apply' or set env vars." >&2
  exit 1
fi

echo "Using ECR URL: $ECR_REPO"

# Derive registry host from repo URL (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com)
ECR_REGISTRY=$(echo "$ECR_REPO" | cut -d'/' -f1)

# Git SHA tag for traceability and rollback
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null) || GIT_SHA="unknown"
SHA_TAG="${ECR_REPO}:git-${GIT_SHA}"

# 1. Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# 2. Build
echo "Building Docker image (linux/amd64)..."
docker build --platform linux/amd64 -t bridge-classifier .

# 3. Tag (both :latest and :git-<sha>)
docker tag bridge-classifier:latest "${ECR_REPO}:latest"
docker tag bridge-classifier:latest "$SHA_TAG"

# 4. Push both tags
echo "Pushing to ECR..."
docker push "${ECR_REPO}:latest"
docker push "$SHA_TAG"

echo ""
echo "Done."
echo "  latest : ${ECR_REPO}:latest"
echo "  sha    : $SHA_TAG"
echo ""
echo "To roll back to this image: docker tag $SHA_TAG ${ECR_REPO}:latest && docker push ${ECR_REPO}:latest"
