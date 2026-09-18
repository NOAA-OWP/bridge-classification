#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Bridge Classification - Build Docker image and push to ECR
#
# Usage:
#   export AWS_PROFILE=my-profile
#   ./scripts/build_and_push.sh
#
# AWS_PROFILE: the account where ECR lives (same account as Batch infra).
# Reads inference_image_repo and aws_region from Terraform outputs, then
# falls back to environment variables. Exits early if required values
# are missing - no hardcoded defaults.
# ---------------------------------------------------------------------------
#
# NOTE: The primary image path is GHCR via GitHub Actions
# (.github/workflows/build-dev-images.yml), which publishes automatically
# on push to main. This script is the manual ECR path for deployments
# using create_ecr = true in Terraform.
# ---------------------------------------------------------------------------

# Check required commands
for cmd in docker aws git; do
  command -v "$cmd" &>/dev/null || { echo "ERROR: $cmd not found" >&2; exit 1; }
done

# Read from terraform outputs
get_terraform_output() {
  local key=$1
  if [ -d "infra/terraform/app" ] && command -v terraform &>/dev/null; then
    terraform -chdir=infra/terraform/app output -raw "$key" 2>/dev/null || return 1
  else
    return 1
  fi
}

_tf_region=$(get_terraform_output aws_region) && AWS_REGION="${AWS_REGION:-$_tf_region}"
_tf_ecr=$(get_terraform_output inference_image_repo) && ECR_REPO="${ECR_REPO:-$_tf_ecr}"
_tf_tag=$(get_terraform_output image_tag) && IMAGE_TAG="${IMAGE_TAG:-$_tf_tag}"
IMAGE_TAG="${IMAGE_TAG:-dev}"

missing=()
[ -z "$AWS_REGION" ]  && missing+=(AWS_REGION)
[ -z "$AWS_PROFILE" ] && missing+=(AWS_PROFILE)
[ -z "$ECR_REPO" ]    && missing+=(ECR_REPO)
if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: Missing: ${missing[*]}" >&2
  echo "Run 'cd infra/terraform/app && terraform init && terraform apply' or set env vars." >&2
  exit 1
fi

echo "Using ECR URL: $ECR_REPO"

# Derive registry host from repo URL (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com)
ECR_REGISTRY="${ECR_REPO%%/*}"

# Git SHA tag for traceability and rollback
GIT_SHA=$(git rev-parse --short HEAD) || { echo "ERROR: not in a git repository" >&2; exit 1; }
SHA_TAG="${ECR_REPO}:sha-${GIT_SHA}"

# 1. Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# 2. Build
echo "Building Docker image (linux/amd64)..."
docker build --platform linux/amd64 -t bridge-classifier .

# 3. Tag (both :$IMAGE_TAG and :sha-<hash>)
docker tag bridge-classifier:latest "${ECR_REPO}:${IMAGE_TAG}"
docker tag bridge-classifier:latest "$SHA_TAG"

# 4. Push both tags
echo "Pushing to ECR..."
docker push "${ECR_REPO}:${IMAGE_TAG}"
docker push "$SHA_TAG"

echo ""
echo "Done."
echo "  ${IMAGE_TAG} : ${ECR_REPO}:${IMAGE_TAG}"
echo "  sha : $SHA_TAG"
echo ""
echo "To roll back to this image: docker tag $SHA_TAG ${ECR_REPO}:${IMAGE_TAG} && docker push ${ECR_REPO}:${IMAGE_TAG}"
