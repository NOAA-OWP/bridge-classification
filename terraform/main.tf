# -----------------------------------------------------------------------------
# Bridge Classification — AWS Batch Infrastructure
#
# Manages: ECR repo, Batch compute environment (SPOT), job queue, job definition,
#          CloudWatch log group.
# IAM roles are NOT managed here — they are referenced by ARN.
# -----------------------------------------------------------------------------

locals {
  tags = {
    Project = var.project_name
  }
}

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

# -----------------------------------------------------------------------------
# CloudWatch Log Group (365-day retention, auto-deleted after 1 year)
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.project_name}"
  retention_in_days = 365
  tags              = local.tags
}

# -----------------------------------------------------------------------------
# ECR Repository
# -----------------------------------------------------------------------------
resource "aws_ecr_repository" "inference" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.tags
}

# -----------------------------------------------------------------------------
# Batch Compute Environment (SPOT or On-Demand)
# -----------------------------------------------------------------------------
resource "aws_batch_compute_environment" "gpu" {
  compute_environment_name = "${var.project_name}-gpu-${var.use_spot ? "spot" : "ec2"}"
  type                     = "MANAGED"
  state                    = "ENABLED"
  service_role             = var.batch_service_role_arn

  compute_resources {
    type                = var.use_spot ? "SPOT" : "EC2"
    allocation_strategy = var.use_spot ? "SPOT_CAPACITY_OPTIMIZED" : "BEST_FIT"
    min_vcpus           = 0
    max_vcpus           = var.max_vcpus
    desired_vcpus       = 0
    instance_type       = var.instance_types

    subnets              = var.subnets
    security_group_ids   = var.security_group_ids
    instance_role        = var.batch_instance_profile
    spot_iam_fleet_role  = var.use_spot ? var.spot_fleet_role_arn : null
  }

  tags = local.tags

  lifecycle {
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# Batch Job Queue
# -----------------------------------------------------------------------------
resource "aws_batch_job_queue" "inference" {
  name     = "${var.project_name}-inference-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu.arn
  }

  tags = local.tags
}

# -----------------------------------------------------------------------------
# Batch Job Definition
# -----------------------------------------------------------------------------
resource "aws_batch_job_definition" "inference" {
  name           = "${var.project_name}-inference"
  type           = "container"
  propagate_tags = true

  timeout {
    attempt_duration_seconds = var.job_timeout_seconds
  }

  retry_strategy {
    attempts = var.retry_attempts

    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*"
    }
    evaluate_on_exit {
      action    = "EXIT"
      on_reason = "*"
    }
  }

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.inference.repository_url}:latest"
    vcpus      = var.job_vcpus
    memory     = var.job_memory
    jobRoleArn = var.batch_job_role_arn
    command    = ["python", "/app/scripts/batch_entrypoint.py"]

    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]

    linuxParameters = {
      sharedMemorySize = var.shared_memory_size
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "inference"
      }
    }

    environment = [
      { name = "S3_BUCKET", value = var.s3_bucket },
      { name = "S3_INPUT_PREFIX", value = var.s3_input_prefix },
      { name = "S3_MANIFEST_URI", value = var.s3_manifest_uri },
      { name = "S3_MODEL_URI", value = var.s3_model_uri },
      { name = "S3_OUTPUT_PREFIX", value = var.s3_output_prefix },
      { name = "INFERENCE_MODE", value = var.inference_mode },
      { name = "BRIDGE_TIMEOUT", value = tostring(var.bridge_timeout) },
    ]
  })

  tags = local.tags
}
