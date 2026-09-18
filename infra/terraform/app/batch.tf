# ----- Launch template (IMDSv2, encrypted EBS) -----
resource "aws_launch_template" "batch" {
  name_prefix = "${var.project_name}-batch-"

  metadata_options {
    http_tokens = "required"
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      encrypted = true
    }
  }
}

# ----- Compute environment (SPOT or on-demand) -----
resource "aws_batch_compute_environment" "gpu" {
  name         = "${var.project_name}-gpu-${var.use_spot ? "spot" : "ec2"}"
  type         = "MANAGED"
  state        = "ENABLED"
  service_role = local.batch_service_role_arn

  compute_resources {
    type                = var.use_spot ? "SPOT" : "EC2"
    allocation_strategy = var.use_spot ? "SPOT_CAPACITY_OPTIMIZED" : "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = var.max_vcpus
    desired_vcpus       = 0
    instance_type       = var.instance_types

    subnets             = var.private_subnet_ids
    security_group_ids  = [aws_security_group.batch.id]
    instance_role       = local.batch_instance_profile_arn
    spot_iam_fleet_role = var.use_spot ? local.spot_fleet_role_arn : null

    launch_template {
      launch_template_id = aws_launch_template.batch.id
      version            = "$Latest"
    }

    # Batch launches these instances at runtime, outside Terraform, so provider
    # default_tags don't reach them -replicate them here for cost/ownership tagging.
    tags = merge({
      ManagedBy = "Terraform"
      Project   = var.project_name
      Stack     = "app"
    }, local.optional_tags)
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [compute_resources[0].desired_vcpus]
  }
}

# ----- Job queue -----
resource "aws_batch_job_queue" "inference" {
  name     = "${var.project_name}-inference-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu.arn
  }
}

# ----- Job definition -----
resource "aws_batch_job_definition" "inference" {
  name                  = "${var.project_name}-inference"
  type                  = "container"
  propagate_tags        = true
  platform_capabilities = ["EC2"]

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
    image      = "${local.inference_image_repo}:${var.image_tag}"
    vcpus      = var.job_vcpus
    memory     = var.job_memory
    jobRoleArn = local.batch_job_role_arn
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
        "awslogs-region"        = var.region
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
}
