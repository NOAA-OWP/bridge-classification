# -----------------------------------------------------------------------------
# AWS / General
# -----------------------------------------------------------------------------
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
  default     = "test-se"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "bridge-classifier"
}

variable "aws_account_id" {
  description = "AWS account ID"
  type        = string
}

# -----------------------------------------------------------------------------
# IAM (existing roles — not managed by this Terraform config)
# -----------------------------------------------------------------------------
variable "batch_job_role_arn" {
  description = "IAM role ARN for Batch job containers (S3 access, etc.)"
  type        = string
}

variable "batch_instance_profile" {
  description = "EC2 instance profile name for Batch compute instances"
  type        = string
}

variable "spot_fleet_role_arn" {
  description = "IAM role ARN for EC2 Spot Fleet requests"
  type        = string
}

variable "batch_service_role_arn" {
  description = "Service-linked role ARN for AWS Batch"
  type        = string
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------
variable "subnets" {
  description = "Subnet IDs for Batch compute instances"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for Batch compute instances"
  type        = list(string)
}

# -----------------------------------------------------------------------------
# Batch Compute
# -----------------------------------------------------------------------------
variable "max_vcpus" {
  description = "Maximum vCPUs for the Batch compute environment"
  type        = number
  default     = 256
}

variable "instance_types" {
  description = "EC2 instance types for Batch compute"
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "use_spot" {
  description = "Use Spot instances (true) or On-Demand (false)"
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# Job Definition
# -----------------------------------------------------------------------------
variable "job_vcpus" {
  description = "vCPUs per job container"
  type        = number
  default     = 3
}

variable "job_memory" {
  description = "Memory (MB) per job container"
  type        = number
  default     = 15000
}

variable "shared_memory_size" {
  description = "Shared memory size (MB) for PyTorch/spconv"
  type        = number
  default     = 4096
}

variable "job_timeout_seconds" {
  description = "Max wall-clock seconds per array child before Batch kills it (prevents runaway GPU costs)"
  type        = number
  default     = 28800 # 8 hours; 150 bridges at ~150s each = ~6.25h, plus buffer for SPOT retries
}

# -----------------------------------------------------------------------------
# S3 / Inference Config (baked into job definition environment; required in tfvars)
# -----------------------------------------------------------------------------
variable "s3_bucket" {
  description = "S3 bucket for input/output data"
  type        = string
}

variable "s3_input_prefix" {
  description = "S3 prefix for source LAZ files"
  type        = string
}

variable "s3_manifest_uri" {
  description = "S3 URI of the manifest file (e.g. s3://bucket/path/manifest.txt)"
  type        = string
}

variable "s3_model_uri" {
  description = "S3 URI of the model checkpoint (e.g. s3://bucket/path/model.ckpt)"
  type        = string
}

variable "s3_output_prefix" {
  description = "S3 prefix for prediction outputs (no trailing slash)"
  type        = string
}

# -----------------------------------------------------------------------------
# Container Image
# -----------------------------------------------------------------------------
variable "image_tag" {
  description = "Docker image tag for the inference container (pin to avoid breaking in-flight jobs)"
  type        = string
  default     = "latest"
}

# -----------------------------------------------------------------------------
# Inference Runtime
# -----------------------------------------------------------------------------
variable "inference_mode" {
  description = "Inference mode: masked (default), raw, or both"
  type        = string
  default     = "masked"
}

variable "bridge_timeout" {
  description = "Per-bridge timeout in seconds before skipping"
  type        = number
  default     = 150
}

variable "retry_attempts" {
  description = "Number of retry attempts for SPOT interruptions"
  type        = number
  default     = 3
}
