# ----- Account / general -----

variable "allowed_account_id" {
  description = "AWS account ID to restrict operations to - prevents accidental apply in the wrong account"
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.allowed_account_id))
    error_message = "allowed_account_id must be a 12-digit AWS account ID."
  }
}

variable "project_name" {
  description = "Project name; prefixes resource names and is the ECR repo name"
  type        = string
  default     = "bridge-classifier"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]*[a-z0-9]$", var.project_name))
    error_message = "project_name must be lowercase letters, digits, and hyphens only."
  }
}

variable "region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]$", var.region))
    error_message = "region must look like an AWS region, e.g. us-east-1."
  }
}

variable "team" {
  description = "Team name for cost-allocation and ownership tagging (omitted from tags if empty)"
  type        = string
  default     = ""
}

variable "poc" {
  description = "Point of contact for these resources (omitted from tags if empty)"
  type        = string
  default     = ""
}

# ----- From the foundation layer (paste from its `terraform output`) -----

variable "vpc_id" {
  description = "VPC ID for security groups (foundation output: vpc_id)"
  type        = string

  validation {
    condition     = can(regex("^vpc-", var.vpc_id))
    error_message = "vpc_id must start with 'vpc-'."
  }
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for Batch compute (foundation output: private_subnet_ids)"
  type        = list(string)

  validation {
    condition     = length(var.private_subnet_ids) > 0 && alltrue([for s in var.private_subnet_ids : can(regex("^subnet-", s))])
    error_message = "private_subnet_ids must be a non-empty list of 'subnet-' IDs."
  }
}

variable "vpce_security_group_id" {
  description = "VPC interface endpoints security group ID (foundation output: vpce_security_group_id). Empty when VPC endpoints are not in use."
  type        = string
  default     = ""

  validation {
    condition     = var.vpce_security_group_id == "" || can(regex("^sg-", var.vpce_security_group_id))
    error_message = "vpce_security_group_id must be empty or start with 'sg-'."
  }
}

# ----- IAM: create roles (default), or reference existing ones -----

variable "create_iam" {
  description = "Create IAM roles for Batch. Set false to reference existing roles via existing_* variables."
  type        = bool
  default     = true
}

variable "create_batch_service_linked_role" {
  description = "Create the AWSServiceRoleForBatch service-linked role. Set false if the account already has it."
  type        = bool
  default     = true
}

variable "data_bucket" {
  description = "S3 bucket the inference workload reads (model, input) and writes (predictions). Scopes the Batch job role."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.data_bucket))
    error_message = "data_bucket must be a valid S3 bucket name (3-63 chars, lowercase)."
  }
}

variable "existing_batch_job_role_arn" {
  description = "Existing Batch job role ARN (required when create_iam = false)"
  type        = string
  default     = ""

  validation {
    condition     = var.create_iam || can(regex("^arn:aws[a-zA-Z-]*:iam::[0-9]{12}:role/", var.existing_batch_job_role_arn))
    error_message = "existing_batch_job_role_arn is required (and must be a role ARN) when create_iam = false."
  }
}

variable "existing_batch_instance_profile_arn" {
  description = "Existing Batch instance profile ARN (required when create_iam = false)"
  type        = string
  default     = ""

  validation {
    condition     = var.create_iam || can(regex("^arn:aws[a-zA-Z-]*:iam::[0-9]{12}:instance-profile/", var.existing_batch_instance_profile_arn))
    error_message = "existing_batch_instance_profile_arn is required (and must be an instance-profile ARN) when create_iam = false."
  }
}

variable "existing_spot_fleet_role_arn" {
  description = "Existing Spot Fleet role ARN (required when create_iam = false)"
  type        = string
  default     = ""

  validation {
    condition     = var.create_iam || can(regex("^arn:aws[a-zA-Z-]*:iam::[0-9]{12}:role/", var.existing_spot_fleet_role_arn))
    error_message = "existing_spot_fleet_role_arn is required (and must be a role ARN) when create_iam = false."
  }
}

variable "existing_batch_service_role_arn" {
  description = "Existing Batch service role ARN (required when create_iam = false)"
  type        = string
  default     = ""

  validation {
    condition     = var.create_iam || can(regex("^arn:aws[a-zA-Z-]*:iam::[0-9]{12}:role/", var.existing_batch_service_role_arn))
    error_message = "existing_batch_service_role_arn is required (and must be a role ARN) when create_iam = false."
  }
}

# ----- Container registry -----

variable "create_ecr" {
  description = "Create ECR repo for the inference image. Set false when using an external registry like GHCR."
  type        = bool
  default     = true
}

variable "inference_image_repo" {
  description = "Image repository for inference (required when create_ecr = false, e.g. ghcr.io/ngwpc/bridge-classification/inference)"
  type        = string
  default     = ""

  validation {
    condition     = var.create_ecr || var.inference_image_repo != ""
    error_message = "inference_image_repo is required when create_ecr = false."
  }
}

# ----- Compute environment -----

variable "max_vcpus" {
  description = "Max vCPUs the compute environment can scale to"
  type        = number
  default     = 256

  validation {
    condition     = var.max_vcpus > 0
    error_message = "max_vcpus must be greater than 0."
  }
}

variable "instance_types" {
  description = "Instance types Batch may launch"
  type        = list(string)
  default     = ["g4dn.xlarge"]

  validation {
    condition     = length(var.instance_types) > 0
    error_message = "instance_types must be a non-empty list."
  }
}

variable "use_spot" {
  description = "Use SPOT (true) or on-demand EC2 (false)"
  type        = bool
  default     = true
}

# ----- Job definition -----

variable "job_vcpus" {
  description = "vCPUs requested per job"
  type        = number
  default     = 3

  validation {
    condition     = var.job_vcpus > 0
    error_message = "job_vcpus must be greater than 0."
  }
}

variable "job_memory" {
  description = "Memory (MB) requested per job"
  type        = number
  default     = 15000

  validation {
    condition     = var.job_memory > 0
    error_message = "job_memory must be greater than 0."
  }
}

variable "shared_memory_size" {
  description = "Shared memory (MB) for the container (/dev/shm)"
  type        = number
  default     = 4096

  validation {
    condition     = var.shared_memory_size > 0
    error_message = "shared_memory_size must be greater than 0."
  }
}

variable "job_timeout_seconds" {
  description = "Per-job attempt timeout (seconds)"
  type        = number
  default     = 28800

  validation {
    condition     = var.job_timeout_seconds >= 60
    error_message = "job_timeout_seconds must be at least 60 (AWS Batch minimum attempt duration)."
  }
}

variable "retry_attempts" {
  description = "Job retry attempts (SPOT interruption handling)"
  type        = number
  default     = 3

  validation {
    condition     = var.retry_attempts >= 1 && var.retry_attempts <= 10
    error_message = "retry_attempts must be between 1 and 10 (AWS Batch range)."
  }
}

variable "image_tag" {
  description = "Image tag to run (pin a sha tag to avoid breaking in-flight jobs)"
  type        = string
  default     = "dev"
}

# ----- Inference runtime + data (S3) -----

variable "inference_mode" {
  description = "Inference output mode: masked, raw, or both"
  type        = string
  default     = "masked"

  validation {
    condition     = contains(["masked", "raw", "both"], var.inference_mode)
    error_message = "inference_mode must be one of: masked, raw, both."
  }
}

variable "bridge_timeout" {
  description = "Per-bridge inference timeout (seconds)"
  type        = number
  default     = 150

  validation {
    condition     = var.bridge_timeout > 0
    error_message = "bridge_timeout must be greater than 0."
  }
}

variable "s3_bucket" {
  description = "S3 bucket holding model/input and receiving predictions (matches data_bucket, used for IAM scoping)"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.s3_bucket))
    error_message = "s3_bucket must be a valid S3 bucket name (3-63 chars, lowercase)."
  }
}

variable "s3_input_prefix" {
  description = "S3 prefix for input source LAZ"
  type        = string
}

variable "s3_manifest_uri" {
  description = "S3 URI of the manifest file (s3://...)"
  type        = string

  validation {
    condition     = can(regex("^s3://", var.s3_manifest_uri))
    error_message = "s3_manifest_uri must start with s3://."
  }
}

variable "s3_model_uri" {
  description = "S3 URI of the model checkpoint (s3://...)"
  type        = string

  validation {
    condition     = can(regex("^s3://", var.s3_model_uri))
    error_message = "s3_model_uri must start with s3://."
  }
}

variable "s3_output_prefix" {
  description = "S3 prefix for prediction outputs"
  type        = string
}

# ----- CloudWatch -----

variable "log_retention_days" {
  description = "CloudWatch log group retention in days"
  type        = number
  default     = 365

  validation {
    condition     = contains([0, 1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653], var.log_retention_days)
    error_message = "log_retention_days must be a valid CloudWatch retention value (0 = never expire)."
  }
}

variable "log_kms_key_arn" {
  description = "KMS key ARN for CloudWatch log group encryption (empty for no encryption)"
  type        = string
  default     = ""
}
