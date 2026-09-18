variable "allowed_account_id" {
  description = "AWS account ID to restrict operations to - prevents accidental apply in the wrong account"
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.allowed_account_id))
    error_message = "allowed_account_id must be a 12-digit AWS account ID."
  }
}

variable "project_name" {
  description = "Project name, used as a prefix for resource names"
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

# --- Networking: create fresh (default), or reference an existing VPC ---

variable "create_networking" {
  description = "Create a VPC with public and private subnets. Set false to reference an existing VPC via existing_* variables."
  type        = bool
  default     = true
}

variable "enable_nat_gateway" {
  description = "Create NAT gateway for private subnet internet access (adds ongoing cost)"
  type        = bool
  default     = true
}

variable "create_vpc_endpoints" {
  description = "Create VPC interface endpoints for ECR (api, dkr) and CloudWatch Logs, so private subnets can reach them without the NAT gateway. Adds per-endpoint hourly and data processing cost. Defaults to false since enable_nat_gateway already covers egress."
  type        = bool
  default     = false
}

variable "vpc_cidr" {
  description = "CIDR for the created VPC (used only when create_networking = true)"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "vpc_cidr must be valid CIDR notation, e.g. 10.0.0.0/16."
  }
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDRs, one per AZ - NAT gateway placement only (used only when create_networking = true)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]

  validation {
    condition     = length(var.public_subnet_cidrs) >= 1
    error_message = "public_subnet_cidrs must have at least one CIDR."
  }

  validation {
    condition     = alltrue([for c in var.public_subnet_cidrs : can(cidrhost(c, 0))])
    error_message = "every entry in public_subnet_cidrs must be valid CIDR notation."
  }
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDRs for all workloads, one per AZ (used only when create_networking = true)"
  type        = list(string)
  default     = ["10.0.3.0/24", "10.0.4.0/24"]

  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "private_subnet_cidrs must have at least two CIDRs (one per AZ, min 2 for Batch)."
  }

  validation {
    condition     = alltrue([for c in var.private_subnet_cidrs : can(cidrhost(c, 0))])
    error_message = "every entry in private_subnet_cidrs must be valid CIDR notation."
  }
}

variable "existing_vpc_id" {
  description = "Existing VPC ID (used only when create_networking = false; informational)"
  type        = string
  default     = ""
}

variable "existing_private_subnet_ids" {
  description = "Existing private subnet IDs for all workloads (required, min 2, when create_networking = false)"
  type        = list(string)
  default     = []

  validation {
    condition     = var.create_networking || length(var.existing_private_subnet_ids) >= 2
    error_message = "existing_private_subnet_ids requires at least 2 subnet IDs when create_networking = false."
  }

  validation {
    condition     = alltrue([for s in var.existing_private_subnet_ids : can(regex("^subnet-", s))])
    error_message = "every existing_private_subnet_ids entry must start with 'subnet-'."
  }
}

variable "existing_vpce_security_group_id" {
  description = "Existing VPC interface endpoints security group ID (used only when create_networking = false; optional, only needed if that VPC has interface endpoints Batch must reach)"
  type        = string
  default     = ""

  validation {
    condition     = var.create_networking || var.existing_vpce_security_group_id == "" || can(regex("^sg-", var.existing_vpce_security_group_id))
    error_message = "existing_vpce_security_group_id must be empty or start with 'sg-'."
  }
}
