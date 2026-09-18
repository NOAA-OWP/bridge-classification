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
