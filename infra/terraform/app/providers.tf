locals {
  optional_tags = merge(
    var.team != "" ? { Team = var.team } : {},
    var.poc != "" ? { POC = var.poc } : {},
  )
}

provider "aws" {
  region              = var.region
  allowed_account_ids = [var.allowed_account_id]

  default_tags {
    tags = merge({
      ManagedBy = "Terraform"
      Project   = var.project_name
      Stack     = "app"
    }, local.optional_tags)
  }
}
