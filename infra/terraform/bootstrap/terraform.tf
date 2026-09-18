terraform {
  required_version = ">= 1.14"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # First-time setup (new AWS account) - the state bucket can't store its own
  # state until it exists, so bootstrap with local state then migrate:
  #   1. cp backend.hcl.example backend.hcl && cp terraform.tfvars.example terraform.tfvars
  #   2. edit both with your account ID, region, and bucket name
  #   3. comment out the `backend "s3" {}` line below
  #   4. terraform init && terraform apply (creates the S3 bucket with local state)
  #   5. uncomment the `backend "s3" {}` line
  #   6. terraform init -backend-config=backend.hcl -migrate-state
  #   7. rm terraform.tfstate terraform.tfstate.backup
  backend "s3" {}
}
