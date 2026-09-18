# Terraform Infrastructure

Infrastructure as code for the bridge classification batch inference pipeline.
Three independent stacks, applied in order: bootstrap, then foundation, then app.
Bootstrap and foundation are **optional** when using existing infrastructure - only the app stack is required.
When using an existing VPC and IAM roles, skip bootstrap and foundation entirely.

Each stack reads two local config files that are gitignored and created from committed `.example` templates.

- `terraform.tfvars` - input variables, copied from `terraform.tfvars.example`
- `backend.hcl` - remote state config, copied from `backend.hcl.example`

## Prerequisites

- Terraform >= 1.14, AWS provider ~> 6.0 (pinned per stack).
- AWS CLI with a profile for the target account.
- Permissions to create S3, IAM, VPC, Batch, ECR, and CloudWatch resources.

## Layout

```
infra/terraform/
├── README.md
├── bootstrap/                     state bucket stack, apply once
│   ├── main.tf                    S3 state bucket - versioning, AES256 encryption, public access block, account-restricted TLS-only policy
│   ├── outputs.tf                 bucket_name, bucket_arn, region
│   ├── providers.tf               AWS provider config, default tags (incl. optional team/poc)
│   ├── terraform.tf               version constraints, S3 backend block, first-time setup notes
│   ├── variables.tf               allowed_account_id, project_name, region, team, poc
│   ├── backend.hcl.example        remote state backend template
│   ├── terraform.tfvars.example   input variable template
│   └── .terraform.lock.hcl        provider version lock (generated, committed)
├── foundation/                    persistent infra stack - networking only
│   ├── networking.tf              VPC, public/private subnets, IGW, NAT gateway, S3 endpoint, optional ECR/CloudWatch Logs interface endpoints, VPCE SG
│   ├── outputs.tf                 vpc_id, private_subnet_ids, vpce_security_group_id
│   ├── providers.tf               AWS provider config, default tags (incl. optional team/poc)
│   ├── terraform.tf               version constraints, S3 backend block
│   ├── variables.tf               create_networking, CIDRs, enable_nat_gateway, create_vpc_endpoints, existing_* fallbacks
│   ├── backend.hcl.example        remote state backend template
│   ├── terraform.tfvars.example   input variable template
│   └── .terraform.lock.hcl        provider version lock (generated, committed)
└── app/                           application stack, ok to destroy and recreate
    ├── batch.tf                   launch template (IMDSv2, encrypted EBS), Batch compute env (GPU SPOT), job queue, job definition
    ├── cloudwatch.tf              log group (configurable retention + optional KMS encryption)
    ├── data.tf                    aws_partition, aws_caller_identity data sources
    ├── ecr.tf                     ECR repo + lifecycle policy (optional, gated by create_ecr)
    ├── iam.tf                     create_iam toggle, Batch IAM roles + instance profile, existing_* fallbacks
    ├── outputs.tf                 ECR/image repo, Batch, CloudWatch, S3 outputs
    ├── providers.tf               AWS provider config, default tags (incl. optional team/poc)
    ├── security_groups.tf         Batch SG + optional VPC endpoint ingress rule
    ├── terraform.tf               version constraints, S3 backend block
    ├── variables.tf               shared + IAM + container registry + compute + inference variables
    ├── backend.hcl.example        remote state backend template
    ├── terraform.tfvars.example   input variable template
    └── .terraform.lock.hcl        provider version lock (generated, committed)
```

## Stacks

### Bootstrap

Creates the S3 bucket that holds Terraform remote state for the other two stacks.
Apply once per AWS account.

Bootstrap is **optional**.
If you have an existing S3 bucket for state storage, skip this stack and set `bucket` in foundation and app `backend.hcl` to that bucket name.

### Foundation

Persistent networking infrastructure that survives an app stack destroy/recreate.
**Optional** when using an existing VPC - skip this stack entirely and pass values directly to the app stack.

Creates:
- VPC with public subnets (NAT gateway placement only, no workloads) and private subnets (all workloads)
- NAT gateway for private subnet internet access
- S3 gateway endpoint (free, attached to both route tables)
- Optional ECR API, ECR DKR, and CloudWatch Logs interface endpoints (for no-NAT deployments)
- VPC endpoint security group (app stack adds ingress rules when `vpce_security_group_id` is provided)

### App

Application infrastructure, safe to destroy and recreate.

Creates:
- AWS Batch GPU SPOT compute environment, job queue, job definition (with launch template for IMDSv2 + encrypted EBS)
- ECR repo with scan-on-push and lifecycle policy (optional, gated by `create_ecr`)
- CloudWatch log group (configurable retention, optional KMS encryption)
- Batch IAM roles + instance profile (optional, gated by `create_iam`)
- Batch security group + optional VPC endpoint ingress rule

IAM is toggleable via `create_iam`.
ECR is toggleable via `create_ecr` - set to `false` when using an external registry like GHCR.
When `create_ecr = false`, the image repository is provided via `inference_image_repo`.
Security groups are always created by this stack.

## Tags

All three stacks apply default tags to every resource:

| Tag | Source | Required |
|---|---|---|
| `ManagedBy` | hardcoded `"Terraform"` | Always |
| `Project` | `var.project_name` | Always |
| `Stack` | hardcoded per stack | Always |
| `Team` | `var.team` | Optional (omitted if empty) |
| `POC` | `var.poc` | Optional (omitted if empty) |

## Toggles

| Toggle | Stack | Default | Controls |
|---|---|---|---|
| `create_networking` | foundation | `true` | VPC, subnets, IGW, NAT gateway, VPC endpoints, VPCE SG |
| `enable_nat_gateway` | foundation | `true` | NAT gateway + private subnet default route |
| `create_vpc_endpoints` | foundation | `false` | ECR + CloudWatch Logs interface endpoints + VPCE SG |
| `create_iam` | app | `true` | Batch IAM roles + instance profile |
| `create_ecr` | app | `true` | ECR repo + lifecycle policy |
| `create_batch_service_linked_role` | app | `true` | Account-global AWSServiceRoleForBatch |

**NAT-off warning.**
Disabling `enable_nat_gateway` without enabling `create_vpc_endpoints` leaves private subnets with no route to ECR or CloudWatch Logs.
Batch jobs will fail at image pull.

## Fresh deployment

Set the AWS profile and confirm the account ID before touching any stack.
Bootstrap and foundation are optional. If using existing networking, skip to step 3 (App).

```bash
export AWS_PROFILE=<your-profile>
aws sts get-caller-identity --query Account --output text
```

### 1. Bootstrap

```bash
cd infra/terraform/bootstrap
cp terraform.tfvars.example terraform.tfvars
cp backend.hcl.example backend.hcl
# edit both: account ID, state bucket name

# Step 1: comment out `backend "s3" {}` in terraform.tf
terraform init
terraform apply

# Step 2: uncomment `backend "s3" {}`
terraform init -backend-config=backend.hcl -migrate-state

# Step 3: delete local state (now in S3)
rm terraform.tfstate terraform.tfstate.backup
```

### 2. Foundation (optional - skip if you have an existing VPC)

```bash
cd infra/terraform/foundation
cp terraform.tfvars.example terraform.tfvars
cp backend.hcl.example backend.hcl
# edit both: account ID, state bucket name

terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

### 3. App

```bash
cd infra/terraform/app
cp terraform.tfvars.example terraform.tfvars
cp backend.hcl.example backend.hcl

# If foundation was deployed, pull its outputs into terraform.tfvars:
#   terraform -chdir=../foundation output
# If using existing infra, fill in vpc_id, private_subnet_ids from your environment

terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

If the account already has the Batch service-linked role, set `create_batch_service_linked_role = false`.

## Foundation outputs

| Output | Provides |
|---|---|
| `vpc_id` | VPC ID (created, or existing VPC ID passed through) |
| `private_subnet_ids` | Private subnet IDs for all workloads |
| `vpce_security_group_id` | VPC endpoint SG ID (empty if not created) |

## App outputs

| Output | Provides |
|---|---|
| `inference_image_repo` | Image repository (ECR URL or external registry) |
| `job_queue_name` | Batch job queue name |
| `job_definition_name` | Batch job definition name |
| `compute_environment_name` | Batch compute environment name |
| `log_group_name` | CloudWatch log group name |
| `s3_manifest_uri` | S3 manifest URI (passthrough) |
| `aws_region` | AWS region (passthrough) |
| `s3_bucket` | S3 data bucket (passthrough) |
| `s3_output_prefix` | S3 output prefix (passthrough) |
