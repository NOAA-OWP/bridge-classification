locals {
  account_id           = data.aws_caller_identity.current.account_id
  partition            = data.aws_partition.current.partition
  data_bucket_arn      = "arn:${local.partition}:s3:::${var.data_bucket}"
  inference_image_repo = var.create_ecr ? aws_ecr_repository.inference[0].repository_url : var.inference_image_repo
}

# ----- Batch job role -----
# Application permissions for the inference container (jobRoleArn): reads model/input,
# writes predictions. Scoped to the single data bucket.

resource "aws_iam_role" "batch_job" {
  count = var.create_iam ? 1 : 0
  name  = "${var.project_name}-batch-job"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "batch_job" {
  count = var.create_iam ? 1 : 0
  name  = "${var.project_name}-batch-job"
  role  = aws_iam_role.batch_job[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "S3List"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = local.data_bucket_arn
      },
      {
        Sid      = "S3ReadWrite"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${local.data_bucket_arn}/*"
      },
    ]
  })
}

# ----- Batch container instance role + instance profile -----
# The ECS agent on each EC2 instance: registers, pulls from ECR, ships awslogs.

resource "aws_iam_role" "batch_instance" {
  count = var.create_iam ? 1 : 0
  name  = "${var.project_name}-batch-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_instance_profile" "batch_instance" {
  count = var.create_iam ? 1 : 0
  name  = "${var.project_name}-batch-instance"
  role  = aws_iam_role.batch_instance[0].name
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  count      = var.create_iam ? 1 : 0
  role       = aws_iam_role.batch_instance[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# awslogs driver runs under the instance role on EC2 launch type - grant scoped logs.
resource "aws_iam_role_policy" "batch_instance_logs" {
  count = var.create_iam ? 1 : 0
  name  = "${var.project_name}-batch-instance-logs"
  role  = aws_iam_role.batch_instance[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "CloudWatchLogs"
      Effect   = "Allow"
      Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
      Resource = aws_cloudwatch_log_group.batch.arn
    }]
  })
}

# ----- Spot Fleet role -----

resource "aws_iam_role" "spot_fleet" {
  count = var.create_iam ? 1 : 0
  name  = "${var.project_name}-spot-fleet"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "spotfleet.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  count      = var.create_iam ? 1 : 0
  role       = aws_iam_role.spot_fleet[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

# ----- Batch service-linked role -----
# Account-global; AWS may have auto-created it already. Toggle off to skip and reference it.

resource "aws_iam_service_linked_role" "batch" {
  count            = var.create_iam && var.create_batch_service_linked_role ? 1 : 0
  aws_service_name = "batch.amazonaws.com"
}

# ----- Resolved values for consumers (batch.tf) -----
# Single source of truth: the resource created above when create_iam = true, the matching
# existing_* input otherwise. Consumers reference these locals, never the resources or the
# existing_* variables directly.

locals {
  batch_job_role_arn         = var.create_iam ? aws_iam_role.batch_job[0].arn : var.existing_batch_job_role_arn
  batch_instance_profile_arn = var.create_iam ? aws_iam_instance_profile.batch_instance[0].arn : var.existing_batch_instance_profile_arn
  spot_fleet_role_arn        = var.create_iam ? aws_iam_role.spot_fleet[0].arn : var.existing_spot_fleet_role_arn

  # 3-way toggle: create_iam = false takes the existing_* input; create_iam = true with
  # create_batch_service_linked_role = false falls back to the well-known ARN of the
  # AWS-managed role (service-linked roles cannot be created twice in one account); both
  # true resolves to the resource created above.
  batch_service_role_arn = var.create_iam ? (
    var.create_batch_service_linked_role
    ? aws_iam_service_linked_role.batch[0].arn
    : "arn:${local.partition}:iam::${local.account_id}:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch"
  ) : var.existing_batch_service_role_arn
}
