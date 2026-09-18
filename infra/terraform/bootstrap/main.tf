data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  bucket_name = "${var.project_name}-terraform-state-${data.aws_caller_identity.current.account_id}"
}

# ----- State bucket -----
resource "aws_s3_bucket" "state" {
  bucket = local.bucket_name

  lifecycle {
    prevent_destroy = true
  }
}

# ----- Versioning (recover prior state versions) -----
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ----- Encryption at rest -----
resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ----- Public access block -----
resource "aws_s3_bucket_public_access_block" "state" {
  bucket = aws_s3_bucket.state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ----- Bucket policy (account-scoped, TLS-only) -----
resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowStateBucketList"
        Effect    = "Allow"
        Principal = { AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "s3:ListBucket"
        Resource  = aws_s3_bucket.state.arn
      },
      {
        Sid       = "AllowStateFileReadWrite"
        Effect    = "Allow"
        Principal = { AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = ["s3:GetObject", "s3:PutObject"]
        Resource  = "${aws_s3_bucket.state.arn}/*.tfstate"
      },
      {
        Sid       = "AllowLockFileReadWriteDelete"
        Effect    = "Allow"
        Principal = { AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource  = "${aws_s3_bucket.state.arn}/*.tflock"
      },
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      },
      {
        Sid       = "DenyAllOtherAccounts"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
        Condition = { StringNotEquals = { "aws:PrincipalAccount" = data.aws_caller_identity.current.account_id } }
      },
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.state]
}
