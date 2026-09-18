output "bucket_name" {
  description = "S3 bucket for Terraform state - use in foundation/ and app/ backend.hcl"
  value       = aws_s3_bucket.state.id
}

output "bucket_arn" {
  description = "S3 bucket ARN for Terraform state"
  value       = aws_s3_bucket.state.arn
}

output "region" {
  description = "AWS region - use in foundation/ and app/ backend.hcl"
  value       = var.region
}
