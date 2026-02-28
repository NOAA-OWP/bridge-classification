output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.inference.repository_url
}

output "job_definition_name" {
  description = "Batch job definition name (use in submit script)"
  value       = aws_batch_job_definition.inference.name
}

output "job_queue_name" {
  description = "Batch job queue name (use in submit script)"
  value       = aws_batch_job_queue.inference.name
}

output "compute_environment_name" {
  description = "Batch compute environment name"
  value       = aws_batch_compute_environment.gpu.compute_environment_name
}

output "s3_manifest_uri" {
  description = "S3 manifest URI (for submit script auto-counting)"
  value       = var.s3_manifest_uri
}

output "log_group_name" {
  description = "CloudWatch log group for Batch job logs"
  value       = aws_cloudwatch_log_group.batch.name
}

output "aws_account_id" {
  description = "AWS account ID (for scripts that need to construct AWS resource URLs)"
  value       = var.aws_account_id
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "aws_profile" {
  description = "AWS CLI profile"
  value       = var.aws_profile
}
