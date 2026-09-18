# ----- CloudWatch log group (Batch container logs) -----
# Referenced directly (by ARN) from the batch_instance_logs IAM policy in iam.tf.
resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.project_name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.log_kms_key_arn != "" ? var.log_kms_key_arn : null
}
