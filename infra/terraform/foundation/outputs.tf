# --- Networking ---

output "vpc_id" {
  description = "VPC ID (created, or the existing VPC ID passed in)"
  value       = var.create_networking ? aws_vpc.main[0].id : var.existing_vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs for all workloads (created, or the existing subnet IDs passed in)"
  value       = var.create_networking ? aws_subnet.private[*].id : var.existing_private_subnet_ids
}

output "vpce_security_group_id" {
  description = "VPC interface endpoints security group ID (empty string if VPC endpoints are not in use)"
  value       = var.create_networking ? (var.create_vpc_endpoints ? aws_security_group.vpc_endpoints[0].id : "") : var.existing_vpce_security_group_id
}
