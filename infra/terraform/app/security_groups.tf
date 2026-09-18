# --- Batch compute ---

resource "aws_security_group" "batch" {
  name_prefix = "${var.project_name}-batch-"
  description = "Batch compute instances: all egress for S3, ECR, CloudWatch"
  vpc_id      = var.vpc_id

  tags = { Name = "${var.project_name}-batch-sg" }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_egress_rule" "batch_all" {
  security_group_id = aws_security_group.batch.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

# --- VPC endpoint ingress ---
# The SG itself is foundation's (created only when create_vpc_endpoints = true); this adds
# the ingress rule that lets Batch compute reach it for ECR pulls and CloudWatch Logs.

resource "aws_vpc_security_group_ingress_rule" "vpce_from_batch" {
  count = var.vpce_security_group_id != "" ? 1 : 0

  security_group_id            = var.vpce_security_group_id
  referenced_security_group_id = aws_security_group.batch.id
  from_port                    = 443
  to_port                      = 443
  ip_protocol                  = "tcp"
}
