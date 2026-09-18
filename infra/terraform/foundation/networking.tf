# All resources here are gated by var.create_networking:
#   true  -> create a VPC + public/private subnets + IGW + NAT + S3 endpoint
#           (+ ECR/CloudWatch Logs interface endpoints when var.create_vpc_endpoints = true)
#   false -> create nothing; the app layer uses var.existing_vpc_id / existing_private_subnet_ids /
#            existing_vpce_security_group_id
#
# Two-tier networking: public subnets exist ONLY for NAT gateway placement (no workloads).
# Private subnets host all workloads (Batch).

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  az_count = max(length(var.public_subnet_cidrs), length(var.private_subnet_cidrs))
  azs      = slice(data.aws_availability_zones.available.names, 0, local.az_count)
}

# --- VPC ---

resource "aws_vpc" "main" {
  count = var.create_networking ? 1 : 0

  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  count = var.create_networking ? 1 : 0

  vpc_id = aws_vpc.main[0].id

  tags = { Name = "${var.project_name}-igw" }
}

# --- Public subnets (NAT gateway placement only, no workloads) ---

resource "aws_subnet" "public" {
  count = var.create_networking ? length(var.public_subnet_cidrs) : 0

  vpc_id                  = aws_vpc.main[0].id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = false

  tags = { Name = "${var.project_name}-public-${local.azs[count.index]}" }
}

resource "aws_route_table" "public" {
  count = var.create_networking ? 1 : 0

  vpc_id = aws_vpc.main[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count = var.create_networking ? length(var.public_subnet_cidrs) : 0

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public[0].id
}

# --- Private subnets (all workloads) ---

resource "aws_subnet" "private" {
  count = var.create_networking ? length(var.private_subnet_cidrs) : 0

  vpc_id            = aws_vpc.main[0].id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = local.azs[count.index]

  tags = { Name = "${var.project_name}-private-${local.azs[count.index]}" }
}

resource "aws_route_table" "private" {
  count = var.create_networking ? 1 : 0

  vpc_id = aws_vpc.main[0].id

  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count = var.create_networking ? length(var.private_subnet_cidrs) : 0

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[0].id
}

# --- NAT gateway (conditional; sits in public subnet, routes private traffic to internet) ---

resource "aws_eip" "nat" {
  count = var.create_networking && var.enable_nat_gateway ? 1 : 0

  domain = "vpc"

  tags = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  count = var.create_networking && var.enable_nat_gateway ? 1 : 0

  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  depends_on = [aws_internet_gateway.main]

  tags = { Name = "${var.project_name}-nat" }
}

resource "aws_route" "private_nat" {
  count = var.create_networking && var.enable_nat_gateway ? 1 : 0

  route_table_id         = aws_route_table.private[0].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.main[0].id
}

# --- S3 gateway endpoint (free; keeps S3-heavy traffic on the AWS backbone) ---

resource "aws_vpc_endpoint" "s3" {
  count = var.create_networking ? 1 : 0

  vpc_id       = aws_vpc.main[0].id
  service_name = "com.amazonaws.${var.region}.s3"

  route_table_ids = [
    aws_route_table.public[0].id,
    aws_route_table.private[0].id,
  ]

  tags = { Name = "${var.project_name}-s3-endpoint" }
}

# --- VPC interface endpoints (optional; avoids NAT gateway cost for private-subnet egress) ---
# Off by default: enable_nat_gateway already covers egress. Turn on to run without a NAT
# gateway, or alongside it to keep ECR/CloudWatch Logs traffic off the public internet.
# Bridge's Batch jobs only need ECR (image pulls) and CloudWatch Logs (log shipping) -
# no Secrets Manager or Batch API endpoints.

resource "aws_security_group" "vpc_endpoints" {
  count = var.create_networking && var.create_vpc_endpoints ? 1 : 0

  name_prefix = "${var.project_name}-vpce-"
  description = "VPC interface endpoints (ingress rule added by app stack)"
  vpc_id      = aws_vpc.main[0].id

  tags = { Name = "${var.project_name}-vpce-sg" }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_endpoint" "ecr_api" {
  count = var.create_networking && var.create_vpc_endpoints ? 1 : 0

  vpc_id              = aws_vpc.main[0].id
  service_name        = "com.amazonaws.${var.region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${var.project_name}-ecr-api-endpoint" }
}

resource "aws_vpc_endpoint" "ecr_dkr" {
  count = var.create_networking && var.create_vpc_endpoints ? 1 : 0

  vpc_id              = aws_vpc.main[0].id
  service_name        = "com.amazonaws.${var.region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${var.project_name}-ecr-dkr-endpoint" }
}

resource "aws_vpc_endpoint" "logs" {
  count = var.create_networking && var.create_vpc_endpoints ? 1 : 0

  vpc_id              = aws_vpc.main[0].id
  service_name        = "com.amazonaws.${var.region}.logs"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]

  tags = { Name = "${var.project_name}-logs-endpoint" }
}
