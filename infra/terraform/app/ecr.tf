# ----- ECR repository (inference image) -----
resource "aws_ecr_repository" "inference" {
  count = var.create_ecr ? 1 : 0

  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "inference" {
  count = var.create_ecr ? 1 : 0

  repository = aws_ecr_repository.inference[0].name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 30 untagged images"
      selection = {
        tagStatus   = "untagged"
        countType   = "imageCountMoreThan"
        countNumber = 30
      }
      action = { type = "expire" }
    }]
  })
}
