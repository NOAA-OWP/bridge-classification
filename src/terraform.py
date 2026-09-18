import os
import subprocess
from typing import Dict

DEFAULT_TERRAFORM_DIR = 'infra/terraform/app'

TERRAFORM_KEYS = [
    'aws_region', 'job_definition_name', 'job_queue_name',
    's3_manifest_uri', 's3_bucket', 's3_output_prefix',
]


def get_terraform_outputs(
    terraform_dir: str = DEFAULT_TERRAFORM_DIR,
    keys: list | None = None,
) -> Dict[str, str]:
    """Read config values from terraform outputs."""
    outputs = {}
    if not os.path.isdir(terraform_dir):
        return outputs

    for key in (keys or TERRAFORM_KEYS):
        try:
            result = subprocess.run(
                ['terraform', 'output', '-raw', key],
                cwd=terraform_dir, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                outputs[key] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return outputs
