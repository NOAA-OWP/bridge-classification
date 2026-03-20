"""
S3 utilities script for Bridge Classification.

Two logical groups:
  - Generic S3 operations: parse_s3_uri, object_exists, download_file,
    upload_file, stream_manifest_lines
  - Bridge path conventions: resolve_input_key, resolve_extension,
    resolve_output_keys, PROBE_EXTENSIONS
"""

import os
from pathlib import PurePosixPath

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from src.constants import AWS_MAX_RETRIES


def create_s3_client(profile=None):
    """Create a boto3 S3 client with standard retry config.

    Args:
        profile: AWS profile name (None uses default credentials).

    Returns:
        boto3 S3 client with adaptive retry (AWS_MAX_RETRIES attempts).
    """
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client('s3', config=BotoConfig(
        retries={'max_attempts': AWS_MAX_RETRIES, 'mode': 'adaptive'}
    ))

# Extensions to try when a manifest line has no extension (most common first).
PROBE_EXTENSIONS = ['.laz', '.las']


# ---------------------------------------------------------------------------
# Generic S3 operations
# ---------------------------------------------------------------------------

def parse_s3_uri(uri):
    """Split 's3://bucket/key' into (bucket, key)."""
    path = uri[5:]  # strip 's3://'
    bucket, _, key = path.partition('/')
    return bucket, key


def object_exists(s3_client, bucket, key):
    """Return True if the S3 object exists, False on 404; re-raises other errors."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def download_file(s3_client, bucket, key, local_path):
    """Download an S3 object to a local path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)


def upload_file(s3_client, local_path, bucket, key):
    """Upload a local file to S3."""
    s3_client.upload_file(local_path, bucket, key)


def stream_manifest_lines(s3_client, manifest_uri):
    """Stream a manifest file from S3, yielding non-empty stripped lines.

    Handles byte decoding transparently.

    Args:
        s3_client: boto3 S3 client.
        manifest_uri: Full S3 URI (s3://bucket/key).

    Yields:
        Non-empty stripped line strings.
    """
    bucket, key = parse_s3_uri(manifest_uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    for raw_line in response['Body'].iter_lines():
        line = raw_line.decode('utf-8').strip() if isinstance(raw_line, bytes) else raw_line.strip()
        if line:
            yield line


# ---------------------------------------------------------------------------
# Bridge path conventions
# ---------------------------------------------------------------------------

def resolve_input_key(s3_client, bucket, input_prefix, manifest_line):
    """Resolve a manifest line to a full S3 input key, probing extensions if needed.

    If the manifest line already ends with .laz or .las, it is used directly.
    Otherwise each extension in PROBE_EXTENSIONS is tried via head_object.

    Args:
        s3_client: boto3 S3 client.
        bucket: S3 bucket name.
        input_prefix: S3 prefix for input files.
        manifest_line: e.g. '02050206/bridge_123' or '02050206/bridge_123.laz'

    Returns:
        Full S3 key string (e.g. 'prefix/02050206/bridge_123.laz').

    Raises:
        FileNotFoundError if no matching object exists in S3.
    """
    p = PurePosixPath(manifest_line)

    if p.suffix in ('.laz', '.las'):
        return f"{input_prefix}/{manifest_line}"

    for ext in PROBE_EXTENSIONS:
        key = f"{input_prefix}/{manifest_line}{ext}"
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return key
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                continue
            raise

    raise FileNotFoundError(
        f"No file found in S3 for manifest line '{manifest_line}' "
        f"(tried extensions: {PROBE_EXTENSIONS})"
    )


def resolve_extension(s3_client, bucket, input_prefix, manifest_line):
    """Determine the actual file extension for a manifest line by probing S3.

    Returns the extension from the manifest line directly if it already has one.
    Otherwise probes S3 via resolve_input_key. Falls back to '.laz' if not found.

    Args:
        s3_client: boto3 S3 client.
        bucket: S3 bucket name.
        input_prefix: S3 prefix for input files.
        manifest_line: Manifest entry.

    Returns:
        Extension string (e.g. '.laz').
    """
    p = PurePosixPath(manifest_line)
    if p.suffix in ('.laz', '.las'):
        return p.suffix
    try:
        key = resolve_input_key(s3_client, bucket, input_prefix, manifest_line)
        return PurePosixPath(key).suffix
    except FileNotFoundError:
        return '.laz'  # default fallback


def resolve_output_keys(output_prefix, manifest_line, ext, mode):
    """Compute the expected S3 output key(s) for a manifest line.

    Args:
        output_prefix: S3 prefix for output files (no trailing slash).
        manifest_line: Manifest entry used to derive HUC ID and stem
                       (e.g. '02050206/bridge_123' or '02050206/bridge_123.laz').
        ext: File extension to use for outputs (e.g. '.laz').
        mode: Inference mode - 'raw', 'masked', or 'both'.

    Returns:
        Dict with keys:
          'primary' - always present (the main output key)
          'masked'  - present only when mode='both'

        Naming convention:
          mode='masked' -> primary: {stem}_bridge_masked{ext}
          mode='raw'    -> primary: {stem}_predicted{ext}
          mode='both'   -> primary: {stem}_predicted{ext}, masked: {stem}_bridge_masked{ext}
    """
    p = PurePosixPath(manifest_line)
    huc_id = str(p.parent)
    stem = p.stem

    if mode == 'masked':
        primary = f"{output_prefix}/{huc_id}/{stem}_bridge_masked{ext}"
    else:
        primary = f"{output_prefix}/{huc_id}/{stem}_predicted{ext}"

    result = {'primary': primary}

    if mode == 'both':
        result['masked'] = f"{output_prefix}/{huc_id}/{stem}_bridge_masked{ext}"

    return result
