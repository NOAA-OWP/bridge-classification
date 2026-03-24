"""Tests for src/s3.py — URI parsing and output key resolution."""

import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError

from src.s3 import parse_s3_uri, resolve_input_key, resolve_output_keys


class TestParseS3Uri:
    def test_simple_key(self):
        bucket, key = parse_s3_uri("s3://my-bucket/path/to/file.laz")
        assert bucket == "my-bucket"
        assert key == "path/to/file.laz"

    def test_nested_key(self):
        bucket, key = parse_s3_uri("s3://my-bucket/a/b/c/d.las")
        assert bucket == "my-bucket"
        assert key == "a/b/c/d.las"

    def test_root_key(self):
        bucket, key = parse_s3_uri("s3://my-bucket/file.txt")
        assert bucket == "my-bucket"
        assert key == "file.txt"

    def test_bucket_only(self):
        bucket, key = parse_s3_uri("s3://my-bucket/")
        assert bucket == "my-bucket"
        assert key == ""


class TestResolveOutputKeys:
    def test_raw_mode_primary_key(self):
        result = resolve_output_keys("s3://bucket/output", "02050206/bridge_123.laz", ".laz", "raw")
        assert result["primary"] == "s3://bucket/output/02050206/bridge_123_predicted.laz"

    def test_masked_mode_primary_key(self):
        result = resolve_output_keys("s3://bucket/output", "02050206/bridge_123.laz", ".laz", "masked")
        assert result["primary"] == "s3://bucket/output/02050206/bridge_123_bridge_masked.laz"

    def test_both_mode_has_primary_and_masked(self):
        result = resolve_output_keys("s3://bucket/output", "02050206/bridge_123.laz", ".laz", "both")
        assert "primary" in result
        assert "masked" in result

    def test_without_extension_in_manifest_line(self):
        """Manifest lines without extension should still produce correct keys."""
        result = resolve_output_keys("output", "02050206/bridge_123", ".laz", "raw")
        assert result["primary"] == "output/02050206/bridge_123_predicted.laz"


class TestResolveInputKey:
    def test_manifest_line_with_laz_extension(self):
        """If manifest line already has .laz, return it directly without probing."""
        s3 = MagicMock()
        key = resolve_input_key(s3, "bucket", "prefix", "02050206/bridge_123.laz")
        assert key == "prefix/02050206/bridge_123.laz"
        s3.head_object.assert_not_called()

    def test_manifest_line_with_las_extension(self):
        """If manifest line already has .las, return it directly without probing."""
        s3 = MagicMock()
        key = resolve_input_key(s3, "bucket", "prefix", "02050206/bridge_123.las")
        assert key == "prefix/02050206/bridge_123.las"
        s3.head_object.assert_not_called()

    def test_probes_laz_first_and_returns_it(self):
        """No extension: probes .laz, finds it, returns .laz key."""
        s3 = MagicMock()
        s3.head_object.return_value = {}

        key = resolve_input_key(s3, "bucket", "prefix", "02050206/bridge_123")
        assert key == "prefix/02050206/bridge_123.laz"
        s3.head_object.assert_called_once_with(Bucket="bucket", Key="prefix/02050206/bridge_123.laz")

    def test_falls_back_to_las_when_laz_missing(self):
        """No extension: .laz returns 404, .las exists."""
        s3 = MagicMock()
        err_404 = ClientError({'Error': {'Code': '404', 'Message': 'Not Found'}}, 'HeadObject')
        s3.head_object.side_effect = [err_404, {}]

        key = resolve_input_key(s3, "bucket", "prefix", "02050206/bridge_123")
        assert key == "prefix/02050206/bridge_123.las"
        assert s3.head_object.call_count == 2

    def test_raises_file_not_found_when_both_missing(self):
        """No extension: both .laz and .las return 404 -> FileNotFoundError."""
        s3 = MagicMock()
        err_404 = ClientError({'Error': {'Code': '404', 'Message': 'Not Found'}}, 'HeadObject')
        s3.head_object.side_effect = [err_404, err_404]

        with pytest.raises(FileNotFoundError):
            resolve_input_key(s3, "bucket", "prefix", "02050206/bridge_123")
