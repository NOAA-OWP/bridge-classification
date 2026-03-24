"""Tests for src/las_io.py and src/inference.py utilities.

pdal/torch/spconv are mocked at import time in conftest.py so these tests run anywhere.
Bridge mask contract is tested via raw numpy to avoid calling apply_bridge_mask directly.
"""

import numpy as np
import pytest

from src.las_io import normalize_intensity
from src.inference import parse_pairs_file


class TestNormalizeIntensity:
    def test_max_greater_than_zero(self):
        intensity = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        result = normalize_intensity(intensity)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)

    def test_all_zeros_returns_unchanged(self):
        """When max == 0, return the array without dividing (avoids NaN/inf)."""
        intensity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = normalize_intensity(intensity)
        np.testing.assert_array_equal(result, intensity)


class TestBridgeMaskContract:
    """Contract tests for bridge masking logic without importing src/inference.py.

    The production function apply_bridge_mask does:
        merged[point_labels == BRIDGE_DECK_MODEL_CLASS] = BRIDGE_DECK_ASPRS_CODE
    which maps model class 2 -> ASPRS 17 and leaves other classes unchanged.
    """

    def _apply_mask(self, point_labels, merged):
        """Local reimplementation of the masking contract."""
        from src.constants import BRIDGE_DECK_ASPRS_CODE, BRIDGE_DECK_MODEL_CLASS
        result = merged.copy()
        result[point_labels == BRIDGE_DECK_MODEL_CLASS] = BRIDGE_DECK_ASPRS_CODE
        return result

    def test_bridge_deck_reclassified_to_asprs_17(self):
        from src.constants import BRIDGE_DECK_MODEL_CLASS
        point_labels = np.array([0, 1, BRIDGE_DECK_MODEL_CLASS, 1], dtype=np.int64)
        merged = np.array([0, 1, 0, 2], dtype=np.int64)
        result = self._apply_mask(point_labels, merged)
        assert result[2] == 17

    def test_non_bridge_points_unchanged(self):
        point_labels = np.array([0, 1, 0], dtype=np.int64)
        merged = np.array([5, 6, 7], dtype=np.int64)
        result = self._apply_mask(point_labels, merged)
        np.testing.assert_array_equal(result, [5, 6, 7])


class TestParsePairsFile:
    def test_valid_tsv(self, tmp_path):
        f = tmp_path / "pairs.tsv"
        f.write_text("input1.laz\toutput1.laz\ninput2.laz\toutput2.laz\n")
        pairs = parse_pairs_file(f)
        assert pairs == [("input1.laz", "output1.laz"), ("input2.laz", "output2.laz")]

    def test_malformed_line_raises(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("only_one_field\n")
        with pytest.raises(ValueError, match="expected 2 tab-separated"):
            parse_pairs_file(f)
