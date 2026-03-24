"""Tests for utils/calculate_weights.py — weight computation and JSON loading."""

import json
import pytest

from utils.calculate_weights import _load_one_json, compute_weights


class TestComputeWeights:
    def test_known_distribution(self):
        """W_c = total / (n_classes * count_c)."""
        counts = {0: 100, 1: 100, 2: 100, 3: 100}
        weights = compute_weights(counts, n_classes=4)
        assert len(weights) == 4
        for w in weights:
            assert w == pytest.approx(1.0, abs=1e-6)

    def test_single_class(self):
        """Single class -> weight = total / (n_classes * count) = 1.0 when n_classes=1."""
        counts = {0: 500}
        weights = compute_weights(counts, n_classes=1)
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0, abs=1e-6)


class TestLoadOneJson:
    def test_valid_json_returns_counts(self, tmp_path):
        meta = {"class_distribution": {"0": 500, "1": 300, "2": 150, "3": 50}}
        path = tmp_path / "bridge.json"
        path.write_text(json.dumps(meta))

        counts, err = _load_one_json(path)
        assert err is None
        assert counts == {0: 500, 1: 300, 2: 150, 3: 50}

    def test_missing_class_distribution_returns_empty(self, tmp_path):
        """JSON without class_distribution key -> empty dict, no error path."""
        meta = {"original_file": "bridge.laz"}
        path = tmp_path / "bridge.json"
        path.write_text(json.dumps(meta))

        counts, err = _load_one_json(path)
        assert err is None
        assert counts == {}
