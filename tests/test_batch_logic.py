"""Tests for scripts/batch_entrypoint.py and scripts/submit_batch_job.py — chunking math."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from batch_entrypoint import compute_chunk
from submit_batch_job import MAX_ARRAY_SIZE, compute_array_size, build_container_overrides


class TestComputeChunk:
    def test_even_division(self):
        """6 lines, 3 children: each gets 2."""
        assert compute_chunk(0, 3, 6) == (0, 2)
        assert compute_chunk(1, 3, 6) == (2, 4)
        assert compute_chunk(2, 3, 6) == (4, 6)

    def test_uneven_division_last_child_gets_remainder(self):
        """7 lines, 3 children: chunk_size=3, last child gets 1."""
        assert compute_chunk(0, 3, 7) == (0, 3)
        assert compute_chunk(1, 3, 7) == (3, 6)
        assert compute_chunk(2, 3, 7) == (6, 7)

    def test_more_children_than_lines(self):
        """3 lines, 5 children: later children get empty range."""
        s, e = compute_chunk(4, 5, 3)
        assert s >= e  # empty range for overflow child

    def test_single_child(self):
        """1 child gets all lines."""
        assert compute_chunk(0, 1, 10) == (0, 10)

    @pytest.mark.parametrize("total,array_size", [
        (1, 1), (10, 3), (100, 7), (134, 5), (1000, 17),
    ])
    def test_all_lines_covered_exactly_once(self, total, array_size):
        """Every line index must appear in exactly one child's range."""
        covered = set()
        for child in range(array_size):
            start, end = compute_chunk(child, array_size, total)
            for i in range(start, end):
                assert i not in covered, f"Line {i} covered twice"
                covered.add(i)
        assert covered == set(range(total))


class TestComputeArraySize:
    def test_below_target_returns_one(self):
        assert compute_array_size(50, 60) == 1

    def test_normal_division(self):
        assert compute_array_size(120, 60) == 2

    def test_capped_at_max_array_size(self):
        result = compute_array_size(10_000_000, 1)
        assert result == MAX_ARRAY_SIZE


class TestBuildContainerOverrides:
    def test_array_size_always_present(self):
        result = build_container_overrides(5)
        names = {item['name'] for item in result['environment']}
        assert 'ARRAY_SIZE' in names

    def test_env_overrides_included(self):
        result = build_container_overrides(3, {'INFERENCE_MODE': 'both', 'BRIDGE_TIMEOUT': '200'})
        env = {item['name']: item['value'] for item in result['environment']}
        assert env['INFERENCE_MODE'] == 'both'
        assert env['BRIDGE_TIMEOUT'] == '200'
