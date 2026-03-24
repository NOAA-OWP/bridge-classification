"""Tests for src/voxelization.py — core voxelization algorithm."""

import numpy as np

from src.voxelization import voxelize


class TestVoxelize:
    def test_single_point_one_voxel(self):
        xyz = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        intensity = np.array([0.7], dtype=np.float32)
        labels = np.array([2], dtype=np.int64)
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=labels)

        assert result.unique_coords.shape == (1, 3)

    def test_two_voxel_count(self, two_voxel_cloud):
        xyz, intensity, labels = two_voxel_cloud
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=labels)

        assert result.unique_coords.shape[0] == 2

    def test_mean_intensity_aggregation(self, two_voxel_cloud):
        """Voxel A mean = 0.4, Voxel B mean = 0.3."""
        xyz, intensity, labels = two_voxel_cloud
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=labels)

        features_sorted = np.sort(result.voxel_features.ravel())
        np.testing.assert_allclose(features_sorted, [0.3, 0.4], atol=1e-5)

    def test_majority_vote_labels(self, two_voxel_cloud):
        """Voxel A [1,1,2] -> 1; Voxel B [2,2,0] -> 2."""
        xyz, intensity, labels = two_voxel_cloud
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=labels)

        assert result.voxel_labels is not None
        assert set(result.voxel_labels.tolist()) == {1, 2}

    def test_inverse_map_all_valid_indices(self, two_voxel_cloud):
        xyz, intensity, labels = two_voxel_cloud
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=labels)

        n_voxels = result.unique_coords.shape[0]
        assert result.inverse_map.min() >= 0
        assert result.inverse_map.max() < n_voxels

    def test_inverse_map_shape(self, two_voxel_cloud):
        xyz, intensity, labels = two_voxel_cloud
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=labels)

        assert result.inverse_map.shape == (6,)

    def test_labels_none_without_input(self, two_voxel_cloud):
        """Inference mode: labels=None -> voxel_labels is None."""
        xyz, intensity, _ = two_voxel_cloud
        result = voxelize(xyz, voxel_size=1.0, intensity=intensity, labels=None)

        assert result.voxel_labels is None
