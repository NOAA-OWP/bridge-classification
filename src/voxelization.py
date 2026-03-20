"""Shared voxelization utilities for training and inference.

Converts raw point clouds into discrete voxel grids with aggregated features.
Used by both the training data loader and inference pipeline.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VoxelResult:
    """Result of voxelizing a point cloud.

    Attributes:
        unique_coords: (M, 3) int32 unique voxel grid coordinates.
        voxel_features: (M, 1) float32 mean intensity per voxel.
        inverse_map: (N,) int array mapping each input point to its voxel index.
        voxel_labels: (M,) int64 majority-vote label per voxel (training only, None at inference).
    """
    unique_coords: np.ndarray
    voxel_features: np.ndarray
    inverse_map: np.ndarray
    voxel_labels: Optional[np.ndarray] = None


def voxelize(xyz: np.ndarray, voxel_size: float, intensity: np.ndarray,
             labels: Optional[np.ndarray] = None) -> VoxelResult:
    """Voxelize a point cloud with mean-intensity aggregation.

    Quantizes xyz into a discrete voxel grid, deduplicates voxels, and
    computes mean intensity per voxel.  When labels are provided (training),
    also computes majority-vote labels per voxel.

    Args:
        xyz: (N, 3) point coordinates, should be shifted so min ~= 0.
        voxel_size: Voxel edge length in meters (e.g. 0.1).
        intensity: (N,) or (N, 1) intensity values.
        labels: Optional (N,) integer class labels for majority-vote aggregation.

    Returns:
        VoxelResult with unique voxel coordinates, aggregated features,
        point-to-voxel inverse map, and optionally majority-vote labels.
    """
    discrete_coords = np.floor(xyz / voxel_size).astype(np.int32)
    unique_coords, inverse_map = np.unique(discrete_coords, axis=0, return_inverse=True)
    n_voxels = len(unique_coords)

    flat_intensity = intensity.ravel().astype(np.float64)
    counts = np.bincount(inverse_map, minlength=n_voxels).astype(np.float64)
    sum_intensity = np.bincount(inverse_map, weights=flat_intensity, minlength=n_voxels)
    voxel_features = (sum_intensity / counts).reshape(-1, 1).astype(np.float32)

    voxel_labels = None
    if labels is not None:
        n_classes = max(4, int(labels.max()) + 1)
        label_votes = np.zeros((n_voxels, n_classes), dtype=np.int32)
        for c in range(n_classes):
            mask = labels == c
            if mask.any():
                label_votes[:, c] = np.bincount(inverse_map[mask], minlength=n_voxels)
        voxel_labels = label_votes.argmax(axis=1).astype(np.int64)

    return VoxelResult(
        unique_coords=unique_coords,
        voxel_features=voxel_features,
        inverse_map=inverse_map,
        voxel_labels=voxel_labels,
    )
