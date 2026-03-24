"""
Shared pytest configuration and fixtures for bridge classification tests.

Adds project root to sys.path so imports work in all tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add project root so imports work without installing the package.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so initial imports work anywhere.
# Must happen before any test module imports src.las_io, src.inference, etc.
#
# Each submodule (e.g. torch.nn) needs its own sys.modules entry.
# ---------------------------------------------------------------------------
_heavy_mods = [
    'pdal',
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.cuda',
    'spconv', 'spconv.pytorch',
]
for _mod in _heavy_mods:
    sys.modules.setdefault(_mod, MagicMock())


# ---------------------------------------------------------------------------
# Point cloud fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_voxel_cloud():
    """6 points collapsing into exactly 2 voxels at voxel_size=1.0.

    Voxel A at integer coord (0,0,0): points at 0.1, 0.3, 0.5
        intensities [0.2, 0.4, 0.6] -> mean 0.4
        labels [1, 1, 2]            -> majority 1
    Voxel B at integer coord (2,0,0): points at 2.1, 2.3, 2.5
        intensities [0.1, 0.3, 0.5] -> mean 0.3
        labels [2, 2, 0]            -> majority 2
    """
    xyz = np.array([
        [0.1, 0.1, 0.1],
        [0.3, 0.3, 0.3],
        [0.5, 0.5, 0.5],
        [2.1, 0.1, 0.1],
        [2.3, 0.3, 0.3],
        [2.5, 0.5, 0.5],
    ], dtype=np.float32)
    intensity = np.array([0.2, 0.4, 0.6, 0.1, 0.3, 0.5], dtype=np.float32)
    labels = np.array([1, 1, 2, 2, 2, 0], dtype=np.int64)
    return xyz, intensity, labels
