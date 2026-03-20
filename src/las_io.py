"""LAS/LAZ file I/O via PDAL.

Shared read/write helpers used by inference, preprocessing, and evaluation.
"""

import json

import numpy as np
import pdal


def read_las(filepath):
    """Read a LAS/LAZ file via PDAL.

    Returns:
        arrays: Structured numpy array with all LAS fields (X, Y, Z, Intensity, Classification, etc.)
        metadata: PDAL pipeline metadata dict.
    """
    pipeline_json = {"pipeline": [{"type": "readers.las", "filename": str(filepath)}]}
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    return pipeline.arrays[0], pipeline.metadata


def write_las(output_path, arrays, srs="EPSG:3857"):
    """Write a LAS/LAZ file via PDAL with standard options.

    Preserves all extra dims and forward headers from the input arrays.
    Creates parent directories if they don't exist.

    Args:
        output_path: Path to write the output file.
        arrays: Structured numpy array (e.g. from read_las, with Classification updated).
        srs: Spatial reference system. Default: EPSG:3857.
    """
    import os
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
    writer_stage = {
        "type": "writers.las",
        "filename": str(output_path),
        "extra_dims": "all",
        "a_srs": srs,
        "forward": "all",
    }
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [writer_stage]}), arrays=[arrays])
    pipeline.execute()


def normalize_intensity(intensity):
    """Normalize intensity values to 0-1 range.

    Args:
        intensity: numpy array of intensity values.

    Returns:
        Normalized array (0-1). Returns unchanged if max is 0.
    """
    max_val = np.max(intensity)
    return intensity / max_val if max_val > 0 else intensity
