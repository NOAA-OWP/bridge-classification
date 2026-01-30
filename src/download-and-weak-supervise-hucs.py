"""
HUC-based Bridge Processing Pipeline

A comprehensive pipeline for processing bridge lidar data organized by Hydrologic
Unit Code (HUC) regions. This script finds intersecting lidar sources, applies
ground filtering and weak supervision rules to generate labeled training data for
machine learning models.

Input Requirements
------------------
Directory Structure:
    hucs_dir/
        {huc_id}/
            osm_bridges_lidar_subset__{huc_id}.gpkg

Usage Examples
-------------
Basic Usage:
    # Process all bridges in all HUCs with default settings
    python src/download-and-weak-supervise-hucs.py

Filtering by HUC:
    # Process bridges in specific HUC regions
    python src/download-and-weak-supervise-hucs.py --hucs 01010001 01010002

Filtering by OSM ID:
    # Process specific bridges by their OpenStreetMap IDs
    python src/download-and-weak-supervise-hucs.py --osm-ids 123456 789012

Custom Configuration:
    # Use custom buffer size and worker count
    python src/download-and-weak-supervise-hucs.py --buffer 15.0 --workers 8

Resume Processing:
    # Skip already processed files (useful for resuming interrupted runs)
    python src/download-and-weak-supervise-hucs.py --skip-existing

Custom Directories:
    # Specify custom input/output directories
    python src/download-and-weak-supervise-hucs.py \\
        --hucs-dir ./data/osm/hucs \\
        --source-dir ./data/ml-data/source \\
        --silver-dir ./data/ml-data/silver_training \\
        --lidar-resources ./data/usgs_entwine/lidar_resources.geojson \\
        --log-dir ./logs

Combined Options:
    # Process specific HUCs with custom settings
    python src/download-and-weak-supervise-hucs.py \\
        --hucs 01010001 01010002 \\
        --buffer 12.0 \\
        --workers 16 \\
        --skip-existing \\
        --no-progress
"""

import geopandas as gpd
import pdal
import json
import os
import numpy as np
import argparse
import multiprocessing
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MatplotlibPath

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Progress bars disabled.")


# Type aliases for commonly used complex types
TaskTuple = Tuple[str, str, str, str, str, float, str, str, Dict[str, Any]]  # Task tuple for multiprocessing


@dataclass
class BridgeProcessingConfig:
    """
    Centralized configuration for bridge processing pipeline.
    Contains all magic numbers, thresholds, and parameters.
    """

    # PDAL Configuration
    # EPT Reader parameters
    pdal_ept_requests: int = 3
    pdal_ept_resolution: float = 0.1

    # SMRF Filter parameters
    pdal_smrf_scalar: float = 1.25
    pdal_smrf_slope: float = 0.05
    pdal_smrf_threshold: float = 0.5
    pdal_smrf_window: float = 10.0
    pdal_smrf_ignore: str = "Classification[7:7]"

    # Writer parameters
    pdal_writer_srs: str = "EPSG:3857"

    # RANSAC Configuration
    ransac_min_samples: int = 10
    ransac_residual_threshold: float = 0.20
    ransac_random_state: int = 27

    # Linearity Check Configuration
    linearity_num_bins: int = 10
    linearity_deviation_threshold: float = 0.8  # Initial check threshold
    linearity_min_z_points: int = 50
    linearity_min_bridge_length: float = 5.0  # meters
    linearity_min_points_per_bin: int = 5
    linearity_min_skeleton_points: int = 3
    linearity_final_deviation_threshold: float = 0.35  # Final check threshold

    # Processing Thresholds
    min_points_for_ransac: int = 20
    min_ransac_inliers: int = 20
    structure_check_min_z: float = -2.0
    structure_check_max_z: float = 2.0
    max_rmse: float = 0.30

    # Classification Rules
    ignore_classes: List[int] = None  # Will default to [7, 9, 18]
    bridge_deck_class: int = 17
    high_noise_class: int = 18
    deck_z_max: float = 0.20
    deck_z_min: float = -0.70
    noise_z_min: float = 0.20
    noise_z_max: float = 15.0

    # PCA Configuration
    pca_n_components: int = 2
    pca_random_state: int = 27

    # Spatial Configuration
    default_buffer_meters: float = 10.0
    epsg_code: int = 3857

    def __post_init__(self):
        """Set default values for mutable types."""
        if self.ignore_classes is None:
            self.ignore_classes = [7, 9, 18]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BridgeProcessingConfig':
        """Create config from dictionary (for multiprocessing serialization)."""
        # Create a copy to avoid mutating the original
        d = d.copy()
        if 'ignore_classes' in d and isinstance(d['ignore_classes'], list):
            d['ignore_classes'] = list(d['ignore_classes'])
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for multiprocessing serialization)."""
        return asdict(self)


# Global logger (will be initialized in main)
logger = None


class LidarSourceFinder:
    """
    Class for finding intersecting lidar sources.
    Loads lidar_resources.geojson and builds spatial index for fast queries.
    """

    def __init__(self, lidar_resources_path: str) -> None:
        """Initialize the finder by loading lidar resources and building spatial index."""
        print(f"[Worker {os.getpid()}] Loading lidar resources from {lidar_resources_path}...")
        self.lidar_gdf = gpd.read_file(str(lidar_resources_path))
        self.lidar_gdf = self.lidar_gdf.to_crs(epsg=3857)

        # Build spatial index for fast queries
        if not self.lidar_gdf.empty:
            self.sindex = self.lidar_gdf.sindex
        else:
            self.sindex = None
        print(f"[Worker {os.getpid()}] Loaded {len(self.lidar_gdf)} lidar sources")

    def find_intersecting_sources(self, bridge_geometry: Any, buffer_meters: float = 10) -> List[Dict[str, str]]:  # Dictionary with 'url' and 'name' keys
        """
        Find all lidar sources that intersect with the buffered bridge geometry.

        Args:
            bridge_geometry: Shapely geometry of the bridge
            buffer_meters: Buffer size in meters

        Returns:
            List of dictionaries with 'url' and 'name' keys for intersecting sources
        """
        if self.lidar_gdf.empty:
            return []

        # Buffer the bridge geometry
        # Bridge geometry should already be in EPSG:3857 (converted in load_bridges)
        buffered = bridge_geometry.buffer(buffer_meters)

        # Ensure buffered geometry is in same CRS as lidar_gdf (3857)
        # If bridge_geometry came from a GeoDataFrame, it should already be in 3857
        # But buffered is a shapely geometry without CRS info, so we assume it's correct

        # Use spatial index for bounding box query first
        if self.sindex is not None:
            possible_matches = list(self.sindex.intersection(buffered.bounds))
            candidates = self.lidar_gdf.iloc[possible_matches]
        else:
            candidates = self.lidar_gdf

        # Perform detailed intersection check
        intersecting = candidates[candidates.intersects(buffered)]

        # Extract url and name
        # GeoPandas flattens GeoJSON properties into columns
        results = []
        for idx, row in intersecting.iterrows():
            # Try direct column access first (geopandas flattens properties)
            url = row.get('url', '') if 'url' in row else ''
            name = row.get('name', '') if 'name' in row else ''

            # If not found, try properties dict (fallback)
            if not url and 'properties' in row and isinstance(row['properties'], dict):
                url = row['properties'].get('url', '')
                name = row['properties'].get('name', '')

            if url:  # Only add if URL exists
                results.append({
                    'url': url,
                    'name': name or f"source_{idx}"  # Use index as fallback name
                })

        return results


class WeakSupervisionPipeline:
    """
    Applies weak supervision rules to bridge lidar data.
    Stateless class that can be used in parallel workers.
    """

    @staticmethod
    def check_bridge_linearity(xy_points: np.ndarray, z_points: np.ndarray, config: BridgeProcessingConfig, deviation_threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Slices the bridge into bins, finds the median Z for each bin (the 'skeleton'),
        and checks if that skeleton deviates from a straight line.

        Args:
            xy_points: XY coordinates of points
            z_points: Z coordinates of points
            config: BridgeProcessingConfig instance
            deviation_threshold: Optional threshold override (defaults to config.linearity_deviation_threshold)

        Returns:
            Tuple of (is_curved: bool, max_deviation: float)
        """
        if deviation_threshold is None:
            deviation_threshold = config.linearity_deviation_threshold

        if len(z_points) < config.linearity_min_z_points:
            return False, 0.0

        # 1. Rotate bridge to align with X-axis using PCA
        pca = PCA(n_components=config.pca_n_components, random_state=config.pca_random_state)
        xy_rotated = pca.fit_transform(xy_points)
        x_axis = xy_rotated[:, 0]

        min_x, max_x = np.min(x_axis), np.max(x_axis)

        # If bridge is too short, assume it's flat/keep it
        if (max_x - min_x) < config.linearity_min_bridge_length:
            return False, 0.0

        # 2. Slice and skeletonize
        bin_edges = np.linspace(min_x, max_x, config.linearity_num_bins + 1)
        skeleton_x = []
        skeleton_z = []

        for i in range(config.linearity_num_bins):
            mask = (x_axis >= bin_edges[i]) & (x_axis < bin_edges[i+1])
            if np.sum(mask) < config.linearity_min_points_per_bin:
                continue  # Skip empty bins

            z_slice = z_points[mask]
            skeleton_z.append(np.median(z_slice))
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2.0
            skeleton_x.append(bin_center)

        if len(skeleton_x) < config.linearity_min_skeleton_points:
            return False, 0.0

        # 3. Linear Fit to the Skeleton
        X_skel = np.array(skeleton_x).reshape(-1, 1)
        z_skel = np.array(skeleton_z)

        model = LinearRegression()
        model.fit(X_skel, z_skel)
        z_predicted = model.predict(X_skel)

        # 4. Measure Deviation
        deviations = np.abs(z_skel - z_predicted)
        max_deviation = np.max(deviations)

        is_curved = max_deviation > deviation_threshold
        return is_curved, max_deviation

    @staticmethod
    def process_bridge(ept_url: str, bridge_geometry: Any, config: BridgeProcessingConfig, buffer_meters: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Process a single bridge with weak supervision rules.

        Args:
            ept_url: EPT URL for the lidar source
            bridge_geometry: Shapely geometry of the bridge
            config: BridgeProcessingConfig instance with all parameters
            buffer_meters: Buffer size in meters (defaults to config.default_buffer_meters)

        Returns:
            Dictionary with keys: 'success' (bool), 'arrays' (numpy structured array),
            'original_arrays' (numpy structured array), 'rmse' (float), 'deviation' (float),
            'error' (str). Returns None if processing failed.
        """
        if buffer_meters is None:
            buffer_meters = config.default_buffer_meters

        buffered_geom = bridge_geometry.buffer(buffer_meters)
        pdal_polygon = buffered_geom.wkt

        # Stage 1: Read only (Get Raw Data)
        read_pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.ept",
                    "filename": ept_url,
                    "polygon": pdal_polygon,
                    "requests": config.pdal_ept_requests,
                    "resolution": config.pdal_ept_resolution
                }
            ]
        }

        try:
            pipeline = pdal.Pipeline(json.dumps(read_pipeline_json))
            count = pipeline.execute()

            if count == 0:
                return {
                    'success': False,
                    'error': 'No points found in lidar data for this bridge geometry'
                }

            original_arrays = pipeline.arrays[0]

            # Stage 2: Process with SMRF
            # Construct PDAL Pipeline (Read + SMRF)
            smrf_pipeline_json = {
                "pipeline": [
                    {
                        "type": "readers.ept",
                        "filename": ept_url,
                        "polygon": pdal_polygon,
                        "requests": config.pdal_ept_requests,
                        "resolution": config.pdal_ept_resolution
                    },
                    {
                        "type": "filters.smrf",
                        "ignore": config.pdal_smrf_ignore,
                        "scalar": config.pdal_smrf_scalar,
                        "slope": config.pdal_smrf_slope,
                        "threshold": config.pdal_smrf_threshold,
                        "window": config.pdal_smrf_window
                    }
                ]
            }

            smrf_pipeline = pdal.Pipeline(json.dumps(smrf_pipeline_json))
            count = smrf_pipeline.execute()

            if count == 0:
                return {
                    'success': False,
                    'error': 'No points found in lidar data for this bridge geometry'
                }

            arrays = smrf_pipeline.arrays[0]

            # Extract Data for Processing
            X = arrays['X']
            Y = arrays['Y']
            Z = arrays['Z']
            Classes = arrays['Classification']

            # --- RANSAC LOGIC ---
            fit_mask = ~np.isin(Classes, config.ignore_classes)

            if np.sum(fit_mask) < config.min_points_for_ransac:
                return {
                    'success': False,
                    'error': f'Not enough points for RANSAC fitting ({np.sum(fit_mask)} < {config.min_points_for_ransac})'
                }

            X_fit = X[fit_mask]
            Y_fit = Y[fit_mask]
            Z_fit = Z[fit_mask]
            xy_fit = np.stack([X_fit, Y_fit], axis=1)

            # Fit RANSAC
            ransac = RANSACRegressor(
                min_samples=config.ransac_min_samples,
                residual_threshold=config.ransac_residual_threshold,
                random_state=config.ransac_random_state
            )
            ransac.fit(xy_fit, Z_fit)
            inlier_mask = ransac.inlier_mask_

            if np.sum(inlier_mask) < config.min_ransac_inliers:
                return {
                    'success': False,
                    'error': f'Not enough RANSAC inliers ({np.sum(inlier_mask)} < {config.min_ransac_inliers})'
                }

            # --- MASKING (CONVEX HULL) ---
            x_inliers = X_fit[inlier_mask]
            y_inliers = Y_fit[inlier_mask]
            xy_inliers = np.stack([x_inliers, y_inliers], axis=1)

            try:
                hull = ConvexHull(xy_inliers)
                hull_vertices = xy_inliers[hull.vertices]
                hull_path = MatplotlibPath(hull_vertices)
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Convex hull generation failed: {str(e)}'
                }

            # Create a Global Lateral Mask for ALL points based on the Hull
            all_xy = np.stack([X, Y], axis=1)
            lateral_mask = hull_path.contains_points(all_xy)

            # Identify points inside the hull that are NOT deep noise
            predicted_z_all = ransac.predict(all_xy)
            dist_from_plane_all = Z - predicted_z_all

            # Points roughly near the bridge plane (for curvature check)
            structure_check_mask = lateral_mask & (dist_from_plane_all > config.structure_check_min_z) & (dist_from_plane_all < config.structure_check_max_z)

            xy_check = all_xy[structure_check_mask]
            z_check = Z[structure_check_mask]

            # --- CURVATURE CHECKS ---
            # Metric 1: Inlier RMSE
            z_pred_inliers = ransac.predict(xy_inliers)
            z_inliers_fit = Z_fit[inlier_mask]
            rmse_inliers = np.sqrt(np.mean((z_inliers_fit - z_pred_inliers)**2))

            if rmse_inliers > config.max_rmse:
                return {
                    'success': False,
                    'error': f'Inlier RMSE too high ({rmse_inliers:.3f}m > {config.max_rmse}m)'
                }

            # Metric 2: Linearity (Global Arch/Sag Check)
            is_curved, deviation = WeakSupervisionPipeline.check_bridge_linearity(
                xy_check, z_check, config, deviation_threshold=config.linearity_final_deviation_threshold
            )
            if is_curved:
                return {
                    'success': False,
                    'error': f'Bridge is curved/arched (max deviation: {deviation:.3f}m)'
                }

            # --- CLASSIFICATION (HEURISTICS) ---
            new_classes = Classes.copy()

            # Rule A: Bridge Deck (Class 17)
            deck_z_mask = (dist_from_plane_all <= config.deck_z_max) & (dist_from_plane_all >= config.deck_z_min)
            final_deck_mask = deck_z_mask & lateral_mask
            # Overwrite SMRF errors (Ground->Bridge) inside the Hull
            new_classes[final_deck_mask] = config.bridge_deck_class

            # Rule B: High Noise / Obstacles (Class 18)
            noise_z_mask = (dist_from_plane_all > config.noise_z_min) & (dist_from_plane_all < config.noise_z_max)
            # Only classify noise if it's inside the bridge hull
            final_noise_mask = noise_z_mask & lateral_mask
            new_classes[final_noise_mask] = config.high_noise_class

            # Update Arrays
            arrays['Classification'] = new_classes

            return {
                'original_arrays': original_arrays,  # Original after SMRF, before classification
                'arrays': arrays,  # Modified after weak supervision classification
                'success': True,
                'rmse': rmse_inliers,
                'deviation': deviation
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Exception during processing: {str(e)}'
            }


class DataManager:
    """Manages file I/O and directory organization."""

    def __init__(self, source_dir: str, silver_dir: str) -> None:
        self.source_dir = Path(source_dir)
        self.silver_dir = Path(silver_dir)

    def get_paths(self, huc_id: str, osmid: str, source_name: str) -> Tuple[str, str]:
        """
        Get file paths for source and silver training outputs.

        Returns:
            Tuple of (source_path, silver_path) as strings for pdal compatibility
        """
        # Sanitize source_name for filename (remove/replace invalid characters)
        safe_source_name = source_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        # Remove any other problematic characters
        safe_source_name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in safe_source_name)

        source_filename = f"bridge_{osmid}_{safe_source_name}.laz"
        silver_filename = f"bridge_{osmid}_{safe_source_name}.laz"

        source_path = self.source_dir / huc_id / source_filename
        silver_path = self.silver_dir / huc_id / silver_filename

        return str(source_path), str(silver_path)

    def ensure_directories(self, huc_id: str) -> None:
        """Create output directories for a HUC if they don't exist."""
        source_huc_dir = self.source_dir / huc_id
        silver_huc_dir = self.silver_dir / huc_id

        source_huc_dir.mkdir(parents=True, exist_ok=True)
        silver_huc_dir.mkdir(parents=True, exist_ok=True)

    def file_exists(self, huc_id: str, osmid: str, source_name: str) -> bool:
        """Check if both source and silver files already exist."""
        source_path, silver_path = self.get_paths(huc_id, osmid, source_name)
        return Path(source_path).exists() and Path(silver_path).exists()

    def save_files(self, original_arrays: Any, modified_arrays: Any, huc_id: str, osmid: str, source_name: str, config: BridgeProcessingConfig) -> bool:
        """
        Save both source and silver training files.

        Args:
            original_arrays: Arrays after SMRF, before weak supervision classification
            modified_arrays: Arrays after weak supervision classification
            config: BridgeProcessingConfig instance

        Returns:
            True if successful, False otherwise
        """
        try:
            self.ensure_directories(huc_id)
            source_path, silver_path = self.get_paths(huc_id, osmid, source_name)

            # Save source (original with SMRF, before classification)
            writer_source_json = {
                "pipeline": [{
                    "type": "writers.las",
                    "filename": source_path,
                    "a_srs": config.pdal_writer_srs,
                    "extra_dims": "all"
                }]
            }
            pdal.Pipeline(json.dumps(writer_source_json), arrays=[original_arrays]).execute()

            # Save silver (labeled after weak supervision classification)
            writer_silver_json = {
                "pipeline": [{
                    "type": "writers.las",
                    "filename": silver_path,
                    "a_srs": config.pdal_writer_srs,
                    "extra_dims": "all"
                }]
            }
            pdal.Pipeline(json.dumps(writer_silver_json), arrays=[modified_arrays]).execute()

            return True
        except Exception as e:
            print(f"Error saving files for {osmid}/{source_name}: {e}")
            return False


def process_bridge_source(args: TaskTuple) -> Dict[str, Any]:
    """
    Process a single (bridge, source) pair.
    This function is called by worker processes.

    Args:
        args: Tuple of (huc_id, osmid, bridge_geometry_wkt, source_url, source_name,
                       buffer_meters, source_dir, silver_dir, config_dict)

    Returns:
        Dictionary with keys: 'success' (bool), 'huc_id' (str), 'osmid' (str),
        'source_name' (str), 'error' (Optional[str]), 'rmse' (Optional[float]),
        'deviation' (Optional[float]), 'skipped' (Optional[bool])
    """
    (huc_id, osmid, bridge_geometry_wkt, source_url, source_name,
     buffer_meters, source_dir, silver_dir, config_dict) = args

    try:
        # Reconstruct config from dict
        config = BridgeProcessingConfig.from_dict(config_dict)

        # Reconstruct geometry from WKT
        from shapely import wkt
        bridge_geometry = wkt.loads(bridge_geometry_wkt)

        # Initialize data manager
        data_manager = DataManager(source_dir, silver_dir)

        # Check if already processed
        if data_manager.file_exists(huc_id, osmid, source_name):
            return {
                'success': True,
                'huc_id': huc_id,
                'osmid': osmid,
                'source_name': source_name,
                'skipped': True,
                'error': None
            }

        # Process with weak supervision
        pipeline = WeakSupervisionPipeline()
        result = pipeline.process_bridge(source_url, bridge_geometry, config, buffer_meters)

        if result is None or not result.get('success', False):
            error_msg = result.get('error', 'Unknown processing error') if result else 'Processing returned None'
            return {
                'success': False,
                'huc_id': huc_id,
                'osmid': osmid,
                'source_name': source_name,
                'error': error_msg
            }

        # Extract both arrays
        original_arrays = result['original_arrays']
        modified_arrays = result['arrays']

        # Save files
        success = data_manager.save_files(
            original_arrays, modified_arrays, huc_id, osmid, source_name, config
        )

        if success:
            return {
                'success': True,
                'huc_id': huc_id,
                'osmid': osmid,
                'source_name': source_name,
                'rmse': result['rmse'],
                'deviation': result['deviation'],
                'error': None
            }
        else:
            error_msg = f'File save failed for {huc_id}/{osmid}/{source_name}'
            return {
                'success': False,
                'huc_id': huc_id,
                'osmid': osmid,
                'source_name': source_name,
                'error': error_msg
            }

    except Exception as e:
        error_msg = f'Exception in process_bridge_source: {str(e)}'
        return {
            'success': False,
            'huc_id': huc_id,
            'osmid': osmid,
            'source_name': source_name,
            'error': error_msg
        }


class BridgeProcessor:
    """Main orchestrator class for processing bridges across HUCs."""

    def __init__(self, hucs_dir: str, lidar_resources_path: str, source_dir: str,
                 silver_dir: str, config: BridgeProcessingConfig, buffer_meters: Optional[float] = None,
                 num_workers: Optional[int] = None, skip_existing: bool = False) -> None:
        self.hucs_dir = Path(hucs_dir)
        self.lidar_resources_path = Path(lidar_resources_path)
        self.source_dir = Path(source_dir)
        self.silver_dir = Path(silver_dir)
        self.config = config
        self.buffer_meters = buffer_meters if buffer_meters is not None else config.default_buffer_meters
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.skip_existing = skip_existing

        # Create base directories
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.silver_dir.mkdir(parents=True, exist_ok=True)

    def find_huc_files(self, huc_ids: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Find all HUC GPKG files.

        Returns:
            List of (huc_id, file_path) tuples where file_path is a string
        """
        huc_files = []

        if not self.hucs_dir.exists():
            print(f"Error: HUCs directory not found: {self.hucs_dir}")
            return huc_files

        for item in self.hucs_dir.iterdir():
            huc_path = self.hucs_dir / item.name
            if huc_path.is_dir():
                huc_id = item.name
                if huc_ids is None or huc_id in huc_ids:
                    gpkg_file = huc_path / f"osm_bridges_lidar_subset__{huc_id}.gpkg"
                    if gpkg_file.exists():
                        huc_files.append((huc_id, str(gpkg_file)))

        return huc_files

    def load_bridges(self, gpkg_path: str, osm_ids: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """Load bridges from GPKG file, optionally filtered by OSM IDs."""
        gdf = gpd.read_file(str(gpkg_path))
        gdf = gdf.to_crs(epsg=self.config.epsg_code)

        if 'osmid' not in gdf.columns:
            print(f"Warning: 'osmid' column not found in {gpkg_path}")
            return gpd.GeoDataFrame()

        gdf['osmid'] = gdf['osmid'].astype(str)

        if osm_ids is not None:
            osm_ids_str = [str(x) for x in osm_ids]
            gdf = gdf[gdf['osmid'].isin(osm_ids_str)]

        return gdf

    def generate_tasks(self, huc_ids: Optional[List[str]] = None,
                      osm_ids: Optional[List[str]] = None) -> List[TaskTuple]:
        """
        Generate all (bridge, source) tasks for processing.

        Returns:
            List of task tuples for process_bridge_source function
        """
        tasks = []
        huc_files = self.find_huc_files(huc_ids)

        if logger:
            logger.info(f"Found {len(huc_files)} HUC file(s) to process")

        # Load lidar finder in main process to find intersections
        finder = LidarSourceFinder(str(self.lidar_resources_path))

        for huc_id, gpkg_path in huc_files:
            if logger:
                logger.info(f"Loading bridges from {gpkg_path}...")
            print(f"Loading bridges from {gpkg_path}...")
            bridges = self.load_bridges(gpkg_path, osm_ids)

            if bridges.empty:
                msg = f"No bridges found in {huc_id}"
                if logger:
                    logger.warning(msg)
                print(msg)
                continue

            msg = f"Found {len(bridges)} bridges in HUC {huc_id}"
            if logger:
                logger.info(msg)
            print(msg)

            for idx, row in bridges.iterrows():
                osmid = row['osmid']
                geom = row.geometry

                # Find intersecting sources
                sources = finder.find_intersecting_sources(geom, self.buffer_meters)

                if not sources:
                    msg = f"No intersecting sources for bridge {osmid} in HUC {huc_id}"
                    if logger:
                        logger.warning(msg)
                    print(msg)
                    continue

                # Create task for each source
                for source in sources:
                    # Convert geometry to WKT for serialization
                    geom_wkt = geom.wkt
                    # Convert config to dict for multiprocessing serialization
                    config_dict = self.config.to_dict()

                    task = (
                        huc_id, osmid, geom_wkt, source['url'], source['name'],
                        self.buffer_meters, str(self.source_dir), str(self.silver_dir), config_dict
                    )
                    tasks.append(task)

        if logger:
            logger.info(f"Generated {len(tasks)} total tasks for processing")
        return tasks

    def process(self, huc_ids: Optional[List[str]] = None,
                osm_ids: Optional[List[str]] = None, show_progress: bool = True) -> None:
        """Main processing method with parallel execution."""
        if logger:
            logger.info("=" * 60)
            logger.info("Starting bridge processing pipeline")
            logger.info(f"HUC IDs: {huc_ids if huc_ids else 'All'}")
            logger.info(f"OSM IDs: {osm_ids if osm_ids else 'All'}")
            logger.info(f"Workers: {self.num_workers}")
            logger.info(f"Buffer: {self.buffer_meters}m")
            logger.info("=" * 60)

        print("Generating tasks...")
        tasks = self.generate_tasks(huc_ids, osm_ids)

        if not tasks:
            msg = "No tasks to process."
            if logger:
                logger.warning(msg)
            print(msg)
            return

        msg = f"Generated {len(tasks)} tasks. Starting parallel processing with {self.num_workers} workers..."
        if logger:
            logger.info(msg)
        print(msg)

        # Filter out existing tasks if skip_existing is True
        if self.skip_existing:
            data_manager = DataManager(self.source_dir, self.silver_dir)
            filtered_tasks = []
            skipped_count = 0

            for task in tasks:
                huc_id, osmid, _, _, source_name, _, _, _, _ = task
                if data_manager.file_exists(huc_id, osmid, source_name):
                    skipped_count += 1
                else:
                    filtered_tasks.append(task)

            tasks = filtered_tasks
            msg = f"Skipped {skipped_count} already processed tasks. {len(tasks)} tasks remaining."
            if logger:
                logger.info(msg)
            print(msg)

        if not tasks:
            msg = "All tasks already processed."
            if logger:
                logger.info(msg)
            print(msg)
            return

        # Process tasks in parallel
        with multiprocessing.Pool(
            processes=self.num_workers
        ) as pool:
            if show_progress and HAS_TQDM:
                results = list(tqdm(
                    pool.imap(process_bridge_source, tasks),
                    total=len(tasks),
                    desc="Processing bridges"
                ))
            else:
                results = pool.map(process_bridge_source, tasks)

        # Log all results in main process
        if logger:
            logger.info("=" * 60)
            logger.info("Processing Results")
            logger.info("=" * 60)
            for result in results:
                if result.get('skipped', False):
                    logger.info(f"[{result['huc_id']}] Skipped OSM ID {result['osmid']} / Source {result['source_name']} (already processed)")
                elif result['success']:
                    rmse = result.get('rmse', 0.0)
                    deviation = result.get('deviation', 0.0)
                    logger.info(f"[{result['huc_id']}] Successfully processed OSM ID {result['osmid']} / Source {result['source_name']} (RMSE: {rmse:.3f}m, Deviation: {deviation:.3f}m)")
                else:
                    logger.error(f"[{result['huc_id']}] OSM ID {result['osmid']} / Source {result['source_name']}: {result['error']}")
            logger.info("=" * 60)

        # Aggregate results
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        skipped_count = sum(1 for r in results if r.get('skipped', False))

        # Log and print summary
        summary_msg = f"\n=== Processing Complete ===\n"
        summary_msg += f"Total tasks: {len(results)}\n"
        summary_msg += f"Successful: {success_count}\n"
        summary_msg += f"Failed: {failed_count}\n"
        summary_msg += f"Skipped (already existed): {skipped_count}"

        if logger:
            logger.info("=" * 60)
            logger.info("Processing Summary")
            logger.info(f"Total tasks: {len(results)}")
            logger.info(f"Successful: {success_count}")
            logger.info(f"Failed: {failed_count}")
            logger.info(f"Skipped (already existed): {skipped_count}")
            logger.info("=" * 60)

        print(summary_msg)

        # Print errors if any
        errors = [r for r in results if not r['success'] and not r.get('skipped', False)]
        if errors:
            error_summary = f"\nErrors encountered:"
            if logger:
                logger.error("=" * 60)
                logger.error("Error Summary")
                for err in errors:
                    logger.error(f"[{err['huc_id']}] OSM ID {err['osmid']} / Source {err['source_name']}: {err['error']}")
                logger.error("=" * 60)

            print(error_summary)
            for err in errors[:10]:  # Show first 10 errors
                print(f"  {err['huc_id']}/{err['osmid']}/{err['source_name']}: {err['error']}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors (see log file for complete list)")


def setup_logging(log_dir: str = './logs') -> logging.Logger:
    """
    Set up logging to both file and console.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    global logger

    # Create logs directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir_path / f'bridge_processing_{timestamp}.log'

    # Create logger
    logger = logging.getLogger('bridge_processing')
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates and close them properly
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler (all messages)
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (INFO and above, but less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    print(f"Log file: {log_file}")

    return logger


def main() -> None:
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Process bridges from HUC subfolders with weak supervision'
    )

    parser.add_argument(
        '--hucs',
        nargs='+',
        help='List of HUC IDs to process (default: all)'
    )

    parser.add_argument(
        '--osm-ids',
        nargs='+',
        help='List of OSM IDs to process (default: all)'
    )

    parser.add_argument(
        '--buffer',
        type=float,
        default=10.0,
        help='Buffer size in meters (default: 10)'
    )

    parser.add_argument(
        '--source-dir',
        default='./data/ml-data/source',
        help='Source output directory (default: ./data/ml-data/source)'
    )

    parser.add_argument(
        '--silver-dir',
        default='./data/ml-data/silver_training',
        help='Silver training output directory (default: ./data/ml-data/silver_training)'
    )

    parser.add_argument(
        '--hucs-dir',
        default='./data/osm/hucs',
        help='HUCs input directory (default: ./data/osm/hucs)'
    )

    parser.add_argument(
        '--lidar-resources',
        default='./data/usgs_entwine/lidar_resources.geojson',
        help='Path to lidar_resources.geojson (default: ./data/usgs_entwine/lidar_resources.geojson)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: CPU count = {multiprocessing.cpu_count()})'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip already processed files (resume capability)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    parser.add_argument(
        '--log-dir',
        default='./logs',
        help='Directory for log files (default: ./logs)'
    )

    args = parser.parse_args()

    # Set up logging once with the specified log directory
    setup_logging(args.log_dir)

    # Create configuration instance
    config = BridgeProcessingConfig()
    # Create processor
    processor = BridgeProcessor(
        hucs_dir=args.hucs_dir,
        lidar_resources_path=args.lidar_resources,
        source_dir=args.source_dir,
        silver_dir=args.silver_dir,
        config=config,
        buffer_meters=args.buffer,
        num_workers=args.workers,
        skip_existing=args.skip_existing
    )

    # Process
    processor.process(
        huc_ids=args.hucs,
        osm_ids=args.osm_ids,
        show_progress=not args.no_progress
    )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
