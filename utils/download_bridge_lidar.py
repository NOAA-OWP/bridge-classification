"""
Download raw lidar point clouds for OSM bridges (no weak supervision).

Reads bridge geometries from GeoPackage files, finds intersecting USGS
Entwine lidar sources, downloads the raw point cloud within a buffered
bridge polygon, and saves as LAZ files organized by HUC.

This is a lightweight alternative to src/download_and_weak_supervise_hucs.py
for bridges that only need raw lidar (e.g. for inference on not-lidar bridges).

Output is **source** LAZ only: unclassified point clouds (buffer + EPT crop +
write). This script does not run RANSAC, SMRF, or weak-supervision labeling,
so there is no silver (classified) output. Use src/download_and_weak_supervise_hucs.py
when you need both source and silver for training. Use this script when you only
need raw lidar (e.g. for inference on not-lidar bridges or for OWP comparison).

Usage:
    # Download lidar for not-lidar bridges (default pattern)
    python utils/download_bridge_lidar.py

    # Specific HUCs, custom output
    python utils/download_bridge_lidar.py --hucs 01010001 01010002 --output-dir ./data/not_lidar_source

    # Resume interrupted run
    python utils/download_bridge_lidar.py --skip-existing

    # Custom gpkg pattern (e.g. for lidar bridges)
    python utils/download_bridge_lidar.py --hucs-dir ./data/osm/hucs \
        --lidar-resources ./data/usgs_entwine/lidar_resources.geojson \
        --output-dir ./data/ml-data/not_lidar_source \
        --gpkg-pattern 'osm_bridges_not_lidar_subset__{huc_id}.gpkg' \
        --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.logging_utils import setup_logging

import geopandas as gpd
import pdal

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings('ignore')

# Global logger (initialized in main)
logger: Optional[logging.Logger] = None




# --- PDAL defaults (same as BridgeProcessingConfig in download_and_weak_supervise_hucs.py) ---
EPT_REQUESTS = 3
EPT_RESOLUTION = 0.1
WRITER_SRS = "EPSG:3857"
DEFAULT_BUFFER = 10.0
EPSG = 3857


# ---------------------------------------------------------------------------
# Lidar source finder (subset of LidarSourceFinder from the weak-supervision
# pipeline, kept self-contained so this script has no cross-imports).
# ---------------------------------------------------------------------------

def load_lidar_index(path: str) -> gpd.GeoDataFrame:
    """Load lidar_resources.geojson and reproject to EPSG:3857."""
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=EPSG)
    return gdf


def find_intersecting_sources(
    lidar_gdf: gpd.GeoDataFrame,
    bridge_geometry: Any,
    buffer_meters: float = DEFAULT_BUFFER,
) -> List[Dict[str, str]]:
    """Return list of {'url': ..., 'name': ...} for lidar sources intersecting the buffered bridge."""
    if lidar_gdf.empty:
        return []
    buffered = bridge_geometry.buffer(buffer_meters)
    possible = list(lidar_gdf.sindex.intersection(buffered.bounds))
    candidates = lidar_gdf.iloc[possible]
    intersecting = candidates[candidates.intersects(buffered)]
    results = []
    for idx, row in intersecting.iterrows():
        url = row.get('url', '') if 'url' in row else ''
        name = row.get('name', '') if 'name' in row else ''
        if not url and 'properties' in row and isinstance(row['properties'], dict):
            url = row['properties'].get('url', '')
            name = row['properties'].get('name', '')
        if url:
            results.append({'url': url, 'name': name or f"source_{idx}"})
    return results


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _safe_source_name(source_name: str) -> str:
    """Sanitize source_name for filenames (mirrors DataManager convention)."""
    safe = source_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in safe)


def _output_path(output_dir: Path, huc_id: str, osmid: str, source_name: str) -> Path:
    return output_dir / huc_id / f"bridge_{osmid}_{_safe_source_name(source_name)}.laz"


def _no_points_path(output_dir: Path, huc_id: str, osmid: str, source_name: str) -> Path:
    return output_dir / huc_id / f"bridge_{osmid}_{_safe_source_name(source_name)}.no_points"


def download_one_bridge(args: tuple) -> Dict[str, Any]:
    """Download raw lidar for a single (bridge, source) pair. Multiprocessing worker."""
    import warnings as _w
    _w.filterwarnings('ignore')

    (huc_id, osmid, geom_wkt, source_url, source_name,
     buffer_meters, output_dir) = args

    out_dir = Path(output_dir)
    laz_path = _output_path(out_dir, huc_id, osmid, source_name)
    sentinel = _no_points_path(out_dir, huc_id, osmid, source_name)

    result: Dict[str, Any] = {
        'huc_id': huc_id,
        'osmid': osmid,
        'source_name': source_name,
        'success': False,
        'skipped': False,
        'error': None,
        'points': 0,
    }

    # Skip if already exists
    if laz_path.exists() or sentinel.exists():
        result['success'] = True
        result['skipped'] = True
        return result

    try:
        from shapely import wkt
        bridge_geometry = wkt.loads(geom_wkt)
        buffered_wkt = bridge_geometry.buffer(buffer_meters).wkt

        pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.ept",
                    "filename": source_url,
                    "polygon": buffered_wkt,
                    "requests": EPT_REQUESTS,
                    "resolution": EPT_RESOLUTION,
                }
            ]
        }

        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        count = pipeline.execute()

        if count == 0:
            laz_path.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("# No points found\n")
            result['error'] = 'No points found'
            return result

        arrays = pipeline.arrays[0]
        result['points'] = count

        # Write LAZ
        laz_path.parent.mkdir(parents=True, exist_ok=True)
        writer_json = {
            "pipeline": [{
                "type": "writers.las",
                "filename": str(laz_path),
                "a_srs": WRITER_SRS,
                "extra_dims": "all",
            }]
        }
        pdal.Pipeline(json.dumps(writer_json), arrays=[arrays]).execute()
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_tasks(
    hucs_dir: Path,
    lidar_resources_path: str,
    output_dir: Path,
    buffer_meters: float,
    gpkg_pattern: str,
    huc_ids: Optional[List[str]] = None,
    osm_ids: Optional[List[str]] = None,
) -> tuple[List[tuple], int]:
    """Build (bridge, source) task tuples for all matching HUCs.

    Returns:
        Tuple of (task_list, no_source_count) where no_source_count is the
        number of bridges that had zero intersecting lidar sources.
    """
    lidar_gdf = load_lidar_index(lidar_resources_path)
    print(f"Loaded {len(lidar_gdf)} lidar sources from {lidar_resources_path}")

    tasks: List[tuple] = []
    no_source_count = 0
    total_bridges = 0
    huc_dirs = sorted(p for p in hucs_dir.iterdir() if p.is_dir())

    for huc_dir in huc_dirs:
        huc_id = huc_dir.name
        if huc_ids and huc_id not in huc_ids:
            continue

        gpkg_name = gpkg_pattern.replace("{huc_id}", huc_id)
        gpkg_path = huc_dir / gpkg_name
        if not gpkg_path.exists():
            continue

        gdf = gpd.read_file(str(gpkg_path))
        gdf = gdf.to_crs(epsg=EPSG)
        if 'osmid' not in gdf.columns:
            continue
        gdf['osmid'] = gdf['osmid'].astype(str)

        if osm_ids:
            osm_ids_str = [str(x) for x in osm_ids]
            gdf = gdf[gdf['osmid'].isin(osm_ids_str)]

        for _, row in gdf.iterrows():
            total_bridges += 1
            osmid = row['osmid']
            geom = row.geometry
            sources = find_intersecting_sources(lidar_gdf, geom, buffer_meters)
            if not sources:
                no_source_count += 1
                if logger:
                    logger.info(f"[{huc_id}] OSM ID {osmid}: no intersecting lidar sources")
                continue
            for src in sources:
                tasks.append((
                    huc_id, osmid, geom.wkt, src['url'], src['name'],
                    buffer_meters, str(output_dir),
                ))

    if logger:
        logger.info(f"Task generation: {total_bridges} bridges scanned, "
                     f"{no_source_count} have no lidar sources, {len(tasks)} tasks created")

    return tasks, no_source_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download raw lidar for OSM bridges (no weak supervision).'
    )
    parser.add_argument('--hucs-dir', default='./data/osm/hucs',
                        help='Directory with HUC subdirs containing gpkg files (default: ./data/osm/hucs)')
    parser.add_argument('--lidar-resources', default='./data/usgs_entwine/lidar_resources.geojson',
                        help='Path to lidar_resources.geojson')
    parser.add_argument('--output-dir', default='./data/ml-data/not_lidar_source',
                        help='Output directory for downloaded LAZ files (default: ./data/ml-data/not_lidar_source)')
    parser.add_argument('--gpkg-pattern', default='osm_bridges_not_lidar_subset__{huc_id}.gpkg',
                        help='GeoPackage filename pattern. {huc_id} is replaced per HUC. '
                             '(default: osm_bridges_not_lidar_subset__{huc_id}.gpkg)')
    parser.add_argument('--buffer', type=float, default=DEFAULT_BUFFER,
                        help=f'Buffer size in meters (default: {DEFAULT_BUFFER})')
    parser.add_argument('--hucs', nargs='+', help='Specific HUC IDs to process (default: all)')
    parser.add_argument('--osm-ids', nargs='+', help='Specific OSM IDs to process (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Number of parallel workers (default: CPU count = {multiprocessing.cpu_count()})')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip already downloaded files and no-points sentinels')
    parser.add_argument('--shuffle-seed', type=int, default=None,
                        help='Seed for task shuffle order (default: random)')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--log-dir', default='./logs',
                        help='Directory for log files (default: ./logs)')
    args = parser.parse_args()

    global logger
    logger = setup_logging('download_bridge_lidar', args.log_dir)

    print(f"Running with args: {args}")
    if logger:
        logger.info(f"Args: {args}")

    hucs_dir = Path(args.hucs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or multiprocessing.cpu_count()

    print(f"Generating tasks from {hucs_dir} (pattern: {args.gpkg_pattern})...")
    tasks, no_source_count = generate_tasks(
        hucs_dir=hucs_dir,
        lidar_resources_path=args.lidar_resources,
        output_dir=output_dir,
        buffer_meters=args.buffer,
        gpkg_pattern=args.gpkg_pattern,
        huc_ids=args.hucs,
        osm_ids=args.osm_ids,
    )
    print(f"Generated {len(tasks)} tasks ({no_source_count} bridges had no lidar sources)")

    if not tasks:
        print("Nothing to do.")
        return

    # Filter already-done tasks
    if args.skip_existing:
        before = len(tasks)
        tasks = [
            t for t in tasks
            if not _output_path(output_dir, t[0], t[1], t[4]).exists()
            and not _no_points_path(output_dir, t[0], t[1], t[4]).exists()
        ]
        skipped_existing = before - len(tasks)
        print(f"Skipped {skipped_existing} existing, {len(tasks)} remaining")
        if logger:
            logger.info(f"Skipped {skipped_existing} existing tasks, {len(tasks)} remaining")

    if not tasks:
        print("All tasks already processed.")
        return

    # Shuffle
    if args.shuffle_seed is not None:
        random.seed(args.shuffle_seed)
    random.shuffle(tasks)

    # Process
    print(f"Downloading with {workers} workers...")
    if logger:
        logger.info(f"Starting download with {workers} workers, {len(tasks)} tasks")

    results: List[Dict[str, Any]] = []
    with multiprocessing.Pool(processes=workers, maxtasksperchild=50) as pool:
        iterator = pool.imap(download_one_bridge, tasks)
        if not args.no_progress and HAS_TQDM:
            iterator = tqdm(iterator, total=len(tasks), desc="Downloading bridges")
        for r in iterator:
            results.append(r)
            # Log each result
            if logger:
                if r.get('skipped'):
                    logger.info(f"[{r['huc_id']}] OSM ID {r['osmid']} / {r['source_name']}: skipped (already exists)")
                elif r['success']:
                    logger.info(f"[{r['huc_id']}] OSM ID {r['osmid']} / {r['source_name']}: downloaded ({r['points']} points)")
                elif r.get('error') == 'No points found':
                    logger.info(f"[{r['huc_id']}] OSM ID {r['osmid']} / {r['source_name']}: no points found (sentinel written)")
                else:
                    logger.error(f"[{r['huc_id']}] OSM ID {r['osmid']} / {r['source_name']}: {r['error']}")

    # Summary
    success = sum(1 for r in results if r['success'])
    skipped = sum(1 for r in results if r.get('skipped'))
    failed = sum(1 for r in results if not r['success'])
    no_points = sum(1 for r in results if r.get('error') == 'No points found')
    total_points = sum(r.get('points', 0) for r in results)

    summary = (
        f"\n{'='*40}\n"
        f"Download Summary\n"
        f"{'='*40}\n"
        f"Total tasks:         {len(results)}\n"
        f"Successful:          {success}\n"
        f"  (skipped):         {skipped}\n"
        f"Failed:              {failed}\n"
        f"  (no points):       {no_points}\n"
        f"No lidar sources:    {no_source_count}\n"
        f"Total points:        {total_points:,}\n"
        f"Output dir:          {output_dir.resolve()}"
    )
    print(summary)
    if logger:
        logger.info(summary)

    errors = [r for r in results if not r['success'] and r.get('error') != 'No points found']
    if errors:
        print(f"\nErrors:")
        for e in errors[:10]:
            msg = f"  {e['huc_id']}/{e['osmid']}/{e['source_name']}: {e['error']}"
            print(msg)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more (see log file for full list)")
        # Log all errors
        if logger:
            logger.error(f"{'='*40}")
            logger.error(f"Error Summary ({len(errors)} errors)")
            for e in errors:
                logger.error(f"[{e['huc_id']}] OSM ID {e['osmid']} / {e['source_name']}: {e['error']}")

    if failed > 0 and failed != no_points:
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
