"""Shared utilities for lidar source discovery, bridge file naming, and sampling.

Depends on geopandas. NOT imported by src/constants.py (which stays lightweight).
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd

# --- Spatial constants ---
EPSG = 3857
DEFAULT_BUFFER = 10.0


# --- Bridge filename utilities ---

def safe_source_name(source_name: str) -> str:
    """Sanitize lidar source name for use in filenames."""
    safe = source_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in safe)


def bridge_stem(osm_id: str, source_name: str) -> str:
    """Construct bridge filename stem: bridge_{osm_id}_{safe_source}."""
    return f"bridge_{osm_id}_{safe_source_name(source_name)}"


def parse_bridge_stem(stem: str) -> Optional[Tuple[str, str]]:
    """Parse bridge_{osm_id}_{source} → (osm_id, source) or None."""
    if not stem.startswith("bridge_"):
        return None
    rest = stem[len("bridge_"):]
    osm_id, sep, source = rest.partition("_")
    if not sep or not osm_id:
        return None
    return (osm_id, source)


# --- Lidar source discovery ---

def load_lidar_index(path: str, epsg: int = EPSG) -> gpd.GeoDataFrame:
    """Load lidar_resources.geojson and reproject."""
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=epsg)
    return gdf


def find_intersecting_sources(
    lidar_gdf: gpd.GeoDataFrame,
    bridge_geometry: Any,
    buffer_meters: float = DEFAULT_BUFFER,
) -> List[Dict[str, str]]:
    """Return list of {'url': ..., 'name': ...} for sources intersecting the buffered bridge.

    bridge_geometry must be in EPSG:3857 (same CRS as lidar_gdf after load_lidar_index).
    """
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


# --- Stratified sampling ---

def stratified_sample(
    entries: List[Dict[str, Any]],
    sample_size: int,
    max_per_group: int,
    seed: int,
    group_key: str = "huc_id",
    id_key: str = "osm_id",
) -> List[Dict[str, Any]]:
    """Round-robin sample across groups, deduplicating by id_key within each group."""
    rng = random.Random(seed)

    by_group = defaultdict(list)
    seen_per_group = defaultdict(set)
    for entry in entries:
        group = entry[group_key]
        entry_id = entry[id_key]
        if entry_id not in seen_per_group[group]:
            seen_per_group[group].add(entry_id)
            by_group[group].append(entry)

    for group in by_group:
        rng.shuffle(by_group[group])

    sample = []
    group_counts = defaultdict(int)
    group_order = sorted(by_group.keys())
    rng.shuffle(group_order)

    for pass_num in range(max_per_group):
        for group in group_order:
            if len(sample) >= sample_size:
                break
            bridges = by_group[group]
            if group_counts[group] < len(bridges) and group_counts[group] < max_per_group:
                sample.append(bridges[group_counts[group]])
                group_counts[group] += 1
        if len(sample) >= sample_size:
            break

    sample.sort(key=lambda x: (x[group_key], x[id_key]))
    return sample
