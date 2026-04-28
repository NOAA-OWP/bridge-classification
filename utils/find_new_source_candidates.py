"""Find bridge+lidar source combinations not in any existing train/val/test split.

Scans per-HUC bridge inventories (GeoPackage files), spatially queries available
lidar sources, diffs against existing split files, and outputs a stratified
sample of unseen candidates for gold annotation.

No S3 access required — all inputs are local files.

Workflow
--------
This script finds bridge+lidar source pairs NOT in any split file. Each
candidate has a `proven_linear` column:

  - proven_linear=True:  The bridge's osm_id IS in train/val/test splits
    (with a different lidar source). The bridge already passed the linearity
    check in the pipeline, so it's guaranteed linear. The candidate uses a
    new lidar survey the model never trained on — same physical bridge,
    different point cloud.

  - proven_linear=False: The bridge's osm_id is NOT in any split file.
    Linearity is unknown — per-bridge pass rate is low (~1%), but the
    candidate pool is large (113K+), so running all candidates through
    the pipeline yields sufficient linear bridges (76 from 48 HUC8s
    observed in first full run, 2026-04-28).

Use --proven-linear to filter before sampling.

Primary approach (truly unseen bridges):
    1. Run with --proven-linear false --sample-size 0 (all candidates)
    2. Process ALL candidates through download_and_weak_supervise_hucs.py
       (rejects complex/curved bridges automatically)
    3. Extract successes from logs (archive/scripts/extract_successful_bridges.py)
    4. Select final ~50 bridges (1 per HUC first for diversity)
    Note: small samples (e.g. --sample-size 250) yield very few passes.
    The full pool is needed to get enough linear bridges.

Fallback (if not enough pass the pipeline):
    1. Run with --proven-linear true --sample-size 50
       (guaranteed linear — same bridge, new lidar source)
    2. Send directly to annotators (no pipeline filtering needed)

Usage:
    # Primary: all truly unseen candidates for pipeline filtering
    python utils/find_new_source_candidates.py \
        --lidar-resources data/usgs_entwine/lidar_resources_apr_27_2026.geojson \
        --hucs-dir data/osm/hucs \
        --split-dir data/ml-data \
        --output-dir data/ml-data/new-source-candidates \
        --proven-linear false --sample-size 0

    # Fallback: proven linear bridges with new lidar source
    python utils/find_new_source_candidates.py \
        --lidar-resources data/usgs_entwine/lidar_resources_apr_27_2026.geojson \
        --hucs-dir data/osm/hucs \
        --split-dir data/ml-data \
        --output-dir data/ml-data/new-source-candidates \
        --proven-linear true --sample-size 50 --max-per-huc 3

    # Process candidates through the pipeline (rejects complex bridges)
    python src/download_and_weak_supervise_hucs.py \
        --hucs <huc_ids_from_csv> \
        --osm-ids <osm_ids_from_csv> \
        --skip-existing
"""

import argparse
import csv
import os
import sys
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import geopandas as gpd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lidar_utils import (
    load_lidar_index, find_intersecting_sources, safe_source_name,
    bridge_stem as make_bridge_stem, stratified_sample, EPSG, DEFAULT_BUFFER,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Split file parsing
# ---------------------------------------------------------------------------

def load_used_bridges(split_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Load all bridge stems and OSM IDs from train/val/test split files.

    Returns:
        (used_stems, used_osm_ids) where used_stems are full bridge stems
        like 'bridge_123_SOURCE' and used_osm_ids are just '123'.
    """
    used_stems = set()
    used_osm_ids = set()

    split_files = [
        split_dir / "split_train_ids.txt",
        split_dir / "split_val_ids.txt",
        split_dir / "split_test_ids.txt",
    ]

    for filepath in split_files:
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping")
            continue
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("/", 1)
                if len(parts) == 2:
                    stem = parts[1]
                    used_stems.add(stem)
                    if stem.startswith("bridge_"):
                        rest = stem[len("bridge_"):]
                        osm_id = rest.split("_", 1)[0]
                        used_osm_ids.add(osm_id)

    return used_stems, used_osm_ids


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def find_candidates(
    hucs_dir: Path,
    lidar_gdf: gpd.GeoDataFrame,
    used_stems: Set[str],
    used_osm_ids: Set[str],
    buffer_meters: float,
    huc_ids: Optional[List[str]] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Scan HUC bridge inventories for bridge+source pairs not in any split."""
    candidates = []
    huc_dirs = sorted(p for p in hucs_dir.iterdir() if p.is_dir())

    if huc_ids:
        huc_ids_set = set(huc_ids)
        huc_dirs = [d for d in huc_dirs if d.name in huc_ids_set]

    iterator = huc_dirs
    if show_progress and HAS_TQDM:
        iterator = tqdm(huc_dirs, desc="Scanning HUCs", unit="huc")

    skipped_no_gpkg = 0
    total_bridges = 0
    total_sources_checked = 0

    for huc_dir in iterator:
        huc_id = huc_dir.name

        gpkg_path = huc_dir / f"osm_bridges_lidar_subset__{huc_id}.gpkg"
        if not gpkg_path.exists():
            skipped_no_gpkg += 1
            continue

        try:
            gdf = gpd.read_file(str(gpkg_path))
        except Exception as e:
            print(f"  Warning: failed to read {gpkg_path}: {e}")
            continue

        if 'osmid' not in gdf.columns:
            continue

        gdf = gdf.to_crs(epsg=EPSG)
        gdf['osmid'] = gdf['osmid'].astype(str)

        for _, row in gdf.iterrows():
            total_bridges += 1
            osmid = row['osmid']
            geom = row.geometry
            sources = find_intersecting_sources(lidar_gdf, geom, buffer_meters)

            for src in sources:
                total_sources_checked += 1
                stem = make_bridge_stem(osmid, src['name'])

                # Skip exact bridge+source combos already in splits;
                # a different source for the same osm_id still passes (proven_linear=True)
                if stem in used_stems:
                    continue

                bridge_name = row.get('name', None) if 'name' in row else None
                if bridge_name and str(bridge_name).lower() in ('none', 'nan', ''):
                    bridge_name = None

                candidates.append({
                    "huc_id": huc_id,
                    "osm_id": osmid,
                    "bridge_name": str(bridge_name) if bridge_name else "",
                    "bridge_type": row.get('bridge_type', '') if 'bridge_type' in row else '',
                    "source_name": src['name'],
                    "source_url": src['url'],
                    "proven_linear": osmid in used_osm_ids,
                })

    print(f"\nScan complete:")
    print(f"  HUCs scanned: {len(huc_dirs) - skipped_no_gpkg}")
    print(f"  HUCs skipped (no gpkg): {skipped_no_gpkg}")
    print(f"  Bridges checked: {total_bridges}")
    print(f"  Bridge+source pairs checked: {total_sources_checked}")
    print(f"  New candidates found: {len(candidates)}")

    return candidates




# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "huc_id", "osm_id", "bridge_name", "bridge_type",
    "source_name", "source_url", "proven_linear",
]


def write_csv(entries: List[Dict[str, Any]], filepath: Path) -> None:
    """Write candidate entries to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(entries, key=lambda x: (x["huc_id"], x["osm_id"])):
            writer.writerow(entry)
    print(f"  Wrote {len(entries)} rows to {filepath}")


def print_summary(
    all_candidates: List[Dict[str, Any]],
    sample: List[Dict[str, Any]],
) -> None:
    """Print summary statistics."""
    all_hucs = set(e["huc_id"] for e in all_candidates)
    all_proven = sum(1 for e in all_candidates if e["proven_linear"])
    all_unknown = len(all_candidates) - all_proven

    sample_hucs = set(e["huc_id"] for e in sample)
    sample_proven = sum(1 for e in sample if e["proven_linear"])
    sample_unknown = len(sample) - sample_proven

    print(f"\n{'='*55}")
    print("Summary")
    print(f"{'='*55}")
    print(f"  All candidates:         {len(all_candidates)}")
    print(f"    Unique HUC8s:         {len(all_hucs)}")
    print(f"    Proven linear:         {all_proven}")
    print(f"    Linearity unknown:     {all_unknown}")
    print(f"  Sample:")
    print(f"    Bridges:              {len(sample)}")
    print(f"    Unique HUC8s:         {len(sample_hucs)}")
    print(f"    Proven linear:         {sample_proven}")
    print(f"    Linearity unknown:     {sample_unknown}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find bridge+lidar source candidates not in existing splits"
    )
    parser.add_argument("--hucs", nargs="+", default=None,
                        help="Limit to specific HUC IDs (for testing)")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of bridges to sample (default: 50, 0 = all)")
    parser.add_argument("--max-per-huc", type=int, default=3,
                        help="Max bridges per HUC8 in sample (default: 3)")
    parser.add_argument("--seed", type=int, default=27,
                        help="Random seed for reproducible sampling (default: 27)")
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER,
                        help=f"Buffer in meters for lidar intersection (default: {DEFAULT_BUFFER})")
    parser.add_argument("--lidar-resources", required=True,
                        help="Path to lidar_resources.geojson")
    parser.add_argument("--hucs-dir", required=True,
                        help="Directory containing per-HUC subdirectories")
    parser.add_argument("--split-dir", required=True,
                        help="Directory containing split_*_ids.txt files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for CSV files")
    parser.add_argument("--proven-linear", choices=["true", "false"],
                        default=None,
                        help="Filter candidates before sampling: "
                             "'true' = osm_id in splits (guaranteed linear), "
                             "'false' = osm_id not in splits (linearity unknown). "
                             "Default: no filter (both types)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")
    args = parser.parse_args()

    hucs_dir = Path(args.hucs_dir)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Load used bridges from split files
    print("Loading split files...")
    used_stems, used_osm_ids = load_used_bridges(split_dir)
    print(f"  Used bridge stems: {len(used_stems):,}")
    print(f"  Used OSM IDs:      {len(used_osm_ids):,}")

    # Step 2: Load lidar index
    print(f"\nLoading lidar index from {args.lidar_resources}...")
    lidar_gdf = load_lidar_index(args.lidar_resources)
    print(f"  Loaded {len(lidar_gdf)} lidar sources")

    # Step 3: Find candidates
    print(f"\nScanning HUCs in {hucs_dir}...")
    candidates = find_candidates(
        hucs_dir=hucs_dir,
        lidar_gdf=lidar_gdf,
        used_stems=used_stems,
        used_osm_ids=used_osm_ids,
        buffer_meters=args.buffer,
        huc_ids=args.hucs,
        show_progress=not args.no_progress,
    )

    if not candidates:
        print("\nNo candidates found. Try different HUCs or check split files.")
        return

    # Step 4: Filter by --proven-linear if specified
    if args.proven_linear is not None:
        keep_proven = args.proven_linear == "true"
        before = len(candidates)
        candidates = [c for c in candidates if c["proven_linear"] == keep_proven]
        label = "proven linear" if keep_proven else "linearity unknown"
        print(f"\n--proven-linear {args.proven_linear}: kept {len(candidates)} "
              f"of {before} ({label})")
        if not candidates:
            print(f"No {label} candidates found.")
            return

    # Step 5: Write full candidate list
    print("\nWriting full candidate list...")
    all_csv = output_dir / "new_source_candidates_all.csv"
    write_csv(candidates, all_csv)

    # Step 6: Stratified sample (0 = no sampling, use all candidates)
    if args.sample_size == 0:
        sample = candidates
        print(f"\n--sample-size 0: using all {len(sample)} candidates (no sampling)")
    else:
        print("\nSampling...")
        sample = stratified_sample(candidates, args.sample_size, args.max_per_huc, args.seed)
        sample_csv = output_dir / f"new_source_candidates_sample_{len(sample)}.csv"
        write_csv(sample, sample_csv)

    # Step 7: Write sample HUC and OSM ID files for pipeline input
    sample_hucs = sorted(set(e["huc_id"] for e in sample))
    sample_osm_ids = sorted(set(e["osm_id"] for e in sample))

    hucs_file = output_dir / "sample_hucs.txt"
    hucs_file.write_text(" ".join(sample_hucs) + "\n")
    print(f"  Wrote {len(sample_hucs)} HUC IDs to {hucs_file}")

    osm_ids_file = output_dir / "sample_osm_ids.txt"
    osm_ids_file.write_text(" ".join(sample_osm_ids) + "\n")
    print(f"  Wrote {len(sample_osm_ids)} OSM IDs to {osm_ids_file}")

    # Step 8: Summary
    print_summary(candidates, sample)

    # Step 9: Print next-step command
    print(f"\nNext step — process candidates through the pipeline:")
    print(f"  python src/download_and_weak_supervise_hucs.py \\")
    print(f"      --hucs $(cat {hucs_file}) \\")
    print(f"      --osm-ids $(cat {osm_ids_file}) \\")
    print(f"      --skip-existing")


if __name__ == "__main__":
    main()
