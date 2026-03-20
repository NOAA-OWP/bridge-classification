"""
Download OSM bridge GeoPackages by HUC from S3.

Modes:
- INFO (dry run): Omit --dir to list HUCs and bridge counts without saving.
- DOWNLOAD: Provide --dir to save organized HUC folders and filtered subsets.

Example:
    python utils/download_osm_hucs.py --profile esip --limit 100
    python utils/download_osm_hucs.py --profile esip --dir ./data/osm/hucs --all
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import TypedDict

import geopandas as gpd
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.s3 import create_s3_client

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class BridgeStats(TypedDict):
    """Statistics collected during HUC processing."""

    processed: int
    skipped_missing: int
    total_bridges: int
    lidar_bridges: int
    hucs_with_lidar: int
    not_lidar_bridges: int
    hucs_with_not_lidar: int


class HUCResult(TypedDict):
    """Result from processing one HUC in a worker process."""

    processed: int
    skipped_missing: int
    total_bridges: int
    lidar_bridges: int
    hucs_with_lidar: int
    not_lidar_bridges: int
    hucs_with_not_lidar: int
    huc_id: str
    error: str | None


def _process_one_huc(work_item: tuple[str, str, str, str | None, str | None, str]) -> HUCResult:
    """
    Process a single HUC: download GPKG from S3, optionally filter and save.

    Worker creates its own boto3 session. All inputs come from work_item.
    """
    import warnings as _w

    _w.filterwarnings('ignore')

    (
        bucket_name,
        base_prefix,
        huc_prefix,
        output_dir,
        profile_name,
        save_subsets,
    ) = work_item
    huc_id = huc_prefix.rstrip('/').split('/')[-1]
    s3_key = f"{huc_prefix}osm_bridges_subset__{huc_id}.gpkg"

    empty: HUCResult = {
        'processed': 0,
        'skipped_missing': 0,
        'total_bridges': 0,
        'lidar_bridges': 0,
        'hucs_with_lidar': 0,
        'not_lidar_bridges': 0,
        'hucs_with_not_lidar': 0,
        'huc_id': huc_id,
        'error': None,
    }

    try:
        s3 = create_s3_client(profile=profile_name)
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        file_content = BytesIO(response['Body'].read())
        gdf = gpd.read_file(file_content)
    except ClientError as e:
        if (
            'Error' in e.response
            and 'Code' in e.response['Error']
            and e.response['Error']['Code'] == 'NoSuchKey'
        ):
            return {**empty, 'skipped_missing': 1}
        return {**empty, 'error': str(e)}
    except Exception as e:
        return {**empty, 'error': str(e)}

    num_bridges = len(gdf)
    num_lidar = 0
    num_not_lidar = 0
    if 'has_lidar_tif' in gdf.columns:
        lidar_gdf = gdf[gdf['has_lidar_tif'] == 'Y']
        not_lidar_gdf = gdf[gdf['has_lidar_tif'] == 'N']
        num_lidar = len(lidar_gdf)
        num_not_lidar = len(not_lidar_gdf)
    else:
        lidar_gdf = gpd.GeoDataFrame()
        not_lidar_gdf = gpd.GeoDataFrame()

    if output_dir:
        huc_dir = os.path.join(output_dir, huc_id)
        os.makedirs(huc_dir, exist_ok=True)
        local_original = os.path.join(huc_dir, f"osm_bridges_subset__{huc_id}.gpkg")
        local_filtered = os.path.join(huc_dir, f"osm_bridges_lidar_subset__{huc_id}.gpkg")
        local_not_lidar = os.path.join(huc_dir, f"osm_bridges_not_lidar_subset__{huc_id}.gpkg")
        gdf.to_file(local_original, driver="GPKG")
        if save_subsets in ('lidar', 'both') and not lidar_gdf.empty:
            lidar_gdf.to_file(local_filtered, driver="GPKG")
        if save_subsets in ('not_lidar', 'both') and not not_lidar_gdf.empty:
            not_lidar_gdf.to_file(local_not_lidar, driver="GPKG")

    return {
        'processed': 1,
        'skipped_missing': 0,
        'total_bridges': num_bridges,
        'lidar_bridges': num_lidar,
        'hucs_with_lidar': 1 if num_lidar > 0 else 0,
        'not_lidar_bridges': num_not_lidar,
        'hucs_with_not_lidar': 1 if num_not_lidar > 0 else 0,
        'huc_id': huc_id,
        'error': None,
    }


def process_bridge_files(
    bucket_name: str,
    base_prefix: str,
    output_dir: str | None = None,
    profile_name: str | None = None,
    scan_all: bool = False,
    limit: int = 100,
    workers: int = 1,
    save_subsets: str = "both",
) -> None:
    """
    Download or list OSM bridge subset GeoPackages by HUC from S3 (parallel).

    For each HUC prefix under base_prefix, fetches osm_bridges_subset__{huc_id}.gpkg,
    optionally filters by has_lidar_tif, and either saves to output_dir or
    prints counts (info-only mode when output_dir is None). Uses a process
    pool to process HUCs in parallel.

    Args:
        bucket_name: S3 bucket name.
        base_prefix: S3 prefix under which HUC prefixes are listed.
        output_dir: If set, download and save GPKGs to this directory; if None, info-only mode.
        profile_name: AWS profile name; None for default credentials.
        scan_all: If True, process all HUCs; otherwise respect limit.
        limit: Max HUCs to process when scan_all is False.
        workers: Number of worker processes (default from CPU count; capped at 64).
        save_subsets: Which filtered subsets to save in DOWNLOAD mode: 'lidar' (has_lidar_tif=='Y'),
            'not_lidar' ('N'), or 'both' (default). Only affects which subset GPKGs are written;
            the full osm_bridges_subset__{huc_id}.gpkg is always saved when output_dir is set.

    Returns:
        None.

    Raises:
        ValueError: If limit <= 0, output_dir is an existing file (not a directory), or workers < 1.

    Note:
        AWS/session failures are printed and the process exits with code 1.
    """
    # Input validation
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if output_dir is not None:
        output_path = os.path.abspath(output_dir)
        if os.path.exists(output_path) and not os.path.isdir(output_path):
            raise ValueError(f"output_dir must be a directory or not exist, got file: {output_path}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    workers = min(workers, 64)

    # 1. Setup Session
    try:
        if profile_name:
            print(f"Using AWS Profile: {profile_name}")
        else:
            print("Using default AWS environment credentials")
        s3 = create_s3_client(profile=profile_name)
    except Exception as e:
        print(f"Error initializing AWS session: {e}")
        sys.exit(1)

    mode = "DOWNLOAD & SAVE" if output_dir else "INFO ONLY (DRY RUN)"
    print(f"--- Running in {mode} mode ---")
    if output_dir:
        print(f"Target Directory: {os.path.abspath(output_dir)}")

    # 2. List HUC prefixes
    print(f"\nListing HUCs in s3://{bucket_name}/{base_prefix}...")

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=base_prefix, Delimiter='/')

    huc_prefixes = []
    for page in page_iterator:
        if 'CommonPrefixes' in page:
            for p in page['CommonPrefixes']:
                huc_prefixes.append(p['Prefix'])

    total_found = len(huc_prefixes)
    if not scan_all:
        huc_prefixes = huc_prefixes[:limit]
        print(f"Found {total_found} HUCs. Processing first {limit}...")
    else:
        print(f"Found {total_found} HUCs. Processing ALL...")

    # Statistics Counters
    stats: BridgeStats = {
        'processed': 0,
        'skipped_missing': 0,
        'total_bridges': 0,
        'lidar_bridges': 0,
        'hucs_with_lidar': 0,
        'not_lidar_bridges': 0,
        'hucs_with_not_lidar': 0,
    }

    work_items: list[tuple[str, str, str, str | None, str | None, str]] = [
        (bucket_name, base_prefix, huc_prefix, output_dir, profile_name, save_subsets)
        for huc_prefix in huc_prefixes
    ]

    if not work_items:
        # Skip pool; summary with zeros is printed below.
        pass
    else:
        results: list[HUCResult] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one_huc, item): item for item in work_items}
            done = 0
            total = len(work_items)
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stats['processed'] += result['processed']
                stats['skipped_missing'] += result['skipped_missing']
                stats['total_bridges'] += result['total_bridges']
                stats['lidar_bridges'] += result['lidar_bridges']
                stats['hucs_with_lidar'] += result['hucs_with_lidar']
                stats['not_lidar_bridges'] += result['not_lidar_bridges']
                stats['hucs_with_not_lidar'] += result['hucs_with_not_lidar']
                if result['error'] is not None:
                    print(f"Error on HUC {result['huc_id']}: {result['error']}")
                done += 1
                if output_dir and done % 50 == 0:
                    print(f"Processed {done} / {total} HUCs...")

        # Info-only mode: print one line per HUC (order arbitrary)
        if output_dir is None:
            for r in results:
                if r['error'] is not None:
                    continue
                if r['skipped_missing']:
                    print(f"HUC {r['huc_id']}: File not found")
                else:
                    print(f"HUC {r['huc_id']}: {r['total_bridges']} bridges | {r['lidar_bridges']} have Lidar | {r['not_lidar_bridges']} not Lidar")

    # 4. Final Summary
    print("\n" + "="*30)
    print("       PROCESSING SUMMARY       ")
    print("="*30)
    print(f"Mode: {'DOWNLOAD' if output_dir else 'INFO ONLY'}")
    print(f"HUCs Processed:       {stats['processed']}")
    print(f"HUCs Missing File:    {stats['skipped_missing']}")
    print("-" * 30)
    print(f"Total Bridges Found:   {stats['total_bridges']}")
    print(f"Total Lidar Bridges:   {stats['lidar_bridges']}")
    print(f"HUCs containing Lidar:{stats['hucs_with_lidar']}")
    print(f"Total Not-Lidar Bridges:{stats['not_lidar_bridges']}")
    print(f"HUCs with Not-Lidar:   {stats['hucs_with_not_lidar']}")

    if output_dir:
        print("-" * 30)
        print(f"Data saved to: {os.path.abspath(output_dir)}")
        print("Structure: {dir}/{huc_id}/osm_bridges_subset__{huc_id}.gpkg")
        print("           {dir}/{huc_id}/osm_bridges_lidar_subset__{huc_id}.gpkg")
        print("           {dir}/{huc_id}/osm_bridges_not_lidar_subset__{huc_id}.gpkg")
        print("           (lidar/not_lidar written per --save-subsets)")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    # Parse CLI and run process_bridge_files.
    parser = argparse.ArgumentParser(description='Download and filter OSM bridge data by HUC.')

    # Made --dir optional
    parser.add_argument('--dir', type=str, default=None, required=False,
                        help='Local destination folder. If omitted, script runs in INFO mode (no save).')

    parser.add_argument('--profile', type=str, default=None,
                        help='AWS CLI profile name')

    parser.add_argument('--bucket', type=str, default='noaa-nws-owp-fim',
                        help='S3 Bucket name')

    parser.add_argument('--prefix', type=str, default='hand_fim/hand_4_8_7_2/',
                        help='Base S3 prefix')

    parser.add_argument('--limit', type=int, default=100,
                        help='Limit number of HUCs to process (default: 100)')

    parser.add_argument('--all', action='store_true',
                        help='Process ALL HUCs (overrides limit)')

    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: min(32, cpu_count + 4))')

    parser.add_argument('--save-subsets', type=str, choices=['lidar', 'not_lidar', 'both'],
                        default='both',
                        help="Which filtered subsets to save: lidar (has_lidar_tif=='Y'), not_lidar ('N'), or both (default).")

    args = parser.parse_args()

    workers = args.workers
    if workers is None:
        workers = min(32, (os.cpu_count() or 4) + 4)

    process_bridge_files(
        bucket_name=args.bucket,
        base_prefix=args.prefix,
        output_dir=args.dir,
        profile_name=args.profile,
        scan_all=args.all,
        limit=args.limit,
        workers=workers,
        save_subsets=args.save_subsets,
    )


# Example usage:
# Info/dry run mode:
#   python utils/download_osm_hucs.py --profile esip --limit 100 --save-subsets lidar
# Download mode (limit 100 HUCs):
#   python utils/download_osm_hucs.py --profile esip --dir ./data/osm/hucs --limit 100 --save-subsets lidar
# Download mode (all HUCs): (use trailing slash in prefix)
#   python utils/download_osm_hucs.py --profile esip --dir ./data/osm/hucs --all --bucket fimc-data --prefix bridge-classification/osm/hucs/ --save-subsets not_lidar
