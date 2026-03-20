"""
Verify that osm_bridges_lidar_subset__{huc_id}.gpkg equals the rows of
osm_bridges_subset__{huc_id}.gpkg where has_lidar_tif == 'Y'.

Works with local data (--dir) or S3 (--s3). Randomly samples HUCs for spot-check.

Usage:
    python utils/verify_lidar_subset.py --dir ./data/osm/hucs --sample 10
    python utils/verify_lidar_subset.py --s3 --profile Data --sample 20 --bucket fimc-data --prefix bridge-classification/osm/hucs/
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from io import BytesIO
from pathlib import Path

import geopandas as gpd

# Optional S3 support
try:
    from botocore.exceptions import ClientError
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.s3 import create_s3_client
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False


def _verify_one_huc_local(huc_dir: Path, huc_id: str) -> tuple[bool, str, int, int]:
    """Load both GPKGs from a local HUC dir and verify lidar subset == full[has_lidar_tif=='Y'].

    Returns:
        (pass: bool, message: str, expected_count: int, actual_count: int)
    """
    full_path = huc_dir / f"osm_bridges_subset__{huc_id}.gpkg"
    lidar_path = huc_dir / f"osm_bridges_lidar_subset__{huc_id}.gpkg"

    if not full_path.exists():
        return False, "osm_bridges_subset__*.gpkg not found", 0, 0
    if not lidar_path.exists():
        # No lidar subset file: expected only when there are zero has_lidar_tif=='Y'
        full_gdf = gpd.read_file(full_path)
        if "has_lidar_tif" not in full_gdf.columns:
            return False, "has_lidar_tif column missing in full subset", 0, 0
        expected = full_gdf[full_gdf["has_lidar_tif"] == "Y"]
        if len(expected) == 0:
            return True, "OK (no lidar subset file, 0 expected)", 0, 0
        return False, f"lidar subset file missing but {len(expected)} expected with has_lidar_tif=='Y'", len(expected), 0

    full_gdf = gpd.read_file(full_path)
    lidar_gdf = gpd.read_file(lidar_path)

    if "has_lidar_tif" not in full_gdf.columns:
        return False, "has_lidar_tif column missing in full subset", 0, len(lidar_gdf)

    expected = full_gdf[full_gdf["has_lidar_tif"] == "Y"]
    expected_count = len(expected)
    actual_count = len(lidar_gdf)

    if expected_count != actual_count:
        return False, f"count mismatch: expected {expected_count}, got {actual_count}", expected_count, actual_count

    if "osmid" in full_gdf.columns and "osmid" in lidar_gdf.columns:
        expected_ids = set(expected["osmid"].astype(str))
        actual_ids = set(lidar_gdf["osmid"].astype(str))
        if expected_ids != actual_ids:
            return False, f"osmid set mismatch (counts equal)", expected_count, actual_count

    return True, "OK", expected_count, actual_count


def _verify_one_huc_s3(
    bucket: str,
    prefix: str,
    huc_id: str,
    profile_name: str | None,
) -> tuple[bool, str, int, int]:
    """Download both GPKGs from S3 for one HUC and verify. Same return as _verify_one_huc_local."""
    if not HAS_BOTO:
        return False, "boto3 not installed", 0, 0

    base = f"{prefix}{huc_id}/"
    full_key = f"{base}osm_bridges_subset__{huc_id}.gpkg"
    lidar_key = f"{base}osm_bridges_lidar_subset__{huc_id}.gpkg"

    try:
        s3 = create_s3_client(profile=profile_name)
        resp_full = s3.get_object(Bucket=bucket, Key=full_key)
        full_gdf = gpd.read_file(BytesIO(resp_full["Body"].read()))

        try:
            resp_lidar = s3.get_object(Bucket=bucket, Key=lidar_key)
            lidar_gdf = gpd.read_file(BytesIO(resp_lidar["Body"].read()))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                if "has_lidar_tif" not in full_gdf.columns:
                    return False, "has_lidar_tif column missing", 0, 0
                expected = full_gdf[full_gdf["has_lidar_tif"] == "Y"]
                if len(expected) == 0:
                    return True, "OK (no lidar subset object, 0 expected)", 0, 0
                return False, f"lidar subset object missing but {len(expected)} expected", len(expected), 0
            raise

        if "has_lidar_tif" not in full_gdf.columns:
            return False, "has_lidar_tif column missing", 0, len(lidar_gdf)

        expected = full_gdf[full_gdf["has_lidar_tif"] == "Y"]
        expected_count = len(expected)
        actual_count = len(lidar_gdf)

        if expected_count != actual_count:
            return False, f"count mismatch: expected {expected_count}, got {actual_count}", expected_count, actual_count

        if "osmid" in full_gdf.columns and "osmid" in lidar_gdf.columns:
            expected_ids = set(expected["osmid"].astype(str))
            actual_ids = set(lidar_gdf["osmid"].astype(str))
            if expected_ids != actual_ids:
                return False, "osmid set mismatch (counts equal)", expected_count, actual_count

        return True, "OK", expected_count, actual_count

    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return False, "osm_bridges_subset__*.gpkg not found on S3", 0, 0
        return False, str(e), 0, 0
    except Exception as e:
        return False, str(e), 0, 0


def list_hucs_local(data_dir: Path) -> list[str]:
    """Return list of HUC IDs (subdir names) that have osm_bridges_subset__*.gpkg."""
    huc_ids = []
    if not data_dir.exists() or not data_dir.is_dir():
        return huc_ids
    for item in data_dir.iterdir():
        if item.is_dir():
            huc_id = item.name
            if (item / f"osm_bridges_subset__{huc_id}.gpkg").exists():
                huc_ids.append(huc_id)
    return sorted(huc_ids)


def list_hucs_s3(bucket: str, prefix: str, profile_name: str | None) -> list[str]:
    """List HUC IDs under prefix (CommonPrefixes). Prefix should end with /."""
    if not HAS_BOTO:
        return []
    s3 = create_s3_client(profile=profile_name)
    paginator = s3.get_paginator("list_objects_v2")
    huc_ids = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            # e.g. prefix = "bridge-classification/osm/hucs/", CommonPrefix is "bridge-classification/osm/hucs/02050206/"
            common = p["Prefix"]
            name = common.rstrip("/").split("/")[-1]
            huc_ids.append(name)
    return sorted(huc_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify osm_bridges_lidar_subset equals full subset filtered by has_lidar_tif=='Y'."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./data/osm/hucs"),
        help="Local directory containing HUC subdirs (default: ./data/osm/hucs)",
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Sample and verify from S3 instead of local",
    )
    parser.add_argument("--profile", type=str, default=None, help="AWS profile (for --s3)")
    parser.add_argument("--bucket", type=str, default="fimc-data", help="S3 bucket (for --s3)")
    parser.add_argument("--prefix", type=str, default="bridge-classification/osm/hucs/", help="S3 prefix (for --s3)")
    parser.add_argument("--sample", type=int, default=10, help="Number of HUCs to sample (default: 10)")
    args = parser.parse_args()

    if args.s3:
        if not HAS_BOTO:
            print("ERROR: --s3 requires boto3")
            sys.exit(1)
        huc_ids = list_hucs_s3(args.bucket, args.prefix, args.profile)
        print(f"Found {len(huc_ids)} HUCs under s3://{args.bucket}/{args.prefix}")
    else:
        huc_ids = list_hucs_local(args.dir)
        print(f"Found {len(huc_ids)} HUCs under {args.dir}")

    if not huc_ids:
        print("No HUCs to verify.")
        sys.exit(0)

    to_check = huc_ids
    if len(huc_ids) > args.sample:
        to_check = random.sample(huc_ids, args.sample)
        print(f"Sampling {args.sample} HUCs for verification...")
    else:
        print(f"Verifying all {len(to_check)} HUCs...")

    passed = 0
    failed = 0
    for huc_id in to_check:
        if args.s3:
            ok, msg, exp, act = _verify_one_huc_s3(args.bucket, args.prefix, huc_id, args.profile)
        else:
            ok, msg, exp, act = _verify_one_huc_local(args.dir / huc_id, huc_id)
        status = "PASS" if ok else "FAIL"
        print(f"  {huc_id}: {status} — {msg} (expected={exp}, actual={act})")
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 40)
    print(f"Summary: {passed} PASS, {failed} FAIL out of {len(to_check)}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
