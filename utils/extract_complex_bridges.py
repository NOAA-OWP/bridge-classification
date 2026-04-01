"""Extract curved/arched bridge rejections from processing logs.

Parses server processing logs for bridges rejected by the RANSAC linearity
check, deduplicates, excludes bridges already in training, and produces:
  1. A full CSV of all unique complex bridges
  2. A random sample stratified across HUCs

Example usage:
    python utils/extract_complex_bridges.py \
        --log-dirs logs/server-logs/ \
        --exclude-ids data/ml-data/split_train_ids.txt data/ml-data/split_val_ids.txt data/ml-data/split_test_ids.txt \
        --source-s3-uri s3://bucket/path/to/source/ \
        --profile my-aws-profile \
        --output data/ml-data/complex_bridges_all.csv \
        --sample-output data/ml-data/complex_bridges_sample_50.csv \
        --sample-size 50 \
        --max-per-huc 2 \
        --seed 27
"""

import argparse
import csv
import re
import random
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath


CURVED_PATTERN = re.compile(
    r"\[(\d+)\].*OSM ID (\d+) / Source ([^:]+): "
    r"Bridge is curved/arched \(max deviation: ([\d.]+)m\)"
)


def parse_logs(log_dir):
    """Parse all log files in a directory for curved/arched rejections.

    Returns:
        list of dicts with keys: huc_id, osm_id, lidar_source, max_deviation_m
        int: raw match count (before dedup)
    """
    entries = []
    raw_count = 0

    log_files = sorted(Path(log_dir).glob("bridge_processing_*.log"))
    if not log_files:
        print(f"  Warning: no log files found in {log_dir}")
        return entries, raw_count

    print(f"  Found {len(log_files)} log files in {log_dir}")

    for log_file in log_files:
        file_count = 0
        with open(log_file, "r") as f:
            for line in f:
                match = CURVED_PATTERN.search(line)
                if match:
                    entries.append({
                        "huc_id": match.group(1),
                        "osm_id": match.group(2),
                        "lidar_source": match.group(3).strip(),
                        "max_deviation_m": float(match.group(4)),
                    })
                    file_count += 1
                    raw_count += 1
        if file_count > 0:
            print(f"    {log_file.name}: {file_count} curved/arched entries")

    return entries, raw_count


def deduplicate(entries):
    """Deduplicate by HUC+OSM ID, keeping entry with highest deviation."""
    best = {}
    for entry in entries:
        key = (entry["huc_id"], entry["osm_id"])
        if key not in best or entry["max_deviation_m"] > best[key]["max_deviation_m"]:
            best[key] = entry
    return list(best.values())


def load_exclude_ids(filepath):
    """Load bridge IDs from a split file, return set of (huc, osm_id) tuples."""
    exclude = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: HUC/bridge_OSMID_SOURCE
            parts = line.split("/", 1)
            if len(parts) == 2:
                huc = parts[0]
                # Extract OSM ID: bridge_{osm_id}_{source}
                stem = parts[1]
                if stem.startswith("bridge_"):
                    rest = stem[len("bridge_"):]
                    # OSM ID is the first numeric segment
                    osm_id = rest.split("_", 1)[0]
                    exclude.add((huc, osm_id))
    return exclude


def stratified_sample(entries, sample_size, max_per_huc, seed):
    """Randomly sample bridges, max N per HUC for geographic diversity."""
    rng = random.Random(seed)

    by_huc = defaultdict(list)
    for entry in entries:
        by_huc[entry["huc_id"]].append(entry)

    # Shuffle within each HUC
    for huc in by_huc:
        rng.shuffle(by_huc[huc])

    # Round-robin across HUCs to fill sample
    sample = []
    huc_counts = defaultdict(int)
    huc_order = sorted(by_huc.keys())
    rng.shuffle(huc_order)

    # Multiple passes until we hit sample_size or exhaust all HUCs
    for pass_num in range(max_per_huc):
        for huc in huc_order:
            if len(sample) >= sample_size:
                break
            bridges = by_huc[huc]
            if huc_counts[huc] < len(bridges) and huc_counts[huc] < max_per_huc:
                sample.append(bridges[huc_counts[huc]])
                huc_counts[huc] += 1
        if len(sample) >= sample_size:
            break

    # Sort by HUC for readability
    sample.sort(key=lambda x: (x["huc_id"], x["osm_id"]))
    return sample


def list_s3_source_keys(s3_uri, profile=None):
    """List all .laz files under an S3 prefix and extract bridge identifiers.

    Args:
        s3_uri: S3 URI (e.g., s3://bucket/path/to/source/)
        profile: AWS profile name (None uses default credentials)

    Returns:
        set of (huc_id, osm_id, lidar_source) tuples found in S3
    """
    # Import here to avoid requiring boto3 when not using S3 validation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.s3 import create_s3_client, parse_s3_uri

    s3_client = create_s3_client(profile=profile)
    bucket, prefix = parse_s3_uri(s3_uri)
    # Ensure prefix ends with /
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    source_keys = set()
    total_files = 0
    paginator = s3_client.get_paginator("list_objects_v2")

    print(f"  Listing S3 objects under s3://{bucket}/{prefix} ...")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".laz"):
                continue
            total_files += 1
            # Extract huc_id/bridge_{osm_id}_{source}.laz from the key
            rel_path = key[len(prefix):]  # e.g., 02050206/bridge_43249811_SOURCE.laz
            p = PurePosixPath(rel_path)
            huc_id = str(p.parent)
            stem = p.stem  # bridge_43249811_SOURCE
            if stem.startswith("bridge_"):
                rest = stem[len("bridge_"):]
                osm_id, _, lidar_source = rest.partition("_")
                if osm_id and lidar_source:
                    source_keys.add((huc_id, osm_id, lidar_source))

    print(f"  Found {total_files} .laz files, parsed {len(source_keys)} unique bridge keys")
    return source_keys


def validate_against_source(entries, source_keys):
    """Check which entries have source data available in S3.

    Adds 'source_available' field to each entry.

    Returns:
        (found_count, missing_count)
    """
    found = 0
    missing = 0
    for entry in entries:
        key = (entry["huc_id"], entry["osm_id"], entry["lidar_source"])
        if key in source_keys:
            entry["source_available"] = True
            found += 1
        else:
            entry["source_available"] = False
            missing += 1
    return found, missing


def write_csv(entries, filepath, include_source=False):
    """Write entries to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["huc_id", "osm_id", "lidar_source", "max_deviation_m"]
    if include_source:
        fieldnames.append("source_available")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(entries, key=lambda x: (x["huc_id"], x["osm_id"])):
            writer.writerow(entry)
    print(f"  Wrote {len(entries)} rows to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Extract complex bridge list from processing logs")
    parser.add_argument("--log-dirs", nargs="+", default=["logs/server-logs/"],
                        help="Directories containing processing log files")
    parser.add_argument("--exclude-ids", nargs="*", default=None,
                        help="Paths to split ID files to exclude (e.g., split_train_ids.txt split_val_ids.txt split_test_ids.txt)")
    parser.add_argument("--output", default="data/ml-data/complex_bridges_all.csv",
                        help="Output path for full CSV")
    parser.add_argument("--sample-output", default="data/ml-data/complex_bridges_sample_100.csv",
                        help="Output path for sampled CSV")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of bridges to sample")
    parser.add_argument("--max-per-huc", type=int, default=3,
                        help="Max bridges per HUC in sample")
    parser.add_argument("--seed", type=int, default=27,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--source-s3-uri", default=None,
                        help="S3 URI to source directory (e.g., s3://bucket/path/to/source/) for data completeness check")
    parser.add_argument("--profile", default=None,
                        help="AWS profile name for S3 access")
    args = parser.parse_args()

    # Step 1: Parse logs
    print("Parsing logs...")
    all_entries = []
    total_raw = 0
    for log_dir in args.log_dirs:
        entries, raw = parse_logs(log_dir)
        all_entries.extend(entries)
        total_raw += raw
    print(f"\nRaw curved/arched entries: {total_raw}")

    # Step 2: Deduplicate
    deduped = deduplicate(all_entries)
    unique_hucs = len(set(e["huc_id"] for e in deduped))
    print(f"Unique HUC+OSM ID pairs after dedup: {len(deduped)}")
    print(f"Unique HUCs: {unique_hucs}")

    # Step 3: Exclude bridges already in train/val/test splits
    excluded_count = 0
    if args.exclude_ids:
        exclude_set = set()
        for id_file in args.exclude_ids:
            ids = load_exclude_ids(id_file)
            print(f"Loaded {len(ids)} IDs from {id_file}")
            exclude_set.update(ids)
        before = len(deduped)
        deduped = [e for e in deduped if (e["huc_id"], e["osm_id"]) not in exclude_set]
        excluded_count = before - len(deduped)
        print(f"Excluded (in existing splits): {excluded_count}")

    print(f"Final count: {len(deduped)} complex bridges across {len(set(e['huc_id'] for e in deduped))} HUCs")

    # Step 4: Validate against S3 source data (optional)
    include_source = False
    source_found = source_missing = 0
    if args.source_s3_uri:
        print("\nValidating source data in S3...")
        source_keys = list_s3_source_keys(args.source_s3_uri, profile=args.profile)
        source_found, source_missing = validate_against_source(deduped, source_keys)
        include_source = True
        pct = source_found / len(deduped) * 100 if deduped else 0
        print(f"  Available: {source_found} / {len(deduped)} ({pct:.1f}%)")
        print(f"  Missing:   {source_missing}")

    # Step 5: Write full CSV
    print("\nWriting full list...")
    write_csv(deduped, args.output, include_source=include_source)

    # Step 6: Write sampled CSV
    print("\nSampling...")
    sample = stratified_sample(deduped, args.sample_size, args.max_per_huc, args.seed)
    sample_hucs = len(set(e["huc_id"] for e in sample))
    print(f"  Sample: {len(sample)} bridges across {sample_hucs} HUCs (max {args.max_per_huc}/HUC)")
    write_csv(sample, args.sample_output, include_source=include_source)

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary")
    print(f"{'='*50}")
    print(f"  Raw log entries:        {total_raw}")
    print(f"  After dedup:            {len(deduped)}")
    print(f"  Excluded (splits):      {excluded_count}")
    if include_source:
        print(f"  Source available (S3):   {source_found}")
        print(f"  Source missing:          {source_missing}")
    print(f"  Full list:              {args.output}")
    print(f"  Sample:                 {args.sample_output} ({len(sample)} bridges, {sample_hucs} HUCs)")


if __name__ == "__main__":
    main()
