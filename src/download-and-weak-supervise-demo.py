"""
Bridge Weak Supervision Demo Script

Standalone demo script that processes a small set of target bridges from a
single lidar dataset. Downloads point cloud data via PDAL/EPT, applies SMRF
ground filtering, fits a RANSAC plane, performs linearity quality checks,
and classifies points into bridge deck vs. obstacles using Z-distance
heuristics.

This script is a simplified, single-dataset version of the full HUC-based
pipeline (download-and-weak-supervise-hucs.py). It is useful for quickly
testing the weak supervision algorithm on a handful of known bridges.

Usage:
    python src/download-and-weak-supervise-demo.py
"""

import geopandas as gpd
import pdal
import json
import os
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# --- CONFIGURATION ---
LIDAR_DATASET = 'GA_Statewide_B4_2018'
# LIDAR_DATASET = 'USGS_LPC_PA_South_Central_B2_2017_LAS_2019'

EPT_URL = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{LIDAR_DATASET}/ept.json"
GPKG_PATH = f'./data/osm/osm_bridges_subset_lidar__{LIDAR_DATASET}.gpkg'

OUTPUT_DIR = f"./data/silver_label_bridges__{LIDAR_DATASET.lower()}"
OUTPUT_DIR_ORIGINAL = f"./data/downloaded_bridges__{LIDAR_DATASET.lower()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ORIGINAL, exist_ok=True)

# Target OSM IDs to process
# USGS_LPC_PA_South_Central_B2_2017_LAS_2019
# TARGET_OSMIDS = [
#     1174852485, 160697314, 160697359, 1174889151, 1084727916,
#     938406797, 15902209, 506845138, 432595664, 501253066, 706009962,
#     792385722, 5069009, 64914035, 274538735, 492364942
# ]

# GA_Statewide_B4_2018
TARGET_OSMIDS = [
    980194216, 28994716, 39810511, 39811603
]

# Buffer size (meters).
BUFFER_METERS = 10


def check_bridge_linearity(xy_points, z_points, num_bins=10, deviation_threshold=0.8):
    """
    Slices the bridge into bins, finds the median Z for each bin (the 'skeleton'),
    and checks if that skeleton deviates from a straight line.
    """
    if len(z_points) < 50: return False, 0.0

    # 1. Rotate bridge to align with X-axis using PCA
    pca = PCA(n_components=2, random_state=27)
    xy_rotated = pca.fit_transform(xy_points)
    x_axis = xy_rotated[:, 0]

    min_x, max_x = np.min(x_axis), np.max(x_axis)

    # If bridge is too short (<5m), assume it's flat/keep it
    if (max_x - min_x) < 5.0: return False, 0.0

    # 2. Slice and skeletonize
    bin_edges = np.linspace(min_x, max_x, num_bins + 1)
    skeleton_x = []
    skeleton_z = []

    for i in range(num_bins):
        mask = (x_axis >= bin_edges[i]) & (x_axis < bin_edges[i+1])
        if np.sum(mask) < 5: continue # Skip empty bins

        z_slice = z_points[mask]
        skeleton_z.append(np.median(z_slice))
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2.0
        skeleton_x.append(bin_center)

    if len(skeleton_x) < 3: return False, 0.0

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

def run_weak_supervision_pipeline():
    print(f"Loading geometry from {GPKG_PATH}...")

    try:
        gdf = gpd.read_file(GPKG_PATH)
        gdf = gdf.to_crs(epsg=3857)

        if 'osmid' not in gdf.columns:
            print("Error: 'osmid' column not found.")
            return
        gdf['osmid'] = gdf['osmid'].astype(str)
        target_ids_str = [str(x) for x in TARGET_OSMIDS]
        bridges = gdf[gdf['osmid'].isin(target_ids_str)]

        if bridges.empty:
            print("No matching OSM IDs found.")
            return
        print(f"Found {len(bridges)} bridges. Starting processing loop...")

    except Exception as e:
        print(f"Error reading GPKG: {e}")
        return

    # 2. Loop through each bridge
    for idx, row in bridges.iterrows():
        osmid = row['osmid']
        geom = row.geometry

        print(f"\n--- Processing OSM ID: {osmid} ---")

        buffered_geom = geom.buffer(BUFFER_METERS)
        pdal_polygon = buffered_geom.wkt

        # Define Filenames
        original_filename = os.path.join(OUTPUT_DIR_ORIGINAL, f"original_bridge_{osmid}.laz")
        output_filename = os.path.join(OUTPUT_DIR, f"labeled_bridge_{osmid}.laz")

        # Construct PDAL Pipeline (Read + SMRF)
        pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.ept",
                    "filename": EPT_URL,
                    "polygon": pdal_polygon,
                    "requests": 3,
                    "resolution": 0.1
                },
                {
                    "type": "filters.smrf",
                    "ignore": "Classification[7:7]",
                    "scalar": 1.25,
                    "slope": 0.05,
                    "threshold": 0.5,
                    "window": 10.0
                }
            ]
        }

        try:
            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            count = pipeline.execute()

            if count == 0:
                print(f"No points found for ID {osmid}. Skipping.")
                continue

            arrays = pipeline.arrays[0]

            # --- SAVE ORIGINAL (Raw download) ---
            # Saving BEFORE sorting/shuffling to preserve original order if desired,
            # though usually saving the sorted/shuffled version is fine too.
            # Here we save the raw pipeline output.
            writer_orig_json = {
                "pipeline": [{
                    "type": "writers.las",
                    "filename": original_filename,
                    "a_srs": "EPSG:3857",
                    "extra_dims": "all"
                }]
            }
            pdal.Pipeline(json.dumps(writer_orig_json), arrays=[arrays]).execute()
            print(f" -> Saved Original: {original_filename}")

            # --- ENFORCE DETERMINISTIC ORDERING ---
            # Sort arrays by X, then Y, then Z to ensure RANSAC sees the
            # exact same indices regardless of OS or download chunking.
            # Using structured array sorting or lexsort.

            # Create a sorting index based on X, Y, Z
            # np.lexsort sorts by the last key passed first, so we pass (Z, Y, X) to sort by X, then Y, then Z
            sort_idx = np.lexsort((arrays['Z'], arrays['Y'], arrays['X']))

            # Reorder the structured array
            arrays = arrays[sort_idx]

            # NOW, shuffle deterministically using a fixed seed
            # This breaks the spatial clusters created by step 1
            rng = np.random.default_rng(seed=27)
            shuffle_idx = rng.permutation(len(arrays))
            arrays = arrays[shuffle_idx]


            # Extract Data for Processing
            X = arrays['X']
            Y = arrays['Y']
            Z = arrays['Z']
            Classes = arrays['Classification']

            # --- CENTER COORDINATES FOR NUMERICAL STABILITY ---
            # RANSAC fails on Ubuntu because X/Y are huge (e.g. 10,000,000).
            # We shift them to 0,0 temporarily for the math to work safely.
            x_center = np.mean(X)
            y_center = np.mean(Y)

            X_local = X - x_center
            Y_local = Y - y_center


            # --- 4 RANSAC LOGIC ---

            # FIT CANDIDATES:
            # We INCLUDE Class 2 (Ground) now to ensure RANSAC finds the real floor (deck),
            # even if SMRF misclassified it as Ground.
            # We only exclude obvious noise (7, 18) and Water (9).
            ignore_classes = [7, 9, 18]
            fit_mask = ~np.isin(Classes, ignore_classes)

            if np.sum(fit_mask) < 20:
                print(f"Not enough points to fit a bridge for {osmid}.")
                continue

            # Use local coordinates for fitting
            X_fit = X_local[fit_mask]
            Y_fit = Y_local[fit_mask]
            Z_fit = Z[fit_mask]
            xy_fit = np.stack([X_fit, Y_fit], axis=1)

            # Fit RANSAC
            # tightened residual_threshold=0.20 to snap tighter to pavement
            ransac = RANSACRegressor(min_samples=10, residual_threshold=0.20, random_state=27)
            ransac.fit(xy_fit, Z_fit)
            inlier_mask = ransac.inlier_mask_

            if np.sum(inlier_mask) < 20:
                print(f" -> Skipping {osmid}: Not enough inliers.")
                continue

            # --- 5 MASKING (CONVEX HULL) ---
            # Use local coordinates because RANSAC and inliers were calculated in local space
            x_inliers = X_fit[inlier_mask]
            y_inliers = Y_fit[inlier_mask]
            xy_inliers = np.stack([x_inliers, y_inliers], axis=1)

            try:
                hull = ConvexHull(xy_inliers)
                hull_vertices = xy_inliers[hull.vertices]
                hull_path = Path(hull_vertices)
            except Exception as e:
                print(f" -> Skipping {osmid}: Hull generation failed ({e})")
                continue

            # Create a Global Lateral Mask for ALL points based on the Hull
            # FIX: Must use Local Coordinates for the check, because Hull is in Local Coordinates!
            xy_local_all = np.stack([X_local, Y_local], axis=1)
            lateral_mask = hull_path.contains_points(xy_local_all)

            # Identify points inside the hull that are NOT deep noise
            # (used for curvature check)
            predicted_z_all = ransac.predict(xy_local_all)
            dist_from_plane_all = Z - predicted_z_all

            # Points roughly near the bridge plane (for curvature check)
            structure_check_mask = lateral_mask & (dist_from_plane_all > -2.0) & (dist_from_plane_all < 2.0)

            # Use local coordinates for curvature check too (numerically safer)
            xy_check = xy_local_all[structure_check_mask]
            z_check = Z[structure_check_mask]

            # --- 6 CURVATURE CHECKS ---

            # Metric 1: Inlier RMSE
            z_pred_inliers = ransac.predict(xy_inliers)
            z_inliers_fit = Z_fit[inlier_mask]
            rmse_inliers = np.sqrt(np.mean((z_inliers_fit - z_pred_inliers)**2))

            MAX_RMSE = 0.30
            if rmse_inliers > MAX_RMSE:
                 print(f" -> Skipping {osmid}: Inlier RMSE too high ({rmse_inliers:.3f}m).")
                 continue

            # Metric 2: Linearity (Global Arch/Sag Check)
            is_curved, deviation = check_bridge_linearity(xy_check, z_check, deviation_threshold=0.35) # 0.80 is too loose
            if is_curved:
                 print(f" -> Skipping {osmid}: Bridge is Curved/Arched (Max Dev={deviation:.3f}m).")
                 continue

            print(f" -> Accepted {osmid}: RMSE={rmse_inliers:.3f}m, Deviation={deviation:.3f}m")


            # --- 7 CLASSIFICATION (HEURISTICS) ---

            new_classes = Classes.copy()

            # Rule A: Bridge Deck (Class 17)
            # Upper: +0.20m (Strict top to exclude curbs/fences)
            # Lower: -0.50m (Thick enough for slab, excludes deep noise)
            deck_z_mask = (dist_from_plane_all <= 0.20) & (dist_from_plane_all >= -0.70)
            final_deck_mask = deck_z_mask & lateral_mask
            # Overwrite SMRF errors (Ground->Bridge) inside the Hull
            new_classes[final_deck_mask] = 17

            # Rule B: High Noise / Obstacles (Class 18)
            # Strictly ABOVE the deck (+0.20m to +15.0m)
            # This captures: Cars, Fences, Lamp Posts, Wires
            noise_z_mask = (dist_from_plane_all > 0.20) & (dist_from_plane_all < 15.0)
            # Only classify noise if it's inside the bridge hull
            final_noise_mask = noise_z_mask & lateral_mask # & (Classes != 2)
            new_classes[final_noise_mask] = 18

            # Update Arrays
            arrays['Classification'] = new_classes

            # 8. Write Result
            print(f"Writing labeled data for {osmid}...")

            writer_json = {
                "pipeline": [
                    {
                        "type": "writers.las",
                        "filename": output_filename,
                        "a_srs": "EPSG:3857",
                        "extra_dims": "all"
                    }
                ]
            }

            writer_pipeline = pdal.Pipeline(json.dumps(writer_json), arrays=[arrays])
            writer_pipeline.execute()
            print(f" -> Success: Saved to {output_filename}")

        except Exception as e:
            print(f" -> Failed to process {osmid}: {e}")

if __name__ == "__main__":
    if not TARGET_OSMIDS:
        print("Please add IDs to the TARGET_OSMIDS list.")
    else:
        run_weak_supervision_pipeline()
