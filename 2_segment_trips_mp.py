import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import math

# NEW IMPORTS:
import geopandas as gpd
import movingpandas as mpd
from shapely.geometry import Point

############################################
# 1. Haversine Distance Utility Function
############################################
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns the distance in nautical miles by default.
    For kilometers, change R to 6371.
    """
    # Earth radius in nautical miles
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


############################################
# 2. Read and Combine All CSVs
############################################
data_path = "./data/rawcsvsmall/"  # Adjust your path as needed
all_files = glob.glob(os.path.join(data_path, "*.csv"))

df_list = []
for file in all_files:
    temp_df = pd.read_csv(
        file,
        parse_dates=["# Timestamp"],
        dayfirst=True,  # Data in dd/mm/yyyy format
        infer_datetime_format=True
    )
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
print("Combined shape:", df.shape)

############################################
# 3. Filter Out Unwanted Mobile Types
############################################
exclude_types = ["AtoN", "Base Station"]  # keep only Class A/Class B
df = df[~df["Type of mobile"].isin(exclude_types)].copy()
print("After filtering Type of mobile:", df.shape)

############################################
# 4. Sort by MMSI and Timestamp
############################################
df.sort_values(by=["MMSI", "# Timestamp"], inplace=True)
df.reset_index(drop=True, inplace=True)


############################################
# 5. Separate Static vs. Dynamic Rows
############################################
def is_static_message(row):
    static_cols = ["Ship type", "Cargo type", "Width", "Length", 
                   "Name", "Destination", "Callsign"]
    for col in static_cols:
        val = str(row[col]).strip().lower()
        if val not in ["unknown", "undefined", "", "nan"]:
            return True
    return False

df["is_static"] = df.apply(is_static_message, axis=1)
static_df = df[df["is_static"]].copy()
dynamic_df = df[~df["is_static"]].copy()

############################################
# 6. Forward-Fill Static Data (merge_asof)
############################################
static_df.sort_values(by=["MMSI", "# Timestamp"], inplace=True)
dynamic_df.sort_values(by=["MMSI", "# Timestamp"], inplace=True)

merged_list = []
for mmsi, group_dyn in dynamic_df.groupby("MMSI", as_index=False):
    group_stat = static_df[static_df["MMSI"] == mmsi]
    
    if group_stat.empty:
        # No static records for this MMSI
        merged_list.append(group_dyn)
    else:
        merged = pd.merge_asof(
            group_dyn,
            group_stat,
            on="# Timestamp",
            by="MMSI",
            direction="backward",
            suffixes=("", "_static")
        )
        merged_list.append(merged)

dynamic_merged = pd.concat(merged_list, ignore_index=True)
print("After asof-merge:", dynamic_merged.shape)

############################################
# 7. Basic Data Cleaning
############################################
# Filter out invalid lat/lon/SOG
mask_valid_lat = dynamic_merged["Latitude"].between(-90, 90)
mask_valid_lon = dynamic_merged["Longitude"].between(-180, 180)
mask_valid_sog = dynamic_merged["SOG"].between(0, 60)  # Example max = 60 knots

dynamic_merged = dynamic_merged[mask_valid_lat & mask_valid_lon & mask_valid_sog].copy()
dynamic_merged.reset_index(drop=True, inplace=True)
print("After cleaning lat/lon/SOG:", dynamic_merged.shape)

# --------------------------------------------------------
#   NEW: Filter by bounding box for your region of interest
#   (like in the example you mentioned)
# --------------------------------------------------------
AIS_MIN_LON = 4.250
AIS_MIN_LAT = 53.6
AIS_MAX_LON = 19.5
AIS_MAX_LAT = 61.0
bbox_mask = (
    (dynamic_merged["Longitude"] >= AIS_MIN_LON) &
    (dynamic_merged["Longitude"] <= AIS_MAX_LON) &
    (dynamic_merged["Latitude"] >= AIS_MIN_LAT) &
    (dynamic_merged["Latitude"] <= AIS_MAX_LAT)
)
dynamic_merged = dynamic_merged[bbox_mask].copy()
dynamic_merged.reset_index(drop=True, inplace=True)
print("After bounding box filter:", dynamic_merged.shape)


############################################
#   NEW: Filter out vessels with < 50 readings
############################################
counts = dynamic_merged.groupby("MMSI")["MMSI"].transform("count")
dynamic_merged = dynamic_merged[counts >= 50].copy()
dynamic_merged.reset_index(drop=True, inplace=True)
print("After filtering MMSI < 50 points:", dynamic_merged.shape)

############################################
#   NEW: Resample AIS data (optional)
#   E.g. to 15-minute intervals. 
#   We'll do it via groupby-apply in pandas.
############################################
def resample_to_15min(group):
    # set index to timestamp
    group = group.set_index("# Timestamp").sort_index()
    # resample and take the FIRST in each 15-min bin
    group = group.resample("15T").first()
    return group.dropna(subset=["MMSI"])

dynamic_merged = dynamic_merged.groupby("MMSI", group_keys=False).apply(resample_to_15min)
dynamic_merged.reset_index(inplace=True)
print("After 15-minute resampling:", dynamic_merged.shape)


############################################
# 8. Convert to GeoDataFrame & Project
############################################
# movingpandas requires a geometry column
gdf = gpd.GeoDataFrame(
    dynamic_merged,
    geometry=gpd.points_from_xy(dynamic_merged["Longitude"], dynamic_merged["Latitude"]),
    crs="EPSG:4326"  # WGS84 lat/lon
)
# PROJECT to a meter-based CRS for stop detection
# Pick one suitable for your area, e.g. EPSG:3857 or EPSG:25832, etc.
gdf = gdf.to_crs(epsg=3857)

############################################
# 9. Build TrajectoryCollection w/ MovingPandas
############################################
trajs = mpd.TrajectoryCollection(
    gdf,
    traj_id_col="MMSI",
    t="# Timestamp",
    min_length=10_000  # optional: minimum length in meters
)
print("Number of initial trajectories:", len(trajs.trajectories))

############################################
# 10. Split by Observation Gaps
############################################
# Instead of your manual time-split, use the built-in ObservationGapSplitter.
# e.g., if a gap > 2 hours => new trajectory segment
gap_hours = 2
obs_splitter = mpd.ObservationGapSplitter(trajs)
gap_splitted_trajs = obs_splitter.split(
    gap=timedelta(hours=gap_hours),
    min_length=10_000  # optional
)
print("After splitting by obs gap:", len(gap_splitted_trajs.trajectories))

############################################
# 11. Stop Detection
############################################
# StopSplitter splits a trajectory if it remains within max_diameter
# for at least min_duration. E.g., max_diameter=1000m, min_duration=3h
stop_splitter = mpd.StopSplitter(gap_splitted_trajs)
stop_splitted_trajs = stop_splitter.split(
    max_diameter=1000,                     # 1 km
    min_duration=timedelta(hours=3),       # 3 hours
    min_length=10_000                      # optional, to skip tiny segments
)
print("After stop detection:", len(stop_splitted_trajs.trajectories))

############################################
# 12. Speed Calculation & Outlier Removal
############################################
# Add speed (units = (distance, time)) => (km, h) or (m, s)
stop_splitted_trajs.add_speed(
    overwrite=True,
    units=("km", "h"),       # so speed is in km/h
    name="speed_kmh"
)
# Convert to final GeoDataFrame
final_gdf = stop_splitted_trajs.to_point_gdf()

# Filter out crazy speed outliers:
speed_threshold = 120  # e.g. 120 km/h
final_gdf = final_gdf[final_gdf["speed_kmh"] < speed_threshold].copy()

# Rebuild TrajectoryCollection after removing outliers
final_trajs = mpd.TrajectoryCollection(
    final_gdf,
    traj_id_col="MMSI",
    t="timestamp"  # note: after .to_point_gdf(), the time col is named "timestamp"
)
print("After speed outlier removal:", len(final_trajs.trajectories))


############################################
# 13. Trip-Level Aggregation
############################################
# The structure of final_trajs.trajectories is a list of mpd.Trajectory objects
# We can build a "trip-level" DataFrame (like your existing compute_trip_metrics)
# by iterating over each trajectory and computing summary stats.
############################################

def summarize_trajectory(traj: mpd.Trajectory):
    """
    Return a dict of metrics for one trajectory (trip).
    """
    # We can use some built-in methods or manual calculations:
    # Convert distance from meters to nautical miles if you prefer
    dist_m = traj.get_length()  # in CRS => meters (since we used EPSG:3857)
    dist_nm = dist_m / 1852.0   # convert meters to NM

    start_time = traj.get_start_time()
    end_time   = traj.get_end_time()
    duration   = (end_time - start_time).total_seconds() / 3600.0  # hours

    # min/max/mean speed in km/h (from 'speed_kmh' we added)
    # We'll extract that from the trajectory's data
    data = traj.df  # DataFrame with columns ["geometry", "speed_kmh", ...]
    speeds = data["speed_kmh"].dropna()
    min_sog = speeds.min() if len(speeds) else np.nan
    max_sog = speeds.max() if len(speeds) else np.nan
    mean_sog = speeds.mean() if len(speeds) else np.nan
    std_sog = speeds.std(ddof=0) if len(speeds) else np.nan

    # We can also pick out the last row to get static fields,
    # if we stored them. Let's see if they're in traj.df
    # If your static fields are in the gdf as well, you can do e.g.:
    last_row = data.iloc[-1]
    ship_type   = getattr(last_row, "Ship type_static", np.nan)
    cargo_type  = getattr(last_row, "Cargo type_static", np.nan)
    vessel_name = getattr(last_row, "Name_static",      np.nan)
    width       = getattr(last_row, "Width_static",     np.nan)
    length      = getattr(last_row, "Length_static",    np.nan)
    callsign    = getattr(last_row, "Callsign_static",  np.nan)
    destination = getattr(last_row, "Destination_static", np.nan)

    return {
        "MMSI": traj.id,            # e.g. "123456789_0_20230101" if splitted
        "trip_start": start_time,
        "trip_end": end_time,
        "duration_hrs": duration,
        "distance_nm": dist_nm,
        "min_sog": min_sog,
        "max_sog": max_sog,
        "mean_sog": mean_sog,
        "std_sog": std_sog,
        "ship_type": ship_type,
        "cargo_type": cargo_type,
        "vessel_name": vessel_name,
        "width": width,
        "length": length,
        "callsign": callsign,
        "destination": destination,
        "num_points": len(data),
    }

trip_level_records = []
for traj in final_trajs:
    rec = summarize_trajectory(traj)
    trip_level_records.append(rec)

trip_df = pd.DataFrame(trip_level_records)
print("Trip-level rows:", trip_df.shape)

# Filter out extremely short or stationary trips if needed
DISTANCE_THRESHOLD = 0.02  # nm
before_count = len(trip_df)
trip_df = trip_df[trip_df["distance_nm"] >= DISTANCE_THRESHOLD].copy()
after_count = len(trip_df)
print(f"Filtered out {before_count - after_count} static/low-speed trips.")
print("Final trip-level rows:", trip_df.shape)

############################################
# 14. Save Final Trip-Level CSV
############################################
output_path = "./data/final_trip_data.csv"
trip_df.to_csv(output_path, index=False)
print(f"Saved trip-level data to {output_path}")
