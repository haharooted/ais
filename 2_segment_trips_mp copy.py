import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import math

# NEW imports for advanced segmentation:
import geopandas as gpd
import movingpandas as mpd
from shapely.geometry import Point

############################################
# 1. Haversine Distance Utility (Optional)
############################################
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns the distance in nautical miles by default.
    For kilometers, change R to 6371.
    """
    R = 3440.065  # earth radius in nautical miles
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
        dayfirst=True,  # Data is in dd/mm/yyyy format
        infer_datetime_format=True
    )
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
print("Combined shape:", df.shape)


############################################
# 3. Filter Out Unwanted Mobile Types
############################################
exclude_types = ["AtoN", "Base Station"]  # keep Class A / Class B
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
    # Columns that typically hold static info
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
# 7. Forward-fill static columns across
#    all dynamic rows for each vessel
#    (important so that resampling doesn't lose them)
############################################
static_cols = [
    "Ship type_static",
    "Cargo type_static",
    "Width_static",
    "Length_static",
    "Name_static",
    "Destination_static",
    "Callsign_static"
]

def forward_fill_static(group):
    group = group.sort_values(by="# Timestamp")
    group[static_cols] = group[static_cols].ffill()
    return group

dynamic_merged = dynamic_merged.groupby("MMSI", group_keys=False).apply(forward_fill_static)


############################################
# 8. Basic Data Cleaning
############################################
# Filter out invalid lat/lon/SOG
mask_valid_lat = dynamic_merged["Latitude"].between(-90, 90)
mask_valid_lon = dynamic_merged["Longitude"].between(-180, 180)
mask_valid_sog = dynamic_merged["SOG"].between(0, 60)  # example max speed = 60 knots

dynamic_merged = dynamic_merged[mask_valid_lat & mask_valid_lon & mask_valid_sog].copy()
dynamic_merged.reset_index(drop=True, inplace=True)
print("After cleaning lat/lon/SOG:", dynamic_merged.shape)

# Optional bounding box filter
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
# 9. Filter vessels with < 50 readings
############################################
counts = dynamic_merged.groupby("MMSI")["MMSI"].transform("count")
dynamic_merged = dynamic_merged[counts >= 50].copy()
dynamic_merged.reset_index(drop=True, inplace=True)
print("After filtering <50 points:", dynamic_merged.shape)


############################################
# 10. Resample to 15-minute intervals
#     (retaining first row in each bin)
############################################
def resample_to_15min(group):
    group = group.set_index("# Timestamp").sort_index()
    return group.resample("15T").first().dropna(subset=["MMSI"])

dynamic_merged = dynamic_merged.groupby("MMSI", group_keys=False).apply(resample_to_15min)
dynamic_merged.reset_index(inplace=True)
print("After 15-minute resampling:", dynamic_merged.shape)


############################################
# 11. Convert to GeoDataFrame & Project
#     for use with MovingPandas StopSplitter
############################################
gdf = gpd.GeoDataFrame(
    dynamic_merged,
    geometry=gpd.points_from_xy(dynamic_merged["Longitude"], dynamic_merged["Latitude"]),
    crs="EPSG:4326"  # WGS84 lat/lon
)
# Reproject to a meter-based CRS, e.g. EPSG:3857 or local UTM
gdf = gdf.to_crs(epsg=3857)


############################################
# 12. Create TrajectoryCollection
############################################
trajs = mpd.TrajectoryCollection(
    gdf,
    traj_id_col="MMSI",       # group by MMSI
    t="# Timestamp",          # time column
    min_length=1_000         # optional: filter out short traj <10km
)
print("Initial trajectories:", len(trajs.trajectories))


############################################
# 13. Split by Observation Gaps (2 hours)
############################################
obs_splitted = mpd.ObservationGapSplitter(trajs).split(
    gap=timedelta(hours=2),
    min_length=1_000  # optional
)
print("Trajectories after gap split:", len(obs_splitted.trajectories))


############################################
# 14. Stop Detection (StopSplitter)
#     e.g., consider a 'stop' if vessel is
#     within 1km diameter for >=3 hours
############################################
stop_splitted = mpd.StopSplitter(obs_splitted).split(
    max_diameter=1000,                # 1 km
    min_duration=timedelta(hours=3),  # 3h
    min_length=1_000                 # optional
)
print("Trajectories after stop detection:", len(stop_splitted.trajectories))


############################################
# 15. Add Speed & Filter Speed Outliers
############################################
stop_splitted.add_speed(
    overwrite=True,
    units=("km", "h"),   # speed in km/h
    name="speed_kmh"
)
stop_gdf = stop_splitted.to_point_gdf()

# Filter out speeds above e.g. 100 km/h
max_speed_kmh = 100
stop_gdf = stop_gdf[stop_gdf["speed_kmh"] < max_speed_kmh].copy()

# Rebuild trajectories after removing outliers
final_trajs = mpd.TrajectoryCollection(
    stop_gdf,
    traj_id_col="MMSI",
    t="timestamp"
)
print("Trajectories after speed outlier removal:", len(final_trajs.trajectories))


############################################
# 16. Trip-Level Aggregation (Summary)
############################################
# We'll replicate your 'compute_trip_metrics' logic,
# but adapt it to movingpandas and your static columns

def summarize_trajectory(traj: mpd.Trajectory):
    """
    Summarize one trajectory (trip). 
    We'll parse out the 'static' columns from the last row.
    """
    dist_m = traj.get_length()      # length in CRS units (meters)
    dist_nm = dist_m / 1852.0       # convert meters -> nautical miles

    start_time = traj.get_start_time()
    end_time   = traj.get_end_time()
    duration_hrs = (end_time - start_time).total_seconds() / 3600.0

    # Speeds from the trajectory's dataframe (e.g., "speed_kmh")
    df_ = traj.df
    speeds = df_["speed_kmh"].dropna()
    min_sog = speeds.min() if len(speeds) else np.nan
    max_sog = speeds.max() if len(speeds) else np.nan
    mean_sog = speeds.mean() if len(speeds) else np.nan
    std_sog = speeds.std(ddof=0) if len(speeds) else np.nan

    # Last row => static fields
    last_row = df_.iloc[-1]
    ship_type   = getattr(last_row, "Ship type_static", np.nan)
    cargo_type  = getattr(last_row, "Cargo type_static", np.nan)
    vessel_name = getattr(last_row, "Name_static",      np.nan)
    width       = getattr(last_row, "Width_static",     np.nan)
    length      = getattr(last_row, "Length_static",    np.nan)
    callsign    = getattr(last_row, "Callsign_static",  np.nan)
    destination = getattr(last_row, "Destination_static", np.nan)

    # traj.id holds something like "209911000.0_0_YYYY-MM-DD..."
    # If you want just the original MMSI or break out the rest, you can parse:
    traj_id_str = str(traj.id)
    # We'll store the entire ID. You can do extra .split('_') logic if needed.

    return {
        "MMSI": traj_id_str,
        "trip_start": start_time,
        "trip_end": end_time,
        "duration_hrs": duration_hrs,
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
        "num_points": len(df_),
    }

records = []
for traj in final_trajs:
    rec = summarize_trajectory(traj)
    records.append(rec)

trip_df = pd.DataFrame(records)
print("Trip-level rows:", trip_df.shape)

# Filter out extremely short or stationary segments
DISTANCE_THRESHOLD = 0.02  # nm
before_count = len(trip_df)
trip_df = trip_df[trip_df["distance_nm"] >= DISTANCE_THRESHOLD].copy()
after_count = len(trip_df)
print(f"Filtered out {before_count - after_count} static/low-speed trips.")
print("Final trip-level rows:", trip_df.shape)


############################################
# 17. Save Final Trip-Level CSV
############################################
output_path = "./data/final_trip_data.csv"
trip_df.to_csv(output_path, index=False)
print(f"Saved trip-level data to {output_path}")
