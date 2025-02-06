import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import math

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
data_path = "./data/rawcsvbackup/"  # Adjust your path as needed
all_files = glob.glob(os.path.join(data_path, "*.csv"))

df_list = []
for file in all_files:
    # Because one column literally has "# Timestamp" in its name,
    # we parse the dates manually with that column name
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
exclude_types = ["AtoN", "Base Station"]  # We only keep Class A (and maybe Class B)
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
    # Columns often indicating static info
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
# Example: filter out invalid lat/lon/SOG
mask_valid_lat = dynamic_merged["Latitude"].between(-90, 90)
mask_valid_lon = dynamic_merged["Longitude"].between(-180, 180)
mask_valid_sog = dynamic_merged["SOG"].between(0, 60)  # Example max speed = 60 knots

dynamic_merged = dynamic_merged[mask_valid_lat & mask_valid_lon & mask_valid_sog].copy()
dynamic_merged.reset_index(drop=True, inplace=True)
print("After cleaning lat/lon/SOG:", dynamic_merged.shape)

############################################
# 8. Trip Segmentation
############################################
time_gap_hours = 2  # threshold for new trip

dynamic_merged["time_diff"] = dynamic_merged.groupby("MMSI")["# Timestamp"].diff()

def assign_trip_ids(group):
    trip_id = 0
    trip_ids = []
    
    for diff in group["time_diff"]:
        if pd.isnull(diff) or diff > pd.Timedelta(hours=time_gap_hours):
            trip_id += 1
        trip_ids.append(trip_id)
    group["trip_id"] = trip_ids
    return group

dynamic_merged = dynamic_merged.groupby("MMSI", group_keys=False).apply(assign_trip_ids)
dynamic_merged.drop(columns="time_diff", inplace=True)

print("Unique trips:", dynamic_merged["trip_id"].nunique())

############################################
# 9. Trip-Level Aggregation
############################################
def compute_trip_metrics(group):
    group = group.sort_values(by="# Timestamp")
    
    # Speed stats
    min_sog  = group["SOG"].min()
    max_sog  = group["SOG"].max()
    mean_sog = group["SOG"].mean()
    std_sog  = group["SOG"].std(ddof=0)  # population std for example
    
    # Distance traveled
    group["lat_shift"] = group["Latitude"].shift()
    group["lon_shift"] = group["Longitude"].shift()
    group["segment_dist"] = haversine_distance(
        group["lat_shift"], group["lon_shift"], 
        group["Latitude"], group["Longitude"]
    )
    total_distance = group["segment_dist"].fillna(0).sum()
    
    group.drop(columns=["lat_shift", "lon_shift", "segment_dist"], inplace=True)
    
    # Trip times
    trip_start = group["# Timestamp"].iloc[0]
    trip_end   = group["# Timestamp"].iloc[-1]
    duration   = (trip_end - trip_start).total_seconds() / 3600.0
    
    # Number of AIS points
    num_points = len(group)

    # Grab static fields from the last row in this group
    last_row_static = group.iloc[-1]
    ship_type   = last_row_static.get("Ship type_static", np.nan)
    cargo_type  = last_row_static.get("Cargo type_static", np.nan)
    vessel_name = last_row_static.get("Name_static", np.nan)
    width       = last_row_static.get("Width_static", np.nan)
    length      = last_row_static.get("Length_static", np.nan)
    callsign    = last_row_static.get("Callsign_static", np.nan)
    destination = last_row_static.get("Destination_static", np.nan)

    return pd.Series({
        "trip_start": trip_start,
        "trip_end": trip_end,
        "duration_hrs": duration,
        "num_points": num_points,
        "min_sog": min_sog,
        "max_sog": max_sog,
        "mean_sog": mean_sog,
        "std_sog": std_sog,
        "distance_nm": total_distance,
        "ship_type": ship_type,
        "cargo_type": cargo_type,
        "vessel_name": vessel_name,
        "width": width,
        "length": length,
        "callsign": callsign,
        "destination": destination
    })

trip_df = dynamic_merged.groupby(["MMSI", "trip_id"]).apply(compute_trip_metrics).reset_index()
print("Trip-level rows:", trip_df.shape)

# Filter out short or stationary trips
DISTANCE_THRESHOLD = 0.02 # nm
 
before_count = len(trip_df)
trip_df = trip_df[
    (trip_df["distance_nm"] >= DISTANCE_THRESHOLD)
]
after_count = len(trip_df)

print(f"Filtered out {before_count - after_count} static/low-speed trips.")

print("Trip-level rows:", trip_df.shape)

############################################
# 10. Save Final Trip-Level CSV
############################################
output_path = "./data/final_trip_data_large.csv"
trip_df.to_csv(output_path, index=False)
print(f"Saved trip-level data to {output_path}")
