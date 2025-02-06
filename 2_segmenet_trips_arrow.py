#!/usr/bin/env python
"""
Example: A pyarrow-based AIS cleaning pipeline that:
  1. Reads many CSVs into a single Arrow Table.
  2. Filters out unwanted “mobile types.”
  3. Sorts by MMSI and Timestamp.
  4. For each MMSI group, drops long periods where the ship does not move (i.e. anchored/moored)
     and attaches static properties (ship type, cargo type, etc.) if any row has a valid value.
  5. Saves the cleaned result to a CSV.
  
Notes:
  - Here we use pyarrow.csv and then (for grouping and “per‑MMSI” processing) convert to pandas.
    (If the dataset is huge you might want to use an Arrow‑only group‑by such as via DuckDB.)
  - Adjust DIST_THRESHOLD (in nautical miles) and STATIC_DURATION_THRESHOLD (in seconds)
    as appropriate for your application.
"""

import glob, os
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import numpy as np
import pandas as pd
from datetime import timedelta
import math

############################################
# 1. Haversine Distance Utility Function
############################################
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance (in nautical miles) between two arrays of lat/lon coordinates.
    Uses the haversine formula.
    """
    R = 3440.065  # Earth radius in nautical miles
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

############################################
# 2. Read and Combine All CSVs using pyarrow
############################################
data_path = "./data/rawcsvsmall/"  # adjust as needed
all_files = glob.glob(os.path.join(data_path, "*.csv"))
tables = []

# Note: we specify that the "# Timestamp" column should be parsed as a timestamp.
for file in all_files:
    # Adjust the convert_options as needed; here we assume timestamps are in milliseconds
    table = csv.read_csv(
        file,
        convert_options=csv.ConvertOptions(
            column_types={"# Timestamp": pa.timestamp("ms")},
            timestamp_parsers=["%d/%m/%Y %H:%M:%S"]  # This matches "25/01/2025 00:00:00"
        )
    )
    tables.append(table)

combined = pa.concat_tables(tables)
print("Combined shape:", combined.num_rows)

############################################
# 3. Filter Out Unwanted Mobile Types
############################################
# For example, exclude "AtoN" and "Base Station"
exclude_types = ["AtoN", "Base Station"]

# pc.is_in returns a boolean array indicating membership. We then invert it.
mask = pc.invert(pc.is_in(combined.column("Type of mobile"),
                            value_set=pa.array(exclude_types)))
filtered = combined.filter(mask)
print("After filtering Type of mobile:", filtered.num_rows)

############################################
# 4. Sort by MMSI and Timestamp
############################################
# Get sorted indices by MMSI then "# Timestamp"
sort_indices = pc.sort_indices(
    filtered,
    sort_keys=[("MMSI", "ascending"), ("# Timestamp", "ascending")]
)
sorted_table = filtered.take(sort_indices)

############################################
# 5. Process Each MMSI Group:
#    - Remove long static periods (anchored/moored)
#    - For each static property (e.g. Ship type), if any row in the group has a valid value
#      (i.e. not "unknown", "undefined", "", or "nan"), attach that value to all rows.
############################################

# For convenience of grouping and “window” calculations,
# we convert the sorted Arrow Table to a pandas DataFrame.
df = sorted_table.to_pandas()

# Define thresholds:
DIST_THRESHOLD = 0.01  # distance (nm) below which we say no movement occurred between successive rows
STATIC_DURATION_THRESHOLD = 2 * 3600  # 2 hours in seconds

# Define the columns that contain static info:
static_cols = ["Ship type", "Cargo type", "Width", "Length", "Name", "Destination", "Callsign"]

# Function to check if a string value is “unknown/undefined” (case insensitive)
def is_valid_static(val):
    s = str(val).strip().lower()
    return s not in {"", "unknown", "undefined", "nan"}

# Process groups by MMSI
processed_dfs = []
for mmsi, group in df.groupby("MMSI"):
    group = group.sort_values(by="# Timestamp").reset_index(drop=True)
    
    # Ensure that "# Timestamp" is a datetime column
    if not np.issubdtype(group["# Timestamp"].dtype, np.datetime64):
        group["# Timestamp"] = pd.to_datetime(group["# Timestamp"])
    
    # Compute distance from previous row (the first row gets NaN)
    group["prev_lat"] = group["Latitude"].shift()
    group["prev_lon"] = group["Longitude"].shift()
    group["distance"] = haversine_distance(
        group["prev_lat"].to_numpy(), group["prev_lon"].to_numpy(),
        group["Latitude"].to_numpy(), group["Longitude"].to_numpy()
    )
    # Compute time difference in seconds (first row = 0)
    group["time_diff_sec"] = group["# Timestamp"].diff().dt.total_seconds().fillna(0)
    
    # Mark rows as “static” if the distance traveled since the previous row is below the threshold.
    group["is_static"] = group["distance"] < DIST_THRESHOLD
    
    # Now, find runs (consecutive rows) where is_static is True.
    # If any run lasts for at least STATIC_DURATION_THRESHOLD seconds, we drop those rows.
    drop_mask = np.zeros(len(group), dtype=bool)
    i = 0
    while i < len(group):
        if group.loc[i, "is_static"]:
            start = i
            while i < len(group) and group.loc[i, "is_static"]:
                i += 1
            end = i - 1
            duration = (group.loc[end, "# Timestamp"] - group.loc[start, "# Timestamp"]).total_seconds()
            if duration >= STATIC_DURATION_THRESHOLD:
                # Mark the entire run to drop
                drop_mask[start:i] = True
        else:
            i += 1
    group_filtered = group[~drop_mask].copy()
    
    # For each static column, check if any row in this group has a “valid” value.
    # If so, create a new column (with suffix "_static") with that value repeated.
    for col in static_cols:
        valid = group_filtered[col].apply(is_valid_static)
        if valid.any():
            # take the first valid value
            value = group_filtered.loc[valid, col].iloc[0]
        else:
            value = np.nan
        group_filtered[col + "_static"] = value

    # (Optional) Drop helper columns if you no longer need them:
    group_filtered.drop(columns=["prev_lat", "prev_lon", "distance", "time_diff_sec", "is_static"], inplace=True)
    
    processed_dfs.append(group_filtered)

# Concatenate all processed groups
final_df = pd.concat(processed_dfs, ignore_index=True)
print("Final cleaned data shape:", final_df.shape)

############################################
# 6. Convert Back to pyarrow Table and Save
############################################
final_table = pa.Table.from_pandas(final_df)

output_path = "./data/final_cleaned_data_arrow.csv"
# (For CSV export we use pandas; pyarrow.csv.write_csv is experimental.)
final_df.to_csv(output_path, index=False)
print(f"Saved final cleaned data to {output_path}")
