#!/usr/bin/env python3

"""
parallel_filter_files.py

Usage Example:
    python parallel_filter_files.py \
        --input_files ./data/rawcsv/aisdk-2025-01-25.csv ./data/rawcsv/aisdk-2025-01-26.csv \
        --output_dir ./data/filtered \
        --polygon_file ./data/polygon.json \
        --lat_col Latitude \
        --lon_col Longitude \
        --num_processes 4 \
        --chunk_size 100000 \
        --count_rows
"""

import argparse
import json
import os
import pandas as pd
import multiprocessing as mp
import time
from shapely.geometry import Polygon
from shapely import wkb, vectorized

def parse_args():
    parser = argparse.ArgumentParser(description="Filter multiple AIS CSV files in parallel using a polygon from a GeoJSON file.")
    parser.add_argument("--input_files", nargs='+', required=True,
                        help="List of input CSV files to process.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where filtered CSV files are written.")
    parser.add_argument("--polygon_file", required=True,
                        help=("Path to a file containing a GeoJSON object representing a Polygon. "
                              "The file should contain a valid GeoJSON, e.g.: "
                              "'{\"type\":\"Polygon\",\"coordinates\":[[[lon,lat], [lon,lat], ...]]}'."))
    parser.add_argument("--lat_col", default="lat", help="Name of the latitude column.")
    parser.add_argument("--lon_col", default="lon", help="Name of the longitude column.")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of parallel processes (files) to run at once.")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Number of rows per chunk when reading CSV.")
    parser.add_argument("--count_rows", action='store_true',
                        help="If set, we'll count total rows in each file to provide chunk progress.")
    return parser.parse_args()

def load_polygon_from_file(polygon_file_path):
    """
    Load a GeoJSON file representing a Polygon and return a Shapely Polygon.
    The GeoJSON file must contain an object with type "Polygon" and a "coordinates" key.
    """
    with open(polygon_file_path, 'r', encoding='utf-8') as f:
        geojson_obj = json.load(f)
    
    if geojson_obj.get("type") != "Polygon":
        raise ValueError("The GeoJSON in the file does not represent a Polygon.")
    
    coords = geojson_obj.get("coordinates")
    if not coords or not isinstance(coords, list) or len(coords) == 0:
        raise ValueError("Invalid coordinates in the GeoJSON file.")
    
    # For a simple polygon, we take the first (exterior) ring.
    exterior_ring = coords[0]
    return Polygon(exterior_ring)

def count_rows_in_csv(file_path):
    """
    Count the total number of lines in the CSV (minus the header),
    so we can estimate total_chunks for chunked processing.
    This can be time-consuming for very large files.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Subtract 1 for the header line
        return sum(1 for _ in f) - 1

def filter_chunk(df_chunk, polygon, lat_col, lon_col):
    """
    Filter a pandas DataFrame chunk so that only points within 'polygon' remain.
    Uses vectorized operations for better performance.
    """
    xs = df_chunk[lon_col].values
    ys = df_chunk[lat_col].values
    mask = vectorized.contains(polygon, xs, ys)
    return df_chunk[mask]

def process_file(file_path, output_dir, polygon_wkb, lat_col, lon_col, chunk_size, do_count_rows):
    """
    Process a single CSV file:
        1) Read it in chunks.
        2) Filter each chunk.
        3) Write out a new CSV in 'output_dir' with only points inside the polygon.
        4) Log progress for each chunk, including chunk number, rows kept, etc.
    """
    start_time = time.time()
    
    # Reconstruct polygon from WKB (since Shapely geometry isn't pickleable directly).
    polygon = wkb.loads(polygon_wkb)

    # Prepare output file name
    base_name = os.path.basename(file_path)
    out_file = os.path.join(output_dir, f"filtered_{base_name}")
    
    # Overwrite if it exists
    if os.path.exists(out_file):
        os.remove(out_file)
    
    # Optionally count total rows to compute total_chunks
    total_chunks = None
    if do_count_rows:
        total_rows = count_rows_in_csv(file_path)
        if total_rows > 0 and chunk_size > 0:
            total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
    
    print(f"[START] File: {file_path} -> {out_file}")
    if do_count_rows and total_chunks:
        print(f"         Found ~{total_rows} rows => ~{total_chunks} chunks @ chunk_size={chunk_size}")

    chunks_processed = 0
    total_kept = 0
    
    # Read CSV in chunks
    for df_chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks_processed += 1
        filtered = filter_chunk(df_chunk, polygon, lat_col, lon_col)
        kept = len(filtered)
        total_kept += kept
        
        mode = 'a'
        header = (chunks_processed == 1)
        filtered.to_csv(out_file, mode=mode, header=header, index=False)
        
        if total_chunks:
            print(f"  [File: {base_name}] Chunk {chunks_processed}/{total_chunks}, "
                  f"read {len(df_chunk)} rows, kept {kept}.")
        else:
            print(f"  [File: {base_name}] Chunk {chunks_processed}, "
                  f"read {len(df_chunk)} rows, kept {kept}.")

    elapsed_time = time.time() - start_time
    print(f"[DONE]  File: {file_path} => Kept total {total_kept} rows, took {elapsed_time:.1f} seconds.\n")
    return out_file

def main():
    args = parse_args()
    
    # Ensure output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the polygon from the GeoJSON file and convert it to WKB for pickling in Pool
    polygon = load_polygon_from_file(args.polygon_file)
    polygon_wkb = polygon.wkb

    files_to_process = args.input_files
    
    if args.num_processes <= 1:
        print(f"Running single-process mode for {len(files_to_process)} file(s).")
        output_files = []
        for fpath in files_to_process:
            out_f = process_file(
                fpath, 
                args.output_dir,
                polygon_wkb,
                args.lat_col, 
                args.lon_col, 
                args.chunk_size,
                args.count_rows
            )
            output_files.append(out_f)
    else:
        print(f"Running multi-process mode with {args.num_processes} processes on {len(files_to_process)} files.")
        pool = mp.Pool(args.num_processes)
        async_results = []
        for fpath in files_to_process:
            res = pool.apply_async(
                process_file,
                (
                    fpath,
                    args.output_dir,
                    polygon_wkb,
                    args.lat_col,
                    args.lon_col,
                    args.chunk_size,
                    args.count_rows
                )
            )
            async_results.append(res)
        pool.close()
        pool.join()
        output_files = [r.get() for r in async_results]
    
    print("\nAll tasks completed. Filtered files:")
    for of in output_files:
        print(f"  - {of}")

if __name__ == "__main__":
    main()


# python parallel_filter_files.py \
#     --input_files ./data/rawcsv/aisdk-2025-01-23.csv ./data/rawcsv/aisdk-2025-01-24.csv ./data/rawcsv/aisdk-2025-01-25.csv ./data/rawcsv/aisdk-2025-01-26.csv ./data/rawcsv/aisdk-2025-01-27.csv ./data/rawcsv/aisdk-2025-01-28.csv ./data/rawcsv/aisdk-2025-01-29.csv ./data/rawcsv/aisdk-2025-01-30.csv \
#     --output_dir ./data/filtered \
#     --polygon_file ./data/polygon.json \
#     --lat_col Latitude \
#     --lon_col Longitude \
#     --num_processes 4 \
#     --chunk_size 1000000 \
#     --count_rows
