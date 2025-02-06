import datetime as dt
import os
import subprocess
import zipfile
from typing import Any, List
from tqdm import tqdm
import concurrent.futures

def download_ais_data(date: dt.date, out_folder: str, verbose: bool = False) -> str:
    """
    Downloads the AIS data for a given date from https://web.ais.dk/aisdata/ 
    and saves it to @out_folder using aria2c for faster, multi-connection downloads.
    """
    date_string = date.strftime("%Y-%m-%d")
    url = f"http://web.ais.dk/aisdata/aisdk-{date_string}.zip"
    file_name = f"aisdk-{date_string}.zip"
    file_path = os.path.join(out_folder, file_name)
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print(f"Downloading {url} to path: {file_path}")
    
    # Use aria2c for parallel downloading. The options -x (connections) and -s (split) can be tuned.
    # You might experiment with different values to find an optimal speed for your environment.
    # -o sets the output file name.
    download_command = (
        f"aria2c --console-log-level=warn "
        f"-x 16 -s 16 "
        f"-d {out_folder} "        # download directory
        f"-o {file_name} "         # output file name
        f"{url}"
    )

    if verbose:
        print(f"File name: {file_name}\nDownload command: {download_command}")
    
    subprocess.run(download_command, shell=True, check=True)
    
    return file_path

def unzip_ais_data(zip_file_path: str, out_folder: str) -> List[str]:
    """
    Unzips the AIS data file to @out_folder and returns the list of extracted file names
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    print(f"Extracting {zip_file_path} to {out_folder}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(out_folder)
        extracted_files = zip_ref.namelist()
    
    return [os.path.join(out_folder, x) for x in extracted_files]

def fetch_AIS_for_day(date: dt.date, temp_folder: str, output_folder: str):
    """
    Downloads, extracts, and removes the ZIP file for a given date.
    Returns a list of the extracted files.
    """
    try:
        zip_file_path = download_ais_data(date, temp_folder)
        extracted_files = unzip_ais_data(zip_file_path, output_folder)
        os.remove(zip_file_path)  # Remove the ZIP file after extraction
        print(f"Successfully processed {date.strftime('%Y-%m-%d')}")
        return extracted_files
    except Exception as e:
        print(f"Error processing {date.strftime('%Y-%m-%d')}: {e}")
        return []

def fetch_ais_data(
    PROCESS_DATES=None, 
    OUT="./data/rawcsv", 
    TEMP_FILE_FOLDER="./data/tmp/", 
    max_workers=4
):
    """
    Fetches AIS data for a list of dates in parallel using ThreadPoolExecutor.
    Each date’s data is downloaded, extracted, and the ZIP is removed.
    """
    if PROCESS_DATES is None:
        PROCESS_DATES = [dt.datetime(2025, 1, day).date() for day in range(25, 26)]
    
    # Optional: convert any datetime.datetime objects in PROCESS_DATES to datetime.date
    PROCESS_DATES = [d.date() if isinstance(d, dt.datetime) else d for d in PROCESS_DATES]

    # Using ThreadPoolExecutor to parallelize
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = []
        for date_item in PROCESS_DATES:
            futures.append(executor.submit(fetch_AIS_for_day, date_item, TEMP_FILE_FOLDER, OUT))
        
        # Collect results with a progress bar
        for _ in tqdm(concurrent.futures.as_completed(futures), 
                      total=len(futures), 
                      desc="Downloading & Extracting AIS data"):
            pass  # The progress bar just steps each time a future completes.

    print("All downloads and extractions are complete!")

if __name__ == "__main__":
    # Example usage
    # Provide a list of dates to download in parallel
    date_list = [dt.date(2025, 1, d) for d in range(23, 27)]
    fetch_ais_data(
        PROCESS_DATES=date_list, 
        OUT="./data/rawcsv",
        TEMP_FILE_FOLDER="./data/tmp/",
        max_workers=4  # adjust for the level of parallelism you want
    )
