# Classifying Vessel Types from AIS data using Machine Learning

## 1. data fetching / loading

1_fetch_ais_data.py

Fetches data from a date range and outputs to ./data/rawcsv

## 2. Dataset

### 2.1 data cleaning

- forward fill static data rarely transmitted to the dynamic data positions (transmitted every minute or so.)
- remove data when vessel is anchored/moored in port (nav_status property)
- filter out AtoN and Base Station mobile types, keep class A and maybe class B
- filter out invalid lat/lon/SOG
    mask_valid_lat = dynamic_merged["Latitude"].between(-90, 90)
    mask_valid_lon = dynamic_merged["Longitude"].between(-180, 180)
    mask_valid_sog = dynamic_merged["SOG"].between(0, 60)  # Example max speed = 60 knots


Rest of details can be read in the paper