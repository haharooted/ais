from enum import Enum

### Constants

# Used for filtered AIS points
AIS_MIN_LON = 4.250
AIS_MIN_LAT = 53.6
AIS_MAX_LON = 19.5
AIS_MAX_LAT = 61.0


class TimeFrequency(Enum):
    """A enum class to represent the different time frequencies for resampling the AIS dataset."""

    min_1 = "1min"
    min_10 = "10min"
    min_15 = "15min"
    min_30 = "30min"
    hour_1 = "1H"


class MapProjection(Enum):
    WGS84 = "EPSG:4326"  # longitude / lattidude
    DNN = "EPSG:5733"  # denmark on shore
    UTMzone32n = "EPSG:25832"  # use this one. Unit: meter


class ShipType(Enum):
    """A enum class to represent the different ship types in the AIS dataset"""

    sailboat = "Sailing"
    # These are the other ship types in the dataset:
    # 'Undefined'
    # 'Cargo'
    # 'Pleasure'
    # 'Passenger'
    # 'Fishing'
    # 'Other',
    # 'Pilot'
    # 'Law enforcement'
    # 'SAR'
    # 'Military'
    # 'Tug'
    # 'Dredging'
    # 'Towing'
    # 'HSC'
    # 'Tanker'
    # 'Not party to conflict'
    # 'Port tender'
    # 'Reserved'
    # 'WIG'
    # 'Diving'
    # 'Spare 1'
    # 'Anti-pollution'
    # 'Towing long/wide'
    # 'Medical'
    # 'Spare 2'