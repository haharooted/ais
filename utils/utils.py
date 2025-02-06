import datetime as dt
import os
from tqdm import tqdm

from utils.project_types import (
    AIS_MAX_LAT,
    AIS_MAX_LON,
    AIS_MIN_LAT,
    AIS_MIN_LON,
    ShipType,
)





def remove_faulty_ais_readings(ais_df: pd.DataFrame) -> pd.DataFrame:
    return ais_df.loc[(ais_df["lon"] != 0.0)]



