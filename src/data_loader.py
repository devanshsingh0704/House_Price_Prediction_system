from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.config import DATA_URL, RAW_DATA_PATH


def download_raw_data(dest=RAW_DATA_PATH, url=DATA_URL):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urlretrieve(url, dest)
    return dest


def load_raw_data(path=RAW_DATA_PATH):
    path = download_raw_data(path)
    return pd.read_csv(path)
