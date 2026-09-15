import re

import pandas as pd

from src.config import (
    ALLOWED_BHK,
    CATEGORICAL_FEATURES,
    MAX_PRICE_LAKH,
    MAX_SQFT,
    MIN_LOCATION_COUNT,
    MIN_PRICE_LAKH,
    MIN_SQFT,
    MIN_SQFT_PER_BHK,
    NUMERIC_FEATURES,
    OUTLIER_N_STD,
    SQ_METER_TO_SQFT,
    SQ_YARD_TO_SQFT,
    TARGET,
)

_LAND_UNITS = ("acre", "cent", "guntha", "perch")


def extract_bhk(size):
    if size is None:
        return None
    text = str(size).strip()
    if not text or re.search(r"\brk\b", text, flags=re.IGNORECASE):
        return None
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def parse_sqft(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return None
    if any(unit in text for unit in _LAND_UNITS):
        return None

    if "meter" in text:
        number = _first_number(text)
        return None if number is None else number * SQ_METER_TO_SQFT
    if "yard" in text:
        number = _first_number(text)
        return None if number is None else number * SQ_YARD_TO_SQFT

    range_match = re.search(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", text)
    if range_match:
        return (float(range_match.group(1)) + float(range_match.group(2))) / 2

    return _first_number(text)


def normalize_location(name):
    if name is None:
        return None
    text = str(name).strip().strip(",").lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    # Doddabommasandra is a different neighbourhood (Yelahanka side).
    if "doddabommasandra" in text:
        return "Doddabommasandra"
    if "bommasandra industrial" in text:
        return "Bommasandra Industrial Area"
    if text == "bommasandra":
        return "Bommasandra"

    if "hsr" in text:
        return "HSR Layout"
    if "electronic" in text and "city" in text:
        if "phase 2" in text or "phase ii" in text:
            return "Electronic City Phase II"
        if "phase 1" in text or "phase i" in text:
            return "Electronic City Phase I"
        return "Electronic City"
    if "whitefield" in text:
        return "Whitefield"
    if "marathahalli" in text:
        return "Marathahalli"
    if "bellandur" in text:
        return "Bellandur"
    if "sarjapur" in text and "road" in text:
        return "Sarjapur Road"
    if "btm" in text:
        return "BTM Layout"
    if "koramangala" in text:
        return "Koramangala"
    if "indira nagar" in text or "indiranagar" in text:
        return "Indira Nagar"
    if "hebbal" in text:
        return "Hebbal"
    if "yelahanka" in text:
        return "Yelahanka"
    if "kr puram" in text:
        return "KR Puram"
    if "brookefield" in text:
        return "Brookefield"
    if text == "chandapura":
        return "Chandapura"

    return re.sub(r"\s+", " ", str(name).strip().strip(","))


def group_rare_locations(series, min_count=MIN_LOCATION_COUNT):
    counts = series.value_counts()
    return series.where(series.map(counts) >= min_count, "Other")


def drop_pps_outliers(df, n_std=OUTLIER_N_STD, target="price", size="total_sqft"):
    """Drop listings whose <target>/sqft is >n_std from the locality mean.

    Uses the target, so it must only ever see training rows — call it AFTER the
    train/test split. Running it on the full frame (as this project used to) both
    leaks test prices into the filter and deletes the hard cases from the test set.

    Defaults are price (lakhs) and total_sqft. Mean and std scale together, so
    which rows survive does not depend on the target units.
    """
    parts = []
    for _, part in df.groupby("location"):
        pps = part[target] / part[size]
        std = pps.std()
        if pd.isna(std) or std == 0:
            parts.append(part)
            continue
        mean = pps.mean()
        parts.append(part[(pps > mean - n_std * std) & (pps < mean + n_std * std)])
    return pd.concat(parts)


def clean_dataframe(df):
    """Row-local parsing only: nothing here looks at another row's price.

    Anything that has to be learned from data — the outlier trim above, median
    imputation of bath/balcony — happens after the split, in train.py.
    """
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out["bhk"] = out["size"].map(extract_bhk)
    out["total_sqft"] = out["total_sqft"].map(parse_sqft)
    out["location"] = out["location"].map(normalize_location)
    out["area_type"] = out["area_type"].astype(str).str.strip()
    out["bath"] = pd.to_numeric(out["bath"], errors="coerce")
    out["balcony"] = pd.to_numeric(out["balcony"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    out = out[out["bhk"].isin(ALLOWED_BHK)]
    out = out[out["total_sqft"].between(MIN_SQFT, MAX_SQFT)]
    out = out[out["price"].between(MIN_PRICE_LAKH, MAX_PRICE_LAKH)]
    out = out[out["total_sqft"] / out["bhk"] >= MIN_SQFT_PER_BHK]
    out = out.dropna(subset=["location", "total_sqft", "price", "bhk"])
    out["location"] = group_rare_locations(out["location"])

    cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES) + [TARGET]
    return out[cols].reset_index(drop=True)


def _first_number(text):
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None
