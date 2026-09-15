"""Move a model estimate from 2017-18 price levels into today's rupees, and project forward.

The model is trained on ~2017-18 listings, so what it returns is a 2017-18 price.
A 2 BHK in Electronic City Phase II is about Rs 33 lakh in that data and about
Rs 70 lakh today: the model is not slightly stale, it is out by roughly 2x.

No amount of tuning fixes that, because the training data has no time column at
all -- every row is from the same frozen moment. Price level over time is a
separate problem, and this module is that second stage: published market rates
for the level correction, a CAGR band for the projection. It is arithmetic on
sourced numbers, not something learned from the listings.
"""

import csv
import json

from src.config import (
    MARKET_RATES_PATH,
    METADATA_PATH,
    PROJECTION_CAGR,
)

CITY_KEY = "Bangalore"


def load_market_rates(path=MARKET_RATES_PATH):
    """Present-day rupees/sqft, hand-collected. See the csv for per-row sources."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["locality"]: float(row["rupees_per_sqft"]) for row in csv.DictReader(handle)}


def load_dataset_rates(path=METADATA_PATH):
    """Median rupees/sqft per locality in the training data, written by train.py."""
    return json.loads(path.read_text())["rate_per_sqft"]


def inflation_factor(location, market=None, dataset=None):
    """How far prices have moved since the dataset was collected.

    Returns (factor, basis). Only a handful of localities have a published
    present-day rate, so most fall back to the city-wide average -- the basis
    string says which happened, because a city-wide factor applied to a specific
    locality is a much weaker claim.
    """
    market = load_market_rates() if market is None else market
    dataset = load_dataset_rates() if dataset is None else dataset
    by_location = dataset["by_location"]
    if location in market and location in by_location and by_location[location]:
        return market[location] / by_location[location], location
    return market[CITY_KEY] / dataset["city"], f"{CITY_KEY} city average"


def project(amount, years, cagr):
    return amount * (1 + cagr) ** years


def project_band(amount, years, cagr=PROJECTION_CAGR):
    """A range, never a point. Four-year forecasts for one micro-market have
    error bars wider than the forecast, and the published growth rates disagree
    with each other badly (see README)."""
    low, high = min(cagr), max(cagr)
    return project(amount, years, low), project(amount, years, high)
