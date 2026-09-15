"""Train the sale-price model. The protocol itself lives in src/protocol.py."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src import protocol
from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_YEAR,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
)
from src.data_loader import load_raw_data
from src.model import FEATURES
from src.preprocess import clean_dataframe


def candidates():
    return {
        # The bar to clear: guess the same rate every time.
        "Baseline(median)": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def train():
    raw = load_raw_data()
    data = clean_dataframe(raw)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Rows: {len(raw)} raw -> {len(data)} parsed")

    # Median rupees/sqft per locality, used by price_index.py to move an estimate
    # from the dataset's 2017-18 price level to the present day.
    pps = data[TARGET] * 100_000 / data["total_sqft"]

    return protocol.run(
        data,
        features=FEATURES,
        target=TARGET,
        candidates=candidates(),
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        numeric=NUMERIC_FEATURES,
        categorical=CATEGORICAL_FEATURES,
        size_col="total_sqft",
        unit="lakhs",
        extra_metadata={
            "dataset_year": DATASET_YEAR,
            "raw_rows": len(raw),
            "rate_per_sqft": {
                "city": round(float(pps.median()), 1),
                "by_location": {
                    loc: round(float(pps[data["location"] == loc].median()), 1)
                    for loc in data["location"].unique()
                },
            },
        },
    )


if __name__ == "__main__":
    train()
