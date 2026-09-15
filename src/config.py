from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "Bengaluru_House_Data.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"
MARKET_RATES_PATH = ROOT / "data" / "external" / "bangalore_market_rates.csv"

# The classification task (src/classify.py): is a flat priced above the typical
# rate for its locality? Separate artifacts -- a different question, a different model.
CLF_MODEL_PATH = ROOT / "models" / "overpriced_classifier.pkl"
CLF_METADATA_PATH = ROOT / "models" / "overpriced_metadata.json"

# Same file as Kaggle amitabhajoy/bengaluru-house-price-data (~13,320 listings).
DATA_URL = (
    "https://raw.githubusercontent.com/codebasics/py/master/"
    "DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv"
)

ALLOWED_BHK = (1, 2, 3)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Width of the interval predict.py prints. 0.8 = the 10th-90th percentile of the
# model's own out-of-fold error, so ~80% of comparable flats land inside it.
COVERAGE = 0.8

# Listings are from ~2017-18, so every model estimate is a 2017-18 price level.
# src/price_index.py corrects for that; see data/external/bangalore_market_rates.csv.
DATASET_YEAR = 2018

# Forward projection band. Bengaluru consensus is 8-10% a year; RBI's decade-long
# HPI implies less, property portals imply more. Always quote the band, never a point.
PROJECTION_CAGR = (0.08, 0.10)

NUMERIC_FEATURES = ("total_sqft", "bhk", "bath", "balcony")
CATEGORICAL_FEATURES = ("area_type", "location")
TARGET = "price"

SQ_METER_TO_SQFT = 10.7639
SQ_YARD_TO_SQFT = 9.0
MIN_SQFT = 200
MAX_SQFT = 5000
MIN_SQFT_PER_BHK = 300
MIN_PRICE_LAKH = 8
MAX_PRICE_LAKH = 400
MIN_LOCATION_COUNT = 10

# Per-locality price/sqft outlier trim, applied to TRAINING ROWS ONLY (see train.py).
# Measured on a held-out, never-filtered test set: 1 std -> MAE 16.93, 3 std -> 15.97.
# Trimming hard at 1 std throws away 23% of the data and makes the model worse.
OUTLIER_N_STD = 3
