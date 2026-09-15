import argparse
import json
from datetime import date

import joblib
import pandas as pd

from src.config import DATASET_YEAR, METADATA_PATH, MODEL_PATH, PROJECTION_CAGR
from src.preprocess import normalize_location
from src.price_index import inflation_factor, project_band


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metadata(path=METADATA_PATH):
    if not path.exists():
        raise SystemExit(f"{path} is missing -- run `python -m src.train` first.")
    return json.loads(path.read_text())


def resolve_location(name, known):
    """Map a user-typed locality onto a label the model was actually trained on.

    Anything unknown has to become "Other" explicitly. Left alone it reaches
    OneHotEncoder(handle_unknown="ignore") and silently becomes an all-zeros
    row, which is a different thing from the "Other" bucket the model learned.
    """
    location = normalize_location(name) or "Other"
    return location if location in known else "Other"


def predict_price(location, bhk, total_sqft, bath=None, balcony=1,
                  area_type="Super built-up Area", model=None, known_locations=None):
    """Price in lakhs, at the price level of the training data (see DATASET_YEAR)."""
    if bhk not in (1, 2, 3):
        raise ValueError("v1 supports 1, 2, or 3 BHK only")
    if known_locations is None:
        known_locations = load_metadata()["locations"]
    if bath is None:
        bath = bhk
    row = pd.DataFrame(
        [
            {
                "total_sqft": float(total_sqft),
                "bhk": int(bhk),
                "bath": float(bath),
                "balcony": float(balcony),
                "area_type": area_type,
                "location": resolve_location(location, known_locations),
            }
        ]
    )
    model = load_model() if model is None else model
    return float(model.predict(row)[0])


def main():
    parser = argparse.ArgumentParser(description="Predict Bangalore flat price (lakhs)")
    parser.add_argument("--location", required=True)
    parser.add_argument("--bhk", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--sqft", type=float, required=True)
    parser.add_argument("--bath", type=float, default=None)
    parser.add_argument("--balcony", type=float, default=1)
    parser.add_argument("--project", type=int, metavar="YEAR",
                        help="also project to this year using the CAGR band in config")
    args = parser.parse_args()

    meta = load_metadata()
    resolved = resolve_location(args.location, meta["locations"])
    base = predict_price(args.location, args.bhk, args.sqft, args.bath, args.balcony,
                         known_locations=meta["locations"])
    factor, basis = inflation_factor(resolved)
    today_year = date.today().year
    today = base * factor

    def line(label, value, note=""):
        print(f"  {label:<36s} {value:>24s}   {note}".rstrip())

    interval = meta["interval"]
    today_low, today_high = today * interval["lo"], today * interval["hi"]
    coverage = 100 * interval["coverage"]

    print(f"\n{args.bhk} BHK, {args.sqft:.0f} sqft, {resolved}")
    if resolved != normalize_location(args.location):
        print(f'  ("{args.location}" is not in the training data -- priced as "Other")')
    line(f"model estimate, {DATASET_YEAR} price levels", f"Rs {base:.2f} lakh")
    line(f"x{factor:.2f} market drift to {today_year}", f"Rs {today:.2f} lakh", f"[via {basis}]")
    line(f"{coverage:.0f}% range, model error only",
         f"Rs {today_low:.2f} - {today_high:.2f} lakh")

    if args.project:
        years = args.project - today_year
        if years <= 0:
            raise SystemExit(f"--project must be after {today_year}")
        low, high = project_band(today, years)
        lo_pct, hi_pct = (100 * c for c in (min(PROJECTION_CAGR), max(PROJECTION_CAGR)))
        print()
        line(f"projected {args.project} @ {lo_pct:.0f}-{hi_pct:.0f}%/yr",
             f"Rs {low:.2f} - {high:.2f} lakh", "[growth assumptions only]")
        # Compounding the point estimate alone would show a tight band sitting on a
        # wide one. Carry the model error through as well.
        both_low, _ = project_band(today_low, years)
        _, both_high = project_band(today_high, years)
        line("...and with model error too",
             f"Rs {both_low:.2f} - {both_high:.2f} lakh")
        print("\n  The projection is a scenario, not a prediction. Published Bangalore")
        print("  growth rates disagree with each other by a factor of three -- see README.")
    print()


if __name__ == "__main__":
    main()
