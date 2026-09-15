"""The honest training protocol, written once.

The sale-price model runs through here. The protocol is a reusable template,
not something to re-derive per dataset.

    split FIRST
      -> trim outliers on TRAINING rows only
      -> select the model by cross-validation on TRAINING rows
      -> calibrate the uncertainty interval out-of-fold
      -> score the test set exactly once, at the end

Every one of those steps is here because the first version of this project got it
wrong. The outlier trim ran before the split, so it both leaked test prices into
the filter and deleted the hardest 23% of the test set. That reported MAE 9.97.
The honest number is 15.67.
"""

import json
from datetime import date

import joblib
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from src.config import COVERAGE, CV_FOLDS, OUTLIER_N_STD, RANDOM_STATE, TEST_SIZE
from src.model import build_pipeline
from src.preprocess import drop_pps_outliers


def run(
    data,
    *,
    features,
    target,
    candidates,
    model_path,
    metadata_path,
    numeric,
    categorical,
    size_col="total_sqft",
    unit="",
    extra_metadata=None,
):
    """Train, select, calibrate and report. Returns the fitted winner."""

    def make_pipe(estimator):
        return build_pipeline(estimator, numeric, categorical, size_col)

    # Split FIRST. Everything learned from data happens on the left side of this line.
    train_df, test_df = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    kept = drop_pps_outliers(train_df, OUTLIER_N_STD, target=target, size=size_col)
    print(
        f"Train {len(train_df)} -> {len(kept)} after +/-{OUTLIER_N_STD} std trim  |  "
        f"Test {len(test_df)} (never filtered)"
    )

    X_train, y_train = kept[features], kept[target]
    X_test, y_test = test_df[features], test_df[target]

    # --- selection: cross-validation on training rows only ---
    print(f"\n{CV_FOLDS}-fold CV on training rows (this is what picks the model):")
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = {}
    for name, estimator in candidates.items():
        scores = -cross_val_score(
            make_pipe(estimator),
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
        )
        cv_mae[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {name:18s} MAE {scores.mean():6.2f} +/- {scores.std():.2f}")

    best_name = min(cv_mae, key=lambda n: cv_mae[n]["mean"])
    print(f"  -> selected {best_name}")

    interval = calibrate_interval(
        make_pipe(candidates[best_name]), X_train, y_train, cv
    )
    print(
        f"\nUncertainty, calibrated out-of-fold: an estimate of X is really "
        f"{interval['lo']:.2f}X to {interval['hi']:.2f}X, {100 * COVERAGE:.0f}% of the time"
    )

    # --- reporting: the test set is scored once, after selection ---
    print("\nHeld-out test set (not used for selection):")
    test_metrics = {}
    best_pipe = best_pred = None
    for name, estimator in candidates.items():
        pipe = make_pipe(estimator).fit(X_train, y_train)
        metrics, pred = evaluate(name, pipe, X_test, y_test, unit)
        test_metrics[name] = metrics
        if name == best_name:
            best_pipe, best_pred = pipe, pred

    # Does the out-of-fold interval actually hold up on data nobody calibrated on?
    inside = ((y_test >= best_pred * interval["lo"]) & (y_test <= best_pred * interval["hi"])).mean()
    print(f"\nInterval covers {100 * inside:.1f}% of test rows (aiming for {100 * COVERAGE:.0f}%)")
    interval["observed_coverage_on_test"] = round(float(inside), 4)

    print_slice_mae(X_test, y_test, best_pred, unit)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "trained_on": date.today().isoformat(),
                "sklearn_version": sklearn.__version__,
                "selected_model": best_name,
                "selected_by": f"lowest {CV_FOLDS}-fold CV MAE on training rows",
                "outlier_n_std": OUTLIER_N_STD,
                "target": target,
                "features": features,
                "rows": {
                    "input": len(data),
                    "train": len(kept),
                    "test": len(test_df),
                },
                # predict.py maps anything outside this list to "Other".
                "locations": sorted(data["location"].unique()),
                "cv_mae": cv_mae,
                "test_metrics": test_metrics,
                # Multipliers predict.py applies to turn a point estimate into a range.
                "interval": interval,
                **(extra_metadata or {}),
            },
            indent=2,
        )
    )
    print(f"\nSaved {best_name} -> {model_path}")
    print(f"Wrote {metadata_path}")
    return best_pipe


def calibrate_interval(pipeline, X_train, y_train, cv):
    """How wrong the model usually is, as a multiplier on its own estimate.

    A single number like "MAE 15.7 lakh" is the wrong shape for this: the error
    is right-skewed and scales with the target, so +/- 15.7 is meaningless on a
    28 lakh flat and too tight on a 200 lakh one. Ratios handle both.

    Calibrated on out-of-fold predictions over the training rows, never the test
    set -- the test set already has one job.
    """
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv)
    ratios = y_train.to_numpy() / oof
    tail = 100 * (1 - COVERAGE) / 2
    return {
        "lo": float(np.percentile(ratios, tail)),
        "hi": float(np.percentile(ratios, 100 - tail)),
        "coverage": COVERAGE,
        "calibrated_on": "out-of-fold predictions over training rows",
    }


def evaluate(name, model, X_test, y_test, unit=""):
    pred = model.predict(X_test)
    metrics = {
        "model": name,
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "r2": r2_score(y_test, pred),
    }
    print(
        f"{name}: MAE={metrics['mae']:.2f} {unit}  "
        f"RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}"
    )
    return metrics, pred


def print_slice_mae(X_test, y_test, pred, unit=""):
    frame = X_test.copy()
    frame["actual"] = y_test.values
    frame["pred"] = pred
    print("\nMAE by BHK:")
    for bhk, part in frame.groupby("bhk"):
        print(f"  {int(bhk)} BHK: {mean_absolute_error(part['actual'], part['pred']):.2f} {unit} (n={len(part)})")
    print("MAE by locality (n>=8):")
    for loc, part in frame.groupby("location"):
        if len(part) < 8:
            continue
        print(f"  {loc}: {mean_absolute_error(part['actual'], part['pred']):.2f} {unit} (n={len(part)})")
