"""Classification: is a flat priced above the typical rate for its locality?

The regression model answers "what is this worth". This answers a different
question -- "is this listing expensive for where it is" -- and it exists because
regression metrics cannot teach precision, recall, or a decision threshold.

The protocol is deliberately written out again rather than shared with
src/protocol.py. Classification needs different metrics, a different baseline and
no ratio interval, and bending one abstraction around both task types would make
each harder to read. Seeing split-first / baseline / CV-select / score-once
appear a second time is the point.

    THE TRAP IN THIS TASK: the label is derived from `price`, so
      1. price must never be a feature, and
      2. the locality medians that define the label must be computed from
         TRAINING rows only.
    Get (2) wrong and accuracy inflates exactly the way MAE did at 9.97.
"""

import json
from datetime import date

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.config import (
    CLF_METADATA_PATH,
    CLF_MODEL_PATH,
    CV_FOLDS,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.model import FEATURES, build_preprocessor
from src.preprocess import clean_dataframe
from sklearn.pipeline import Pipeline

LABELS = ["at/below median", "above median"]


def price_per_sqft(df):
    return df[TARGET] * 100_000 / df["total_sqft"]


def locality_medians(train_df):
    """Thresholds that define the label. TRAINING rows only -- see module docstring."""
    pps = price_per_sqft(train_df)
    return pps.groupby(train_df["location"]).median(), pps.median()


def label_above_median(df, medians, fallback):
    """1 if this flat's price/sqft beats its locality's median, else 0.

    Test rows are labelled with the *training* thresholds, never their own.
    """
    threshold = df["location"].map(medians).fillna(fallback)
    return (price_per_sqft(df) > threshold).astype(int)


def candidates():
    return {
        # The bar to clear: always guess the bigger class.
        "Baseline(most frequent)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def build_pipeline(estimator):
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def train():
    data = clean_dataframe(load_raw_data())

    # Split FIRST, then derive the label. Doing it the other way round leaks.
    train_df, test_df = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    medians, fallback = locality_medians(train_df)
    y_train = label_above_median(train_df, medians, fallback)
    y_test = label_above_median(test_df, medians, fallback)
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]

    print(f"Rows: {len(data)} parsed | train {len(train_df)} | test {len(test_df)}")
    print(
        f"Class balance: train {y_train.mean():.1%} above median | "
        f"test {y_test.mean():.1%} (roughly even, so accuracy is readable)"
    )

    # --- selection: CV on training rows, scored by ROC-AUC ---
    # AUC, not accuracy: it judges the ranking rather than one arbitrary
    # 0.5 threshold, which is the thing we actually care about here.
    print(f"\n{CV_FOLDS}-fold CV on training rows (this is what picks the model):")
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = {}
    for name, estimator in candidates().items():
        scores = cross_val_score(
            build_pipeline(estimator), X_train, y_train, cv=cv, scoring="roc_auc"
        )
        cv_auc[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {name:24s} ROC-AUC {scores.mean():.3f} +/- {scores.std():.3f}")

    best_name = max(cv_auc, key=lambda n: cv_auc[n]["mean"])
    print(f"  -> selected {best_name}")

    # --- reporting: the test set is scored once, after selection ---
    print("\nHeld-out test set (not used for selection):")
    test_metrics = {}
    best_pipe = None
    for name, estimator in candidates().items():
        pipe = build_pipeline(estimator).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        test_metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        m = test_metrics[name]
        print(
            f"  {name:24s} acc={m['accuracy']:.3f}  AUC={m['roc_auc']:.3f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        )
        if name == best_name:
            best_pipe, best_pred = pipe, pred

    matrix = confusion_matrix(y_test, best_pred)
    print(f"\nConfusion matrix for {best_name} (rows=actual, cols=predicted):")
    print(f"                    {'pred below':>12s} {'pred above':>12s}")
    for name, row in zip(LABELS, matrix):
        print(f"  actual {name:16s} {row[0]:12d} {row[1]:12d}")
    print()
    print(classification_report(y_test, best_pred, target_names=LABELS, zero_division=0))
    tn, fp, fn, tp = matrix.ravel()
    print(
        f"Read it this way: of {tp + fn} genuinely above-median flats it catches {tp} "
        f"and misses {fn}.\nTightening the threshold would catch more, at the cost of "
        f"more false alarms than the {fp} it already raises."
    )

    CLF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, CLF_MODEL_PATH)
    CLF_METADATA_PATH.write_text(
        json.dumps(
            {
                "trained_on": date.today().isoformat(),
                "sklearn_version": sklearn.__version__,
                "task": "binary classification: price/sqft above locality median",
                "label_thresholds_from": "training rows only",
                "selected_model": best_name,
                "selected_by": f"highest {CV_FOLDS}-fold CV ROC-AUC on training rows",
                "features": FEATURES,
                "rows": {"train": len(train_df), "test": len(test_df)},
                "class_balance": {
                    "train_positive": round(float(y_train.mean()), 4),
                    "test_positive": round(float(y_test.mean()), 4),
                },
                "cv_roc_auc": cv_auc,
                "test_metrics": test_metrics,
                "confusion_matrix": {
                    "labels": LABELS,
                    "rows_actual_cols_predicted": matrix.tolist(),
                },
            },
            indent=2,
        )
    )
    print(f"\nSaved {best_name} -> {CLF_MODEL_PATH}")
    print(f"Wrote {CLF_METADATA_PATH}")
    return best_pipe


if __name__ == "__main__":
    train()
