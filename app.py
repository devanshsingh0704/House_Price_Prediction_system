"""HPPS dashboard.

An area explorer and price estimator over the trained model -- deliberately not a
property portal. It lists no flats for sale, because the dataset is 13,320
anonymised 2017-18 listings with no address and no current availability.

Run:  streamlit run app.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CLF_METADATA_PATH,
    DATASET_YEAR,
    METADATA_PATH,
    MODEL_PATH,
    PROJECTION_CAGR,
)
from src.predict import predict_price, resolve_location
from src.price_index import inflation_factor, load_market_rates, project_band

st.set_page_config(page_title="HPPS — Bangalore flat prices", layout="wide")

THIS_YEAR = 2026


# Streamlit re-runs this script on every interaction and the model is 122 MB,
# so both loads have to be cached or the UI crawls.
@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_metadata():
    return json.loads(METADATA_PATH.read_text())


@st.cache_resource
def get_classifier_metadata():
    """Report-card data only. The classifier itself is never loaded here -- see
    the Model report tab for why it is not used to judge individual flats."""
    if not CLF_METADATA_PATH.exists():
        return None
    return json.loads(CLF_METADATA_PATH.read_text())


if not METADATA_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

meta = get_metadata()
rates = load_market_rates()
tabs = st.tabs(["Areas", "Estimate", "Model report", "How this works"])


# --------------------------------------------------------------------- Areas
with tabs[0]:
    st.header("Areas")
    st.caption(
        f"What the model saw in {DATASET_YEAR}, against today's market rate. "
        "Only a handful of localities have a rate anyone published — the rest are "
        "derived from the city-wide average, and are marked as such."
    )

    by_location = meta["rate_per_sqft"]["by_location"]
    rows = []
    for locality, then in sorted(by_location.items()):
        factor, basis = inflation_factor(locality, rates, meta["rate_per_sqft"])
        measured = basis == locality
        rows.append(
            {
                "Locality": locality,
                f"{DATASET_YEAR} Rs/sqft": round(then),
                f"{THIS_YEAR} Rs/sqft": round(then * factor),
                "Drift": f"x{factor:.2f}",
                "Basis": "measured" if measured else "city average",
            }
        )
    areas = pd.DataFrame(rows)

    measured_count = int((areas["Basis"] == "measured").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Localities", len(areas))
    c2.metric("With a measured 2026 rate", measured_count)
    c3.metric("Derived from city average", len(areas) - measured_count)

    if st.checkbox("Show only localities with a measured rate"):
        areas = areas[areas["Basis"] == "measured"]

    st.dataframe(
        areas.style.map(
            lambda v: "color: #999" if v == "city average" else "", subset=["Basis"]
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        f"**Read the Basis column.** {measured_count} of {len(by_location)} localities have "
        "a rate sourced from a property portal. The other "
        f"{len(by_location) - measured_count} apply the city-wide drift factor, which says "
        "nothing specific about that locality. A table makes all 205 look equally "
        "researched; they are not."
    )


# ------------------------------------------------------------------ Estimate
with tabs[1]:
    st.header("Estimate")
    c1, c2 = st.columns([1, 2])

    with c1:
        locality = st.selectbox("Locality", sorted(meta["locations"]), index=None,
                                placeholder="Type to search...")
        bhk = st.radio("BHK", [1, 2, 3], index=1, horizontal=True)
        sqft = st.number_input("Built-up area (sqft)", 200, 5000, 1000, step=50)
        bath = st.number_input("Bathrooms", 1, 5, bhk)
        balcony = st.number_input("Balconies", 0, 5, 1)
        project_to = st.slider("Project to year", THIS_YEAR + 1, THIS_YEAR + 10, 2030)
        asking = st.number_input(
            "Asking price (lakhs, optional)", 0.0, 2000.0, 0.0, step=5.0,
            help="Enter a real listing price to see whether it falls inside the "
                 "model's range. Leave at 0 to skip.",
        )

    with c2:
        if not locality:
            st.info("Pick a locality to see an estimate.")
        else:
            resolved = resolve_location(locality, meta["locations"])
            base = predict_price(locality, bhk, sqft, bath, balcony,
                                 model=get_model(), known_locations=meta["locations"])
            factor, basis = inflation_factor(resolved, rates, meta["rate_per_sqft"])
            today = base * factor
            iv = meta["interval"]
            low, high = today * iv["lo"], today * iv["hi"]

            st.metric(f"Estimated value, {THIS_YEAR}", f"Rs {today:,.1f} lakh",
                      help=f"Model said Rs {base:,.1f} lakh at {DATASET_YEAR} prices, "
                           f"multiplied by {factor:.2f}")
            st.caption(
                f"**{iv['coverage']:.0%} range: Rs {low:,.1f} - {high:,.1f} lakh.** "
                f"The single number above is the midpoint of that, and the range is the "
                f"honest answer — this model's typical error is about 16%."
            )
            if basis != resolved:
                st.warning(f"No published 2026 rate for {resolved}. Using the "
                           f"city-wide drift factor, so treat this as approximate.")

            if asking > 0:
                st.subheader("Against the asking price")
                gap = 100 * (asking - today) / today
                if asking > high:
                    st.error(
                        f"**Rs {asking:,.1f} lakh is ABOVE the model's range** "
                        f"(Rs {low:,.1f} - {high:,.1f} lakh), by {asking - high:,.1f} lakh. "
                        f"That is {gap:+.0f}% against the midpoint."
                    )
                elif asking < low:
                    st.success(
                        f"**Rs {asking:,.1f} lakh is BELOW the model's range** "
                        f"(Rs {low:,.1f} - {high:,.1f} lakh), by {low - asking:,.1f} lakh. "
                        f"That is {gap:+.0f}% against the midpoint. Worth asking why."
                    )
                else:
                    st.info(
                        f"**Rs {asking:,.1f} lakh is inside the model's range** "
                        f"(Rs {low:,.1f} - {high:,.1f} lakh), {gap:+.0f}% against the midpoint. "
                        f"Unremarkable for this size and locality."
                    )
                st.caption(
                    "The range covers 80% of comparable flats, so roughly 1 in 5 genuine "
                    "listings falls outside it through nothing but ordinary variation. "
                    "Outside the range means *worth a question*, not *wrong*."
                )

            years = project_to - THIS_YEAR
            p_low, p_high = project_band(today, years)
            b_low, _ = project_band(low, years)
            _, b_high = project_band(high, years)
            lo_pct, hi_pct = (100 * c for c in (min(PROJECTION_CAGR), max(PROJECTION_CAGR)))

            st.subheader(f"Projected to {project_to}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Assumption": f"growth only ({lo_pct:.0f}-{hi_pct:.0f}%/yr)",
                         "Range": f"Rs {p_low:,.1f} - {p_high:,.1f} lakh"},
                        {"Assumption": "growth + model error",
                         "Range": f"Rs {b_low:,.1f} - {b_high:,.1f} lakh"},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.warning(
                "**This is a scenario, not a forecast.** It is compound arithmetic on "
                "published growth rates, with no uncertainty model behind it. Those "
                "published rates contradict each other badly — see the last tab."
            )


# -------------------------------------------------------- Model report (clf)
with tabs[2]:
    st.header("Model report — classification")
    clf_meta = get_classifier_metadata()

    if clf_meta is None:
        st.info("No classifier trained yet. Run `python -m src.classify`.")
    else:
        st.caption(
            "A second, separate task: **is a flat priced above the typical rate for its "
            "locality?** It exists because MAE and R² cannot teach precision, recall or "
            "where to put a decision threshold."
        )
        st.warning(
            "**This is a report card, not a tool — and the distinction matters.** The "
            f"classifier's inputs are `{'`, `'.join(clf_meta['features'])}`. There is no "
            "asking price among them, so it cannot judge whether a *specific listing* is "
            "overpriced; it only describes what flats of this type tend to do. For an "
            "actual listing, use the asking-price box on the **Estimate** tab, which "
            "compares a real price against the regression model's range."
        )

        scores = pd.DataFrame(clf_meta["test_metrics"]).T
        scores.columns = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"]
        st.dataframe(scores.style.format("{:.3f}"), width="stretch")
        st.caption(
            f"Selected by {clf_meta['selected_by']}. Classes are close to even "
            f"({clf_meta['class_balance']['train_positive']:.1%} positive in training), so "
            "accuracy is readable here — it would not be on an imbalanced problem."
        )

        st.subheader("Confusion matrix")
        cm = clf_meta["confusion_matrix"]
        labels = cm["labels"]
        matrix = pd.DataFrame(
            cm["rows_actual_cols_predicted"],
            index=[f"actual: {n}" for n in labels],
            columns=[f"predicted: {n}" for n in labels],
        )
        st.dataframe(matrix, width="stretch")

        (tn, fp), (fn, tp) = cm["rows_actual_cols_predicted"]
        best = clf_meta["test_metrics"][clf_meta["selected_model"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Caught", f"{tp} of {tp + fn}", help="recall — genuinely above-median flats it found")
        c2.metric("Missed", f"{fn}", help="false negatives")
        c3.metric("False alarms", f"{fp}", help="called above-median but weren't")
        st.write(
            f"Precision {best['precision']:.2f}, recall {best['recall']:.2f}. Lowering the "
            f"decision threshold would catch more of the {fn} it misses, at the cost of more "
            f"than the {fp} false alarms it already raises. **That tradeoff is the whole "
            "reason this task is here** — there is no equivalent knob in regression."
        )
        st.caption(
            f"Overall accuracy {best['accuracy']:.1%}, so roughly one call in three is wrong. "
            "Treat individual predictions as weak evidence."
        )

        st.subheader("The trap in this task")
        st.write(
            "The label is derived from `price`, which makes it the MAE-9.97 bug in different "
            "clothes. Two rules hold it off: `price` is never a feature, and the locality "
            f"medians defining the label come from **{clf_meta['label_thresholds_from']}** — a "
            "test flat is judged against the training threshold, never its own. "
            "`tests/test_classify.py` fails if either changes."
        )


# -------------------------------------------------------------- How it works
with tabs[3]:
    st.header("How this works")
    st.subheader("Two stages")
    st.code(
        "predicted price = model(features)                 # 2017-18 price level\n"
        "                x market_rate_now / dataset_rate  # level correction\n"
        "                x (1 + growth) ** years           # forward projection",
        language="text",
    )
    st.write(
        f"The model is trained on ~13,320 listings from about {DATASET_YEAR}, and the raw "
        "data has **no time column at all** — every row is from one frozen moment. So it is "
        "a cross-sectional model: good at saying flat A costs more than flat B, structurally "
        "incapable of forecasting. Stage two handles time separately."
    )

    st.subheader("Honest scores")
    scores = pd.DataFrame(meta["test_metrics"]).T[["mae", "rmse", "r2"]]
    scores.columns = ["MAE (lakhs)", "RMSE", "R2"]
    st.dataframe(scores.style.format("{:.2f}"), width="stretch")
    st.caption(
        f"Selected by {meta['selected_by']}; the test set is scored once, afterwards. "
        f"The interval covers {meta['interval']['observed_coverage_on_test']:.1%} of held-out "
        f"prices against a {meta['interval']['coverage']:.0%} target."
    )

    st.subheader("An earlier version claimed MAE 9.97")
    st.write(
        "That number was not real. The outlier filter ran *before* the train/test split, so "
        "it computed each locality's mean using test prices **and** deleted the hardest 23% "
        "of listings from the test set. Scored on a test set that is never filtered, the same "
        "model gets 15.67. Nothing got worse; the measurement got honest."
    )

    st.subheader("What didn't work")
    st.dataframe(
        pd.DataFrame(
            [
                ("+/-1 std outlier trim -> +/-3 std", "16.93 -> 15.97", "kept"),
                ("price -> price-per-sqft target", "15.96 -> 15.67", "kept"),
                ("HistGradientBoosting, tuned", "16.39", "worse"),
                ("sqft_per_bhk, bath_per_bhk features", "16.40", "no effect"),
                ("TargetEncoder on location", "16.94", "worse"),
                ("log1p target", "16.61", "worse"),
                ("adding the unused availability column", "15.87", "noise"),
                ("min_samples_leaf=5 (to shrink the model)", "17.69", "costs 2 lakh"),
            ],
            columns=["Change", "Test MAE", "Verdict"],
        ),
        width="stretch",
        hide_index=True,
    )
    st.write(
        "The ceiling is the data, not the algorithm: 6 features, "
        f"{len(meta['locations'])} localities, a median of 20 training rows each. Two flats "
        "with the same sqft, BHK and locality can legitimately differ by 30 lakh and the "
        "model has no column that can tell them apart."
    )

    st.subheader("What this is not")
    st.write(
        "- **Not a property portal.** It lists no flats for sale; the dataset is anonymised "
        "with no addresses and no current availability.\n"
        "- **Not investment advice.** The projections are scenarios on contradictory public "
        "growth rates. 99acres reports Electronic City Phase 2 at +5.4% over one year but "
        "+127.9% over three and +61.5% over five, which is arithmetically impossible, while "
        "RBI's Housing Price Index implies a far lower 1-9% decade CAGR than any portal."
    )
