# Bangalore House Price Prediction (HPPS)

Predicts **1 / 2 / 3 BHK** flat prices in Bengaluru (output in **lakhs**).

## Data we use (and why)

| Source | Use? | Why |
|---|---|---|
| [Bengaluru House price data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) (same CSV on [codebasics/py](https://github.com/codebasics/py/blob/master/DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv)) | **Yes — training data** | 13,320 unit listings with `location`, `size`, `total_sqft`, `bath`, `balcony`, `price`. Includes HSR, Electronic City Phase I/II, Bommasandra. |
| [Bangalore Property Listings 2026](https://www.kaggle.com/datasets/kaushal01railway/bangalore-property-listings-dataset-2026) | No | Project-level ads (`1.75 - 2.25 Cr`, `3, 4 BHK`), not one row per flat. |
| Karnataka RERA project dumps | No | Registrations, not sale prices for a 2 BHK. |
| `data/external/bangalore_market_rates.csv` | **Yes — level correction** | Present-day ₹/sqft, hand-collected, one source per row. Not training data — it only moves the *level* of an estimate (see below). |

Listings are from around 2017–2018, and that vintage is baked into every model estimate.
What the model learned about *relative* pricing — HSR above Whitefield above Electronic City —
is the durable part; the absolute rupee level is not, and is corrected separately.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Raw CSV is downloaded automatically on first train if `data/raw/Bengaluru_House_Data.csv` is missing.

## Run

```bash
python -m unittest discover tests -v
python -m src.train
python -m src.predict --location "Electronic City Phase II" --bhk 2 --sqft 1000 --project 2030
python -m src.classify
```

EDA notebook: `notebooks/01_eda_and_modeling.ipynb`

## Scope

**In:**

- BHK: 1, 2, 3 only. RK and 4+ BHK are dropped during parsing.
- Localities: aliases merged (HSR sectors, EC Phase I/II, Bommasandra). Rare names (`<10` rows) → `Other`. `Doddabommasandra` is **not** merged into Bommasandra — different neighbourhood, different prices.
- Target: **price per sqft**, multiplied back to a total at predict time.
- Outlier trim: ±3 std on price/sqft within a locality, applied to **training rows only**.
- Models: Linear Regression vs Random Forest, both measured against a median-rate baseline, selected by 5-fold CV.
- Two stages: the model prices flats relative to each other; `src/price_index.py` moves the level to today and projects forward.

**Out, deliberately:**

- **Forecast confidence.** The projection is compound arithmetic on published growth rates. There is no uncertainty model behind the band — it is two CAGR assumptions, not a prediction interval.
- **Anything the six features can't see:** floor, building age, furnishing, amenities, distance to metro or tech park. This is the accuracy ceiling, and no model change gets past it.
- **A current training set.** The listings stay 2017-18; only the level correction is current. Retraining on 2025-26 per-flat data would be the real v2.
- **Deployment.** No API, no container, no experiment tracker. On 8,500 rows they would be decoration.

## Test scores (80/20 split, 10,802 parsed rows)

| Model | MAE (lakhs) | RMSE | R² |
|---|---|---|---|
| Baseline — median rate × sqft | 23.25 | 40.71 | 0.407 |
| Linear Regression | 17.80 | 28.75 | 0.705 |
| **Random Forest (saved)** | **15.67** | 27.79 | 0.724 |

The model is selected by 5-fold CV on the training rows; the test set is scored once,
afterwards. Median error is about 16%, best in the 50–100 lakh band where the data is
dense and worst above 200 lakh where it is thin.

### Uncertainty

A single MAE is the wrong shape for reporting this: the error is right-skewed and grows
with price, so "± 15.7 lakh" is meaningless on a 28 lakh flat and far too tight on a
200 lakh one. Instead `train.py` calibrates **error ratios** on out-of-fold predictions
over the training rows — never the test set — and `predict.py` turns them into a range:

```
an estimate of X is really 0.73X to 1.33X, 80% of the time
```

Checked against the held-out test set, that interval covers **79.4%** of real prices
against a 80% target. `tests/test_price_index.py` fails if those two drift apart.

### Why these numbers are worse than they used to be

An earlier version of this README claimed MAE 9.97 / R² 0.837. That number was not real.
`drop_pps_outliers` ran *before* `train_test_split`, so it (a) computed each locality's
price/sqft mean and std using test prices and (b) deleted the hardest 23% of listings from
the test set as well as the training set. Scored on a test set that is never filtered, the
same model gets MAE ≈ 16. Nothing got worse; the measurement got honest.

### What didn't work

Measured on the same fixed, never-filtered test set. Recorded so nobody re-runs them:

| Change | MAE | |
|---|---|---|
| ±1 std outlier trim → ±3 std | 16.93 → 15.97 | ✅ kept — trimming at 1 std throws away 23% of the data and *hurts* |
| price → price-per-sqft target | 15.96 → 15.67 | ✅ kept |
| HistGradientBoosting, tuned | 16.39 | ❌ worse than plain Random Forest |
| `sqft_per_bhk`, `bath_per_bhk` features | 16.40 | ❌ no effect |
| `TargetEncoder` on location | 16.94 | ❌ worse |
| `log1p` target | 16.61 | ❌ worse |
| Adding the unused `availability` column | 15.87 | ❌ noise |
| `min_samples_leaf=5` (to shrink the 124 MB model) | 17.69 | ❌ costs 2 lakh of accuracy |

The ceiling here is the data, not the algorithm: 6 features, 205 localities, a median of
20 training rows each. Two flats with the same sqft, BHK and locality can legitimately
differ by 30 lakh and the model has no column that can tell them apart. Getting meaningfully
better means more features (floor, age, furnishing, distance to metro), not more models.

## The classification task

Everything above is regression — *what is this worth*. `src/classify.py` answers a different
question: **is this flat priced above the typical rate for its locality?** It exists because
MAE and R² cannot teach precision, recall, or where to put a decision threshold.

```bash
python -m src.classify
```

| Model | Accuracy | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Baseline — always guess the bigger class | 0.526 | 0.500 | — | — |
| Logistic Regression | 0.639 | 0.663 | 0.660 | 0.493 |
| **Random Forest (saved)** | **0.662** | **0.723** | 0.669 | 0.571 |

```
Confusion matrix (rows=actual, cols=predicted):
                      pred below   pred above
  actual at/below median      846          290
  actual above median         440          585
```

Of 1,025 genuinely above-median flats it catches 585 and misses 440 — recall 0.57 against
precision 0.67. Tightening the threshold catches more and raises more false alarms. That
tradeoff is the entire point of the exercise; there is no equivalent knob in regression.

**The trap here is the 9.97 bug wearing different clothes.** The label is derived from
`price`, so two rules are non-negotiable: `price` is never a feature, and the locality
medians that define the label come from *training* rows only — a test flat is judged against
the training threshold, never its own. `tests/test_classify.py` fails if that changes.

The protocol is written out again in `classify.py` rather than shared with `protocol.py`.
Classification needs different metrics, a different baseline and no ratio interval; bending
one abstraction around both task types would make each harder to read. Seeing split-first /
baseline / CV-select / score-once appear a second time is deliberate.

## The 2017-18 problem, and the second stage

Every listing here is from about 2017-18, and there is no time column anywhere in the raw
CSV — `availability` holds possession dates, not listing dates. So the model is *cross-sectional*:
it learned what makes flat A cost more than flat B at one frozen moment. It cannot be tuned into
answering "what will this cost in 2030", because there is no time variable to advance.

`src/price_index.py` is the second stage. The model's relative pricing is the durable part;
the price *level* is corrected separately using published market rates:

```
predicted price = model(features)                    # 2017-18 price level
                × market_rate_now / dataset_rate     # level correction
                × (1 + growth) ** years              # forward projection
```

```
$ python -m src.predict --location "Electronic City Phase II" --bhk 2 --sqft 1000 --project 2030

2 BHK, 1000 sqft, Electronic City Phase II
  model estimate, 2018 price levels               Rs 27.83 lakh
  x2.61 market drift to 2026                      Rs 72.67 lakh   [via Electronic City Phase II]
  80% range, model error only             Rs 52.91 - 96.64 lakh

  projected 2030 @ 8-10%/yr              Rs 98.86 - 106.39 lakh   [growth assumptions only]
  ...and with model error too            Rs 71.99 - 141.49 lakh
```

Sanity check: the ₹72.67 lakh figure lands next to the ~₹70.2 lakh that portals quote for an
average 2 BHK there today, having started from a model that thought it was worth ₹27.83 lakh.

The last two lines are deliberately both there. Compounding only the point estimate produces a
reassuringly tight 2030 band that sits on top of a range twice its width — so the second line
carries the model's own error through the projection. That is the honest number, and it is wide.

**Treat the projection as a scenario, not a forecast.** The published growth rates disagree
badly — 99acres reports Electronic City Phase 2 at +5.4% over one year but +127.9% over three
and +61.5% over five, which is arithmetically impossible, while RBI's Housing Price Index implies
a far lower 1–9% decade CAGR than any property portal. Rates live in
`data/external/bangalore_market_rates.csv`, one source per row; refresh them by hand.

## Dashboard

```bash
streamlit run app.py
```

Four tabs: **Areas** (all 205 localities, 2018 vs 2026 rate, drift), **Estimate** (the CLI
with sliders, plus an asking-price check), **Model report** (the classifier's report card),
**How this works** (the honest scores, the 9.97 story, the what-didn't-work table).

The Areas tab carries a **Basis** column, and it matters. Only 18 of 205 localities have a
2026 rate anyone published; the rest apply the city-wide drift factor. At the CLI that
is visible in the `[via ...]` tag, but a 205-row table makes every row look equally
researched. The column, the greyed-out styling and the counter above the table all exist to
stop the UI from overclaiming.

**Asking-price check.** Enter a real listing price on the Estimate tab and it is compared
against the calibrated 80% range rather than a point estimate:

```
Rs 130.0 lakh is ABOVE the model's range (Rs 52.9 - 96.6 lakh), by 33.4 lakh. That is +79%.
Rs 180.0 lakh is inside the model's range (Rs 135.8 - 248.0 lakh), -3% against the midpoint.
```

Because the range covers 80% of comparable flats, roughly 1 in 5 genuine listings falls
outside it through ordinary variation. Outside means *worth a question*, not *wrong*, and
the UI says so.

**Why the classifier is a report card, not a tool.** Its inputs are the six physical features
— there is no asking price among them. So it cannot judge whether a *specific listing* is
overpriced; it only describes what flats of that type tend to do, and at 66% accuracy about
one call in three is wrong. A tab labelled "is this flat overpriced?" would overclaim exactly
the way an unlabelled Areas table would. The asking-price check above is the honest version
of that question — and it needs no classifier at all.

**This is not a property portal.** It lists no flats for sale — the dataset is anonymised,
with no addresses and no current availability.

## Layout

| Path | |
|---|---|
| `src/preprocess.py` | row-local parsing only — nothing here looks at another row's price |
| `src/model.py` | the estimator: pipeline, imputer, per-sqft wrapper |
| `src/protocol.py` | **the honest protocol, written once** — split, CV, calibrate, test once |
| `src/train.py` | a thin caller: loads data, names the columns, hands off to the protocol |
| `src/predict.py` | resolves localities against the trained vocabulary, applies the index |
| `src/classify.py` | the classification task — its own protocol, on purpose |
| `src/price_index.py` | level correction and forward projection (arithmetic, not learned) |
| `app.py` | the Streamlit dashboard |
| `models/metadata.json` | what the saved `.pkl` was trained on, so it can't silently drift |
