from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "cleaned" / "wqi_next_sample_lags.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, parse_dates=["sample_date", "next_sample_date"])

FEATURES = ["previous_wqi", "wqi_two_samples_ago"]
TARGET = "next_sample_wqi"

# Sort by the date of the observation being predicted.
df = df.sort_values("next_sample_date").reset_index(drop=True)

# Use the latest 20% of target dates as a chronological test period.
unique_dates = np.sort(df["next_sample_date"].dropna().unique())

if len(unique_dates) < 5:
    raise ValueError("Not enough unique target dates for a chronological split.")

split_index = int(len(unique_dates) * 0.8)
split_index = min(max(split_index, 1), len(unique_dates) - 1)
test_start_date = unique_dates[split_index]

train = df[df["next_sample_date"] < test_start_date].copy()
test = df[df["next_sample_date"] >= test_start_date].copy()

if train.empty or test.empty:
    raise ValueError("Training or test set is empty.")

print("=" * 60)
print("BASELINE WQI FORECASTING")
print("=" * 60)
print("Training examples:", len(train))
print("Testing examples:", len(test))
print("Test period starts:", pd.Timestamp(test_start_date).date())
print("Training target dates:", train["next_sample_date"].nunique())
print("Testing target dates:", test["next_sample_date"].nunique())

# Baseline 1: persistence
test["persistence_prediction"] = test["previous_wqi"]

# Baseline 2: linear regression using only past WQI values
model = LinearRegression()
model.fit(train[FEATURES], train[TARGET])
test["linear_prediction"] = model.predict(test[FEATURES])

def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

results = []

for name, prediction_column in [
    ("Persistence", "persistence_prediction"),
    ("Linear Regression", "linear_prediction"),
]:
    metrics = calculate_metrics(
        test[TARGET],
        test[prediction_column],
    )
    results.append({
        "model": name,
        **metrics,
    })

results_df = pd.DataFrame(results)

print("\nOVERALL TEST RESULTS")
print(results_df.round(3).to_string(index=False))

# Station-level results help reveal whether performance varies by station.
station_results = []

for (river, station), group in test.groupby(["river", "station_id"]):
    if len(group) < 2:
        continue

    for name, prediction_column in [
        ("Persistence", "persistence_prediction"),
        ("Linear Regression", "linear_prediction"),
    ]:
        metrics = calculate_metrics(
            group[TARGET],
            group[prediction_column],
        )
        station_results.append({
            "river": river,
            "station_id": station,
            "model": name,
            "test_examples": len(group),
            **metrics,
        })

station_results_df = pd.DataFrame(station_results)

# Save predictions and evaluation results.
test.to_csv(OUTPUT_DIR / "baseline_wqi_test_predictions.csv", index=False)
results_df.to_csv(OUTPUT_DIR / "baseline_wqi_metrics.csv", index=False)
station_results_df.to_csv(
    OUTPUT_DIR / "baseline_wqi_station_metrics.csv",
    index=False,
)

print("\nSaved files:")
print(OUTPUT_DIR / "baseline_wqi_test_predictions.csv")
print(OUTPUT_DIR / "baseline_wqi_metrics.csv")
print(OUTPUT_DIR / "baseline_wqi_station_metrics.csv")

print("\nBASELINE EVALUATION COMPLETE")