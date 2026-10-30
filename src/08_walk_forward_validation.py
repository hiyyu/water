from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "cleaned" / "wqi_next_sample_lags.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["sample_date", "next_sample_date"]
)

FEATURES = ["previous_wqi", "wqi_two_samples_ago"]
TARGET = "next_sample_wqi"

df = df.dropna(
    subset=FEATURES + [TARGET, "next_sample_date"]
).copy()

df = df.sort_values("next_sample_date").reset_index(drop=True)

dates = np.sort(df["next_sample_date"].unique())

# Begin testing after the first 60% of distinct target dates.
initial_train_dates = int(len(dates) * 0.60)

if initial_train_dates < 2 or initial_train_dates >= len(dates):
    raise ValueError("Not enough dates for walk-forward validation.")

predictions = []

for date in dates[initial_train_dates:]:
    train = df[df["next_sample_date"] < date].copy()
    test = df[df["next_sample_date"] == date].copy()

    if train.empty or test.empty:
        continue

    # Persistence: the next WQI equals the latest observed WQI.
    persistence_pred = test["previous_wqi"].to_numpy()

    # Linear regression trained only on earlier target dates.
    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])
    linear_pred = model.predict(test[FEATURES])

    for row_index, (_, row) in enumerate(test.iterrows()):
        predictions.append({
            "river": row["river"],
            "station_id": row["station_id"],
            "sample_date": row["sample_date"],
            "next_sample_date": row["next_sample_date"],
            "actual_wqi": row[TARGET],
            "persistence_prediction": persistence_pred[row_index],
            "linear_prediction": linear_pred[row_index],
            "training_examples": len(train),
        })

pred_df = pd.DataFrame(predictions)

if pred_df.empty:
    raise ValueError("No walk-forward predictions were generated.")

def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2
            else np.nan
        ),
    }

summary = []

for model_name, column in [
    ("Persistence", "persistence_prediction"),
    ("Linear Regression", "linear_prediction"),
]:
    summary.append({
        "model": model_name,
        **metrics(pred_df["actual_wqi"], pred_df[column]),
    })

summary_df = pd.DataFrame(summary)

pred_path = OUTPUT_DIR / "walk_forward_predictions.csv"
metrics_path = OUTPUT_DIR / "walk_forward_metrics.csv"

pred_df.to_csv(pred_path, index=False)
summary_df.to_csv(metrics_path, index=False)

print("=" * 60)
print("EXPANDING-WINDOW WALK-FORWARD VALIDATION")
print("=" * 60)
print("Distinct target dates:", len(dates))
print("Initial training dates:", initial_train_dates)
print("Test dates evaluated:", pred_df["next_sample_date"].nunique())
print("Total predictions:", len(pred_df))
print("First test date:", pred_df["next_sample_date"].min().date())
print("Last test date:", pred_df["next_sample_date"].max().date())

print("\nOVERALL WALK-FORWARD RESULTS")
print(summary_df.round(3).to_string(index=False))

print("\nSaved files:")
print(pred_path)
print(metrics_path)

print("\nWALK-FORWARD VALIDATION COMPLETE")