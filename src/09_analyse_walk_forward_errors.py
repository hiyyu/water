
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "evaluation" / "walk_forward_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

df = pd.read_csv(INPUT_FILE, parse_dates=["next_sample_date"])

models = {
    "Persistence": "persistence_prediction",
    "Linear Regression": "linear_prediction",
}

def calculate_metrics(group, prediction_column):
    actual = group["actual_wqi"]
    predicted = group[prediction_column]

    return {
        "examples": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "mean_error": (predicted - actual).mean(),
    }

# Results by river
river_results = []

for river, group in df.groupby("river"):
    for model_name, prediction_column in models.items():
        river_results.append({
            "river": river,
            "model": model_name,
            **calculate_metrics(group, prediction_column),
        })

river_df = pd.DataFrame(river_results)

# Results by station
station_results = []

for (river, station), group in df.groupby(["river", "station_id"]):
    for model_name, prediction_column in models.items():
        station_results.append({
            "river": river,
            "station_id": station,
            "model": model_name,
            **calculate_metrics(group, prediction_column),
        })

station_df = pd.DataFrame(station_results)

river_path = OUTPUT_DIR / "walk_forward_metrics_by_river.csv"
station_path = OUTPUT_DIR / "walk_forward_metrics_by_station.csv"

river_df.to_csv(river_path, index=False)
station_df.to_csv(station_path, index=False)

print("=" * 60)
print("WALK-FORWARD ERROR ANALYSIS")
print("=" * 60)

print("\nRESULTS BY RIVER")
print(river_df.round(3).to_string(index=False))

print("\nRESULTS BY STATION")
print(station_df.round(3).to_string(index=False))

print("\nMEANING OF MEAN ERROR")
print("Positive: predictions tend to be too high.")
print("Negative: predictions tend to be too low.")

print("\nSaved files:")
print(river_path)
print(station_path)

print("\nERROR ANALYSIS COMPLETE")