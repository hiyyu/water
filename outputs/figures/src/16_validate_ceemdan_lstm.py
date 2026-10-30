"""
16_validate_ceemdan_lstm.py

Leakage-controlled walk-forward validation for CEEMDAN-LSTM.

Method:
1. Process each river station separately.
2. At each forecast point, use only observations available before
   the target observation.
3. Apply CEEMDAN to that historical prefix only.
4. Train an LSTM to forecast all decomposed components jointly.
5. Sum the predicted components to reconstruct predicted WQI.
6. Compare against a persistence baseline (next WQI = latest WQI).
7. Save predictions and evaluation metrics.

IMPORTANT:
The dataset contains only about 12 observations per station.
Results are preliminary and should not be interpreted as final
evidence of model generalisation.
"""

from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd

from PyEMD import CEEMDAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "outputs" / "combined_doe_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "ceemdan_lstm_walk_forward_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "ceemdan_lstm_walk_forward_metrics.json"
RIVER_METRICS_FILE = OUTPUT_DIR / "ceemdan_lstm_walk_forward_by_river.csv"

RANDOM_SEED = 42

# The LSTM uses the previous three observations of each component.
LOOKBACK = 3

# At least eight historical WQI observations are required before
# the first forecast is attempted.
MIN_TRAIN_POINTS = 8

# Maximum number of CEEMDAN IMFs, plus one residual component.
MAX_IMFS = 4
N_COMPONENTS = MAX_IMFS + 1

CEEMDAN_TRIALS = 5

# A small number of epochs is intentional because each station has
# very little training data. This does not eliminate overfitting.
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001

warnings.filterwarnings("ignore", category=FutureWarning)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}\n\n"
            "Check that outputs/combined_doe_dataset.csv exists."
        )

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    required_columns = ["SUNGAI", "SMP-DAT", "WQI"]

    # Prefer the newer station ID, with a fallback to the older ID.
    if "ID STN BARU" in df.columns:
        station_column = "ID STN BARU"
    elif "ID STN (2016)" in df.columns:
        station_column = "ID STN (2016)"
    else:
        raise ValueError(
            "Could not find either 'ID STN BARU' or 'ID STN (2016)'."
        )

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["SUNGAI"] = df["SUNGAI"].astype(str).str.strip()
    df["station_id"] = df[station_column].astype(str).str.strip()
    df["sample_date"] = pd.to_datetime(df["SMP-DAT"], errors="coerce")
    df["WQI"] = pd.to_numeric(df["WQI"], errors="coerce")

    df = df.dropna(
        subset=["SUNGAI", "station_id", "sample_date", "WQI"]
    ).copy()

    # Exclude blank-like station identifiers.
    df = df[
        df["station_id"].ne("")
        & df["station_id"].str.lower().ne("nan")
    ].copy()

    # WQI should be in the official 0-100 range.
    df = df[df["WQI"].between(0, 100)].copy()

    # One record per station and sampling date.
    df = (
        df.sort_values(["SUNGAI", "station_id", "sample_date"])
        .drop_duplicates(
            subset=["SUNGAI", "station_id", "sample_date"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid WQI observations remain after cleaning.")

    print(f"Loaded records: {len(df)}")
    print(f"Rivers: {df['SUNGAI'].nunique()}")
    print(f"Stations: {df['station_id'].nunique()}")
    print(
        "Date range:",
        df["sample_date"].min().date(),
        "to",
        df["sample_date"].max().date(),
    )

    return df


# ============================================================
# 3. CEEMDAN DECOMPOSITION
# ============================================================

def decompose_wqi(values):
    """
    Decompose one historical WQI sequence.

    The decomposition receives only the observations available
    at the current forecast origin. No future target value is passed
    into CEEMDAN.

    Returns a fixed-size matrix:
        [IMF 1, IMF 2, IMF 3, IMF 4, residual]
    """

    values = np.asarray(values, dtype=np.float64)

    decomposer = CEEMDAN(
        trials=CEEMDAN_TRIALS,
        parallel=False,
    )
    decomposer.noise_seed(RANDOM_SEED)

    imfs = decomposer.ceemdan(
        values,
        max_imf=MAX_IMFS,
    )

    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)

    # Ensure there are at most MAX_IMFS IMF rows.
    imfs = imfs[:MAX_IMFS]

    # Pad missing IMF rows with zeros to keep the model input shape fixed.
    padded_imfs = np.zeros(
        (MAX_IMFS, len(values)),
        dtype=np.float64,
    )

    if imfs.size:
        padded_imfs[:len(imfs)] = imfs

    # The residual ensures that the components sum back to the
    # original WQI sequence at each historical time step.
    residual = values - padded_imfs.sum(axis=0)

    components = np.vstack([padded_imfs, residual])

    # Convert to time-major shape: rows = observations,
    # columns = component values.
    return components.T


# ============================================================
# 4. BUILD LSTM TRAINING WINDOWS
# ============================================================

def create_sequences(scaled_components):
    X, y = [], []

    for i in range(LOOKBACK, len(scaled_components)):
        X.append(scaled_components[i - LOOKBACK:i])
        y.append(scaled_components[i])

    return np.asarray(X), np.asarray(y)


# ============================================================
# 5. TRAIN AND FORECAST ONE STEP
# ============================================================

def forecast_next_wqi(historical_wqi, river, station_id, target_date):
    """
    Forecast the next WQI using only historical_wqi.

    CEEMDAN is rerun for each forecast origin. A new model is trained
    for that origin, so later observations cannot enter the training
    data for an earlier prediction.
    """

    historical_wqi = np.asarray(historical_wqi, dtype=np.float64)

    if len(historical_wqi) < MIN_TRAIN_POINTS:
        return np.nan

    components = decompose_wqi(historical_wqi)

    scaler = StandardScaler()
    scaled_components = scaler.fit_transform(components)

    X, y = create_sequences(scaled_components)

    if len(X) < 2:
        return np.nan

    # Each training example contains previous component values and
    # predicts the next component vector.
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    model = Sequential([
        Input(shape=(LOOKBACK, N_COMPONENTS)),
        LSTM(16),
        Dense(8, activation="relu"),
        Dense(N_COMPONENTS),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, len(X)),
        shuffle=False,
        verbose=0,
    )

    # Predict the next decomposed component vector.
    last_window = scaled_components[-LOOKBACK:].reshape(
        1, LOOKBACK, N_COMPONENTS
    )

    scaled_prediction = model.predict(last_window, verbose=0)

    predicted_components = scaler.inverse_transform(
        scaled_prediction
    )[0]

    # Reconstruct the predicted WQI by summing all components.
    predicted_wqi = float(np.sum(predicted_components))

    # WQI is bounded from 0 to 100.
    predicted_wqi = float(np.clip(predicted_wqi, 0, 100))

    return predicted_wqi


# ============================================================
# 6. WALK-FORWARD VALIDATION
# ============================================================

def run_walk_forward_validation(df):
    predictions = []

    grouped = df.groupby(
        ["SUNGAI", "station_id"],
        sort=True,
    )

    total_stations = len(grouped)

    for station_number, ((river, station_id), station_df) in enumerate(
        grouped,
        start=1,
    ):
        station_df = (
            station_df.sort_values("sample_date")
            .reset_index(drop=True)
        )

        values = station_df["WQI"].to_numpy(dtype=float)

        print(
            f"\n[{station_number}/{total_stations}] "
            f"{river} / {station_id}: {len(values)} observations",
            flush=True,
        )

        if len(values) <= MIN_TRAIN_POINTS:
            print(
                "  Skipped: not enough observations for walk-forward "
                "evaluation.",
                flush=True,
            )
            continue

        # The first prediction uses observations 0 through 7 to predict
        # observation 8. Each subsequent fold adds one observed sample.
        for target_index in range(MIN_TRAIN_POINTS, len(station_df)):
            historical_values = values[:target_index]

            target_row = station_df.iloc[target_index]
            target_wqi = float(target_row["WQI"])
            target_date = target_row["sample_date"]

            previous_wqi = float(historical_values[-1])

            print(
                f"  Forecasting target "
                f"{target_date.date()} "
                f"(history={len(historical_values)})...",
                flush=True,
            )

            try:
                predicted_wqi = forecast_next_wqi(
                    historical_wqi=historical_values,
                    river=river,
                    station_id=station_id,
                    target_date=target_date,
                )

                error_message = ""

            except Exception as exc:
                # Preserve the failed fold in the output for transparency.
                predicted_wqi = np.nan
                error_message = f"{type(exc).__name__}: {exc}"

                print(
                    f"  WARNING: forecast failed: {error_message}",
                    flush=True,
                )

            predictions.append({
                "river": river,
                "station_id": station_id,
                "target_date": target_date,
                "training_observations": len(historical_values),
                "previous_wqi": previous_wqi,
                "actual_wqi": target_wqi,
                "predicted_wqi": predicted_wqi,
                "persistence_prediction": previous_wqi,
                "ceemdan_lstm_error": (
                    predicted_wqi - target_wqi
                    if np.isfinite(predicted_wqi)
                    else np.nan
                ),
                "persistence_error": previous_wqi - target_wqi,
                "forecast_error_message": error_message,
            })

    return pd.DataFrame(predictions)


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "n_predictions": 0,
            "MAE": None,
            "RMSE": None,
            "R2": None,
        }

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # R² is undefined when there is only one target or no target variance.
    r2 = (
        float(r2_score(actual, predicted))
        if len(actual) >= 2 and np.var(actual) > 0
        else None
    )

    return {
        "n_predictions": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(rmse),
        "R2": r2,
    }


def evaluate_and_save(predictions):
    if predictions.empty:
        raise RuntimeError(
            "No forecasts were generated. Check the dataset and "
            "minimum training requirements."
        )

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    valid_model = predictions.dropna(subset=["predicted_wqi"]).copy()

    model_metrics = calculate_metrics(
        valid_model["actual_wqi"],
        valid_model["predicted_wqi"],
    )

    baseline_metrics = calculate_metrics(
        predictions["actual_wqi"],
        predictions["persistence_prediction"],
    )

    # Calculate metrics separately for each river.
    river_results = []

    for river, river_df in predictions.groupby("river"):
        model_df = river_df.dropna(subset=["predicted_wqi"])

        river_results.append({
            "river": river,
            "ceemdan_lstm": calculate_metrics(
                model_df["actual_wqi"],
                model_df["predicted_wqi"],
            ),
            "persistence": calculate_metrics(
                river_df["actual_wqi"],
                river_df["persistence_prediction"],
            ),
        })

    # Save detailed per-river results as a flat CSV.
    flat_river_results = []

    for result in river_results:
        flat_river_results.append({
            "river": result["river"],
            "ceemdan_lstm_n": result["ceemdan_lstm"]["n_predictions"],
            "ceemdan_lstm_MAE": result["ceemdan_lstm"]["MAE"],
            "ceemdan_lstm_RMSE": result["ceemdan_lstm"]["RMSE"],
            "ceemdan_lstm_R2": result["ceemdan_lstm"]["R2"],
            "persistence_n": result["persistence"]["n_predictions"],
            "persistence_MAE": result["persistence"]["MAE"],
            "persistence_RMSE": result["persistence"]["RMSE"],
            "persistence_R2": result["persistence"]["R2"],
        })

    pd.DataFrame(flat_river_results).to_csv(
        RIVER_METRICS_FILE,
        index=False,
    )

    metrics = {
        "experiment": "Leakage-controlled station-wise walk-forward validation",
        "target": "Next observed WQI",
        "input_file": str(INPUT_FILE),
        "lookback": LOOKBACK,
        "minimum_training_observations": MIN_TRAIN_POINTS,
        "max_imfs": MAX_IMFS,
        "ceemdan_trials": CEEMDAN_TRIALS,
        "lstm_epochs": EPOCHS,
        "forecasting_method": (
            "CEEMDAN is recomputed on each station's historical prefix. "
            "A new LSTM is trained at each forecast origin."
        ),
        "model_metrics": model_metrics,
        "persistence_baseline_metrics": baseline_metrics,
        "failed_model_predictions": int(
            predictions["predicted_wqi"].isna().sum()
        ),
        "prediction_file": str(PREDICTIONS_FILE),
        "river_metrics_file": str(RIVER_METRICS_FILE),
        "limitations": [
            "Only a small number of observations are available per station.",
            "Each station-specific LSTM has very few training windows.",
            "CEEMDAN components may be unstable for short sequences.",
            "Results are preliminary and require evaluation on additional DOE data.",
            "This validation evaluates one-step-ahead forecasts of the next "
            "observed sample, not a fixed daily or monthly horizon.",
        ],
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 60)

    print("\nCEEMDAN-LSTM:")
    print(json.dumps(model_metrics, indent=4))

    print("\nPersistence baseline:")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nSaved files:")
    print(f"  Predictions: {PREDICTIONS_FILE}")
    print(f"  Metrics:     {METRICS_FILE}")
    print(f"  By river:    {RIVER_METRICS_FILE}")

    if model_metrics["MAE"] is not None and baseline_metrics["MAE"] is not None:
        if model_metrics["MAE"] < baseline_metrics["MAE"]:
            print(
                "\nPreliminary result: CEEMDAN-LSTM has lower MAE "
                "than persistence on this validation set."
            )
        else:
            print(
                "\nPreliminary result: CEEMDAN-LSTM does not beat "
                "the persistence baseline on MAE."
            )

    print(
        "\nReminder: compare all metrics, not MAE alone. "
        "Do not conclude that the complex model is better based on "
        "one metric or this small dataset."
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CEEMDAN-LSTM WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Output folder: {OUTPUT_DIR}")

    df = load_dataset()

    predictions = run_walk_forward_validation(df)

    if predictions.empty:
        raise RuntimeError(
            "No predictions were generated. Check station histories."
        )

    evaluate_and_save(predictions)


if __name__ == "__main__":
    main()