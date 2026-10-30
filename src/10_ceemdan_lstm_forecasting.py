from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "cleaned" / "wqi_forecasting_base.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 3
MAX_IMFS = 4
COMPONENT_COUNT = MAX_IMFS + 1
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("EXPERIMENTAL CEEMDAN-LSTM WQI FORECASTING")
print("=" * 65)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["sample_date"]
)

df = df.dropna(
    subset=["sample_date", "river", "station_id", "wqi"]
).copy()

df["wqi"] = pd.to_numeric(df["wqi"], errors="coerce")
df = df.dropna(subset=["wqi"])

df = df.sort_values(
    ["river", "station_id", "sample_date"]
).reset_index(drop=True)

# Prepare each station's observed WQI series.
stations = []
component_records = []

for (river, station_id), group in df.groupby(
    ["river", "station_id"]
):
    group = group.sort_values("sample_date")

    values = group["wqi"].to_numpy(dtype=float)

    if len(values) < LOOKBACK + 3:
        print(f"Skipping {river} {station_id}: insufficient history")
        continue

    # Decompose the observed WQI series using CEEMDAN.
    print(f"  Decomposing {river} / {station_id}...", flush=True)

    decomposer = CEEMDAN(trials=5, parallel=False)
    decomposer.noise_seed(RANDOM_SEED)

    imfs = decomposer.ceemdan(
        values,
        max_imf=MAX_IMFS
    )

    print(f"  Decomposition complete: {station_id}", flush=True)

    # Keep a fixed number of components for consistent processing.
    components = np.zeros((COMPONENT_COUNT, len(values)))

    number_of_imfs = min(len(imfs), MAX_IMFS)

    if number_of_imfs > 0:
        components[:number_of_imfs] = imfs[:number_of_imfs]

    # The remaining signal is represented by the residual.
    components[-1] = values - components[:-1].sum(axis=0)

    station_key = (river, station_id)

    stations.append({
        "river": river,
        "station_id": station_id,
        "dates": group["sample_date"].to_numpy(),
        "values": values,
        "components": components,
    })

    for component_index in range(COMPONENT_COUNT):
        component_records.append({
            "station_key": station_key,
            "component_index": component_index,
            "values": components[component_index].copy(),
        })

print("Stations prepared:", len(stations))
print("Components per station:", COMPONENT_COUNT)

# Train one shared LSTM per component type using historical windows.
# Each station's component series is scaled independently.
models = {}
scalers = {}

for component_index in range(COMPONENT_COUNT):
    X_train = []
    y_train = []

    for record in component_records:
        if record["component_index"] != component_index:
            continue

        series = record["values"]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(
            series.reshape(-1, 1)
        ).flatten()

        # Use earlier windows to predict the next component value.
        for i in range(LOOKBACK, len(scaled)):
            X_train.append(scaled[i - LOOKBACK:i])
            y_train.append(scaled[i])

        scalers[
            (record["station_key"], component_index)
        ] = scaler

    if not X_train:
        print(f"No training windows for component {component_index}")
        continue

    X_train = np.asarray(X_train).reshape(
        -1, LOOKBACK, 1
    )
    y_train = np.asarray(y_train)

    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(16, activation="tanh"),
        Dense(8, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    early_stopping = EarlyStopping(
        monitor="loss",
        patience=8,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=60,
        batch_size=16,
        verbose=0,
        callbacks=[early_stopping],
        shuffle=False,
    )

    models[component_index] = model

    print(
        f"Component {component_index}: "
        f"trained on {len(X_train)} windows"
    )

# Forecast the next observation for each station.
forecast_rows = []

for station in stations:
    component_forecasts = []

    for component_index in range(COMPONENT_COUNT):
        model = models.get(component_index)

        if model is None:
            component_forecasts.append(0.0)
            continue

        component_series = station["components"][component_index]

        scaler = scalers[
            ((station["river"], station["station_id"]), component_index)
        ]

        scaled = scaler.transform(
            component_series.reshape(-1, 1)
        ).flatten()

        last_window = scaled[-LOOKBACK:].reshape(
            1, LOOKBACK, 1
        )

        predicted_scaled = model.predict(
            last_window,
            verbose=0,
        )

        predicted_component = scaler.inverse_transform(
            predicted_scaled.reshape(-1, 1)
        )[0, 0]

        component_forecasts.append(float(predicted_component))

    predicted_wqi = float(np.sum(component_forecasts))

    # WQI is bounded between 0 and 100.
    predicted_wqi = float(np.clip(predicted_wqi, 0, 100))

    last_date = pd.Timestamp(station["dates"][-1])
    typical_gap = pd.Series(
        pd.to_datetime(station["dates"])
    ).diff().dt.days.median()

    if pd.isna(typical_gap):
        typical_gap = 61

    forecast_date = last_date + pd.Timedelta(
        days=int(round(typical_gap))
    )

    forecast_rows.append({
        "river": station["river"],
        "station_id": station["station_id"],
        "last_sample_date": last_date,
        "forecast_date_estimate": forecast_date,
        "last_observed_wqi": station["values"][-1],
        "predicted_next_wqi": predicted_wqi,
        "persistence_baseline": station["values"][-1],
        "number_of_observations": len(station["values"]),
        "model": "Experimental CEEMDAN-LSTM",
    })

forecast_df = pd.DataFrame(forecast_rows)

OUTPUT_FILE = OUTPUT_DIR / "ceemdan_lstm_forecasts.csv"
forecast_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FORECAST SUMMARY")
print("=" * 65)

print("Forecasts generated:", len(forecast_df))

if not forecast_df.empty:
    print(
        forecast_df[
            [
                "river",
                "station_id",
                "last_observed_wqi",
                "predicted_next_wqi",
                "persistence_baseline",
            ]
        ].round(2).to_string(index=False)
    )

print("\nSaved to:")
print(OUTPUT_FILE)

print(
    "\nNOTE: These are experimental forecasts, not yet "
    "validated model performance estimates."
)