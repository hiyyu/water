from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"
OUTPUT_FILE = (
    ROOT / "outputs" / "cleaned" /
    "risk_next_sample_dataset.csv"
)

df = pd.read_csv(INPUT_FILE)

# Normalize column names, including embedded newlines.
df.columns = [
    re.sub(r"\s+", " ", str(c)).strip()
    for c in df.columns
]

print("=" * 65)
print("NEXT-SAMPLE RISK DATASET PREPARATION")
print("=" * 65)
print("\nAvailable columns:")
print(df.columns.tolist())


def find_column(candidates, description):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Cannot identify {description} column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


date_col = find_column(
    ["SMP-DAT", "SMP DAT", "SAMPLE DATE", "DATE"],
    "sampling date",
)

river_col = find_column(
    ["SUNGAI", "RIVER"],
    "river",
)

station_col = find_column(
    [
        "ID STN BARU",
        "ID STN (2016)",
        "STATION ID",
        "STATION_ID",
        "STATION CODE",
        "STATION",
        "STN",
        "STN CODE",
        "LOKASI",
    ],
    "station identifier",
)

wqi_col = find_column(["WQI"], "WQI")

status_col = find_column(
    ["RIVER STATUS"],
    "river status",
)

# Parse the source's date format.
df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce",
)

df[wqi_col] = pd.to_numeric(
    df[wqi_col],
    errors="coerce",
)

df[status_col] = (
    df[status_col]
    .astype("string")
    .str.strip()
)

df[river_col] = df[river_col].astype("string").str.strip()
df[station_col] = df[station_col].astype("string").str.strip()

df = df.dropna(
    subset=[
        date_col, river_col, station_col,
        wqi_col, status_col,
    ]
).copy()

# Exclude empty labels and invalid WQI values.
df = df[
    df[status_col].ne("")
    & df[wqi_col].between(0, 100)
].copy()

# Standardize status text without changing its meaning.
status_lookup = {
    value.casefold(): value
    for value in df[status_col].unique()
}
df[status_col] = df[status_col].map(
    lambda value: status_lookup.get(value.casefold(), value)
)

# Sort each station's observations chronologically.
df = df.sort_values(
    [river_col, station_col, date_col]
).reset_index(drop=True)

group_cols = [river_col, station_col]
grouped = df.groupby(group_cols, sort=False)

# Features are from the previous observation only.
df["previous_wqi"] = grouped[wqi_col].shift(0)
df["wqi_two_samples_ago"] = grouped[wqi_col].shift(1)
df["previous_status"] = grouped[status_col].shift(0)

# Target comes from the next recorded observation.
df["target_status"] = grouped[status_col].shift(-1)
df["target_date"] = grouped[date_col].shift(-1)
df["days_until_next_sample"] = (
    df["target_date"] - df[date_col]
).dt.days

# Require two past WQI observations, and a valid next status.
forecast_df = df.dropna(
    subset=[
        "previous_wqi",
        "wqi_two_samples_ago",
        "previous_status",
        "target_status",
        "target_date",
        "days_until_next_sample",
    ]
).copy()

forecast_df = forecast_df[
    forecast_df["days_until_next_sample"] > 0
].copy()

forecast_df = forecast_df.rename(
    columns={
        river_col: "river",
        station_col: "station_id",
        date_col: "previous_sample_date",
    }
)

# Keep only model inputs, target and useful audit fields.
output_columns = [
    "river",
    "station_id",
    "previous_sample_date",
    "target_date",
    "days_until_next_sample",
    "previous_wqi",
    "wqi_two_samples_ago",
    "previous_status",
    "target_status",
]

forecast_df = forecast_df[output_columns].sort_values(
    ["target_date", "river", "station_id"]
).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
forecast_df.to_csv(OUTPUT_FILE, index=False)

print("\nDetected columns:")
print("Date:", date_col)
print("River:", river_col)
print("Station:", station_col)
print("WQI:", wqi_col)
print("Status:", status_col)

print("\nDataset summary")
print("Raw usable observations:", len(df))
print("Forecast examples:", len(forecast_df))
print("Unique stations:", forecast_df["station_id"].nunique())
print(
    "Target date range:",
    forecast_df["target_date"].min(),
    "to",
    forecast_df["target_date"].max(),
)

print("\nTarget status distribution:")
print(forecast_df["target_status"].value_counts().to_string())

print("\nTarget status by river:")
print(
    pd.crosstab(
        forecast_df["river"],
        forecast_df["target_status"],
        margins=True,
    ).to_string()
)

print("\nSampling interval (days):")
print(forecast_df["days_until_next_sample"].describe().to_string())

print("\nFirst five examples:")
print(forecast_df.head().to_string(index=False))

print("\nSaved to:", OUTPUT_FILE)