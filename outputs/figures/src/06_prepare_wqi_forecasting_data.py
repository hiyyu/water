from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PREPARING WQI FORECASTING DATASET")
print("=" * 65)

# 1. Load the original combined DOE data
df = pd.read_csv(INPUT_FILE)

required_columns = [
    "SMP-DAT",
    "SUNGAI",
    "ID STN BARU",
    "WQI",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# 2. Keep the essential fields for this first forecasting task
data = df[required_columns].copy()

data = data.rename(columns={
    "SMP-DAT": "sample_date",
    "SUNGAI": "river",
    "ID STN BARU": "station_id",
    "WQI": "wqi",
})

# 3. Parse dates and WQI without changing the original CSV
data["sample_date"] = pd.to_datetime(
    data["sample_date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

data["wqi"] = pd.to_numeric(
    data["wqi"],
    errors="coerce"
)

data["river"] = data["river"].astype("string").str.strip()
data["station_id"] = (
    data["station_id"].astype("string").str.strip()
)

# 4. Report missing essential fields
print("\nMissing values in essential fields:")
print(data.isna().sum().to_string())

print("\nRecords before filtering:", len(data))

# Exclude only records missing essential fields for this task
data = data.dropna(
    subset=["sample_date", "river", "station_id", "wqi"]
).copy()

# Remove time of day for sampling-date ordering
data["sample_date"] = data["sample_date"].dt.normalize()

# 5. Check WQI values before using them as targets
print("\nWQI descriptive statistics:")
print(data["wqi"].describe().round(3).to_string())

outside_range = data[
    (data["wqi"] < 0) | (data["wqi"] > 100)
]

print("\nWQI values outside 0-100:", len(outside_range))

if not outside_range.empty:
    print(outside_range.to_string(index=False))

# 6. Check duplicate station-date records
duplicate_mask = data.duplicated(
    subset=["river", "station_id", "sample_date"],
    keep=False
)

duplicates = data.loc[duplicate_mask].sort_values(
    ["river", "station_id", "sample_date"]
)

print("\nRecords involved in duplicate station-date groups:",
      len(duplicates))

if not duplicates.empty:
    duplicates.to_csv(
        OUTPUT_DIR / "wqi_duplicate_station_dates_review.csv",
        index=False
    )
    print("Duplicate review saved to:")
    print(OUTPUT_DIR / "wqi_duplicate_station_dates_review.csv")

# Do not automatically remove duplicates: review them first.
# Stop if duplicates exist, so we don't create ambiguous sequences.
if not duplicates.empty:
    raise ValueError(
        "Duplicate station-date records found. "
        "Review wqi_duplicate_station_dates_review.csv before continuing."
    )

# 7. Sort each station chronologically
data = data.sort_values(
    ["river", "station_id", "sample_date"]
).reset_index(drop=True)

# 8. Create lag features and the next-observation target
group_columns = ["river", "station_id"]
grouped = data.groupby(group_columns, sort=False)

data["previous_wqi"] = grouped["wqi"].shift(1)
data["wqi_two_samples_ago"] = grouped["wqi"].shift(2)

data["next_sample_date"] = grouped["sample_date"].shift(-1)
data["next_sample_wqi"] = grouped["wqi"].shift(-1)

data["days_until_next_sample"] = (
    data["next_sample_date"] - data["sample_date"]
).dt.days

# Keep only rows with two historical WQI values and a future target
forecast_data = data.dropna(
    subset=[
        "previous_wqi",
        "wqi_two_samples_ago",
        "next_sample_wqi",
        "days_until_next_sample",
    ]
).copy()

# 9. Save the clean chronological data and forecasting examples
base_file = OUTPUT_DIR / "wqi_forecasting_base.csv"
forecast_file = OUTPUT_DIR / "wqi_next_sample_lags.csv"

data.to_csv(base_file, index=False)
forecast_data.to_csv(forecast_file, index=False)

# 10. Print a summary for checking
print("\n" + "=" * 65)
print("FORECASTING DATASET SUMMARY")
print("=" * 65)

print("Usable base records:", len(data))
print("Forecasting examples:", len(forecast_data))
print("Rivers:", data["river"].nunique())
print("Stations:", data["station_id"].nunique())
print(
    "Date range:",
    data["sample_date"].min().date(),
    "to",
    data["sample_date"].max().date()
)

print("\nRecords per station:")
print(
    data.groupby(["river", "station_id"])
    .size()
    .describe()
    .round(2)
    .to_string()
)

print("\nForecast target interval (days):")
print(
    forecast_data["days_until_next_sample"]
    .describe()
    .round(2)
    .to_string()
)

print("\nExamples per station:")
print(
    forecast_data.groupby(["river", "station_id"])
    .size()
    .describe()
    .round(2)
    .to_string()
)

print("\nSaved files:")
print(base_file)
print(forecast_file)

print("\nDATA PREPARATION COMPLETE")