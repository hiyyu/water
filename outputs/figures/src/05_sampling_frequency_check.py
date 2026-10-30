from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Identify columns using their original names
date_col = "SMP-DAT"
river_col = "SUNGAI"
station_col = "ID STN BARU"

required = [date_col, river_col, station_col]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Parse dates explicitly and remove time-of-day for interval analysis
df["sample_date"] = pd.to_datetime(
    df[date_col],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
).dt.normalize()

if df["sample_date"].isna().any():
    print("WARNING: Some dates could not be parsed.")

df = df.dropna(subset=["sample_date"]).copy()

# Sort records chronologically within each station
df = df.sort_values(
    [river_col, station_col, "sample_date"]
).reset_index(drop=True)

# Calculate days between consecutive sampling dates
df["days_since_previous_sample"] = (
    df.groupby([river_col, station_col])["sample_date"]
    .diff()
    .dt.days
)

print("=" * 65)
print("SAMPLING FREQUENCY CHECK")
print("=" * 65)

print("\nOverall interval statistics (days):")
print(
    df["days_since_previous_sample"]
    .dropna()
    .describe()
    .round(2)
    .to_string()
)

print("\nInterval distribution (days):")
print(
    df["days_since_previous_sample"]
    .dropna()
    .value_counts()
    .sort_index()
    .to_string()
)

# Summarise each station
summary = (
    df.groupby([river_col, station_col])
    .agg(
        records=("sample_date", "size"),
        unique_dates=("sample_date", "nunique"),
        first_date=("sample_date", "min"),
        last_date=("sample_date", "max"),
        mean_gap_days=("days_since_previous_sample", "mean"),
        median_gap_days=("days_since_previous_sample", "median"),
        min_gap_days=("days_since_previous_sample", "min"),
        max_gap_days=("days_since_previous_sample", "max"),
    )
    .reset_index()
)

for col in [
    "mean_gap_days",
    "median_gap_days",
    "min_gap_days",
    "max_gap_days",
]:
    summary[col] = summary[col].round(2)

print("\nSampling interval by station:")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "sampling_frequency_by_station.csv",
    index=False
)

# Count records by calendar month
df["sample_month"] = df["sample_date"].dt.to_period("M").astype(str)

monthly = (
    df.groupby([river_col, "sample_month"])
    .size()
    .reset_index(name="record_count")
)

monthly.to_csv(
    OUTPUT_DIR / "sampling_records_by_month.csv",
    index=False
)

print("\nRecords per calendar month:")
print(monthly.to_string(index=False))

print("\nSaved reports:")
print(OUTPUT_DIR / "sampling_frequency_by_station.csv")
print(OUTPUT_DIR / "sampling_records_by_month.csv")

print("\nSAMPLING FREQUENCY CHECK COMPLETE")