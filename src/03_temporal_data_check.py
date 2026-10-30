from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"
CLEAN_DIR = ROOT / "outputs" / "cleaned"

df = pd.read_csv(DATA_FILE)
df.columns = [
    str(c).strip().lower().replace("\n", " ")
    for c in df.columns
]

# --------------------------------------------------
# 1. Inspect sampling dates
# --------------------------------------------------

print("=" * 60)
print("TEMPORAL DATA CHECK")
print("=" * 60)

print("\nDate column sample:")
print(df["smp-dat"].head(15).to_string(index=False))

parsed_dates = pd.to_datetime(
    df["smp-dat"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

print("\nDate parsing results:")
print("Total records:", len(df))
print("Successfully parsed:", parsed_dates.notna().sum())
print("Unparsed dates:", parsed_dates.isna().sum())

df["parsed_date"] = parsed_dates

print("\nRecords by year:")
print(df.groupby(parsed_dates.dt.year, dropna=False).size())

print("\nUnique sampling dates by river:")
print(
    df.groupby("sungai")["parsed_date"]
    .nunique()
    .to_string()
)

print("\nRecords per station:")
station_counts = (
    df.groupby(["sungai", "id stn baru"], dropna=False)
    .agg(
        records=("parsed_date", "size"),
        unique_dates=("parsed_date", "nunique"),
        first_date=("parsed_date", "min"),
        last_date=("parsed_date", "max"),
    )
    .reset_index()
)

print(station_counts.to_string(index=False))

station_counts.to_csv(
    CLEAN_DIR / "station_temporal_summary.csv",
    index=False
)

# --------------------------------------------------
# 2. Inspect detection-limit values in the raw CSV
# --------------------------------------------------

print("\n" + "=" * 60)
print("DETECTION-LIMIT VALUE CHECK")
print("=" * 60)

for col in ["BOD\nmg/l", "SS\nmg/l", "NH3-N\nmg/l"]:
    if col in pd.read_csv(DATA_FILE, nrows=0).columns:
        raw = pd.read_csv(DATA_FILE, usecols=[col])[col].astype(str)
        below_limit = raw[raw.str.strip().str.startswith("<")]

        print(f"\n{col}:")
        print("Values beginning with '<':", len(below_limit))
        print(below_limit.value_counts().to_string())

print("\nTemporal check complete.")
print("Station summary saved to outputs/cleaned/")