from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"

df = pd.read_csv(DATA_FILE)

# Find the original sampling-date column
date_col = next(
    (col for col in df.columns if str(col).strip().lower() == "smp-dat"),
    None
)

if date_col is None:
    raise ValueError(
        "Could not find the SMP-DAT column. "
        f"Available columns: {df.columns.tolist()}"
    )

dates = df[date_col]

print("=" * 65)
print("INSPECTING ORIGINAL SAMPLING DATE VALUES")
print("=" * 65)

print("\nOriginal column name:", repr(date_col))
print("Original data type:", dates.dtype)
print("Total records:", len(dates))

print("\nFirst 30 original values:")
print(dates.head(30).to_string())

print("\nMost common original values:")
print(dates.value_counts(dropna=False).head(30).to_string())

# Attempt ordinary date parsing
parsed = pd.to_datetime(
    dates,
    errors="coerce",
    dayfirst=False
)

failed = dates[parsed.isna() & dates.notna()]

print("\n" + "=" * 65)
print("VALUES THAT FAILED DATE PARSING")
print("=" * 65)

print("Failed non-empty values:", len(failed))

if len(failed) > 0:
    print("\nFirst 60 failed values:")
    print(failed.head(60).to_string())

    print("\nUnique failed values (up to 100):")
    print(failed.astype(str).drop_duplicates().head(100).to_string(index=False))

print("\n" + "=" * 65)
print("DATE FORMAT CHECK COMPLETE")
print("=" * 65)