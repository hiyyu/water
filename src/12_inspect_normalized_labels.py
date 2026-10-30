from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"

df = pd.read_csv(INPUT_FILE)

# Normalize column names, including embedded line breaks.
df.columns = [
    re.sub(r"\s+", " ", str(column)).strip()
    for column in df.columns
]

print("=" * 60)
print("NORMALIZED CONTAMINATION LABEL INSPECTION")
print("=" * 60)

# Normalize CLASS labels such as "III" and "CLASS III".
def normalize_class(value):
    if pd.isna(value):
        return "Missing"

    value = re.sub(r"\s+", " ", str(value)).strip().upper()
    value = re.sub(r"^CLASS\s+", "", value)

    if value in {"I", "II", "III", "IV", "V"}:
        return f"Class {value}"

    return value


if "CLASS" in df.columns:
    df["CLASS_NORMALIZED"] = df["CLASS"].apply(normalize_class)

    print("\n--- Normalized CLASS distribution ---")
    print(
        df["CLASS_NORMALIZED"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\n--- Normalized CLASS by river ---")
    if "SUNGAI" in df.columns:
        print(
            pd.crosstab(
                df["SUNGAI"],
                df["CLASS_NORMALIZED"],
                margins=True,
            ).to_string()
        )

# Find and inspect the status column.
status_columns = [
    column for column in df.columns
    if "RIVER STATUS" in column.upper()
]

if status_columns:
    for column in status_columns:
        print(f"\n--- {column} ---")
        print("Missing values:", df[column].isna().sum())
        print(df[column].value_counts(dropna=False).to_string())

        if "SUNGAI" in df.columns:
            print("\nStatus by river:")
            print(
                pd.crosstab(
                    df["SUNGAI"],
                    df[column],
                    dropna=False,
                    margins=True,
                ).to_string()
            )
else:
    print("\nNo RIVER STATUS column found after normalization.")
    print("\nAvailable columns:")
    for column in df.columns:
        print("-", repr(column))

print("\nLABEL INSPECTION COMPLETE")