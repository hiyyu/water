from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("CONTAMINATION-RISK LABEL INSPECTION")
print("=" * 60)

for column in ["CLASS", "RIVER STATUS"]:
    print(f"\n--- {column} ---")

    if column not in df.columns:
        print("Column not found.")
        continue

    print("Missing values:", df[column].isna().sum())

    print("\nUnique labels:")
    print(df[column].value_counts(dropna=False).to_string())

    print("\nLabel distribution by river:")
    if "SUNGAI" in df.columns:
        print(
            pd.crosstab(
                df["SUNGAI"],
                df[column],
                dropna=False,
                margins=True,
            ).to_string()
        )

print("\nRelevant water-quality columns:")
for column in df.columns:
    if any(
        keyword in column.upper()
        for keyword in [
            "WQI", "CLASS", "STATUS", "DO", "BOD",
            "COD", "SS", "PH", "NH3", "INDEX"
        ]
    ):
        print("-", column)

print("\nLABEL INSPECTION COMPLETE")