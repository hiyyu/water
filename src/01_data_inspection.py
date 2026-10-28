from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

EXCEL_FILE = DATA_DIR / "Lampiran A - Klang & Skudai.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not EXCEL_FILE.exists():
        print("Excel workbook not found.")
        print(f"Expected location: {EXCEL_FILE}")
        print("Place the DOE workbook inside the data folder.")
        return

    # Read all worksheets
    worksheets = pd.read_excel(EXCEL_FILE, sheet_name=None)

    frames = []

    for sheet_name, df in worksheets.items():
        if df.empty:
            continue

        df.columns = df.columns.astype(str).str.strip()
        df["SOURCE_SHEET"] = sheet_name
        frames.append(df)

        print(f"{sheet_name}: {len(df)} records")

    if not frames:
        print("No records found in the workbook.")
        return

    # Combine worksheets
    data = pd.concat(frames, ignore_index=True, sort=False)

    # Inspect the dataset
    print("\n--- DATASET SUMMARY ---")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print("\nColumn names:")
    print(data.columns.tolist())

    print("\n--- MISSING VALUES ---")
    missing = data.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.any() else "No blank cells detected.")

    print("\n--- DUPLICATE RECORDS ---")
    print(f"Exact duplicate rows: {data.duplicated().sum()}")

    # River counts
    if "SUNGAI" in data.columns:
        print("\n--- RECORDS BY RIVER ---")
        print(data["SUNGAI"].value_counts(dropna=False))

    # Station counts
    station_col = next(
        (c for c in ["ID STN BARU", "ID STN (2016)", "LOCATION"]
         if c in data.columns),
        None
    )

    if station_col and "SUNGAI" in data.columns:
        print("\n--- UNIQUE STATIONS BY RIVER ---")
        print(data.groupby("SUNGAI")[station_col].nunique())

    # Save preliminary outputs
    data.to_csv(OUTPUT_DIR / "combined_doe_dataset.csv", index=False)
    missing.rename("missing_count").to_csv(
        OUTPUT_DIR / "missing_values_report.csv"
    )

    print("\nInspection complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()