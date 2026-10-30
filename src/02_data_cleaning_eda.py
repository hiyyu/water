from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "combined_doe_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
FIGURE_DIR = OUTPUT_DIR / "figures"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD THE ORIGINAL DATASET
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Run src/01_data_inspection.py first."
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("WATER QUALITY DATA CLEANING AND EDA")
print("=" * 60)
print(f"Original shape: {df.shape}")


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

def clean_column_name(name):
    """Convert column names into consistent snake_case names."""
    name = str(name).strip().lower()
    name = re.sub(r"[\r\n]+", " ", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


df.columns = [clean_column_name(col) for col in df.columns]

print("\nStandardized columns:")
print(df.columns.tolist())


# ============================================================
# 4. CLEAN TEXT FIELDS
# ============================================================

text_columns = df.select_dtypes(include=["object"]).columns.tolist()

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
        }
    )

# Standardize river names.
if "sungai" in df.columns:
    df["sungai"] = df["sungai"].str.upper()

# Standardize class labels without changing their meaning.
if "class" in df.columns:
    df["class"] = (
        df["class"]
        .str.upper()
        .str.replace(r"^CLASS\s*", "", regex=True)
        .str.strip()
    )

if "river_status" in df.columns:
    df["river_status"] = (
        df["river_status"]
        .str.strip()
        .str.title()
    )


# ============================================================
# 5. CONVERT MEASUREMENT COLUMNS TO NUMERIC
# ============================================================

# These names are produced by the column-name cleaning above.
measurement_columns = [
    "longitude",
    "latitude",
    "freq",
    "do_sat",
    "do_mg_l",
    "bod_mg_l",
    "cod_mg_l",
    "ss_mg_l",
    "ph",
    "nh3_n_mg_l",
    "do_index_index",
    "bod_index_index",
    "cod_index_index",
    "an_index",
    "ss_index",
    "ph_index",
    "wqi",
]

# Save the original strings that need special handling.
detection_limit_report = []

for col in measurement_columns:
    if col not in df.columns:
        continue

    original = df[col].astype("string").str.strip()

    # Identify measurements reported below a detection limit,
    # such as "<1.000" or "<0.010".
    mask = original.str.match(r"^<\s*-?\d+(\.\d+)?$", na=False)

    if mask.any():
        for value in original[mask].unique():
            detection_limit_report.append(
                {
                    "column": col,
                    "original_value": value,
                    "count": int((original == value).sum()),
                    "handling": "Not converted; requires review",
                }
            )

    # Remove thousands separators if present.
    # Do not silently replace "<limit" values with zero.
    cleaned = original.str.replace(",", "", regex=False)
    cleaned = cleaned.mask(mask, pd.NA)

    df[col] = pd.to_numeric(cleaned, errors="coerce")

# Save the detection-limit report.
pd.DataFrame(
    detection_limit_report,
    columns=["column", "original_value", "count", "handling"],
).to_csv(
    CLEAN_DIR / "detection_limit_report.csv",
    index=False,
)

print("\nNumeric conversion completed.")
print("Detection-limit values were preserved in a separate report.")
print("They are missing in the numeric analysis columns until reviewed.")


# ============================================================
# 6. REPORT MISSING VALUES AFTER CONVERSION
# ============================================================

missing_report = pd.DataFrame(
    {
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2),
    }
)

missing_report = missing_report.sort_values(
    "missing_count", ascending=False
)

missing_report.to_csv(
    CLEAN_DIR / "missing_values_after_conversion.csv"
)

print("\nMissing values after conversion:")
print(missing_report[missing_report["missing_count"] > 0])

print("\nData types:")
print(df.dtypes)


# ============================================================
# 7. CHECK WQI AND MEASUREMENT RANGES
# ============================================================

range_checks = {
    "wqi": (0, 100),
    "ph": (0, 14),
    "do_sat": (0, 200),
    "do_mg_l": (0, None),
    "bod_mg_l": (0, None),
    "cod_mg_l": (0, None),
    "ss_mg_l": (0, None),
    "nh3_n_mg_l": (0, None),
}

range_issues = []

for col, (lower, upper) in range_checks.items():
    if col not in df.columns:
        continue

    invalid = pd.Series(False, index=df.index)

    if lower is not None:
        invalid |= df[col].notna() & (df[col] < lower)

    if upper is not None:
        invalid |= df[col].notna() & (df[col] > upper)

    for idx in df.index[invalid]:
        range_issues.append(
            {
                "row_index": int(idx),
                "column": col,
                "value": df.at[idx, col],
                "issue": f"Outside preliminary check range {lower} to {upper}",
            }
        )

pd.DataFrame(
    range_issues,
    columns=["row_index", "column", "value", "issue"],
).to_csv(
    CLEAN_DIR / "range_check_issues.csv",
    index=False,
)

print(f"\nPotential range issues: {len(range_issues)}")


# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

numeric_df = df.select_dtypes(include=[np.number])

numeric_df.describe().T.to_csv(
    CLEAN_DIR / "descriptive_statistics.csv"
)

print("\nDescriptive statistics saved.")


# ============================================================
# 9. WQI SUMMARY BY RIVER AND YEAR
# ============================================================

if "wqi" in df.columns and "sungai" in df.columns:
    grouping_columns = ["sungai"]

    if "source_sheet" in df.columns:
        grouping_columns.append("source_sheet")

    wqi_summary = (
        df.groupby(grouping_columns, dropna=False)["wqi"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
        .reset_index()
    )

    wqi_summary.to_csv(
        CLEAN_DIR / "wqi_summary_by_river_year.csv",
        index=False,
    )

    print("\nWQI summary by river and year:")
    print(wqi_summary.to_string(index=False))


# ============================================================
# 10. WQI DISTRIBUTION
# ============================================================

if "wqi" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="wqi",
        bins=20,
        kde=True,
    )
    plt.title("Distribution of Water Quality Index (WQI)")
    plt.xlabel("WQI")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "wqi_distribution.png", dpi=300)
    plt.close()


# ============================================================
# 11. WQI COMPARISON BY RIVER
# ============================================================

if {"wqi", "sungai"}.issubset(df.columns):
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="sungai",
        y="wqi",
    )
    plt.title("WQI Comparison Between Rivers")
    plt.xlabel("River")
    plt.ylabel("WQI")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "wqi_by_river.png", dpi=300)
    plt.close()


# ============================================================
# 12. WQI COMPARISON BY YEAR
# ============================================================

if {"wqi", "source_sheet"}.issubset(df.columns):
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="source_sheet",
        y="wqi",
    )
    plt.title("WQI Comparison by Year")
    plt.xlabel("Source Year")
    plt.ylabel("WQI")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "wqi_by_year.png", dpi=300)
    plt.close()


# ============================================================
# 13. CORRELATION MATRIX
# ============================================================

correlation_columns = [
    col for col in [
        "do_sat",
        "do_mg_l",
        "bod_mg_l",
        "cod_mg_l",
        "ss_mg_l",
        "ph",
        "nh3_n_mg_l",
        "do_index_index",
        "bod_index_index",
        "cod_index_index",
        "an_index",
        "ss_index",
        "ph_index",
        "wqi",
    ]
    if col in df.columns
]

if len(correlation_columns) >= 2:
    correlation_matrix = df[correlation_columns].corr(numeric_only=True)

    correlation_matrix.to_csv(
        CLEAN_DIR / "correlation_matrix.csv"
    )

    plt.figure(
        figsize=(
            max(10, len(correlation_columns) * 0.8),
            max(8, len(correlation_columns) * 0.7),
        )
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title("Correlation Matrix of Water Quality Variables")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "correlation_matrix.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# 14. SAVE CLEANED WORKING DATASET
# ============================================================

cleaned_file = CLEAN_DIR / "doe_dataset_cleaned_working.csv"
df.to_csv(cleaned_file, index=False)

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"Working dataset: {cleaned_file}")
print(f"Reports folder: {CLEAN_DIR}")
print(f"Figures folder: {FIGURE_DIR}")
print("\nImportant:")
print("- Original combined CSV was not modified.")
print("- Detection-limit values need a documented treatment decision.")
print("- Range checks flag possible issues; they do not delete records.")
print("- This script does not split data or train a model.")