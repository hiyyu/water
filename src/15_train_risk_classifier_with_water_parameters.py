from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "combined_doe_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

PREDICTIONS_FILE = (
    OUTPUT_DIR / "risk_classifier_extended_predictions.csv"
)
METRICS_FILE = OUTPUT_DIR / "risk_classifier_extended_metrics.json"
REPORT_FILE = OUTPUT_DIR / "risk_classifier_extended_report.txt"
CONFUSION_FILE = (
    OUTPUT_DIR / "risk_classifier_extended_confusion_matrix.csv"
)


# ------------------------------------------------------------
# 1. LOAD DATA AND IDENTIFY COLUMNS
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)
df.columns = [
    re.sub(r"\s+", " ", str(column)).strip()
    for column in df.columns
]

def find_column(candidates, required=True):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


date_col = find_column(["SMP-DAT", "SMP DAT", "SAMPLE DATE"])
river_col = find_column(["SUNGAI", "RIVER"])
new_station_col = find_column(["ID STN BARU"], required=False)
old_station_col = find_column(["ID STN (2016)"], required=False)
status_col = find_column(["RIVER STATUS"])
wqi_col = find_column(["WQI"])

# Use the new station identifier when available, otherwise
# fall back to the older station identifier for that row.
if new_station_col is not None:
    station = df[new_station_col].astype("string").str.strip()
else:
    station = pd.Series(pd.NA, index=df.index, dtype="string")

if old_station_col is not None:
    old_station = (
        df[old_station_col].astype("string").str.strip()
    )
    station = station.fillna(old_station)

station = station.mask(station.eq(""))

df["__station_id"] = station
df["__sample_date"] = pd.to_datetime(
    df[date_col], errors="coerce"
)
df["__river"] = df[river_col].astype("string").str.strip()
df["__status"] = df[status_col].astype("string").str.strip()
df["__wqi"] = pd.to_numeric(df[wqi_col], errors="coerce")

# Candidate predictors are measurements from the previous sample.
measurement_candidates = {
    "DO % Sat": ["DO % Sat"],
    "DO mg/l": ["DO mg/l"],
    "BOD mg/l": ["BOD mg/l"],
    "COD mg/l": ["COD mg/l"],
    "SS mg/l": ["SS mg/l"],
    "pH": ["pH"],
    "NH3-N mg/l": ["NH3-N mg/l"],
    "DO INDEX": ["DO INDEX INDEX"],
    "BOD INDEX": ["BOD INDEX INDEX"],
    "COD INDEX": ["COD INDEX INDEX"],
    "AN INDEX": ["AN INDEX"],
    "SS INDEX": ["SS INDEX"],
    "pH Index": ["pH Index"],
}

measurement_columns = {}

for feature_name, candidates in measurement_candidates.items():
    column = find_column(candidates, required=False)
    if column is not None:
        df[f"__feature_{feature_name}"] = pd.to_numeric(
            df[column], errors="coerce"
        )
        measurement_columns[feature_name] = (
            f"__feature_{feature_name}"
        )

df = df.dropna(
    subset=[
        "__sample_date",
        "__river",
        "__station_id",
        "__wqi",
        "__status",
    ]
).copy()

df = df[
    df["__river"].ne("")
    & df["__station_id"].ne("")
    & df["__status"].ne("")
    & df["__wqi"].between(0, 100)
].copy()

# If the same station has multiple records with the same timestamp,
# inspect and resolve those records before final model evaluation.
duplicates = df.duplicated(
    ["__river", "__station_id", "__sample_date"],
    keep=False,
)

if duplicates.any():
    print(
        "Warning: duplicate station/timestamp records detected:",
        int(duplicates.sum()),
    )
    print(
        "Keeping the first record per station and timestamp "
        "for this preliminary experiment."
    )
    df = df.drop_duplicates(
        ["__river", "__station_id", "__sample_date"],
        keep="first",
    )

df = df.sort_values(
    ["__river", "__station_id", "__sample_date"]
).reset_index(drop=True)

# Standardize status labels by trimming whitespace and
# case-insensitive matching.
status_lookup = {
    value.casefold(): value
    for value in df["__status"].dropna().unique()
}
df["__status"] = df["__status"].map(
    lambda value: status_lookup.get(value.casefold(), value)
)

print("=" * 65)
print("EXTENDED NEXT-SAMPLE RISK CLASSIFIER")
print("=" * 65)
print(f"Usable raw observations: {len(df)}")
print(f"Unique stations: {df['__station_id'].nunique()}")
print(f"Water-quality features available: {list(measurement_columns)}")


# ------------------------------------------------------------
# 2. BUILD NEXT-SAMPLE FORECASTING EXAMPLES
# ------------------------------------------------------------

group_columns = ["__river", "__station_id"]
grouped = df.groupby(group_columns, sort=False)

df["previous_wqi"] = df["__wqi"]
df["wqi_two_samples_ago"] = grouped["__wqi"].shift(1)
df["previous_status"] = df["__status"]
df["target_status"] = grouped["__status"].shift(-1)
df["target_date"] = grouped["__sample_date"].shift(-1)
df["days_until_next_sample"] = (
    df["target_date"] - df["__sample_date"]
).dt.total_seconds() / 86400

# The water measurements on the current row are the previous
# sample's measurements relative to target_status.
for feature_name, source_column in measurement_columns.items():
    df[feature_name] = df[source_column]

forecast = df.dropna(
    subset=[
        "wqi_two_samples_ago",
        "target_status",
        "target_date",
        "days_until_next_sample",
    ]
).copy()

forecast = forecast[
    forecast["days_until_next_sample"] > 0
].copy()

forecast = forecast.rename(
    columns={
        "__river": "river",
        "__station_id": "station_id",
        "__sample_date": "previous_sample_date",
    }
)

forecast["_target_day"] = forecast["target_date"].dt.normalize()
forecast = forecast.sort_values(
    ["target_date", "river", "station_id"]
).reset_index(drop=True)

if len(forecast) < 30:
    raise ValueError(
        f"Only {len(forecast)} forecasting examples were created. "
        "Inspect station identifiers and dates."
    )

print(f"\nForecast examples: {len(forecast)}")
print(f"Target classes:\n{forecast['target_status'].value_counts()}")


# ------------------------------------------------------------
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ------------------------------------------------------------

unique_days = np.array(sorted(forecast["_target_day"].unique()))

split_index = int(len(unique_days) * 0.80)
split_index = max(1, min(split_index, len(unique_days) - 1))
cutoff_day = pd.Timestamp(unique_days[split_index])

train = forecast[
    forecast["_target_day"] < cutoff_day
].copy()
test = forecast[
    forecast["_target_day"] >= cutoff_day
].copy()

if train.empty or test.empty:
    raise ValueError("Training or test set is empty.")

print("\nChronological split:")
print(f"Training examples: {len(train)}")
print(f"Testing examples: {len(test)}")
print(f"Training ends: {train['target_date'].max()}")
print(f"Testing begins: {test['target_date'].min()}")

assert train["_target_day"].max() < test["_target_day"].min()

# Ensure the target classes in the test period were represented
# in training. Otherwise, the model cannot learn those classes.
encoder = LabelEncoder()
y_train_text = train["target_status"].astype(str)
y_test_text = test["target_status"].astype(str)

y_train = encoder.fit_transform(y_train_text)

unseen = sorted(set(y_test_text) - set(encoder.classes_))
if unseen:
    raise ValueError(
        f"Test period has classes absent from training: {unseen}. "
        "A different evaluation design or more data is needed."
    )

y_test = encoder.transform(y_test_text)


# ------------------------------------------------------------
# 4. DEFINE PREVIOUS-SAMPLE FEATURES
# ------------------------------------------------------------

numeric_features = [
    "previous_wqi",
    "wqi_two_samples_ago",
] + list(measurement_columns.keys())

categorical_features = [
    "river",
    "previous_status",
]

numeric_features = [
    feature for feature in numeric_features
    if feature in forecast.columns
]

feature_columns = numeric_features + categorical_features

X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()

# Baseline: predict that the status remains unchanged.
baseline_predictions = test["previous_status"].astype(str).to_numpy()
actual_text = y_test_text.to_numpy()

baseline_metrics = {
    "accuracy": float(
        accuracy_score(actual_text, baseline_predictions)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(actual_text, baseline_predictions)
    ),
    "macro_f1": float(
        f1_score(
            actual_text,
            baseline_predictions,
            labels=encoder.classes_,
            average="macro",
            zero_division=0,
        )
    ),
}

print("\nPersistence baseline:")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.4f}")


# ------------------------------------------------------------
# 5. TRAIN XGBOOST
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

classifier = XGBClassifier(
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    eval_metric="mlogloss",
    n_estimators=200,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=2.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

print("\nTraining extended XGBoost model...")
model.fit(
    X_train,
    y_train,
    classifier__sample_weight=weights,
)

predicted_codes = model.predict(X_test).astype(int)
predictions = encoder.inverse_transform(predicted_codes)

xgb_metrics = {
    "accuracy": float(accuracy_score(actual_text, predictions)),
    "balanced_accuracy": float(
        balanced_accuracy_score(actual_text, predictions)
    ),
    "macro_f1": float(
        f1_score(
            actual_text,
            predictions,
            labels=encoder.classes_,
            average="macro",
            zero_division=0,
        )
    ),
}

print("\nExtended XGBoost results:")
for metric, value in xgb_metrics.items():
    print(f"{metric}: {value:.4f}")

report = classification_report(
    actual_text,
    predictions,
    labels=encoder.classes_,
    zero_division=0,
)
print("\nClassification report:")
print(report)


# ------------------------------------------------------------
# 6. SAVE RESULTS
# ------------------------------------------------------------

class_names = list(encoder.classes_)
matrix = confusion_matrix(
    actual_text,
    predictions,
    labels=class_names,
)

confusion_df = pd.DataFrame(
    matrix,
    index=[f"Actual_{name}" for name in class_names],
    columns=[f"Predicted_{name}" for name in class_names],
)
confusion_df.to_csv(CONFUSION_FILE)

prediction_output = test[
    [
        "river",
        "station_id",
        "previous_sample_date",
        "target_date",
        "previous_wqi",
        "wqi_two_samples_ago",
        "previous_status",
        "target_status",
    ]
].copy()

for feature in measurement_columns:
    prediction_output[feature] = test[feature]

prediction_output["baseline_prediction"] = baseline_predictions
prediction_output["xgboost_prediction"] = predictions
prediction_output["xgboost_correct"] = (
    prediction_output["target_status"]
    == prediction_output["xgboost_prediction"]
)

prediction_output.to_csv(PREDICTIONS_FILE, index=False)

metrics = {
    "task": "Next-sample contamination-status classification",
    "features": feature_columns,
    "examples": int(len(forecast)),
    "training_examples": int(len(train)),
    "testing_examples": int(len(test)),
    "unique_stations": int(forecast["station_id"].nunique()),
    "test_start": str(test["target_date"].min()),
    "test_end": str(test["target_date"].max()),
    "classes": class_names,
    "persistence_baseline": baseline_metrics,
    "xgboost": xgb_metrics,
    "warning": (
        "Preliminary experiment using a small dataset. "
        "Results require further validation and additional data."
    ),
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("EXTENDED CONTAMINATION STATUS CLASSIFIER\n")
    file.write("=" * 55 + "\n\n")
    file.write(f"Features: {feature_columns}\n\n")
    file.write("PERSISTENCE BASELINE\n")
    file.write(json.dumps(baseline_metrics, indent=4))
    file.write("\n\nEXTENDED XGBOOST\n")
    file.write(json.dumps(xgb_metrics, indent=4))
    file.write("\n\nCLASSIFICATION REPORT\n")
    file.write(report)
    file.write("\n\nCONFUSION MATRIX\n")
    file.write(confusion_df.to_string())
    file.write("\n\nLIMITATION\n")
    file.write(metrics["warning"] + "\n")

print("\nSaved:")
print(PREDICTIONS_FILE)
print(METRICS_FILE)
print(REPORT_FILE)
print(CONFUSION_FILE)
print("\nFinished.")