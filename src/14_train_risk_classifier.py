from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "outputs" / "cleaned" / "risk_next_sample_dataset.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "risk_classifier_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "risk_classifier_metrics.json"
REPORT_FILE = OUTPUT_DIR / "risk_classifier_report.txt"
CONFUSION_FILE = OUTPUT_DIR / "risk_classifier_confusion_matrix.csv"

RANDOM_SEED = 42


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("NEXT-SAMPLE CONTAMINATION STATUS CLASSIFICATION")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "river",
    "station_id",
    "previous_sample_date",
    "target_date",
    "previous_wqi",
    "wqi_two_samples_ago",
    "previous_status",
    "target_status",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Parse dates and numeric values.
df["previous_sample_date"] = pd.to_datetime(
    df["previous_sample_date"], errors="coerce"
)
df["target_date"] = pd.to_datetime(
    df["target_date"], errors="coerce"
)

for column in ["previous_wqi", "wqi_two_samples_ago"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Keep only usable examples.
df = df.dropna(subset=required_columns).copy()

df["river"] = df["river"].astype(str).str.strip()
df["previous_status"] = (
    df["previous_status"].astype(str).str.strip()
)
df["target_status"] = df["target_status"].astype(str).str.strip()

df = df[
    df["target_status"].ne("")
    & df["previous_status"].ne("")
    & df["previous_wqi"].between(0, 100)
    & df["wqi_two_samples_ago"].between(0, 100)
    & (df["target_date"] > df["previous_sample_date"])
].copy()

if df.empty:
    raise ValueError("No usable forecasting examples remain.")

# Keep date order consistent.
df = df.sort_values(
    ["target_date", "river", "station_id"]
).reset_index(drop=True)

print(f"\nUsable examples: {len(df)}")
print(f"Unique stations: {df['station_id'].nunique()}")
print(f"Target date range: {df['target_date'].min()} to "
      f"{df['target_date'].max()}")

print("\nOverall target class distribution:")
print(df["target_status"].value_counts())



# ============================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# Split by calendar day, not exact timestamp.
# This keeps samples from the same target day together.
df["_target_day"] = df["target_date"].dt.normalize()

unique_dates = np.array(sorted(df["_target_day"].unique()))

if len(unique_dates) < 5:
    raise ValueError(
        "Too few unique target dates for a meaningful "
        "chronological split."
    )

split_index = int(len(unique_dates) * 0.80)
split_index = max(1, min(split_index, len(unique_dates) - 1))

cutoff_date = pd.Timestamp(unique_dates[split_index])

train_df = df[df["_target_day"] < cutoff_date].copy()
test_df = df[df["_target_day"] >= cutoff_date].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or test set is empty.")

print("\nChronological split by calendar day:")
print(f"Training examples: {len(train_df)}")
print(f"Testing examples:  {len(test_df)}")
print(
    f"Training target dates: {train_df['target_date'].min()} "
    f"to {train_df['target_date'].max()}"
)
print(
    f"Testing target dates:  {test_df['target_date'].min()} "
    f"to {test_df['target_date'].max()}"
)
print(f"Test period begins on: {cutoff_date.date()}")

assert (
    train_df["_target_day"].max()
    < test_df["_target_day"].min()
), "Training and test calendar days overlap."

print("\nTraining class distribution:")
print(train_df["target_status"].value_counts())

print("\nTesting class distribution:")
print(test_df["target_status"].value_counts())


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

# Only use information available from previous samples.
# Do not use target_date, station_id, or target_status as
# model predictors.
numeric_features = [
    "previous_wqi",
    "wqi_two_samples_ago",
]

categorical_features = [
    "river",
    "previous_status",
]

feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

y_train_text = train_df["target_status"].astype(str)
y_test_text = test_df["target_status"].astype(str)

# Encode class names as integers for XGBoost.
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_text)

# The encoder is fitted only on training labels.
# Unknown classes in the test set cannot be evaluated by this model.
known_classes = set(label_encoder.classes_)
unknown_test_classes = sorted(set(y_test_text) - known_classes)

if unknown_test_classes:
    raise ValueError(
        "The test period contains status classes absent from "
        f"training: {unknown_test_classes}. A model cannot learn "
        "classes it has never seen. Review the chronological "
        "split and available data."
    )

y_test = label_encoder.transform(y_test_text)

print("\nClasses learned from training data:")
print(list(label_encoder.classes_))


# ============================================================
# 5. PERSISTENCE BASELINE
# ============================================================

# Baseline assumption: next status stays the same as the
# most recently observed status.
baseline_predictions = test_df["previous_status"].astype(str).to_numpy()
actual_text = y_test_text.to_numpy()

baseline_accuracy = accuracy_score(
    actual_text, baseline_predictions
)
baseline_balanced_accuracy = balanced_accuracy_score(
    actual_text, baseline_predictions
)
baseline_macro_f1 = f1_score(
    actual_text,
    baseline_predictions,
    labels=label_encoder.classes_,
    average="macro",
    zero_division=0,
)

print("\n" + "-" * 65)
print("PERSISTENCE BASELINE")
print("-" * 65)
print(f"Accuracy:          {baseline_accuracy:.4f}")
print(f"Balanced accuracy: {baseline_balanced_accuracy:.4f}")
print(f"Macro F1-score:    {baseline_macro_f1:.4f}")


# ============================================================
# 6. BUILD AND TRAIN XGBOOST
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
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
        ("classifier", xgb_model),
    ]
)

# Balance the training classes without changing the test set.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

print("\nTraining XGBoost classifier...")

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)

predicted_codes = model.predict(X_test).astype(int)
xgb_predictions = label_encoder.inverse_transform(predicted_codes)

# Metrics
xgb_accuracy = accuracy_score(actual_text, xgb_predictions)
xgb_balanced_accuracy = balanced_accuracy_score(
    actual_text, xgb_predictions
)
xgb_macro_f1 = f1_score(
    actual_text,
    xgb_predictions,
    labels=label_encoder.classes_,
    average="macro",
    zero_division=0,
)

print("\n" + "-" * 65)
print("XGBOOST RESULTS")
print("-" * 65)
print(f"Accuracy:          {xgb_accuracy:.4f}")
print(f"Balanced accuracy: {xgb_balanced_accuracy:.4f}")
print(f"Macro F1-score:    {xgb_macro_f1:.4f}")

report = classification_report(
    actual_text,
    xgb_predictions,
    labels=label_encoder.classes_,
    zero_division=0,
)

print("\nClassification report:")
print(report)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

class_names = list(label_encoder.classes_)

matrix = confusion_matrix(
    actual_text,
    xgb_predictions,
    labels=class_names,
)

confusion_df = pd.DataFrame(
    matrix,
    index=[f"Actual_{name}" for name in class_names],
    columns=[f"Predicted_{name}" for name in class_names],
)

confusion_df.to_csv(CONFUSION_FILE)


# ============================================================
# 8. SAVE PREDICTIONS AND METRICS
# ============================================================

results_df = test_df[
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

results_df["baseline_prediction"] = baseline_predictions
results_df["xgboost_prediction"] = xgb_predictions
results_df["baseline_correct"] = (
    results_df["baseline_prediction"] == results_df["target_status"]
)
results_df["xgboost_correct"] = (
    results_df["xgboost_prediction"] == results_df["target_status"]
)

results_df.to_csv(PREDICTIONS_FILE, index=False)

metrics = {
    "task": "Next-sample river contamination status classification",
    "input_examples": int(len(df)),
    "training_examples": int(len(train_df)),
    "testing_examples": int(len(test_df)),
    "unique_stations": int(df["station_id"].nunique()),
    "training_target_start": str(train_df["target_date"].min()),
    "training_target_end": str(train_df["target_date"].max()),
    "testing_target_start": str(test_df["target_date"].min()),
    "testing_target_end": str(test_df["target_date"].max()),
    "classes": class_names,
    "baseline": {
        "accuracy": float(baseline_accuracy),
        "balanced_accuracy": float(baseline_balanced_accuracy),
        "macro_f1": float(baseline_macro_f1),
    },
    "xgboost": {
        "accuracy": float(xgb_accuracy),
        "balanced_accuracy": float(xgb_balanced_accuracy),
        "macro_f1": float(xgb_macro_f1),
    },
    "warning": (
        "Preliminary results from a small dataset. "
        "Not a validated operational water-quality warning system."
    ),
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("NEXT-SAMPLE CONTAMINATION STATUS CLASSIFICATION\n")
    file.write("=" * 60 + "\n\n")
    file.write("PERSISTENCE BASELINE\n")
    file.write(f"Accuracy: {baseline_accuracy:.4f}\n")
    file.write(
        f"Balanced accuracy: {baseline_balanced_accuracy:.4f}\n"
    )
    file.write(f"Macro F1-score: {baseline_macro_f1:.4f}\n\n")
    file.write("XGBOOST\n")
    file.write(f"Accuracy: {xgb_accuracy:.4f}\n")
    file.write(
        f"Balanced accuracy: {xgb_balanced_accuracy:.4f}\n"
    )
    file.write(f"Macro F1-score: {xgb_macro_f1:.4f}\n\n")
    file.write("CLASSIFICATION REPORT\n")
    file.write(report)
    file.write("\n\nCONFUSION MATRIX\n")
    file.write(confusion_df.to_string())
    file.write("\n\nIMPORTANT LIMITATION\n")
    file.write(metrics["warning"] + "\n")

print("\nSaved files:")
print(f"Predictions:     {PREDICTIONS_FILE}")
print(f"Metrics:         {METRICS_FILE}")
print(f"Report:          {REPORT_FILE}")
print(f"Confusion matrix:{CONFUSION_FILE}")

print("\nFinished.")