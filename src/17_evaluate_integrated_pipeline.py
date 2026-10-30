"""
17_evaluate_integrated_pipeline.py

Evaluates:
1. Status persistence baseline.
2. XGBoost using historical features only.
3. XGBoost using historical features + CEEMDAN-LSTM predicted WQI.

The script reuses the existing leakage-controlled CEEMDAN-LSTM
walk-forward predictions. It does not use actual target WQI as
an XGBoost feature.

IMPORTANT:
The available dataset is small. Treat results as preliminary.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = PROJECT_ROOT / "outputs" / "combined_doe_dataset.csv"

WQI_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "ceemdan_lstm_walk_forward_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTEGRATED_PREDICTIONS_FILE = (
    OUTPUT_DIR / "integrated_pipeline_predictions.csv"
)
METRICS_FILE = OUTPUT_DIR / "integrated_pipeline_metrics.json"
CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR / "integrated_pipeline_confusion_matrix.csv"
)
REPORT_FILE = OUTPUT_DIR / "integrated_pipeline_classification_report.txt"

RANDOM_SEED = 42

# Chronological split by target date.
# Earlier dates train the classifier; later dates test it.
TRAIN_DATE_FRACTION = 0.70

STATUS_CLASSES = ["Clean", "Polluted", "Slightly Polluted"]

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 2. HELPERS
# ============================================================

def normalize_status(value):
    """Standardise the source's contamination-status labels."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    mapping = {
        "clean": "Clean",
        "polluted": "Polluted",
        "slightly polluted": "Slightly Polluted",
    }

    return mapping.get(value, np.nan)


def calculate_wqi_metrics(actual, predicted):
    """Calculate WQI regression metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "n_predictions": 0,
            "MAE": None,
            "RMSE": None,
            "R2": None,
        }

    r2 = None
    if len(actual) >= 2 and np.var(actual) > 0:
        r2 = float(r2_score(actual, predicted))

    return {
        "n_predictions": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2,
    }


def calculate_status_metrics(actual, predicted):
    """Calculate classification metrics over the fixed status classes."""

    return {
        "n_predictions": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=STATUS_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "class_report": classification_report(
            actual,
            predicted,
            labels=STATUS_CLASSES,
            zero_division=0,
            output_dict=True,
        ),
    }


# ============================================================
# 3. LOAD CEEMDAN-LSTM FORECASTS
# ============================================================

def load_wqi_predictions():
    if not WQI_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing CEEMDAN-LSTM predictions:\n"
            f"{WQI_PREDICTIONS_FILE}\n\n"
            "Run src/16_validate_ceemdan_lstm.py first."
        )

    forecasts = pd.read_csv(WQI_PREDICTIONS_FILE)
    forecasts.columns = forecasts.columns.str.strip()

    required = [
        "river",
        "station_id",
        "target_date",
        "actual_wqi",
        "predicted_wqi",
    ]

    missing = [column for column in required if column not in forecasts]
    if missing:
        raise ValueError(
            f"Missing columns in CEEMDAN-LSTM predictions: {missing}"
        )

    forecasts["river"] = forecasts["river"].astype(str).str.strip()
    forecasts["station_id"] = (
        forecasts["station_id"].astype(str).str.strip()
    )
    forecasts["target_date"] = pd.to_datetime(
        forecasts["target_date"], errors="coerce"
    )
    forecasts["actual_wqi"] = pd.to_numeric(
        forecasts["actual_wqi"], errors="coerce"
    )
    forecasts["predicted_wqi"] = pd.to_numeric(
        forecasts["predicted_wqi"], errors="coerce"
    )

    forecasts = forecasts.dropna(
        subset=[
            "river",
            "station_id",
            "target_date",
            "actual_wqi",
            "predicted_wqi",
        ]
    ).copy()

    forecasts = forecasts.drop_duplicates(
        subset=["river", "station_id", "target_date"]
    )

    print(f"CEEMDAN-LSTM forecasts loaded: {len(forecasts)}")

    return forecasts


# ============================================================
# 4. PREPARE NEXT-SAMPLE RISK DATA
# ============================================================

def prepare_integrated_dataset(forecasts):
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw combined dataset not found:\n{RAW_DATA_FILE}"
        )

    df = pd.read_csv(RAW_DATA_FILE)

    # Normalize column names, including embedded line breaks
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("Normalized source columns:")
    print(df.columns.tolist())

    required = ["SUNGAI", "SMP-DAT", "WQI", "RIVER STATUS"]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw dataset: {missing}")

    if "ID STN BARU" in df.columns:
        station_column = "ID STN BARU"
    elif "ID STN (2016)" in df.columns:
        station_column = "ID STN (2016)"
    else:
        raise ValueError("Could not find a station ID column.")

    df["river"] = df["SUNGAI"].astype(str).str.strip()
    df["station_id"] = df[station_column].astype(str).str.strip()
    df["sample_date"] = pd.to_datetime(
        df["SMP-DAT"], errors="coerce"
    )
    df["WQI"] = pd.to_numeric(df["WQI"], errors="coerce")
    df["status"] = df["RIVER STATUS"].apply(normalize_status)

    df = df.dropna(
        subset=[
            "river",
            "station_id",
            "sample_date",
            "WQI",
            "status",
        ]
    ).copy()

    df = df[df["WQI"].between(0, 100)].copy()

    df = (
        df.sort_values(["river", "station_id", "sample_date"])
        .drop_duplicates(
            subset=["river", "station_id", "sample_date"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Build features from the previous sample(s), then attach the
    # status of the current target sample.
    station_rows = []

    for (river, station_id), station_df in df.groupby(
        ["river", "station_id"],
        sort=True,
    ):
        station_df = station_df.sort_values("sample_date").reset_index(
            drop=True
        )

        for target_index in range(2, len(station_df)):
            previous_row = station_df.iloc[target_index - 1]
            two_back_row = station_df.iloc[target_index - 2]
            target_row = station_df.iloc[target_index]

            station_rows.append({
                "river": river,
                "station_id": station_id,
                "target_date": target_row["sample_date"],
                "previous_wqi": float(previous_row["WQI"]),
                "wqi_two_samples_ago": float(two_back_row["WQI"]),
                "previous_status": previous_row["status"],
                "actual_status": target_row["status"],
                "actual_wqi_from_source": float(target_row["WQI"]),
                "previous_sample_date": previous_row["sample_date"],
            })

    risk = pd.DataFrame(station_rows)

    if risk.empty:
        raise ValueError("No next-sample risk examples were generated.")

    # Match the CEEMDAN-LSTM forecast to the same station and target date.
    forecasts = forecasts.rename(
        columns={
            "actual_wqi": "wqi_actual_from_forecast_file",
            "predicted_wqi": "predicted_wqi",
        }
    )

    merged = risk.merge(
        forecasts[
            [
                "river",
                "station_id",
                "target_date",
                "predicted_wqi",
                "wqi_actual_from_forecast_file",
            ]
        ],
        on=["river", "station_id", "target_date"],
        how="inner",
        validate="one_to_one",
    )

    # Safety check: the actual target WQI in both files should agree.
    merged["actual_wqi_match"] = np.isclose(
        merged["actual_wqi_from_source"],
        merged["wqi_actual_from_forecast_file"],
        atol=1e-6,
    )

    mismatches = int((~merged["actual_wqi_match"]).sum())

    if mismatches:
        raise ValueError(
            f"{mismatches} matched rows have inconsistent actual WQI. "
            "Check the source data and forecast file before proceeding."
        )

    merged = merged.drop(
        columns=["wqi_actual_from_forecast_file", "actual_wqi_match"]
    )

    merged = merged.sort_values(
        ["target_date", "river", "station_id"]
    ).reset_index(drop=True)

    print(f"Risk examples before forecast matching: {len(risk)}")
    print(f"Matched integrated examples: {len(merged)}")
    print(
        "Matched target dates:",
        merged["target_date"].min().date(),
        "to",
        merged["target_date"].max().date(),
    )
    print("\nTarget status distribution:")
    print(merged["actual_status"].value_counts().to_string())

    return merged


# ============================================================
# 5. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def chronological_split(df):
    unique_dates = sorted(df["target_date"].dropna().unique())

    if len(unique_dates) < 3:
        raise ValueError(
            "Too few unique target dates for a chronological split."
        )

    split_index = int(len(unique_dates) * TRAIN_DATE_FRACTION)
    split_index = max(1, min(split_index, len(unique_dates) - 1))

    cutoff_date = pd.Timestamp(unique_dates[split_index])

    train_df = df[df["target_date"] < cutoff_date].copy()
    test_df = df[df["target_date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty partition.")

    print("\nChronological split")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows:  {len(test_df)}")
    print(f"Training ends before: {cutoff_date.date()}")
    print(f"Test starts:         {test_df['target_date'].min().date()}")
    print(f"Test ends:           {test_df['target_date'].max().date()}")

    return train_df, test_df, cutoff_date


# ============================================================
# 6. TRAIN XGBOOST MODELS
# ============================================================

def make_feature_matrices(train_df, test_df, include_forecast):
    """
    One-hot encode categorical historical features.

    The model never receives actual target WQI or actual target status.
    """

    feature_columns = [
        "river",
        "previous_status",
        "previous_wqi",
        "wqi_two_samples_ago",
    ]

    if include_forecast:
        feature_columns.append("predicted_wqi")

    X_train_raw = train_df[feature_columns].copy()
    X_test_raw = test_df[feature_columns].copy()

    categorical = ["river", "previous_status"]

    X_train = pd.get_dummies(
        X_train_raw,
        columns=categorical,
        dtype=int,
    )
    X_test = pd.get_dummies(
        X_test_raw,
        columns=categorical,
        dtype=int,
    )

    # Ensure train and test use identical columns.
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    return X_train, X_test, feature_columns


def train_xgboost(train_df, test_df, include_forecast):
    X_train, X_test, used_features = make_feature_matrices(
        train_df,
        test_df,
        include_forecast,
    )

    y_train_status = train_df["actual_status"].astype(str)
    y_test_status = test_df["actual_status"].astype(str)

    # Encode only classes seen in training.
    # If a class first appears in the test period, the model cannot
    # learn it from training; this will be visible in the report.
    training_classes = sorted(y_train_status.unique())
    class_to_number = {
        class_name: index
        for index, class_name in enumerate(training_classes)
    }
    number_to_class = {
        index: class_name
        for class_name, index in class_to_number.items()
    }

    y_train = y_train_status.map(class_to_number).astype(int)

    model = XGBClassifier(
        objective=(
            "binary:logistic"
            if len(training_classes) == 2
            else "multi:softprob"
        ),
        num_class=(
            len(training_classes)
            if len(training_classes) > 2
            else None
        ),
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        eval_metric="mlogloss" if len(training_classes) > 2 else "logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    encoded_predictions = model.predict(X_test).astype(int)
    predictions = [
        number_to_class.get(code, "Unknown")
        for code in encoded_predictions
    ]

    result = {
        "actual_status": y_test_status.to_numpy(),
        "predicted_status": np.asarray(predictions),
        "model": model,
        "features": used_features,
        "training_classes": training_classes,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

    return result


# ============================================================
# 7. EVALUATE AND SAVE RESULTS
# ============================================================

def evaluate_pipeline(merged, train_df, test_df, cutoff_date):
    # Baseline: next status is assumed to equal previous status.
    persistence_predictions = test_df["previous_status"].astype(str).to_numpy()
    actual_status = test_df["actual_status"].astype(str).to_numpy()

    history_result = train_xgboost(
        train_df,
        test_df,
        include_forecast=False,
    )

    integrated_result = train_xgboost(
        train_df,
        test_df,
        include_forecast=True,
    )

    baseline_metrics = calculate_status_metrics(
        actual_status,
        persistence_predictions,
    )

    history_metrics = calculate_status_metrics(
        actual_status,
        history_result["predicted_status"],
    )

    integrated_metrics = calculate_status_metrics(
        actual_status,
        integrated_result["predicted_status"],
    )

    # Evaluate the WQI model on these same integrated test examples.
    wqi_metrics = calculate_wqi_metrics(
        test_df["actual_wqi_from_source"],
        test_df["predicted_wqi"],
    )

    # Save row-level predictions for dashboard charts and tables.
    output = test_df[
        [
            "river",
            "station_id",
            "previous_sample_date",
            "target_date",
            "previous_wqi",
            "wqi_two_samples_ago",
            "predicted_wqi",
            "actual_wqi_from_source",
            "previous_status",
            "actual_status",
        ]
    ].copy()

    output["persistence_status_prediction"] = persistence_predictions
    output["xgboost_history_prediction"] = history_result[
        "predicted_status"
    ]
    output["xgboost_integrated_prediction"] = integrated_result[
        "predicted_status"
    ]

    # Add class probabilities from the integrated model.
    integrated_model = integrated_result["model"]
    X_train, X_test, _ = make_feature_matrices(
        train_df,
        test_df,
        include_forecast=True,
    )
    probabilities = integrated_model.predict_proba(X_test)

    for class_index, class_name in enumerate(
        integrated_result["training_classes"]
    ):
        output[f"probability_{class_name.lower().replace(' ', '_')}"] = (
            probabilities[:, class_index]
        )

    output.to_csv(INTEGRATED_PREDICTIONS_FILE, index=False)

    # Confusion matrix for the integrated XGBoost configuration.
    matrix = confusion_matrix(
        actual_status,
        integrated_result["predicted_status"],
        labels=STATUS_CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in STATUS_CLASSES],
        columns=[f"predicted_{name}" for name in STATUS_CLASSES],
    )
    matrix_df.to_csv(CONFUSION_MATRIX_FILE)

    report = classification_report(
        actual_status,
        integrated_result["predicted_status"],
        labels=STATUS_CLASSES,
        zero_division=0,
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write("Integrated XGBoost classification report\n")
        file.write(f"Training rows: {len(train_df)}\n")
        file.write(f"Testing rows: {len(test_df)}\n")
        file.write(f"Test period begins: {test_df['target_date'].min()}\n\n")
        file.write(report)

    metrics = {
        "experiment": "CEEMDAN-LSTM + XGBoost integrated evaluation",
        "data_source": str(RAW_DATA_FILE),
        "ceemdan_lstm_predictions_source": str(WQI_PREDICTIONS_FILE),
        "split_method": "Chronological split by target date",
        "training_date_cutoff": str(cutoff_date.date()),
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "test_period_start": str(test_df["target_date"].min().date()),
        "test_period_end": str(test_df["target_date"].max().date()),
        "wqi_forecasting_metrics_on_integrated_test_rows": wqi_metrics,
        "risk_classification": {
            "persistence_baseline": baseline_metrics,
            "xgboost_history_only": history_metrics,
            "xgboost_history_plus_predicted_wqi": integrated_metrics,
        },
        "files": {
            "predictions": str(INTEGRATED_PREDICTIONS_FILE),
            "confusion_matrix": str(CONFUSION_MATRIX_FILE),
            "classification_report": str(REPORT_FILE),
        },
        "limitations": [
            "Only dates with available CEEMDAN-LSTM walk-forward forecasts are evaluated.",
            "The integrated test set is small and contains only later target dates.",
            "A class absent from training cannot be learned by XGBoost.",
            "Results depend on the current 2023-2024 DOE dataset.",
            "The status labels reflect the supplied dataset's categories; "
            "they are not a separate independently measured contamination-risk score.",
            "More DOE data and repeated chronological validation are needed "
            "before drawing final conclusions.",
        ],
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)

    # Print concise, comparable results.
    print("\n" + "=" * 68)
    print("INTEGRATED PIPELINE EVALUATION")
    print("=" * 68)
    print(f"Test rows: {len(test_df)}")
    print(
        f"Test dates: {test_df['target_date'].min().date()} "
        f"to {test_df['target_date'].max().date()}"
    )

    print("\nWQI forecast metrics on the integrated test rows:")
    print(json.dumps(wqi_metrics, indent=4))

    print("\nContamination-status classification:")
    print(
        f"{'Model':42s} {'Accuracy':>10s} "
        f"{'Balanced Acc.':>14s} {'Macro F1':>10s}"
    )
    print("-" * 80)

    for name, result in [
        ("Status persistence baseline", baseline_metrics),
        ("XGBoost: history only", history_metrics),
        ("XGBoost: history + predicted WQI", integrated_metrics),
    ]:
        print(
            f"{name:42s} "
            f"{result['accuracy']:10.3f} "
            f"{result['balanced_accuracy']:14.3f} "
            f"{result['macro_f1']:10.3f}"
        )

    print("\nIntegrated model classification report:")
    print(report)

    print("\nSaved files:")
    print(f"  Predictions:      {INTEGRATED_PREDICTIONS_FILE}")
    print(f"  Metrics:          {METRICS_FILE}")
    print(f"  Confusion matrix: {CONFUSION_MATRIX_FILE}")
    print(f"  Report:           {REPORT_FILE}")

    print(
        "\nInterpretation: compare history-only XGBoost against "
        "history + predicted WQI on the same test rows. "
        "If the integrated model does not improve balanced accuracy "
        "or macro F1, the forecast may not add useful risk-classification "
        "information in this dataset."
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 68)
    print("INTEGRATED CEEMDAN-LSTM-XGBOOST EVALUATION")
    print("=" * 68)

    forecasts = load_wqi_predictions()
    merged = prepare_integrated_dataset(forecasts)

    if len(merged) < 10:
        raise RuntimeError(
            "Too few matched examples for a meaningful preliminary split."
        )

    train_df, test_df, cutoff_date = chronological_split(merged)

    if train_df["actual_status"].nunique() < 2:
        raise RuntimeError(
            "Training data contains fewer than two status classes. "
            "More training data is required."
        )

    evaluate_pipeline(
        merged=merged,
        train_df=train_df,
        test_df=test_df,
        cutoff_date=cutoff_date,
    )


if __name__ == "__main__":
    main()