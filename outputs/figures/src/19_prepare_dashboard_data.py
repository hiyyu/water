from pathlib import Path
import json
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_DIR / "outputs" / "evaluation"
RAW_DATA_FILE = PROJECT_DIR / "outputs" / "combined_doe_dataset.csv"
DASHBOARD_DIR = PROJECT_DIR / "outputs" / "dashboard"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Clean", "Polluted", "Slightly Polluted"]


def read_csv(filename):
    """Read a CSV file if it exists."""
    path = EVALUATION_DIR / filename

    if not path.exists():
        print(f"[SKIP] Missing file: {filename}")
        return None

    df = pd.read_csv(path)
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    print(f"[OK] {filename}: {len(df)} rows")
    return df


def read_json(filename):
    """Read a JSON file if it exists."""
    path = EVALUATION_DIR / filename

    if not path.exists():
        print(f"[SKIP] Missing file: {filename}")
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_date(series):
    """Standardize timestamps for safe date matching."""
    return pd.to_datetime(series, errors="coerce")


def save_csv(df, filename):
    """Save a dashboard CSV."""
    path = DASHBOARD_DIR / filename
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(df)} rows)")


# ============================================================
# 1. LATEST WQI FORECASTS
# ============================================================

def prepare_latest_wqi_forecasts():
    forecasts = read_csv("ceemdan_lstm_forecasts.csv")

    if forecasts is None:
        return pd.DataFrame()

    required = [
        "river",
        "station_id",
        "last_sample_date",
        "forecast_date_estimate",
        "last_observed_wqi",
        "predicted_next_wqi",
        "persistence_baseline",
    ]

    missing = [col for col in required if col not in forecasts.columns]

    if missing:
        print(f"[WARNING] Latest forecasts missing columns: {missing}")
        return pd.DataFrame()

    result = forecasts[required].copy()

    result = result.rename(
        columns={
            "last_sample_date": "last_observation_date",
            "forecast_date_estimate": "forecast_date",
            "last_observed_wqi": "last_observed_wqi",
            "predicted_next_wqi": "predicted_wqi",
            "persistence_baseline": "persistence_prediction",
        }
    )

    result["model"] = "CEEMDAN-LSTM"
    result["forecast_type"] = "Latest station-level estimate"
    result["forecast_date"] = normalize_date(result["forecast_date"])
    result["last_observation_date"] = normalize_date(
        result["last_observation_date"]
    )

    result["predicted_wqi"] = pd.to_numeric(
        result["predicted_wqi"], errors="coerce"
    )
    result["last_observed_wqi"] = pd.to_numeric(
        result["last_observed_wqi"], errors="coerce"
    )

    # These are forecasts, so actual future WQI is not available here.
    result["actual_wqi"] = pd.NA

    return result


# ============================================================
# 2. HISTORICAL WQI EVALUATION PREDICTIONS
# ============================================================

def prepare_wqi_evaluation_predictions():
    frames = []

    # CEEMDAN-LSTM walk-forward predictions
    ceemdan = read_csv("ceemdan_lstm_walk_forward_predictions.csv")

    if ceemdan is not None:
        required = [
            "river", "station_id", "target_date",
            "actual_wqi", "predicted_wqi",
            "persistence_prediction",
        ]

        if all(col in ceemdan.columns for col in required):
            for model_name, prediction_col in [
                ("CEEMDAN-LSTM", "predicted_wqi"),
                ("Persistence (CEEMDAN evaluation)", "persistence_prediction"),
            ]:
                part = ceemdan[
                    ["river", "station_id", "target_date", "actual_wqi",
                     prediction_col]
                ].copy()

                part = part.rename(
                    columns={
                        "target_date": "sample_date",
                        "actual_wqi": "actual_wqi",
                        prediction_col: "predicted_wqi",
                    }
                )

                part["model"] = model_name
                part["evaluation_source"] = (
                    "CEEMDAN-LSTM station-wise walk-forward evaluation"
                )
                frames.append(part)

    # Walk-forward persistence and linear regression predictions
    baseline = read_csv("walk_forward_predictions.csv")

    if baseline is not None:
        required = [
            "river", "station_id", "next_sample_date", "actual_wqi",
            "persistence_prediction", "linear_prediction",
        ]

        if all(col in baseline.columns for col in required):
            for model_name, prediction_col in [
                ("Persistence (baseline walk-forward)", "persistence_prediction"),
                ("Linear Regression (walk-forward)", "linear_prediction"),
            ]:
                part = baseline[
                    ["river", "station_id", "next_sample_date",
                     "actual_wqi", prediction_col]
                ].copy()

                part = part.rename(
                    columns={
                        "next_sample_date": "sample_date",
                        prediction_col: "predicted_wqi",
                    }
                )

                part["model"] = model_name
                part["evaluation_source"] = "Baseline walk-forward evaluation"
                frames.append(part)

    # Original single chronological split
    single_split = read_csv("baseline_wqi_test_predictions.csv")

    if single_split is not None:
        required = [
            "river", "station_id", "next_sample_date", "next_sample_wqi",
            "persistence_prediction", "linear_prediction",
        ]

        if all(col in single_split.columns for col in required):
            for model_name, prediction_col in [
                ("Persistence (single split)", "persistence_prediction"),
                ("Linear Regression (single split)", "linear_prediction"),
            ]:
                part = single_split[
                    ["river", "station_id", "next_sample_date",
                     "next_sample_wqi", prediction_col]
                ].copy()

                part = part.rename(
                    columns={
                        "next_sample_date": "sample_date",
                        "next_sample_wqi": "actual_wqi",
                        prediction_col: "predicted_wqi",
                    }
                )

                part["model"] = model_name
                part["evaluation_source"] = "Single chronological test split"
                frames.append(part)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["sample_date"] = normalize_date(result["sample_date"])
    result["actual_wqi"] = pd.to_numeric(
        result["actual_wqi"], errors="coerce"
    )
    result["predicted_wqi"] = pd.to_numeric(
        result["predicted_wqi"], errors="coerce"
    )

    result = result.dropna(
        subset=["river", "station_id", "sample_date",
                "actual_wqi", "predicted_wqi"]
    )

    result["error"] = result["predicted_wqi"] - result["actual_wqi"]
    result["absolute_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2

    # Keep distinct experiments separate. Do not deduplicate by date alone.
    return result


# ============================================================
# 3. CONTAMINATION-STATUS EVALUATION PREDICTIONS
# ============================================================

def prepare_risk_evaluation_predictions():
    frames = []

    sources = [
        (
            "risk_classifier_predictions.csv",
            "XGBoost (history only)",
            "target_status",
            "xgboost_prediction",
        ),
        (
            "risk_classifier_extended_predictions.csv",
            "XGBoost (historical water parameters)",
            "target_status",
            "xgboost_prediction",
        ),
        (
            "integrated_pipeline_predictions.csv",
            "XGBoost (integrated with predicted WQI)",
            "actual_status",
            "xgboost_integrated_prediction",
        ),
    ]

    for filename, model_name, actual_col, predicted_col in sources:
        df = read_csv(filename)

        if df is None:
            continue

        required = [
            "river", "station_id", "target_date",
            actual_col, predicted_col,
        ]

        if not all(col in df.columns for col in required):
            print(f"[WARNING] Required columns missing in {filename}")
            continue

        columns = [
            "river", "station_id", "target_date",
            actual_col, predicted_col,
        ]

        # Keep integrated probabilities if they are present.
        for col in [
            "probability_clean",
            "probability_polluted",
            "probability_slightly_polluted",
        ]:
            if col in df.columns:
                columns.append(col)

        part = df[columns].copy()
        part = part.rename(
            columns={
                "target_date": "sample_date",
                actual_col: "actual_status",
                predicted_col: "predicted_status",
            }
        )

        part["model"] = model_name
        part["evaluation_source"] = filename
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["sample_date"] = normalize_date(result["sample_date"])

    return result


# ============================================================
# 4. UNIFIED MODEL METRICS
# ============================================================

def prepare_model_metrics():
    rows = []

    def add_metric_row(
        model, task, evaluation, n=None, mae=None, rmse=None,
        r2=None, accuracy=None, balanced_accuracy=None, macro_f1=None
    ):
        rows.append({
            "model": model,
            "task": task,
            "evaluation": evaluation,
            "n_predictions": n,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        })

    # Single-split WQI baseline metrics
    baseline_metrics = read_csv("baseline_wqi_metrics.csv")

    if baseline_metrics is not None:
        for _, row in baseline_metrics.iterrows():
            add_metric_row(
                row["model"], "WQI forecasting", "Single chronological split",
                mae=row["MAE"], rmse=row["RMSE"], r2=row["R2"]
            )

    # Walk-forward WQI baseline metrics
    walk_metrics = read_csv("walk_forward_metrics.csv")

    if walk_metrics is not None:
        for _, row in walk_metrics.iterrows():
            add_metric_row(
                row["model"], "WQI forecasting", "Walk-forward validation",
                mae=row["MAE"], rmse=row["RMSE"], r2=row["R2"]
            )

    # CEEMDAN-LSTM overall metrics
    ceemdan_metrics = read_json("ceemdan_lstm_walk_forward_metrics.json")

    if ceemdan_metrics:
        m = ceemdan_metrics.get("model_metrics", {})
        add_metric_row(
            "CEEMDAN-LSTM", "WQI forecasting", "Walk-forward validation",
            n=m.get("n_predictions"),
            mae=m.get("MAE"), rmse=m.get("RMSE"), r2=m.get("R2")
        )

        m = ceemdan_metrics.get("persistence_baseline_metrics", {})
        add_metric_row(
            "Persistence", "WQI forecasting",
            "CEEMDAN evaluation comparison",
            n=m.get("n_predictions"),
            mae=m.get("MAE"), rmse=m.get("RMSE"), r2=m.get("R2")
        )

    # Contamination-status classification metrics
    risk_metrics = read_json("risk_classifier_metrics.json")

    if risk_metrics:
        for key, label in [
            ("baseline", "Persistence status baseline"),
            ("xgboost", "XGBoost (history only)"),
        ]:
            m = risk_metrics.get(key, {})
            add_metric_row(
                label, "Contamination-status classification",
                "Chronological test split",
                n=risk_metrics.get("testing_examples"),
                accuracy=m.get("accuracy"),
                balanced_accuracy=m.get("balanced_accuracy"),
                macro_f1=m.get("macro_f1"),
            )

    extended_metrics = read_json("risk_classifier_extended_metrics.json")

    if extended_metrics:
        for key, label in [
            ("persistence_baseline", "Persistence status baseline"),
            ("xgboost", "XGBoost (historical water parameters)"),
        ]:
            m = extended_metrics.get(key, {})
            add_metric_row(
                label, "Contamination-status classification",
                "Chronological test split",
                n=extended_metrics.get("testing_examples"),
                accuracy=m.get("accuracy"),
                balanced_accuracy=m.get("balanced_accuracy"),
                macro_f1=m.get("macro_f1"),
            )

    integrated_metrics = read_json("integrated_pipeline_metrics.json")

    if integrated_metrics:
        risk = integrated_metrics.get("risk_classification", {})

        for key, label in [
            ("persistence_baseline", "Persistence status baseline"),
            ("xgboost_history_only", "XGBoost (history only, integrated test)"),
            (
                "xgboost_history_plus_predicted_wqi",
                "XGBoost (integrated with predicted WQI)",
            ),
        ]:
            m = risk.get(key, {})
            add_metric_row(
                label, "Contamination-status classification",
                "Integrated chronological test split",
                n=m.get("n_predictions"),
                accuracy=m.get("accuracy"),
                balanced_accuracy=m.get("balanced_accuracy"),
                macro_f1=m.get("macro_f1"),
            )

        m = integrated_metrics.get(
            "wqi_forecasting_metrics_on_integrated_test_rows", {}
        )
        add_metric_row(
            "CEEMDAN-LSTM", "WQI forecasting",
            "Integrated test subset",
            n=m.get("n_predictions"),
            mae=m.get("MAE"), rmse=m.get("RMSE"), r2=m.get("R2")
        )

    return pd.DataFrame(rows)


# ============================================================
# 5. CONFUSION MATRICES IN LONG FORMAT
# ============================================================

def prepare_confusion_matrices():
    files = [
        ("risk_classifier_confusion_matrix.csv",
         "XGBoost (history only)"),
        ("risk_classifier_extended_confusion_matrix.csv",
         "XGBoost (historical water parameters)"),
        ("integrated_pipeline_confusion_matrix.csv",
         "XGBoost (integrated with predicted WQI)"),
    ]

    rows = []

    for filename, model_name in files:
        df = read_csv(filename)

        if df is None or df.empty:
            continue

        actual_col = df.columns[0]

        for _, record in df.iterrows():
            actual_label = str(record[actual_col]).replace("actual_", "")
            actual_label = actual_label.replace("Actual_", "")

            for column in df.columns[1:]:
                if not column.lower().startswith("predicted_"):
                    continue

                predicted_label = column[len("predicted_"):]
                count = int(record[column])

                rows.append({
                    "model": model_name,
                    "actual_status": actual_label,
                    "predicted_status": predicted_label,
                    "count": count,
                })

    return pd.DataFrame(rows)


# ============================================================
# 6. LATEST OBSERVED WATER QUALITY
# ============================================================

def prepare_latest_observations():
    if not RAW_DATA_FILE.exists():
        print(f"[SKIP] Raw data not found: {RAW_DATA_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(RAW_DATA_FILE)
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    required = ["SUNGAI", "SMP-DAT", "WQI", "RIVER STATUS"]

    if not all(col in df.columns for col in required):
        print("[WARNING] Raw data does not contain all expected columns.")
        return pd.DataFrame()

    station_col = (
        "ID STN BARU"
        if "ID STN BARU" in df.columns
        else "ID STN (2016)"
    )

    df["SMP-DAT"] = normalize_date(df["SMP-DAT"])
    df["WQI"] = pd.to_numeric(df["WQI"], errors="coerce")
    df["river"] = df["SUNGAI"].astype(str).str.strip().str.upper()
    df["station_id"] = df[station_col].astype(str).str.strip()

    df = df.dropna(subset=["SMP-DAT", "WQI"])
    df = df.sort_values("SMP-DAT")

    latest = df.groupby(
        ["river", "station_id"], as_index=False
    ).tail(1).copy()

    result = latest.rename(
        columns={
            "SMP-DAT": "last_sample_date",
            "WQI": "last_observed_wqi",
            "RIVER STATUS": "last_observed_status",
            "LOCATION": "station_location",
        }
    )

    wanted = [
        "river", "station_id", "station_location",
        "last_sample_date", "last_observed_wqi",
        "last_observed_status",
    ]

    wanted = [col for col in wanted if col in result.columns]

    return result[wanted].sort_values(["river", "station_id"])


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PREPARING WATER QUALITY DASHBOARD DATA")
    print("=" * 70)

    latest_forecasts = prepare_latest_wqi_forecasts()
    wqi_predictions = prepare_wqi_evaluation_predictions()
    risk_predictions = prepare_risk_evaluation_predictions()
    metrics = prepare_model_metrics()
    confusion = prepare_confusion_matrices()
    latest_observations = prepare_latest_observations()

    if not latest_forecasts.empty:
        save_csv(latest_forecasts, "wqi_latest_forecasts.csv")

    if not wqi_predictions.empty:
        save_csv(wqi_predictions, "wqi_evaluation_predictions.csv")

    if not risk_predictions.empty:
        save_csv(risk_predictions, "risk_evaluation_predictions.csv")

    if not metrics.empty:
        save_csv(metrics, "model_metrics.csv")

    if not confusion.empty:
        save_csv(confusion, "classification_confusion_matrices.csv")

    if not latest_observations.empty:
        save_csv(latest_observations, "latest_station_observations.csv")

    # Dashboard metadata
    metadata = {
        "project": "Water Quality Index and Contamination Status Prediction",
        "dashboard_data_generated": pd.Timestamp.now().isoformat(),
        "data_period": "2023-2024",
        "rivers_in_current_dataset": ["KLANG", "SKUDAI"],
        "station_count_in_latest_forecasts": (
            int(latest_forecasts["station_id"].nunique())
            if not latest_forecasts.empty else 0
        ),
        "latest_forecast_rows": len(latest_forecasts),
        "wqi_evaluation_rows": len(wqi_predictions),
        "risk_evaluation_rows": len(risk_predictions),
        "model_metric_rows": len(metrics),
        "confusion_matrix_rows": len(confusion),
        "status_classes": CLASSES,
        "notes": [
            "Latest forecasts are estimates, not observed future WQI values.",
            "Historical evaluation predictions are separate from latest forecasts.",
            "Different evaluation splits must not be compared as if they were identical test sets.",
            "Risk status is the supplied RIVER STATUS category, not an independently measured risk score.",
            "Current outputs are preliminary and based on the available DOE dataset.",
        ],
    }

    metadata_path = DASHBOARD_DIR / "dashboard_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[SAVED] {metadata_path}")

    print("\n" + "=" * 70)
    print("DASHBOARD DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Output folder: {DASHBOARD_DIR}")
    print("Review the generated CSV and JSON files before dashboard development.")


if __name__ == "__main__":
    main()