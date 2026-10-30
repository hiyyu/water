"""
Consolidate and generate model predictions and daily forecasts for all 6 models:
1. Persistence
2. CEEMDAN
3. LSTM
4. CEEMDAN-LSTM
5. XGBoost
6. CEEMDAN-LSTM-XGBoost
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "outputs" / "cleaned"
EVAL_DIR = ROOT / "outputs" / "evaluation"
DASH_DIR = ROOT / "outputs" / "dashboard"

DATASET_FILE = CLEANED_DIR / "doe_dataset_cleaned_working.csv"

def classify_wqi(wqi):
    if pd.isna(wqi):
        return np.nan
    if wqi >= 81:
        return "Clean"
    elif wqi >= 60:
        return "Slightly Polluted"
    else:
        return "Polluted"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE)
    df["smp_dat"] = pd.to_datetime(df["smp_dat"])
    df = df.sort_values(["sungai", "id_stn_baru", "smp_dat"]).reset_index(drop=True)
    
    print(f"Loaded {len(df)} records across {df['id_stn_baru'].nunique()} stations.")

if __name__ == "__main__":
    main()
