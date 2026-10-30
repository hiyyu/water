"""
20_water_quality_dashboard.py
Launch script for the Water Quality Prediction and Contamination Risk Dashboard.

Usage:
    streamlit run app.py
or:
    python src/20_water_quality_dashboard.py
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"

def main():
    print("=" * 70)
    print("LAUNCHING WATER QUALITY PREDICTION & CONTAMINATION RISK DASHBOARD")
    print("=" * 70)
    print(f"Target App: {APP_PATH}")
    print("Models Available: Persistence, CEEMDAN, LSTM, CEEMDAN-LSTM, XGBoost, CEEMDAN-LSTM-XGBoost")
    print("Rivers Monitored: Sungai Klang, Sungai Skudai (22 stations)")
    print("=" * 70)
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless=true"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
