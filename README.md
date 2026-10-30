# Prediction of Water Quality Index and Contamination Risk Using Machine Learning Techniques

An interactive, multi-model machine learning dashboard for predicting the Water Quality Index (WQI) and classifying contamination risk across Malaysian river basins (Sungai Klang and Sungai Skudai), based on official Department of Environment (DOE) monitoring standards.

---

## 🌟 Key Features

- **Overview tab**: status hero card, data-freshness banner, weekly forecast strip, monthly outlook, latest water parameters with NWQS Class III alert, station map, and permitted water uses for the forecast class.
- **Forecast tabs (Daily / Weekly / Monthly)**: aggregate by all stations, state (Selangor, W.P. Kuala Lumpur, Johor), river (Sungai Klang / Skudai) or a single station. Each shows DOE-coloured WQI bands, a ±1 RMSE uncertainty band from the model backtest, a table and CSV download. Optionally overlay other models for comparison.
- **State & River Map**: state boundaries shaded by mean station WQI, plus every station coloured by DOE status. View the next-sampling forecast, the latest DOE sample, or any date; zoom to Klang Valley or Johor.

> **How daily/weekly/monthly values are produced.** DOE samples each station about every 60 days and every model forecasts one value per station (the next sampling date). The daily series is a straight-line trajectory between real points; weekly and monthly values are averages of it. Forecast horizon is therefore about 60 days. For true daily/weekly/monthly forecasts, retrain on higher-frequency data.
>
> **Keeping it current.** The forecasts are for the sampling date after the newest DOE sample in the data. To see current conditions, add newer DOE records to `data/` and re-run the pipeline (`src/02_...` onward, finishing with `src/train_all_models.py`). The Overview banner shows how old the data is.

- **Interactive Water Quality Trend & Forecast Charts** (legacy "Trend & Date Snapshot" tab):
  - Colored horizontal background bands representing official Malaysian DOE standards:
    - 🔵 **Clean**: WQI 81 – 100
    - 🟡 **Slightly Polluted**: WQI 60 – 80
    - 🔴 **Polluted**: WQI 0 – 59
  - Continuous time-series curves combining historical measurements with future daily forecast trajectories.
  - Interactive date pinpoint marker displaying immediate WQI, DOE class, and contamination risk status.

- **Support for All 6 Machine Learning & Baseline Models**:
  1. **Persistence Baseline**: Lag-1 constant forecast ($y_t = y_{t-1}$)
  2. **CEEMDAN**: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
  3. **LSTM**: Long Short-Term Memory recurrent neural network
  4. **CEEMDAN-LSTM**: Decomposition-coupled deep sequence forecasting (MAE: 6.12, $R^2$: 0.659)
  5. **XGBoost**: Gradient-boosted decision trees for WQI regression and risk classification
  6. **CEEMDAN-LSTM-XGBoost**: Integrated two-stage hybrid ensemble framework (MAE: 5.64, $R^2$: 0.692)

- **Station & Basin Filtering (Faithfully Matching the Wireframe)**:
  - **Show both stations (default)**: Parallel side-by-side / stacked views for Sungai Skudai and Sungai Klang.
  - **Sungai Skudai**: Dedicated view across all 9 Skudai monitoring stations in Johor.
  - **Sungai Klang**: Dedicated view across all 13 Klang monitoring stations in Selangor.
  - **Specific Station**: High-resolution focus on any individual monitoring station.

- **Parameter Filtering**:
  - WQI Composite Overview
  - Dissolved Oxygen (DO in mg/L and % Saturation)
  - Biochemical Oxygen Demand (BOD in mg/L)
  - Chemical Oxygen Demand (COD in mg/L)
  - Ammoniacal Nitrogen ($NH_3\text{-N}$ in mg/L)
  - Total Suspended Solids (SS in mg/L)
  - pH Level

- **"Do It Better" Advanced Capabilities**:
  - 🗺️ **Geographic Station Map**: Interactive map with real GPS coordinates of all 22 monitoring stations, color-coded by real-time contamination risk.
  - 🔮 **Daily Multi-Day Forecast Horizon Explorer**: Day-by-day projected WQI table up to 90 days ahead with downloadable CSV reports.
  - 🔬 **Multi-Model Benchmark Arena**: Side-by-side metric tables (MAE, RMSE, $R^2$, Accuracy, Macro F1) and comparative visual charts.
  - 🧪 **Interactive What-If Scenario Simulator**: Adjust DO, BOD, COD, $NH_3\text{-N}$, SS, and pH sliders to compute real-time Malaysian DOE Sub-indices (SIDO, SIBOD, SICOD, SIAN, SISS, SIpH), composite WQI, DOE class (I–V), and predicted risk category.

---

## 🚀 Quick Start

### 1. Launch the Dashboard

Run the Streamlit application directly:

```bash
streamlit run app.py
```

Or execute via the workflow script:

```bash
python src/20_water_quality_dashboard.py
```

Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
├── app.py                                # Main Streamlit Dashboard Application
├── requirements.txt                      # Dependencies
├── README.md                             # Documentation
├── data/                                 # Raw DOE datasets (Excel workbook)
├── outputs/
│   ├── cleaned/                          # Processed and cleaned datasets
│   │   ├── doe_dataset_cleaned_working.csv
│   │   ├── wqi_forecasting_base.csv
│   │   └── wqi_next_sample_lags.csv
│   ├── dashboard/                        # Unified dashboard data & forecasts
│   │   ├── all_models_latest_forecasts.csv
│   │   ├── all_models_metrics.csv
│   │   ├── all_models_wqi_predictions.csv
│   │   └── latest_station_observations.csv
│   └── evaluation/                       # Model evaluation outputs & reports
├── assets/malaysia_states.geojson        # State boundaries for the map
└── src/
    ├── 01_data_inspection.py
    ├── 02_data_cleaning_eda.py
    ├── 06_prepare_wqi_forecasting_data.py
    ├── 07_baseline_wqi_forecasting.py
    ├── 10_ceemdan_lstm_forecasting.py
    ├── 14_train_risk_classifier.py
    ├── 16_validate_ceemdan_lstm.py
    ├── 17_evaluate_integrated_pipeline.py
    ├── 19_prepare_dashboard_data.py
    ├── 20_water_quality_dashboard.py      # Dashboard launcher
    ├── model_engine.py                   # Unified backend inference engine (series + aggregation)
    ├── forecast_views.py                 # Plotly figures for forecast tabs and map
    ├── test_forecast_series.py           # Sanity checks: python src/test_forecast_series.py
    └── train_all_models.py               # All-model unification pipeline
```

---

## 📊 Malaysian DOE Water Quality Index Formula

$$\text{WQI} = 0.22 \times \text{SIDO} + 0.19 \times \text{SIBOD} + 0.16 \times \text{SICOD} + 0.15 \times \text{SIAN} + 0.16 \times \text{SISS} + 0.12 \times \text{SIpH}$$

### Risk Status & River Classification:
- **Clean**: $\text{WQI} \ge 81$ (Class I & II)
- **Slightly Polluted**: $60 \le \text{WQI} \le 80$ (Class III)
- **Polluted**: $\text{WQI} < 60$ (Class IV & V)
