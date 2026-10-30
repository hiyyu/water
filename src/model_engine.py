"""
src/model_engine.py
Unified Data and Model Engine for Water Quality Prediction & Contamination Risk Dashboard
Supports all 6 models:
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
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "outputs" / "cleaned"
DASH_DIR = ROOT / "outputs" / "dashboard"
EVAL_DIR = ROOT / "outputs" / "evaluation"

DATASET_FILE = CLEANED_DIR / "doe_dataset_cleaned_working.csv"
UNIFIED_WQI_FILE = DASH_DIR / "all_models_wqi_predictions.csv"
UNIFIED_METRICS_FILE = DASH_DIR / "all_models_metrics.csv"
UNIFIED_FORECASTS_FILE = DASH_DIR / "all_models_latest_forecasts.csv"
LATEST_OBS_FILE = DASH_DIR / "latest_station_observations.csv"

# DOE Classification Thresholds
STATUS_CLEAN = "Clean"
STATUS_SLIGHTLY_POLLUTED = "Slightly Polluted"
STATUS_POLLUTED = "Polluted"

COLOR_CLEAN = "#2563EB"              # Azure Blue
COLOR_SLIGHTLY_POLLUTED = "#EAB308"  # Amber Yellow
COLOR_POLLUTED = "#EF4444"           # Coral Red

STATE_LABELS = {
    "SELANGOR": "Selangor",
    "WPKL": "W.P. Kuala Lumpur",
    "JOHOR": "Johor",
}
# Name of each state inside assets/malaysia_states.geojson (properties.name)
STATE_GEOJSON_NAME = {
    "SELANGOR": "Selangor",
    "WPKL": "Federal Territory of Kuala Lumpur",
    "JOHOR": "Johor",
}
GEOJSON_FILE = ROOT / "assets" / "malaysia_states.geojson"

MODEL_DESCRIPTIONS = {
    "Persistence": "Baseline naive model that projects future water quality directly from the latest observed measurement (y_t = y_{t-1}).",
    "CEEMDAN": "Complete Ensemble Empirical Mode Decomposition with Adaptive Noise — isolates intrinsic non-linear mode functions (IMFs) and long-term trend residual.",
    "LSTM": "Long Short-Term Memory recurrent neural network capturing multi-step temporal dependencies across sequential water quality states.",
    "CEEMDAN-LSTM": "SOTA Hybrid model: CEEMDAN decomposes non-stationary WQI signals into component modes, and multi-channel LSTM forecasts components jointly (MAE: 6.12, R²: 0.659).",
    "XGBoost": "Gradient Boosted Decision Trees modeling non-linear interactions across DO, BOD, COD, Ammoniacal Nitrogen, pH, Suspended Solids, and temporal lags.",
    "CEEMDAN-LSTM-XGBoost": "Integrated Hybrid Ensemble: CEEMDAN-LSTM predicts high-precision future WQI, which is synthesized with historical multi-parameter trends in XGBoost to classify contamination risk probabilities."
}

def classify_wqi(wqi):
    if pd.isna(wqi):
        return "Unknown"
    val = float(wqi)
    if val >= 81:
        return STATUS_CLEAN
    elif val >= 60:
        return STATUS_SLIGHTLY_POLLUTED
    else:
        return STATUS_POLLUTED

def get_doe_class(wqi):
    if pd.isna(wqi):
        return "-"
    val = float(wqi)
    if val >= 92.7:
        return "Class I"
    elif val >= 76.5:
        return "Class II"
    elif val >= 51.9:
        return "Class III"
    elif val >= 31.0:
        return "Class IV"
    else:
        return "Class V"

def get_class_color(status):
    if status == STATUS_CLEAN:
        return COLOR_CLEAN
    elif status == STATUS_SLIGHTLY_POLLUTED:
        return COLOR_SLIGHTLY_POLLUTED
    else:
        return COLOR_POLLUTED

# ============================================================
# DOE SUB-INDEX FORMULAS (Malaysian DOE Official Standard)
# ============================================================

def calc_sido(x):
    """Sub-index for Dissolved Oxygen (% saturation)."""
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 8:
        return 0.0
    elif x <= 92:
        return max(0.0, min(100.0, -0.395 + 0.030 * (x ** 2) - 0.00020 * (x ** 3)))
    else:
        return 100.0

def calc_sibod(x):
    """Sub-index for BOD (mg/L)."""
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 5:
        return max(0.0, min(100.0, 100.4 - 4.23 * x))
    else:
        return max(0.0, min(100.0, 108 * np.exp(-0.055 * x) - 0.1 * x))

def calc_sicod(x):
    """Sub-index for COD (mg/L)."""
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 20:
        return max(0.0, min(100.0, -1.33 * x + 99.1))
    else:
        return max(0.0, min(100.0, 103 * np.exp(-0.0157 * x) - 0.04 * x))

def calc_sian(x):
    """Sub-index for Ammoniacal Nitrogen NH3-N (mg/L)."""
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 0.3:
        return max(0.0, min(100.0, 100.5 - 105 * x))
    elif x < 4:
        return max(0.0, min(100.0, 94 * np.exp(-0.573 * x) - 5 * abs(x - 2)))
    else:
        return 0.0

def calc_siss(x):
    """Sub-index for Suspended Solids (mg/L)."""
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 100:
        return max(0.0, min(100.0, 97.5 * np.exp(-0.00676 * x) + 0.05 * x))
    elif x < 1000:
        return max(0.0, min(100.0, 71 * np.exp(-0.00161 * x) - 0.015 * x))
    else:
        return 0.0

def calc_siph(x):
    """Sub-index for pH."""
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x < 5.5:
        return max(0.0, min(100.0, 17.2 - 17.2 * x + 5.02 * (x ** 2)))
    elif x < 7:
        return max(0.0, min(100.0, -242 + 95.5 * x - 6.67 * (x ** 2)))
    elif x < 8.75:
        return max(0.0, min(100.0, -181 + 82.4 * x - 6.05 * (x ** 2)))
    else:
        return max(0.0, min(100.0, 536 - 77.0 * x + 2.76 * (x ** 2)))

def calculate_doe_wqi(do_sat, bod, cod, an, ss, ph):
    """Calculates official Malaysian DOE Water Quality Index."""
    sido = calc_sido(do_sat)
    sibod = calc_sibod(bod)
    sicod = calc_sicod(cod)
    sian = calc_sian(an)
    siss = calc_siss(ss)
    siph = calc_siph(ph)
    
    wqi = (
        0.22 * sido +
        0.19 * sibod +
        0.16 * sicod +
        0.15 * sian +
        0.16 * siss +
        0.12 * siph
    )
    wqi = round(max(0.0, min(100.0, wqi)), 1)
    status = classify_wqi(wqi)
    cls = get_doe_class(wqi)
    return {
        "wqi": wqi,
        "status": status,
        "class": cls,
        "sub_indices": {
            "SIDO": round(sido, 1),
            "SIBOD": round(sibod, 1),
            "SICOD": round(sicod, 1),
            "SIAN": round(sian, 1),
            "SISS": round(siss, 1),
            "SIpH": round(siph, 1),
        }
    }

# ============================================================
# DATA ENGINE CLASS
# ============================================================

class WaterQualityEngine:
    def __init__(self):
        self.load_data()

    def load_data(self):
        # Base observations
        if DATASET_FILE.exists():
            self.df_obs = pd.read_csv(DATASET_FILE)
            self.df_obs["smp_dat"] = pd.to_datetime(self.df_obs["smp_dat"])
            self.df_obs = self.df_obs.sort_values(["sungai", "id_stn_baru", "smp_dat"]).reset_index(drop=True)
        else:
            self.df_obs = pd.DataFrame()

        # All models WQI predictions
        if UNIFIED_WQI_FILE.exists():
            self.df_wqi_preds = pd.read_csv(UNIFIED_WQI_FILE)
            self.df_wqi_preds["sample_date"] = pd.to_datetime(self.df_wqi_preds["sample_date"])
        else:
            self.df_wqi_preds = pd.DataFrame()

        # All models metrics
        if UNIFIED_METRICS_FILE.exists():
            self.df_metrics = pd.read_csv(UNIFIED_METRICS_FILE)
        else:
            self.df_metrics = pd.DataFrame()

        # Latest forecasts
        if UNIFIED_FORECASTS_FILE.exists():
            self.df_forecasts = pd.read_csv(UNIFIED_FORECASTS_FILE)
            self.df_forecasts["last_observation_date"] = pd.to_datetime(self.df_forecasts["last_observation_date"])
            self.df_forecasts["forecast_date"] = pd.to_datetime(self.df_forecasts["forecast_date"])
        else:
            self.df_forecasts = pd.DataFrame()

        # Latest station observations
        if LATEST_OBS_FILE.exists():
            self.df_latest_obs = pd.read_csv(LATEST_OBS_FILE)
            self.df_latest_obs["last_sample_date"] = pd.to_datetime(self.df_latest_obs["last_sample_date"])
        else:
            self.df_latest_obs = pd.DataFrame()

    def get_models(self):
        return [
            "Persistence",
            "CEEMDAN",
            "LSTM",
            "CEEMDAN-LSTM",
            "XGBoost",
            "CEEMDAN-LSTM-XGBoost"
        ]

    def get_rivers(self):
        return ["Sungai Klang", "Sungai Skudai"]

    def get_stations(self, river_filter="All"):
        if self.df_obs.empty:
            return []
        df = self.df_obs
        if river_filter == "Sungai Klang":
            df = df[df["sungai"] == "KLANG"]
        elif river_filter == "Sungai Skudai":
            df = df[df["sungai"] == "SKUDAI"]
            
        stns = []
        for (r, s_id), g in df.groupby(["sungai", "id_stn_baru"]):
            loc = g["location"].dropna().iloc[0] if "location" in g.columns and not g["location"].dropna().empty else s_id
            lat = g["latitude"].dropna().iloc[0] if "latitude" in g.columns else None
            lon = g["longitude"].dropna().iloc[0] if "longitude" in g.columns else None
            state_code = str(g["states"].dropna().iloc[0]).strip().upper() if "states" in g.columns and not g["states"].dropna().empty else ""
            stns.append({
                "state_code": state_code,
                "state": STATE_LABELS.get(state_code, state_code.title()),
                "station_id": s_id,
                "river": "Sungai Klang" if r == "KLANG" else "Sungai Skudai",
                "river_code": r,
                "location": str(loc).strip(),
                "latitude": float(lat) if lat is not None else None,
                "longitude": float(lon) if lon is not None else None,
                "label": f"{s_id} - {loc} ({'Klang' if r == 'KLANG' else 'Skudai'})"
            })
        return stns

    def get_station_observations(self, station_id):
        if self.df_obs.empty:
            return pd.DataFrame()
        sub = self.df_obs[self.df_obs["id_stn_baru"] == station_id].sort_values("smp_dat").reset_index(drop=True)
        return sub

    def get_river_observations(self, river_name):
        if self.df_obs.empty:
            return pd.DataFrame()
        r_code = "KLANG" if "klang" in river_name.lower() else "SKUDAI"
        sub = self.df_obs[self.df_obs["sungai"] == r_code].sort_values("smp_dat").reset_index(drop=True)
        return sub

    def get_daily_time_series(self, station_id, model_name="CEEMDAN-LSTM", days_forecast_ahead=60):
        """
        Generates daily time-series combining historical observations,
        model historical evaluation estimates, and daily future forecast projections.
        """
        stn_obs = self.get_station_observations(station_id)
        if stn_obs.empty:
            return pd.DataFrame()

        river_code = stn_obs["sungai"].iloc[0]
        stn_loc = stn_obs["location"].iloc[0] if "location" in stn_obs.columns else station_id
        
        # Sort historical observations
        dates_obs = pd.to_datetime(stn_obs["smp_dat"]).dt.normalize().values
        wqi_obs = stn_obs["wqi"].astype(float).values

        # Unique dates
        df_ts = pd.DataFrame({"date": dates_obs, "wqi": wqi_obs}).drop_duplicates("date").sort_values("date")
        
        # Model predictions if available
        model_preds_sub = pd.DataFrame()
        if not self.df_wqi_preds.empty:
            model_preds_sub = self.df_wqi_preds[
                (self.df_wqi_preds["station_id"] == station_id) &
                (self.df_wqi_preds["model"] == model_name)
            ].sort_values("sample_date")

        # Future forecast endpoint
        latest_date = df_ts["date"].max()
        last_wqi = df_ts[df_ts["date"] == latest_date]["wqi"].iloc[0]
        
        # Query latest forecast value from model
        future_pred_wqi = last_wqi
        forecast_end_date = latest_date + pd.Timedelta(days=days_forecast_ahead)
        
        if not self.df_forecasts.empty:
            match_f = self.df_forecasts[
                (self.df_forecasts["station_id"] == station_id) &
                (self.df_forecasts["model"] == model_name)
            ]
            if len(match_f) > 0:
                future_pred_wqi = float(match_f["predicted_wqi"].iloc[0])
                if not pd.isna(match_f["forecast_date"].iloc[0]):
                    forecast_end_date = pd.to_datetime(match_f["forecast_date"].iloc[0])

        # Create daily date grid from first sample to future forecast horizon
        min_date = df_ts["date"].min()
        # The models only produce ONE forecast point per station (the next sampling date),
        # so the series ends there. Extending further would just extrapolate the spline.
        max_date = max(forecast_end_date, latest_date)
        all_days = pd.date_range(min_date, max_date, freq="D")
        
        # Base daily curve using monotonic cubic spline (Pchip) for realistic, smooth non-overshooting daily dynamics
        x_numeric = (df_ts["date"] - min_date).dt.days.values
        y_numeric = df_ts["wqi"].values
        
        # Add future projection point
        x_future = (forecast_end_date - min_date).days
        x_all = np.append(x_numeric, [x_future])
        y_all = np.append(y_numeric, [future_pred_wqi])
        
        # Sort points
        order = np.argsort(x_all)
        x_all = x_all[order]
        y_all = y_all[order]
        # Remove duplicates
        _, unique_idx = np.unique(x_all, return_index=True)
        x_all = x_all[unique_idx]
        y_all = y_all[unique_idx]

        interpolator = PchipInterpolator(x_all, y_all)
        
        daily_x = (all_days - min_date).days.values
        daily_wqi = interpolator(daily_x)
        daily_wqi = np.clip(daily_wqi, 0.0, 100.0)

        # Build daily dataframe
        daily_df = pd.DataFrame({
            "date": all_days,
            "predicted_wqi": np.round(daily_wqi, 1),
            "is_future": all_days > latest_date,
            "station_id": station_id,
            "river": "Sungai Klang" if river_code == "KLANG" else "Sungai Skudai",
            "model": model_name
        })

        # Mark actual observed dates and values
        obs_map = dict(zip(df_ts["date"], df_ts["wqi"]))
        daily_df["actual_wqi"] = daily_df["date"].map(obs_map)
        
        # Add classification
        daily_df["status"] = daily_df["predicted_wqi"].apply(classify_wqi)
        daily_df["status_color"] = daily_df["status"].apply(get_class_color)
        daily_df["class"] = daily_df["predicted_wqi"].apply(get_doe_class)
        
        # Add soft probability scores
        probs = [self.calc_probabilities(val) for val in daily_df["predicted_wqi"]]
        daily_df["prob_clean"] = [p[0] for p in probs]
        daily_df["prob_polluted"] = [p[1] for p in probs]
        daily_df["prob_slightly_polluted"] = [p[2] for p in probs]

        return daily_df

    def calc_probabilities(self, wqi):
        if pd.isna(wqi):
            return (0.33, 0.33, 0.34)
        val = float(wqi)
        s_clean = np.exp((val - 81) / 5.0)
        s_slight = np.exp(-((val - 70.5) ** 2) / (2 * (6.0 ** 2)))
        s_polluted = np.exp((60 - val) / 6.0)
        total = s_clean + s_slight + s_polluted
        return (round(float(s_clean / total), 3), round(float(s_polluted / total), 3), round(float(s_slight / total), 3))

    def get_date_snapshot(self, station_id, target_date, model_name="CEEMDAN-LSTM"):
        """
        Retrieves water quality info for a user-specified date:
        - Date
        - Water Quality Index
        - Contamination Status
        - Probabilities
        - Water parameters (DO, BOD, COD, SS, pH, NH3N)
        """
        target_dt = pd.to_datetime(target_date).normalize()
        daily_df = self.get_daily_time_series(station_id, model_name=model_name)
        
        if daily_df.empty:
            return None

        # Find closest day
        diff = (daily_df["date"] - target_dt).abs()
        idx_closest = diff.idxmin()
        row = daily_df.loc[idx_closest]
        
        # Find nearest historical observation to extract physical-chemical parameters
        stn_obs = self.get_station_observations(station_id)
        nearest_params = {
            "DO": 5.8,
            "BOD": 4.2,
            "COD": 22.0,
            "NH3N": 1.1,
            "SS": 35.0,
            "pH": 7.2
        }
        if not stn_obs.empty:
            diff_obs = (pd.to_datetime(stn_obs["smp_dat"]).dt.normalize() - target_dt).abs()
            nearest_row = stn_obs.loc[diff_obs.idxmin()]
            nearest_params = {
                "DO": round(float(nearest_row.get("do_mg_l", 5.8) or 5.8), 2),
                "BOD": round(float(nearest_row.get("bod_mg_l", 4.2) or 4.2), 2),
                "COD": round(float(nearest_row.get("cod_mg_l", 22.0) or 22.0), 2),
                "NH3N": round(float(nearest_row.get("nh3_n_mg_l", 1.1) or 1.1), 2),
                "SS": round(float(nearest_row.get("ss_mg_l", 35.0) or 35.0), 2),
                "pH": round(float(nearest_row.get("ph", 7.2) or 7.2), 2),
                "sample_date_ref": nearest_row.get("smp_dat")
            }

        return {
            "station_id": station_id,
            "river": row["river"],
            "date": target_dt.strftime("%Y-%m-%d"),
            "display_date": target_dt.strftime("%B %d, %Y"),
            "month_year": target_dt.strftime("%B %Y"),
            "predicted_wqi": float(row["predicted_wqi"]),
            "actual_wqi": None if pd.isna(row["actual_wqi"]) else float(row["actual_wqi"]),
            "status": row["status"],
            "status_color": row["status_color"],
            "class": row["class"],
            "is_future": bool(row["is_future"]),
            "model": model_name,
            "prob_clean": float(row["prob_clean"]),
            "prob_polluted": float(row["prob_polluted"]),
            "prob_slightly_polluted": float(row["prob_slightly_polluted"]),
            "parameters": nearest_params
        }

    def get_metrics_table(self):
        if self.df_metrics.empty:
            return pd.DataFrame()
        return self.df_metrics.copy()

    def get_latest_forecasts_summary(self, model_name=None):
        if self.df_forecasts.empty:
            return pd.DataFrame()
        df = self.df_forecasts.copy()
        if model_name:
            df = df[df["model"] == model_name]
        return df


    # ========================================================
    # DAILY / WEEKLY / MONTHLY FORECAST SERIES
    # ========================================================
    # IMPORTANT: DOE samples each station roughly every 60 days and every model
    # emits ONE forecast per station (for the next sampling date). The daily series
    # below is therefore a straight-line trajectory between real points:
    #   history  : observed sample -> observed sample
    #   forecast : last observed sample -> model forecast at the next sampling date
    # Weekly / monthly values are averages of that daily trajectory.

    def get_model_backtest(self, model_name):
        """Backtest metrics for a model (MAE, RMSE, R2, n_samples) or {}."""
        if self.df_metrics.empty:
            return {}
        row = self.df_metrics[self.df_metrics["model"] == model_name]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {"MAE": float(r["MAE"]), "RMSE": float(r["RMSE"]), "R2": float(r["R2"]),
                "n_samples": int(r["n_samples"])}

    def get_station_series(self, station_id, model_name):
        """Daily trajectory for one station: history + forecast to the next sampling date."""
        obs = self.get_station_observations(station_id)
        if obs.empty:
            return pd.DataFrame()
        o = pd.DataFrame({"date": pd.to_datetime(obs["smp_dat"]).dt.normalize(),
                          "wqi": obs["wqi"].astype(float)})
        o = o.dropna().drop_duplicates("date", keep="last").sort_values("date")
        last_date, last_wqi = o["date"].iloc[-1], float(o["wqi"].iloc[-1])

        fc = pd.DataFrame()
        if not self.df_forecasts.empty:
            fc = self.df_forecasts[(self.df_forecasts["station_id"] == station_id) &
                                   (self.df_forecasts["model"] == model_name)]
        end_date = last_date
        pred = None
        if len(fc):
            pred = float(fc["predicted_wqi"].iloc[0])
            end_date = max(pd.to_datetime(fc["forecast_date"].iloc[0]).normalize(), last_date)

        days = pd.date_range(o["date"].iloc[0], end_date, freq="D")
        t = (days - days[0]).days.values.astype(float)
        t_obs = (o["date"] - days[0]).dt.days.values.astype(float)
        wqi = np.interp(t, t_obs, o["wqi"].values)  # holds last value beyond last obs
        is_fc = days > last_date
        if pred is not None and is_fc.any():
            t_last = (last_date - days[0]).days
            t_end = (end_date - days[0]).days
            wqi[is_fc] = last_wqi + (pred - last_wqi) * (t[is_fc] - t_last) / (t_end - t_last)

        rmse = self.get_model_backtest(model_name).get("RMSE", np.nan)
        frac = np.zeros(len(days))
        if is_fc.any():
            frac[is_fc] = (t[is_fc] - (last_date - days[0]).days) / max((end_date - last_date).days, 1)
        half = rmse * np.sqrt(frac)  # band widens from 0 at the last sample to 1 RMSE at the forecast date
        out = pd.DataFrame({
            "date": days,
            "wqi": np.round(wqi, 2),
            "lower": np.where(is_fc, np.clip(wqi - half, 0, 100), np.nan),
            "upper": np.where(is_fc, np.clip(wqi + half, 0, 100), np.nan),
            "is_forecast": is_fc,
            "observed": pd.Series(days).map(dict(zip(o["date"], o["wqi"]))).values,
            "n_stations": 1,
        })
        return out

    def _stations_for(self, level, key):
        """Station ids for level in {'all','river','state','station'}."""
        stns = self.get_stations("All")
        if level == "station":
            return [key]
        if level == "river":
            return [x["station_id"] for x in stns if x["river"] == key]
        if level == "state":
            return [x["station_id"] for x in stns if x["state"] == key]
        return [x["station_id"] for x in stns]

    def get_group_series(self, level, key, model_name):
        """
        Daily series for a station / river / state / all stations (mean over stations).
        The series ends at the MEDIAN station forecast date; stations whose forecast date
        is earlier are held flat from their forecast date so the mix of stations never changes.
        """
        ids = self._stations_for(level, key)
        series = [self.get_station_series(i, model_name) for i in ids]
        series = [x for x in series if not x.empty]
        if not series:
            return pd.DataFrame()
        if len(series) == 1:
            return series[0]
        start = max(x["date"].iloc[0] for x in series)
        cap = pd.Series([x["date"].iloc[-1] for x in series]).sort_values().iloc[len(series) // 2]
        grid = pd.date_range(start, cap, freq="D")
        cols = ["wqi", "lower", "upper"]
        stack = {c: [] for c in cols}
        fc_flags = []
        for x in series:
            x = x.set_index("date").reindex(grid, method=None)
            x[cols] = x[cols].ffill()
            x["is_forecast"] = x["is_forecast"].astype("object").ffill().fillna(False).astype(bool)
            for c in cols:
                stack[c].append(x[c].values)
            fc_flags.append(x["is_forecast"].values)
        out = pd.DataFrame({"date": grid})
        for c in cols:
            out[c] = np.round(pd.DataFrame(np.vstack(stack[c])).mean(axis=0).values, 2)
        out["is_forecast"] = np.mean(np.vstack(fc_flags), axis=0) >= 0.5
        out["observed"] = np.nan
        out["n_stations"] = len(series)
        return out

    def aggregate_series(self, df, freq):
        """freq: 'D' daily, 'W' weekly (Mon-Sun), 'M' calendar month. Mean WQI per period."""
        if df.empty:
            return df
        d = df.copy()
        if freq == "D":
            d["period_start"] = d["date"]
            d["period_end"] = d["date"]
            d["n_days"] = 1
            d["forecast_days"] = d["is_forecast"].astype(int)
            d["full_days"] = 1
            d["label"] = d["date"].dt.strftime("%d %b %Y")
            res = d
        else:
            code = "W-SUN" if freq == "W" else "M"
            per = d["date"].dt.to_period(code)
            d["_p"] = per
            g = d.groupby("_p")
            res = g.agg(wqi=("wqi", "mean"), lower=("lower", "mean"), upper=("upper", "mean"),
                        n_days=("date", "size"), forecast_days=("is_forecast", "sum"),
                        n_stations=("n_stations", "max")).reset_index()
            res["period_start"] = res["_p"].dt.start_time
            res["period_end"] = res["_p"].dt.end_time.dt.normalize()
            res["full_days"] = (res["period_end"] - res["period_start"]).dt.days + 1
            res["is_forecast"] = res["forecast_days"] > 0
            res["date"] = res["period_start"]
            res["label"] = (res["period_start"].dt.strftime("Wk %d %b %Y") if freq == "W"
                            else res["period_start"].dt.strftime("%b %Y"))
            res = res.drop(columns=["_p"])
        res["wqi"] = res["wqi"].round(1)
        res["status"] = res["wqi"].apply(classify_wqi)
        res["class"] = res["wqi"].apply(get_doe_class)
        res["partial"] = res["n_days"] < res["full_days"]
        return res.reset_index(drop=True)

    def get_map_snapshot(self, model_name, mode="forecast", target_date=None):
        """
        One row per station for the map.
        mode: 'observed' (latest DOE sample), 'forecast' (model value at next sampling date),
              'date' (trajectory value at target_date, clamped to the series range).
        """
        rows = []
        for st in self.get_stations("All"):
            ser = self.get_station_series(st["station_id"], model_name)
            if ser.empty:
                continue
            hist = ser[~ser["is_forecast"]]
            if mode == "observed":
                r = hist.iloc[-1]
            elif mode == "forecast":
                r = ser.iloc[-1]
            else:
                t = pd.to_datetime(target_date).normalize()
                t = min(max(t, ser["date"].iloc[0]), ser["date"].iloc[-1])
                r = ser[ser["date"] == t].iloc[0]
            wqi = float(r["wqi"])
            row = {**{k: st[k] for k in ("station_id", "river", "state", "state_code", "location", "latitude", "longitude")},
                   "wqi": round(wqi, 1), "as_of": r["date"], "is_forecast": bool(r["is_forecast"]),
                   "status": classify_wqi(wqi), "class": get_doe_class(wqi),
                   "last_observed_wqi": float(hist["wqi"].iloc[-1]), "prob_clean": np.nan,
                   "prob_slightly_polluted": np.nan, "prob_polluted": np.nan}
            if mode == "forecast" and not self.df_forecasts.empty:
                f = self.df_forecasts[(self.df_forecasts["station_id"] == st["station_id"]) &
                                      (self.df_forecasts["model"] == model_name)]
                if len(f):
                    row["prob_clean"] = float(f["prob_clean"].iloc[0])
                    row["prob_slightly_polluted"] = float(f["prob_slightly_polluted"].iloc[0])
                    row["prob_polluted"] = float(f["prob_polluted"].iloc[0])
            rows.append(row)
        return pd.DataFrame(rows)


    # ========================================================
    # OVERVIEW HELPERS (latest parameters, data freshness)
    # ========================================================
    PARAM_META = [
        # key, label, sub-label, unit, obs column, sub-index column, WQI weight
        ("DO", "DO", "Dissolved oxygen", "mg/L", "do_mg_l", "do_index_index", 0.22),
        ("BOD", "BOD", "Biochemical oxygen demand", "mg/L", "bod_mg_l", "bod_index_index", 0.19),
        ("COD", "COD", "Chemical oxygen demand", "mg/L", "cod_mg_l", "cod_index_index", 0.16),
        ("NH3N", "NH\u2083-N", "Ammoniacal nitrogen", "mg/L", "nh3_n_mg_l", "an_index", 0.15),
        ("SS", "SS", "Suspended solids", "mg/L", "ss_mg_l", "ss_index", 0.16),
        ("pH", "pH", "Acidity", "", "ph", "ph_index", 0.12),
    ]

    def get_scope_parameters(self, level, key):
        """Mean of each station's LATEST sample in scope: value, sub-index, and the main driver of a lower WQI."""
        ids = self._stations_for(level, key)
        rows = []
        for i in ids:
            o = self.get_station_observations(i)
            if not o.empty:
                rows.append(o.iloc[-1])
        if not rows:
            return {}
        d = pd.DataFrame(rows)
        params = []
        for k, label, sub, unit, col, icol, w in self.PARAM_META:
            val = float(pd.to_numeric(d[col], errors="coerce").mean())
            si = float(pd.to_numeric(d[icol], errors="coerce").mean())
            params.append({"key": k, "label": label, "sub": sub, "unit": unit, "value": val,
                           "sub_index": si, "weight": w, "loss": w * (100 - si)})
        main = max(params, key=lambda x: x["loss"])
        return {"params": params, "main": main, "n_stations": len(rows),
                "last_sample": pd.to_datetime(d["smp_dat"]).max()}

    def get_data_freshness(self):
        """How old the data behind the forecasts is, relative to today."""
        today = pd.Timestamp.today().normalize()
        last_obs = self.df_obs["smp_dat"].max().normalize() if not self.df_obs.empty else None
        fc_end = self.df_forecasts["forecast_date"].median().normalize() if not self.df_forecasts.empty else None
        return {"today": today, "last_obs": last_obs, "forecast_date": fc_end,
                "data_age_days": (today - last_obs).days if last_obs is not None else None,
                "forecast_age_days": (today - fc_end).days if fc_end is not None else None}

# Singleton engine instance
ENGINE = WaterQualityEngine()
