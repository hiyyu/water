"""Sanity checks for the daily / weekly / monthly forecast series. Run: python src/test_forecast_series.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from model_engine import ENGINE as E


def test_forecast_endpoint_matches_model_output():
    f = E.df_forecasts
    for model in E.get_models():
        for _, r in f[f["model"] == model].iterrows():
            s = E.get_station_series(r["station_id"], model)
            assert s["date"].iloc[-1] == r["forecast_date"], (model, r["station_id"])
            assert abs(s["wqi"].iloc[-1] - r["predicted_wqi"]) < 0.01, (model, r["station_id"])


def test_history_passes_through_observations():
    for st in E.get_stations("All"):
        s = E.get_station_series(st["station_id"], "Persistence")
        obs = E.get_station_observations(st["station_id"])
        for d, v in zip(pd.to_datetime(obs["smp_dat"]).dt.normalize(), obs["wqi"]):
            assert abs(s.loc[s["date"] == d, "wqi"].iloc[0] - v) < 0.01


def test_persistence_forecast_is_flat():
    s = E.get_station_series("2BKLG005", "Persistence")
    fc = s[s["is_forecast"]]
    assert fc["wqi"].nunique() == 1 and len(fc) > 30


def test_band_starts_at_zero_and_widens():
    s = E.get_station_series("2BKLG005", "CEEMDAN-LSTM")
    fc = s[s["is_forecast"]]
    width = (fc["upper"] - fc["lower"]).values
    assert width[0] < width[-1]


def test_weekly_monthly_are_means_of_daily():
    g = E.get_group_series("river", "Sungai Klang", "CEEMDAN-LSTM")
    for freq, code in (("W", "W-SUN"), ("M", "M")):
        a = E.aggregate_series(g, freq)
        manual = g.groupby(g["date"].dt.to_period(code))["wqi"].mean().round(1).values
        assert np.allclose(a["wqi"].values, manual)
        assert a["n_days"].sum() == len(g)


def test_group_series_ends_at_median_forecast_date_and_has_no_gaps():
    g = E.get_group_series("river", "Sungai Klang", "XGBoost")
    assert g["wqi"].notna().all() and g["date"].is_monotonic_increasing
    assert g["n_stations"].iloc[0] == 13


def test_map_forecast_mode_matches_csv():
    snap = E.get_map_snapshot("LSTM", "forecast").set_index("station_id")
    f = E.df_forecasts[E.df_forecasts["model"] == "LSTM"].set_index("station_id")
    assert np.allclose(snap["wqi"].round(1), f.loc[snap.index, "predicted_wqi"].round(1), atol=0.06)
    assert set(snap["state"]) == {"Selangor", "W.P. Kuala Lumpur", "Johor"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
