"""
src/forecast_views.py
Plotly figure builders for the daily / weekly / monthly forecast tabs and the state/river map.
Kept separate from app.py so the chart logic can be tested without Streamlit.
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model_engine import (
    COLOR_CLEAN, COLOR_SLIGHTLY_POLLUTED, COLOR_POLLUTED,
    STATUS_CLEAN, STATUS_SLIGHTLY_POLLUTED, STATUS_POLLUTED,
    STATE_GEOJSON_NAME, GEOJSON_FILE, get_class_color,
)

STATUS_COLORS = {
    STATUS_CLEAN: COLOR_CLEAN,
    STATUS_SLIGHTLY_POLLUTED: COLOR_SLIGHTLY_POLLUTED,
    STATUS_POLLUTED: COLOR_POLLUTED,
}
# One line colour per model so comparisons stay readable
MODEL_COLORS = {
    "Persistence": "#64748b",
    "CEEMDAN": "#0d9488",
    "LSTM": "#db2777",
    "CEEMDAN-LSTM": "#7c3aed",
    "XGBoost": "#ea580c",
    "CEEMDAN-LSTM-XGBoost": "#0f172a",
}


def _add_doe_bands(fig):
    fig.add_hrect(y0=81, y1=100, fillcolor="rgba(37,99,235,0.12)", line_width=0)
    fig.add_hrect(y0=60, y1=81, fillcolor="rgba(234,179,8,0.14)", line_width=0)
    fig.add_hrect(y0=0, y1=60, fillcolor="rgba(239,68,68,0.12)", line_width=0)


def _base_layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1e293b")),
        yaxis=dict(title="Water Quality Index (WQI)", range=[0, 102],
                   tickvals=[0, 20, 40, 60, 81, 100], gridcolor="rgba(226,232,240,0.7)"),
        xaxis=dict(gridcolor="rgba(226,232,240,0.7)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40), height=height,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
    )
    return fig


def daily_figure(series, model, title, compare=None, history_days=270):
    """Daily trajectory: observed→observed (history), observed→forecast (dashed) with ±RMSE band."""
    compare = compare or {}
    fig = go.Figure()
    _add_doe_bands(fig)
    last_obs = series.loc[~series["is_forecast"], "date"].max()
    start = last_obs - pd.Timedelta(days=history_days)
    s = series[series["date"] >= start]
    hist, fc = s[~s["is_forecast"]], s[s["is_forecast"]]
    color = MODEL_COLORS.get(model, "#7c3aed")

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["wqi"], mode="lines", name="History (interpolated between samples)",
                             line=dict(color="#475569", width=2), hovertemplate="%{y:.1f}"))
    obs = hist.dropna(subset=["observed"])
    if not obs.empty:
        fig.add_trace(go.Scatter(x=obs["date"], y=obs["observed"], mode="markers", name="DOE sample",
                                 marker=dict(size=8, color="#2563eb", line=dict(width=1.5, color="#fff")),
                                 hovertemplate="%{y:.0f}"))
    if not fc.empty:
        bridge = pd.concat([hist.tail(1), fc])
        fig.add_trace(go.Scatter(x=pd.concat([bridge["date"], bridge["date"][::-1]]),
                                 y=pd.concat([bridge["upper"].fillna(bridge["wqi"]), bridge["lower"].fillna(bridge["wqi"])[::-1]]),
                                 fill="toself", fillcolor="rgba(124,58,237,0.14)", line=dict(width=0),
                                 hoverinfo="skip", name="±1 RMSE (backtest)"))
        fig.add_trace(go.Scatter(x=bridge["date"], y=bridge["wqi"], mode="lines", name=f"{model} forecast",
                                 line=dict(color=color, width=3, dash="dash"), hovertemplate="%{y:.1f}"))
        end = fc.iloc[-1]
        fig.add_trace(go.Scatter(x=[end["date"]], y=[end["wqi"]], mode="markers+text", showlegend=False,
                                 marker=dict(size=12, color=color, symbol="star", line=dict(width=1.5, color="#fff")),
                                 text=[f"{end['wqi']:.0f}"], textposition="top center", hoverinfo="skip"))
    for m, cdf in compare.items():
        c = cdf[(cdf["date"] >= start) & cdf["is_forecast"]]
        if c.empty:
            continue
        bridge = pd.concat([hist.tail(1), c[["date", "wqi"]]])
        fig.add_trace(go.Scatter(x=bridge["date"], y=bridge["wqi"], mode="lines", name=m,
                                 line=dict(color=MODEL_COLORS.get(m, "#999"), width=1.8, dash="dot"),
                                 hovertemplate="%{y:.1f}"))
    fig.add_vline(x=last_obs, line_dash="dot", line_color="rgba(15,23,42,0.5)",
                  annotation_text="Last DOE sample", annotation_position="top left")
    return _base_layout(fig, title)


def period_figure(agg, model, title, compare_aggs=None, unit="week", periods_history=8):
    """Weekly / monthly bars (colour = DOE status), error bars on forecast periods, optional model overlay."""
    compare_aggs = compare_aggs or {}
    first_fc = agg.index[agg["is_forecast"]].min()
    lo = max(0, (first_fc if pd.notna(first_fc) else len(agg)) - periods_history)
    a = agg.iloc[lo:].copy()
    fig = go.Figure()
    _add_doe_bands(fig)

    for is_fc, name in ((False, f"Past {unit}s (interpolated from DOE samples)"), (True, f"Forecast {unit}s")):
        sub = a[a["is_forecast"] == is_fc]
        if sub.empty:
            continue
        err = None
        if is_fc:
            err = dict(type="data", symmetric=False,
                       array=(sub["upper"] - sub["wqi"]).fillna(0).values,
                       arrayminus=(sub["wqi"] - sub["lower"]).fillna(0).values,
                       color="#334155", thickness=1.4, width=4)
        fig.add_trace(go.Bar(
            x=sub["label"], y=sub["wqi"], name=name, error_y=err,
            marker=dict(color=[STATUS_COLORS[s] for s in sub["status"]], opacity=0.9 if is_fc else 0.45,
                        line=dict(color="#0f172a" if is_fc else "rgba(0,0,0,0)", width=1)),
            text=[f"{v:.0f}" + ("*" if p else "") for v, p in zip(sub["wqi"], sub["partial"])],
            textposition="inside", insidetextanchor="end", textfont=dict(color="#0f172a", size=12),
            customdata=np.stack([sub["status"], sub["n_days"], sub["forecast_days"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>WQI %{y:.1f}<br>%{customdata[0]}<br>"
                          "days covered: %{customdata[1]} (forecast days: %{customdata[2]})<extra></extra>",
        ))
    for m, cagg in compare_aggs.items():
        c = cagg[cagg["label"].isin(a["label"]) & cagg["is_forecast"]]
        if c.empty:
            continue
        fig.add_trace(go.Scatter(x=c["label"], y=c["wqi"], mode="markers", name=m,
                                 marker=dict(size=11, symbol="diamond", color=MODEL_COLORS.get(m, "#999"),
                                             line=dict(width=1.5, color="#fff")),
                                 hovertemplate="%{y:.1f}"))
    fig.update_xaxes(type="category")
    fig = _base_layout(fig, title, height=400)
    fig.update_layout(bargap=0.25, hovermode="closest")
    return fig


@lru_cache(maxsize=1)
def load_states_geojson():
    with open(GEOJSON_FILE, encoding="utf-8") as f:
        return json.load(f)


MAP_FOCUS = {
    "All monitored areas": dict(center=dict(lat=2.4, lon=102.5), zoom=6.1),
    "Klang Valley (Selangor + KL)": dict(center=dict(lat=3.08, lon=101.60), zoom=9.2),
    "Johor (Sungai Skudai)": dict(center=dict(lat=1.60, lon=103.60), zoom=9.6),
}


def state_summary(snap):
    g = snap.groupby(["state_code", "state"]).agg(
        stations=("station_id", "size"), wqi=("wqi", "mean"),
        polluted=("status", lambda s: int((s == STATUS_POLLUTED).sum()))).reset_index()
    g["wqi"] = g["wqi"].round(1)
    from model_engine import classify_wqi
    g["status"] = g["wqi"].apply(classify_wqi)
    return g


def river_summary(snap):
    from model_engine import classify_wqi
    g = snap.groupby("river").agg(
        stations=("station_id", "size"), wqi=("wqi", "mean"),
        polluted=("status", lambda s: int((s == STATUS_POLLUTED).sum()))).reset_index()
    g["wqi"] = g["wqi"].round(1)
    g["status"] = g["wqi"].apply(classify_wqi)
    return g


def map_figure(snap, focus="All monitored areas", title=""):
    """State polygons (shaded by mean WQI status) + station dots (coloured by status)."""
    geo = load_states_geojson()
    states = state_summary(snap)
    fig = go.Figure()

    # Every other Malaysian state: neutral grey, for geographic context only
    mon_names = {STATE_GEOJSON_NAME[c] for c in states["state_code"]}
    others = [f["properties"]["name"] for f in geo["features"] if f["properties"]["name"] not in mon_names]
    fig.add_trace(go.Choroplethmap(
        geojson=geo, featureidkey="properties.name", locations=others, z=[0] * len(others),
        colorscale=[[0, "#e2e8f0"], [1, "#e2e8f0"]], showscale=False, marker_opacity=0.35,
        marker_line_width=0.6, marker_line_color="#94a3b8", hoverinfo="skip", name="Other states"))

    # Monitored states: shaded by the state's mean-station DOE status
    names = [STATE_GEOJSON_NAME[c] for c in states["state_code"]]
    fig.add_trace(go.Choroplethmap(
        geojson=geo, featureidkey="properties.name", locations=names, z=[0] * len(names),
        colorscale=[[0, "#ffffff"], [1, "#ffffff"]], showscale=False, marker_opacity=0.0,
        marker_line_width=2.2, marker_line_color="#0f172a", hoverinfo="skip", name="State outline"))
    for _, r in states.iterrows():
        fig.add_trace(go.Choroplethmap(
            geojson=geo, featureidkey="properties.name", locations=[STATE_GEOJSON_NAME[r["state_code"]]], z=[0],
            colorscale=[[0, STATUS_COLORS[r["status"]]], [1, STATUS_COLORS[r["status"]]]], showscale=False,
            marker_opacity=0.28, marker_line_width=0, showlegend=False,
            hovertemplate=(f"<b>{r['state']}</b><br>Mean WQI {r['wqi']:.1f} ({r['status']})<br>"
                           f"{r['stations']} stations, {r['polluted']} polluted<extra></extra>")))

    # Stations
    for status in (STATUS_CLEAN, STATUS_SLIGHTLY_POLLUTED, STATUS_POLLUTED):
        sub = snap[snap["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermap(
            lat=sub["latitude"], lon=sub["longitude"], mode="markers", name=f"{status} stations",
            marker=dict(size=13, color=STATUS_COLORS[status]),
            customdata=np.stack([sub["station_id"], sub["river"], sub["state"], sub["class"],
                                 sub["as_of"].dt.strftime("%d %b %Y"), sub["wqi"].round(1).astype(str)], axis=-1),
            text=sub["location"],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}<br>"
                          "WQI %{customdata[5]} · %{customdata[3]}<br>as of %{customdata[4]}<extra></extra>"))
    f = MAP_FOCUS[focus]
    fig.update_layout(map=dict(style="open-street-map", center=f["center"], zoom=f["zoom"]),
                      margin=dict(l=0, r=0, t=36, b=0), height=560, title=dict(text=title, font=dict(size=14)),
                      legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                                  bgcolor="rgba(255,255,255,0.85)"))
    return fig
