"""
Water Quality Prediction and Contamination Risk Dashboard
Malaysian Department of Environment (DOE) River Monitoring System
Supports: Persistence | CEEMDAN | LSTM | CEEMDAN-LSTM | XGBoost | CEEMDAN-LSTM-XGBoost
"""

from datetime import datetime, date
from pathlib import Path
import sys

# Ensure src is on Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import overview_ui as ui
from forecast_views import (
    MODEL_COLORS, MAP_FOCUS, daily_figure, period_figure, map_figure, state_summary, river_summary,
)
from model_engine import (
    ENGINE,
    STATUS_CLEAN,
    STATUS_SLIGHTLY_POLLUTED,
    STATUS_POLLUTED,
    COLOR_CLEAN,
    COLOR_SLIGHTLY_POLLUTED,
    COLOR_POLLUTED,
    MODEL_DESCRIPTIONS,
    classify_wqi,
    get_doe_class,
    get_class_color,
    calculate_doe_wqi,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Water Quality & Contamination Risk Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.3rem;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    
    /* Status Badges */
    .badge-clean {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #bfdbfe;
    }
    .badge-slightly-polluted {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #fde68a;
    }
    .badge-polluted {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid #fecaca;
    }
    
    /* Legend Banner */
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        padding: 0.8rem 1.2rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        align-items: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 4px;
    }
    
    /* Search Info Box */
    .info-callout {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(ui.CSS, unsafe_allow_html=True)


# ============================================================
# HELPER: STATUS BADGE RENDERER
# ============================================================

def render_status_badge(status):
    if status == STATUS_CLEAN:
        return f'<span class="badge-clean">● {status}</span>'
    elif status == STATUS_SLIGHTLY_POLLUTED:
        return f'<span class="badge-slightly-polluted">▲ {status}</span>'
    elif status == STATUS_POLLUTED:
        return f'<span class="badge-polluted">■ {status}</span>'
    else:
        return f'<span style="background:#e2e8f0; color:#475569; padding:4px 12px; border-radius:9999px; font-weight:600;">{status}</span>'


# ============================================================
# SIDEBAR FILTERS & MODEL CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    st.markdown("Configure river stations, dates, and prediction models.")
    st.markdown("---")

    # 1. Model Selector
    st.markdown("#### 🤖 Select Prediction Model")
    selected_model = st.selectbox(
        "Active Machine Learning Model",
        options=ENGINE.get_models(),
        index=3, # Default: CEEMDAN-LSTM
        help="Select any forecasting or risk classification model."
    )
    st.caption(f"ℹ️ **{selected_model}**: {MODEL_DESCRIPTIONS.get(selected_model, '')}")
    _bt = ENGINE.get_model_backtest(selected_model)
    if _bt:
        st.caption(f"Backtest: MAE {_bt['MAE']:.1f} · RMSE {_bt['RMSE']:.1f} · R² {_bt['R2']:.2f} (n={_bt['n_samples']})")
    if selected_model == "CEEMDAN-LSTM-XGBoost":
        st.info("This model's WQI forecast is identical to CEEMDAN-LSTM; the XGBoost stage only changes the risk class.")
    compare_models = st.multiselect(
        "Compare with (optional)",
        options=[m for m in ENGINE.get_models() if m != selected_model],
        default=[],
        help="Overlay other models on the Daily, Weekly and Monthly forecast tabs.",
    )
    st.markdown("---")

    # Forecast scope used by the Daily / Weekly / Monthly tabs
    st.markdown("#### 🎯 Forecast Scope")
    _stns = ENGINE.get_stations("All")
    scope_level_label = st.radio("Aggregate forecast by", ["All stations", "State", "River (Sungai)", "Single station"], index=2)
    if scope_level_label == "All stations":
        scope_level, scope_key, scope_title = "all", None, "All 22 stations"
    elif scope_level_label == "State":
        _opts = sorted({x["state"] for x in _stns})
        scope_key = st.selectbox("State", _opts)
        scope_level, scope_title = "state", scope_key
    elif scope_level_label == "River (Sungai)":
        scope_key = st.selectbox("River", ENGINE.get_rivers())
        scope_level, scope_title = "river", scope_key
    else:
        _lab = st.selectbox("Station", [x["label"] for x in _stns])
        _sel = next(x for x in _stns if x["label"] == _lab)
        scope_level, scope_key, scope_title = "station", _sel["station_id"], f"{_sel['station_id']} · {_sel['location']}"
    st.markdown("---")

    # 2. Station Filter (as drawn in user wireframe)
    st.markdown("#### 📍 River Basin & Station")
    station_scope = st.radio(
        "Stations Display Mode",
        options=["Show both stations (default)", "Sungai Skudai", "Sungai Klang", "Specific Station"],
        index=0,
        help="Choose whether to view both major river systems side-by-side or focus on one."
    )

    specific_station_id = None
    if station_scope == "Specific Station":
        all_stns = ENGINE.get_stations("All")
        stn_labels = [s["label"] for s in all_stns]
        chosen_label = st.selectbox("Choose Monitoring Station", options=stn_labels, index=0)
        specific_station_id = next(s["station_id"] for s in all_stns if s["label"] == chosen_label)

    st.markdown("---")

    # 3. Parameter Filter (as drawn in user wireframe)
    st.markdown("#### 🧪 Water Quality Parameter")
    parameter_filter = st.selectbox(
        "Parameter View",
        options=["Show all (WQI Overview)", "DO (Dissolved Oxygen)", "BOD (Biochemical Oxygen Demand)", 
                 "COD (Chemical Oxygen Demand)", "NH3N (Ammoniacal Nitrogen)", "SS/TSS (Suspended Solids)", "pH"],
        index=0,
        help="Filter the chart view to the full WQI composite or an individual DOE sub-parameter."
    )

    st.markdown("---")

    st.caption("🔮 Forecast horizon = each model's next DOE sampling date (about 60 days after the last sample).")
    forecast_days = 60  # kept for the legacy trend tab; the series now ends at the real forecast date

    st.markdown("---")
    st.markdown("### 📊 Malaysian DOE Standards")
    st.markdown("""
    - **Clean**: WQI 81 - 100
    - **Slightly Polluted**: WQI 60 - 80
    - **Polluted**: WQI 0 - 59
    """)


# ============================================================
# TREND TAB CONTROLS (date picker + legend from the original wireframe)
# ============================================================

def _render_trend_controls():
    # Top Search Bar & Date Picker (Faithfully implements the top search bar from the sketch)
    search_col1, search_col2, search_col3, search_col4 = st.columns([1.5, 1.8, 1.5, 1.2])

    with search_col1:
        search_station_mode = st.selectbox(
            "🔍 Station Filter",
            options=["Both Rivers (Klang & Skudai)", "Sungai Skudai Only", "Sungai Klang Only"],
            index=0 if station_scope == "Show both stations (default)" else (1 if station_scope == "Sungai Skudai" else 2),
            key="top_search_station"
        )

    with search_col2:
        # Quick date selection preset or custom calendar
        date_preset = st.selectbox(
            "📅 Select Date / Period",
            options=[
                "July 2024 (as sketched in wireframe)",
                "Latest Available Observation (Nov/Dec 2024)",
                "Today / Current Forecast",
                "Custom Date Picker"
            ],
            index=0,
            key="date_preset_select"
        )

    with search_col3:
        if date_preset == "July 2024 (as sketched in wireframe)":
            user_selected_date = date(2024, 7, 15)
            st.date_input("Target Date", value=user_selected_date, disabled=True)
        elif date_preset == "Latest Available Observation (Nov/Dec 2024)":
            user_selected_date = date(2024, 11, 15)
            st.date_input("Target Date", value=user_selected_date, disabled=True)
        elif date_preset == "Today / Current Forecast":
            user_selected_date = date.today()
            st.date_input("Target Date", value=user_selected_date, disabled=True)
        else:
            user_selected_date = st.date_input("Target Date", value=date(2024, 7, 15), min_value=date(2023, 1, 1), max_value=date(2025, 6, 30))

    with search_col4:
        st.write("")
        st.write("")
        apply_search = st.button("Enter 🔍", type="primary", width="stretch")

    st.markdown("""
    <div class="info-callout">
        💡 <strong>Showing selected date's Water Quality info</strong>: Live prediction and contamination classification dynamically updates according to user date selection and active model (<strong>{}</strong>).
    </div>
    """.format(selected_model), unsafe_allow_html=True)

    # Legend Bar (Matching Sketch)
    st.markdown("""
    <div class="legend-container">
        <span style="font-size:0.95rem; font-weight:700; color:#334155; margin-right: 0.5rem;">Water Quality Status Legend:</span>
        <div class="legend-item"><div class="legend-dot" style="background:#2563EB;"></div><span>Clean (WQI 81 - 100)</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:#EAB308;"></div><span>Slightly Polluted (WQI 60 - 80)</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:#EF4444;"></div><span>Polluted (WQI 0 - 59)</span></div>
        <span style="margin-left:auto; font-size:0.85rem; color:#64748b;">Thresholds: Malaysian Department of Environment (DOE) Standard</span>
    </div>
    """, unsafe_allow_html=True)
    return user_selected_date


# ============================================================
# FUNCTION: RENDER WATER QUALITY TREND CHART WITH COLOR BANDS
# ============================================================

def create_trend_figure(daily_df, river_title, selected_date, param_view="Show all (WQI Overview)"):
    fig = go.Figure()

    # 1. Background Colored Zones (Clean: Blue, Slightly Polluted: Yellow, Polluted: Red)
    # Matching the sketch's three colored horizontal bands!
    fig.add_hrect(
        y0=81, y1=100,
        fillcolor="rgba(37, 99, 235, 0.16)",
        line_width=0,
        annotation_text="Clean (81-100)",
        annotation_position="top left",
        annotation_font=dict(color="#1d4ed8", size=10, weight="bold")
    )
    fig.add_hrect(
        y0=60, y1=81,
        fillcolor="rgba(234, 179, 8, 0.18)",
        line_width=0,
        annotation_text="Slightly Polluted (60-80)",
        annotation_position="top left",
        annotation_font=dict(color="#b45309", size=10, weight="bold")
    )
    fig.add_hrect(
        y0=0, y1=60,
        fillcolor="rgba(239, 68, 68, 0.16)",
        line_width=0,
        annotation_text="Polluted (0-59)",
        annotation_position="bottom left",
        annotation_font=dict(color="#b91c1c", size=10, weight="bold")
    )

    # 2. Historical / Known Model Trend Line
    hist_df = daily_df[~daily_df["is_future"]]
    future_df = daily_df[daily_df["is_future"]]

    fig.add_trace(go.Scatter(
        x=hist_df["date"],
        y=hist_df["predicted_wqi"],
        mode="lines",
        line=dict(color="#0f172a", width=2.5),
        name="Historical WQI Trend",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>WQI:</b> %{y:.1f}<extra></extra>"
    ))

    # 3. Future Daily Forecast Line (Dashed)
    if not future_df.empty:
        # Connect last historical to first future
        connect_df = pd.concat([hist_df.tail(1), future_df])
        fig.add_trace(go.Scatter(
            x=connect_df["date"],
            y=connect_df["predicted_wqi"],
            mode="lines",
            line=dict(color="#7c3aed", width=2.5, dash="dash"),
            name="Daily Forecast Horizon",
            hovertemplate="<b>Forecast Date:</b> %{x|%Y-%m-%d}<br><b>Projected WQI:</b> %{y:.1f}<extra></extra>"
        ))

    # 4. Actual Sampling Observations (Discrete markers)
    obs_points = daily_df.dropna(subset=["actual_wqi"])
    if not obs_points.empty:
        fig.add_trace(go.Scatter(
            x=obs_points["date"],
            y=obs_points["actual_wqi"],
            mode="markers",
            marker=dict(
                size=8,
                color="#2563eb",
                line=dict(width=1.5, color="#ffffff"),
                symbol="circle"
            ),
            name="DOE Sampling Observations",
            hovertemplate="<b>Observed Date:</b> %{x|%Y-%m-%d}<br><b>Actual WQI:</b> %{y:.1f}<extra></extra>"
        ))

    # 5. Highlight Selected Date Point
    sel_dt = pd.to_datetime(selected_date).normalize()
    if daily_df["date"].min() <= sel_dt <= daily_df["date"].max():
        closest_row = daily_df.iloc[(daily_df["date"] - sel_dt).abs().argmin()]
        fig.add_trace(go.Scatter(
            x=[closest_row["date"]],
            y=[closest_row["predicted_wqi"]],
            mode="markers+text",
            marker=dict(
                size=14,
                color="#ef4444",
                symbol="star",
                line=dict(width=2, color="#ffffff")
            ),
            text=[f"  Target: {closest_row['predicted_wqi']:.1f}"],
            textposition="top center",
            textfont=dict(color="#0f172a", size=11, weight="bold"),
            name="Selected Date",
            hovertemplate="<b>Target Date:</b> %{x|%Y-%m-%d}<br><b>WQI:</b> %{y:.1f}<extra></extra>"
        ))
        # Add vertical line for target date
        fig.add_vline(
            x=closest_row["date"],
            line_width=1.5,
            line_dash="dot",
            line_color="rgba(15, 23, 42, 0.6)"
        )

    # Layout styling
    fig.update_layout(
        title=dict(
            text=f"<b>{river_title}</b> — Water Quality Index (WQI) Trend & Daily Forecast",
            font=dict(size=15, color="#1e293b")
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.6)",
            zeroline=False
        ),
        yaxis=dict(
            title="Water Quality Index (WQI)",
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.6)",
            tickvals=[0, 20, 40, 60, 81, 100],
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    return fig


# ============================================================
# FUNCTION: RENDER RIVER PANEL (CHART + OUTPUT KPI CARDS)
# Matches the wireframe: Chart on top, user input + output below
# ============================================================

def render_river_section(river_name, default_station_id, target_date, model_name, param_filter):
    st.subheader(f"📊 {river_name}")

    # Available stations for this river
    stns = ENGINE.get_stations(river_name)
    stn_options = {s["label"]: s["station_id"] for s in stns}
    
    col_sel, col_empty = st.columns([2.5, 3.5])
    with col_sel:
        stn_label = st.selectbox(
            f"Select Station in {river_name}:",
            options=list(stn_options.keys()),
            index=0,
            key=f"select_{river_name}"
        )
    station_id = stn_options[stn_label]

    # Daily time-series and snapshot
    daily_df = ENGINE.get_daily_time_series(station_id, model_name=model_name, days_forecast_ahead=forecast_days)
    snapshot = ENGINE.get_date_snapshot(station_id, target_date, model_name=model_name)

    if daily_df.empty or snapshot is None:
        st.warning(f"No monitoring records available for {station_id}.")
        return

    # Render Trend Chart
    fig = create_trend_figure(daily_df, f"{river_name} ({station_id})", target_date, param_filter)
    st.plotly_chart(fig, width="stretch")

    # OUTPUT SECTION (Faithfully matching user's sketch!)
    # Wireframe specifies:
    # "Date: July" <--- user input
    # "Water Quality Index: 88" } output
    # "Contamination: Polluted" } output
    # "Showing today's Water Quality info, change according to user input in search bar"
    st.markdown("##### 📌 Station Status at Selected Date")
    
    out_col1, out_col2, out_col3, out_col4, out_col5 = st.columns(5)

    with out_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📅 Selected Date</div>
            <div class="metric-value" style="font-size:1.4rem;">{snapshot['month_year']}</div>
            <div class="metric-delta" style="color:#64748b;">Target: {snapshot['display_date']}</div>
        </div>
        """, unsafe_allow_html=True)

    with out_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💧 Water Quality Index (WQI)</div>
            <div class="metric-value">{snapshot['predicted_wqi']}</div>
            <div class="metric-delta" style="color:#0284c7;">DOE {snapshot['class']}</div>
        </div>
        """, unsafe_allow_html=True)

    with out_col3:
        status_html = render_status_badge(snapshot['status'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Contamination Risk</div>
            <div style="margin-top:0.6rem; margin-bottom:0.4rem;">{status_html}</div>
            <div class="metric-delta" style="color:#64748b;">Model: {snapshot['model']}</div>
        </div>
        """, unsafe_allow_html=True)

    with out_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Indicative Class Probabilities</div>
            <div style="font-size:0.85rem; margin-top:0.3rem;">
                <span style="color:#2563eb; font-weight:700;">Clean:</span> {snapshot['prob_clean']*100:.1f}%<br>
                <span style="color:#d97706; font-weight:700;">Slightly:</span> {snapshot['prob_slightly_polluted']*100:.1f}%<br>
                <span style="color:#dc2626; font-weight:700;">Polluted:</span> {snapshot['prob_polluted']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with out_col5:
        p = snapshot['parameters']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧪 Key Parameters</div>
            <div style="font-size:0.82rem; margin-top:0.3rem; line-height:1.4;">
                <b>DO:</b> {p['DO']} mg/L | <b>BOD:</b> {p['BOD']} mg/L<br>
                <b>COD:</b> {p['COD']} mg/L | <b>NH3N:</b> {p['NH3N']} mg/L<br>
                <b>SS:</b> {p['SS']} mg/L | <b>pH:</b> {p['pH']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN DISPLAY LOGIC (Based on station_scope selection)
# ============================================================

tabs = st.tabs([
    "🏠 Overview",
    "🗺️ State & River Map",
    "📅 Daily Forecast",
    "🗓️ Weekly Forecast",
    "📆 Monthly Forecast",
    "📈 Trend & Date Snapshot",
    "🔬 Model Benchmark",
    "🧪 What-If Simulator"
])

# ------------------------------------------------------------
# TAB 1: WIREFRAME VIEW (Trend Charts + Date Snapshot)
# ------------------------------------------------------------
with tabs[5]:
    user_selected_date = _render_trend_controls()
    if station_scope == "Show both stations (default)":
        # Wireframe layout: Sungai Skudai on top, Sungai Klang below (or side-by-side)
        render_river_section(
            river_name="Sungai Skudai",
            default_station_id="3JSKU001",
            target_date=user_selected_date,
            model_name=selected_model,
            param_filter=parameter_filter
        )
        st.markdown("---")
        render_river_section(
            river_name="Sungai Klang",
            default_station_id="2BKLG005",
            target_date=user_selected_date,
            model_name=selected_model,
            param_filter=parameter_filter
        )
    elif station_scope == "Sungai Skudai":
        render_river_section(
            river_name="Sungai Skudai",
            default_station_id="3JSKU001",
            target_date=user_selected_date,
            model_name=selected_model,
            param_filter=parameter_filter
        )
    elif station_scope == "Sungai Klang":
        render_river_section(
            river_name="Sungai Klang",
            default_station_id="2BKLG005",
            target_date=user_selected_date,
            model_name=selected_model,
            param_filter=parameter_filter
        )
    else:
        # Specific Station Selected
        all_stns = ENGINE.get_stations("All")
        stn_match = next((s for s in all_stns if s["station_id"] == specific_station_id), None)
        river_name = stn_match["river"] if stn_match else "Sungai Klang"
        render_river_section(
            river_name=river_name,
            default_station_id=specific_station_id,
            target_date=user_selected_date,
            model_name=selected_model,
            param_filter=parameter_filter
        )


# ------------------------------------------------------------
# SHARED HELPERS FOR THE FORECAST TABS
# ------------------------------------------------------------
HOW_TO_READ = (
    "DOE samples each station about every 60 days, and every model forecasts **one** value per station: the next sampling date. "
    "The daily line is therefore a straight-line trajectory between real points (sample → sample for the past, last sample → model "
    "forecast for the future). Weekly and monthly values are averages of that daily line. The shaded band / error bars show ±1 RMSE "
    "from the model's backtest, widening from 0 at the last sample. Treat the shape between points as an illustration, not a measurement."
)


def _scope_series(model):
    return ENGINE.get_group_series(scope_level, scope_key, model)


def _compare_series():
    return {m: _scope_series(m) for m in compare_models}


def _forecast_kpis(series):
    hist = series[~series["is_forecast"]]
    last = series.iloc[-1]
    prev = float(hist["wqi"].iloc[-1])
    delta = float(last["wqi"]) - prev
    bt = ENGINE.get_model_backtest(selected_model)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Last observed WQI", f"{prev:.1f}", help=f"DOE sample on {hist['date'].iloc[-1]:%d %b %Y}")
    k2.metric(f"Forecast WQI ({last['date']:%d %b %Y})", f"{float(last['wqi']):.1f}", f"{delta:+.1f} vs last sample")
    k3.markdown("**Forecast status**")
    k3.markdown(render_status_badge(classify_wqi(last["wqi"])), unsafe_allow_html=True)
    k4.metric("Backtest RMSE", f"{bt.get('RMSE', float('nan')):.1f}", help="Typical size of the model's error, in WQI points.")
    if scope_level != "station":
        st.caption(f"Averaged over {int(series['n_stations'].iloc[0])} stations. The series ends at the median station forecast date; "
                   "stations with an earlier forecast date are held flat after it.")


def _render_period_tab(freq, unit, title_word):
    series = _scope_series(selected_model)
    if series.empty:
        st.warning("No data for this scope.")
        return
    _forecast_kpis(series)
    cmp_series = _compare_series()
    agg = ENGINE.aggregate_series(series, freq)
    cmp_agg = {m: ENGINE.aggregate_series(x, freq) for m, x in cmp_series.items()}
    st.plotly_chart(period_figure(agg, selected_model, f"<b>{scope_title}</b> — {title_word} WQI forecast ({selected_model})",
                                  cmp_agg, unit=unit, periods_history=8 if freq == "W" else 6), width="stretch")
    st.caption("* partial period (fewer days than a full " + unit + " are covered by the data).  " + HOW_TO_READ)

    view = agg[agg["is_forecast"]].copy()
    first_fc = agg.index[agg["is_forecast"]].min()
    context = agg.iloc[max(0, first_fc - 2):first_fc]
    table = pd.concat([context, view])
    out = pd.DataFrame({
        title_word: table["label"].values + np.where(table["partial"].values, " *", ""),
        "Type": np.where(table["is_forecast"], np.where(table["forecast_days"] < table["n_days"], "Mixed", "Forecast"), "Past"),
        f"{selected_model} WQI": table["wqi"].values,
        "Low (−1 RMSE)": table["lower"].round(1).values,
        "High (+1 RMSE)": table["upper"].round(1).values,
        "Status": table["status"].values,
        "DOE Class": table["class"].values,
        "Days covered": table["n_days"].values,
    })
    for m, a in cmp_agg.items():
        out[f"{m} WQI"] = a.set_index("label")["wqi"].reindex(table["label"]).values
    st.dataframe(out, hide_index=True, width="stretch")
    st.download_button(f"📥 Download {title_word.lower()} forecast CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"wqi_{title_word.lower()}_forecast_{scope_level}_{selected_model}.csv", mime="text/csv",
                       key=f"dl_{freq}")


# ------------------------------------------------------------
# TAB: OVERVIEW (hero + forecast strip + parameters + map + uses)
# ------------------------------------------------------------
def _map_focus_for_scope():
    if scope_level == "state":
        return "Johor (Sungai Skudai)" if scope_key == "Johor" else "Klang Valley (Selangor + KL)"
    if scope_level == "river":
        return "Johor (Sungai Skudai)" if "Skudai" in scope_key else "Klang Valley (Selangor + KL)"
    if scope_level == "station":
        return "Johor (Sungai Skudai)" if scope_key.startswith("3J") else "Klang Valley (Selangor + KL)"
    return "All monitored areas"


with tabs[0]:
    series = _scope_series(selected_model)
    if series.empty:
        st.warning("No data for this scope.")
    else:
        hist = series[~series["is_forecast"]]
        last = series.iloc[-1]
        info = ENGINE.get_scope_parameters(scope_level, scope_key)
        fr = ENGINE.get_data_freshness()
        bt = ENGINE.get_model_backtest(selected_model)
        if scope_level == "station":
            _s = next(x for x in ENGINE.get_stations("All") if x["station_id"] == scope_key)
            crumb = f"Malaysia / {_s['state']} / {_s['river']}"
        elif scope_level in ("state", "river"):
            crumb = f"Malaysia / {scope_key}"
        else:
            crumb = "Malaysia / Klang Valley &amp; Johor"
        left_html, right_html = ui.hero(scope_title, crumb, selected_model, float(last["wqi"]), last["date"],
                                        float(hist["wqi"].iloc[-1]), hist["date"].iloc[-1], bt.get("RMSE", float("nan")),
                                        info.get("main"))
        h1, h2 = st.columns([1.7, 1])
        h1.markdown(left_html, unsafe_allow_html=True)
        h2.markdown(right_html, unsafe_allow_html=True)
        st.markdown(ui.freshness(fr), unsafe_allow_html=True)

        agg_w = ENGINE.aggregate_series(series, "W")
        st.markdown('<div class="card"><h4>Weekly forecast</h4><div class="cap">Average WQI per week · colour = DOE status · range = ±1 RMSE from the model backtest. '
                    'See the Daily, Weekly and Monthly tabs for tables and model comparison.</div>' + ui.weekly_strip(agg_w) + '</div>',
                    unsafe_allow_html=True)

        c_left, c_right = st.columns([1.25, 1])
        with c_left:
            agg_m = ENGINE.aggregate_series(series, "M")
            st.markdown('<div class="card"><h4>Monthly forecast</h4><div class="cap">Bars show the ±1 RMSE range on a 0–100 scale.</div>'
                        + ui.monthly_rows(agg_m) + '</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("#### Station map")
                st.caption(f"{selected_model} forecast at each station's next sampling date. Open the State & River Map tab for more views.")
                _snap = ENGINE.get_map_snapshot(selected_model, mode="forecast")
                _fig = map_figure(_snap, _map_focus_for_scope())
                _fig.update_layout(height=400)
                st.plotly_chart(_fig, width="stretch", key="overview_map")
        with c_right:
            if info:
                st.markdown(f'<div class="card"><h4>Water parameters</h4><div class="cap">Latest DOE sample, averaged over {info["n_stations"]} station(s) '
                            f'(most recent {info["last_sample"]:%d %b %Y}). Dot colour = DOE sub-index band.</div>'
                            + ui.parameter_rows(info) + '</div>', unsafe_allow_html=True)
                st.markdown(ui.parameter_alert(info), unsafe_allow_html=True)
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            uses_html, cls = ui.uses_rows(float(last["wqi"]))
            st.markdown(f'<div class="card"><h4>What this water can be used for</h4><div class="cap">Forecast {cls} · beneficial uses under Malaysia\'s National Water Quality Standards (NWQS).</div>'
                        + uses_html + '</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB: STATE & RIVER MAP
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("🗺️ Water Quality by State and River")
    m_col1, m_col2, m_col3 = st.columns([2, 2, 2])
    with m_col1:
        map_mode_label = st.radio("Show on map", ["Forecast (next sampling date)", "Latest DOE sample", "Pick a date"],
                                  horizontal=False, key="map_mode")
    with m_col2:
        map_focus = st.selectbox("Zoom to", list(MAP_FOCUS.keys()), key="map_focus")
    with m_col3:
        map_date = None
        if map_mode_label == "Pick a date":
            map_date = st.date_input("Date", value=date(2024, 12, 1), min_value=date(2023, 1, 16), max_value=date(2025, 1, 6),
                                     key="map_date", help="Values come from the daily trajectory; dates outside a station's range use its nearest end.")
    map_mode = {"Forecast (next sampling date)": "forecast", "Latest DOE sample": "observed", "Pick a date": "date"}[map_mode_label]
    snap = ENGINE.get_map_snapshot(selected_model, mode=map_mode, target_date=map_date)

    if snap.empty:
        st.info("No station data available.")
    else:
        fig_map = map_figure(snap, map_focus, title=f"{map_mode_label} — {selected_model}")
        st.plotly_chart(fig_map, width="stretch")
        st.caption("State shading and outline = state's mean station WQI; dots = individual stations, coloured by DOE status. "
                   "Only states/rivers in the DOE dataset (Selangor, W.P. Kuala Lumpur, Johor) have data; other states are shown for context. "
                   "The map needs internet access to load OpenStreetMap tiles.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### By state")
            ss = state_summary(snap).rename(columns={"state": "State", "stations": "Stations", "wqi": "Mean WQI", "status": "Status", "polluted": "Polluted stations"})
            st.dataframe(ss[["State", "Stations", "Mean WQI", "Status", "Polluted stations"]], hide_index=True, width="stretch")
        with c2:
            st.markdown("##### By river")
            rs = river_summary(snap).rename(columns={"river": "River", "stations": "Stations", "wqi": "Mean WQI", "status": "Status", "polluted": "Polluted stations"})
            st.dataframe(rs[["River", "Stations", "Mean WQI", "Status", "Polluted stations"]], hide_index=True, width="stretch")
        st.markdown("##### Stations")
        tbl = snap[["station_id", "location", "river", "state", "wqi", "status", "class", "as_of", "last_observed_wqi"]].copy()
        tbl["as_of"] = tbl["as_of"].dt.strftime("%Y-%m-%d")
        tbl.columns = ["Station", "Location", "River", "State", "WQI", "Status", "DOE Class", "As of", "Last observed WQI"]
        st.dataframe(tbl.sort_values("WQI"), hide_index=True, width="stretch")


# ------------------------------------------------------------
# TAB: DAILY FORECAST
# ------------------------------------------------------------
with tabs[2]:
    st.subheader(f"📅 Daily forecast — {scope_title}")
    series = _scope_series(selected_model)
    if series.empty:
        st.warning("No data for this scope.")
    else:
        _forecast_kpis(series)
        cmp_series = _compare_series()
        st.plotly_chart(daily_figure(series, selected_model, f"<b>{scope_title}</b> — daily WQI ({selected_model})", cmp_series), width="stretch")
        st.caption(HOW_TO_READ)

        fc = series[series["is_forecast"]].copy()
        tbl = pd.DataFrame({
            "Date": fc["date"].dt.strftime("%Y-%m-%d"),
            f"{selected_model} WQI": fc["wqi"].round(1),
            "Low (−1 RMSE)": fc["lower"].round(1),
            "High (+1 RMSE)": fc["upper"].round(1),
            "Status": fc["wqi"].apply(classify_wqi),
            "DOE Class": fc["wqi"].apply(get_doe_class),
        })
        for m, cs in cmp_series.items():
            tbl[f"{m} WQI"] = cs.set_index("date")["wqi"].reindex(fc["date"]).round(1).values
        st.dataframe(tbl, hide_index=True, width="stretch", height=320)
        st.download_button("📥 Download daily forecast CSV", tbl.to_csv(index=False).encode("utf-8"),
                           file_name=f"wqi_daily_forecast_{scope_level}_{selected_model}.csv", mime="text/csv", key="dl_D")


# ------------------------------------------------------------
# TAB: WEEKLY FORECAST
# ------------------------------------------------------------
with tabs[3]:
    st.subheader(f"🗓️ Weekly forecast — {scope_title}")
    _render_period_tab("W", "week", "Week")


# ------------------------------------------------------------
# TAB: MONTHLY FORECAST
# ------------------------------------------------------------
with tabs[4]:
    st.subheader(f"📆 Monthly forecast — {scope_title}")
    _render_period_tab("M", "month", "Month")


# ------------------------------------------------------------
# TAB 4: MULTI-MODEL BENCHMARK ARENA (Do It Better!)
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("🔬 Multi-Model Performance Benchmark Arena")
    st.markdown("Comprehensive comparison of all 6 models implemented across WQI regression and Contamination Risk classification.")

    metrics_df = ENGINE.get_metrics_table()
    if not metrics_df.empty:
        m_col1, m_col2 = st.columns([3, 2])
        
        with m_col1:
            st.markdown("##### 📋 Unified Evaluation Metrics Table")
            st.warning("Models were scored on different numbers of test samples (see n_samples), so these metrics are indicative, not a like-for-like ranking. "
                       "Persistence (R² 0.64) is a strong baseline; plain LSTM and XGBoost scored below it on their smaller test sets.")
            display_m = metrics_df.copy()
            st.dataframe(
                display_m.style.format({
                    "MAE": "{:.3f}",
                    "RMSE": "{:.3f}",
                    "R2": "{:.3f}",
                    "accuracy": "{:.3f}",
                    "balanced_accuracy": "{:.3f}",
                    "macro_f1": "{:.3f}"
                }).highlight_min(subset=["MAE", "RMSE"], color="#dcfce7")
                  .highlight_max(subset=["R2", "accuracy", "macro_f1"], color="#dcfce7"),
                width="stretch"
            )

        with m_col2:
            st.markdown("##### 🏆 WQI Regression Accuracy (R² Score)")
            fig_r2 = px.bar(
                metrics_df,
                x="model",
                y="R2",
                color="model",
                text="R2",
                title="Model R² Score Comparison (Higher is Better)",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_r2.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_r2, width="stretch")

        st.markdown("---")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            fig_mae = px.bar(
                metrics_df,
                x="model",
                y="MAE",
                color="model",
                text="MAE",
                title="Mean Absolute Error (MAE - Lower is Better)",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_mae.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_mae.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_mae, width="stretch")

        with c_col2:
            fig_acc = px.bar(
                metrics_df,
                x="model",
                y="accuracy",
                color="model",
                text="accuracy",
                title="Contamination Risk Classification Accuracy",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_acc.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_acc.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_acc, width="stretch")
    else:
        st.info("Model metrics are currently being consolidated.")


# ------------------------------------------------------------
# TAB 5: WHAT-IF SCENARIO SIMULATOR (Do It Better!)
# ------------------------------------------------------------
with tabs[7]:
    st.subheader("🧪 Interactive What-If Water Parameter Simulator")
    st.markdown("Adjust physical-chemical water parameters below to simulate instant real-time DOE Water Quality Index, DOE Class, and Contamination Risk prediction.")

    sim_col1, sim_col2 = st.columns([3, 2])

    with sim_col1:
        st.markdown("##### 🎚️ Environmental Parameter Controls")
        sim_do = st.slider("Dissolved Oxygen DO (% Saturation)", min_value=0.0, max_value=150.0, value=75.0, step=1.0)
        sim_bod = st.slider("Biochemical Oxygen Demand BOD (mg/L)", min_value=0.0, max_value=25.0, value=3.5, step=0.1)
        sim_cod = st.slider("Chemical Oxygen Demand COD (mg/L)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        sim_nh3n = st.slider("Ammoniacal Nitrogen NH3-N (mg/L)", min_value=0.0, max_value=8.0, value=0.4, step=0.05)
        sim_ss = st.slider("Total Suspended Solids SS (mg/L)", min_value=0.0, max_value=300.0, value=40.0, step=2.0)
        sim_ph = st.slider("pH Value", min_value=4.0, max_value=10.0, value=7.2, step=0.1)

    sim_result = calculate_doe_wqi(sim_do, sim_bod, sim_cod, sim_nh3n, sim_ss, sim_ph)

    with sim_col2:
        st.markdown("##### 🎯 Real-Time Simulated Output")
        sim_wqi = sim_result["wqi"]
        sim_status = sim_result["status"]
        sim_class = sim_result["class"]
        
        st.markdown(f"""
        <div class="metric-card" style="text-align:center; padding: 2rem;">
            <div class="metric-label">Simulated Water Quality Index</div>
            <div class="metric-value" style="font-size:3.5rem; color:{get_class_color(sim_status)};">{sim_wqi}</div>
            <div style="margin-top:1rem; margin-bottom:0.5rem;">{render_status_badge(sim_status)}</div>
            <div style="font-weight:700; color:#475569; font-size:1.1rem; margin-top:0.5rem;">DOE {sim_class}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### 📐 Calculated DOE Sub-Indices")
        sub_df = pd.DataFrame(list(sim_result["sub_indices"].items()), columns=["Sub-Index", "Score (0-100)"])
        st.dataframe(sub_df, width="stretch", height=220)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; color:#64748b; font-size:0.85rem;">
    <span>🌊 Malaysian Department of Environment (DOE) River Water Quality Monitoring System</span>
    <span>Models: Persistence | CEEMDAN | LSTM | CEEMDAN-LSTM | XGBoost | CEEMDAN-LSTM-XGBoost</span>
</div>
""", unsafe_allow_html=True)
