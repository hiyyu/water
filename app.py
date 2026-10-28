"""Water Quality FYP dashboard — two tabs, backed by saved model outputs.
Run: python -m streamlit run app.py
"""
from pathlib import Path
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DASH = ROOT / 'outputs' / 'dashboard'
EVAL = ROOT / 'outputs' / 'evaluation'
STATUS_COLORS = {'Clean': '#2788e8', 'Slightly Polluted': '#e9b700', 'Polluted': '#ee5050'}

st.set_page_config(page_title='Water Quality Prediction', page_icon='💧', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.7rem; max-width: 1500px}
[data-testid="stMetric"] {border:1px solid #dce6ee; padding:13px; border-radius:12px}
h1,h2,h3 {letter-spacing:-.02em}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def csv_file(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    for col in df.columns:
        if ('date' in col.lower() or col.lower() in ['smp-dat']) and not col.lower().endswith('_id'):
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().any():
                df[col] = converted
    return df

def col(df, *names):
    normalized = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
    for name in names:
        key = re.sub(r'[^a-z0-9]', '', name.lower())
        if key in normalized:
            return normalized[key]
    return None

def format_number(value, digits=2):
    try:
        return f'{float(value):.{digits}f}' if pd.notna(value) else '—'
    except (ValueError, TypeError):
        return '—'

def normalize_river(value):
    s = str(value).strip().upper()
    if 'KLANG' in s: return 'Sungai Klang'
    if 'SKUDAI' in s: return 'Sungai Skudai'
    return str(value).strip()

def prepare_river(df):
    df = df.copy()
    rc = col(df, 'river', 'sungai', 'river_name')
    if rc: df['_river'] = df[rc].map(normalize_river)
    return df

def filter_river(df, chosen):
    if '_river' not in df or chosen == 'Both Rivers': return df.copy()
    return df[df['_river'] == chosen].copy()

def filter_station(df, chosen):
    sc = col(df, 'station_id', 'id stn baru', 'station', 'station_name')
    if not sc or chosen == 'All Stations': return df.copy()
    return df[df[sc].astype(str) == chosen].copy()

def filter_dates(df, start, end):
    dc = col(df, 'target_date', 'sample_date', 'date', 'smp-dat', 'last_sample_date')
    if not dc or not pd.api.types.is_datetime64_any_dtype(df[dc]): return df
    return df[df[dc].between(pd.Timestamp(start), pd.Timestamp(end))].copy()

def date_options(*frames):
    vals = []
    for frame in frames:
        dc = col(frame, 'target_date', 'sample_date', 'date', 'smp-dat', 'last_sample_date')
        if dc and pd.api.types.is_datetime64_any_dtype(frame[dc]):
            vals.extend(frame[dc].dropna().dt.date.tolist())
    return sorted(set(vals))

def model_names(df):
    mc = col(df, 'model', 'model_name')
    return sorted(df[mc].dropna().astype(str).unique()) if mc else []

def show_table(df, label, filename):
    st.subheader(label)
    if df.empty:
        st.info('No saved records match the current selection.')
    else:
        st.dataframe(df.drop(columns=['_river'], errors='ignore'), use_container_width=True, hide_index=True)
        st.download_button('Download CSV', df.drop(columns=['_river'], errors='ignore').to_csv(index=False), filename, mime='text/csv', key=f'dl_{filename}')

def metric_subset(df, name):
    mc = col(df, 'model', 'model_name')
    if not mc or not name: return df.iloc[:0]
    return df[df[mc].astype(str).str.casefold() == str(name).casefold()]

def show_metric_cards(frame):
    if frame.empty:
        st.caption('No matching saved evaluation metrics.')
        return
    row = frame.iloc[0]
    wanted = [('MAE', 'mae'), ('RMSE', 'rmse'), ('R²', 'r2'), ('Accuracy', 'accuracy'), ('Macro F1', 'macro_f1'), ('Balanced accuracy', 'balanced_accuracy')]
    available = [(label, col(frame, key)) for label, key in wanted]
    available = [(label, key) for label, key in available if key and pd.notna(row[key])]
    if not available:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        return
    for batch_start in range(0, len(available), 3):
        cells = st.columns(min(3, len(available) - batch_start))
        for cell, (label, key) in zip(cells, available[batch_start:batch_start+3]):
            cell.metric(label, format_number(row[key], 3))
    if len(frame) > 1:
        with st.expander('All evaluation records for this model'):
            st.dataframe(frame, use_container_width=True, hide_index=True)

def plot_wqi_history(df, title, parameter='WQI'):
    dc = col(df, 'sample_date', 'target_date', 'date', 'smp-dat', 'last_sample_date')
    value_candidates = {
        'WQI': ['wqi', 'actual_wqi', 'last_observed_wqi'],
        'DO': ['do', 'dissolved_oxygen'],
        'BOD': ['bod'], 'COD': ['cod'], 'NH3N': ['nh3n', 'nh3-n', 'ammoniacal_nitrogen'],
        'SS/TSS': ['ss/tss', 'ss', 'tss', 'suspended_solids'], 'pH': ['ph']
    }
    vc = col(df, *value_candidates.get(parameter, ['wqi']))
    if not dc or not vc:
        st.info(f'Historical {parameter} observations are not available in the loaded dashboard CSVs.')
        return
    data = df[[dc, vc]].copy()
    data[vc] = pd.to_numeric(data[vc], errors='coerce')
    data = data.dropna().sort_values(dc)
    if data.empty:
        st.info(f'No historical {parameter} observations for this selection.')
        return
    # Aggregate by sampling date when multiple stations are selected.
    data = data.groupby(dc, as_index=False)[vc].mean()
    fig = px.line(data, x=dc, y=vc, markers=True, title=title)
    fig.update_layout(xaxis_title='Sampling date', yaxis_title=parameter, height=320, margin=dict(t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Historical observations; multiple stations are averaged per sampling date when applicable.')

def wqi_category(value):
    """DOE WQI classification; an interpretation, not an XGBoost forecast."""
    try:
        x = float(value)
        if pd.isna(x): return 'Unknown'
        if x > 80: return 'Clean'
        if x >= 60: return 'Slightly Polluted'
        return 'Polluted'
    except (TypeError, ValueError):
        return 'Unknown'

def forecast_cards(frame, river, station, history):
    data = filter_station(filter_river(frame, river), station)
    pred = col(data, 'predicted_wqi', 'predicted_next_wqi')
    observed = col(data, 'last_observed_wqi', 'previous_wqi')
    forecast_date = col(data, 'forecast_date_estimate', 'forecast_date')
    station_key = col(data, 'station_id', 'station')
    if data.empty or not pred:
        st.info('No saved next-sampling WQI forecast for this model and station.')
        return
    data = data.copy()
    data[pred] = pd.to_numeric(data[pred], errors='coerce')
    data = data.dropna(subset=[pred])
    if data.empty:
        st.info('No numeric forecast values are available.')
        return
    data['Estimated category'] = data[pred].map(wqi_category)
    st.markdown('#### Latest available forecast')
    if len(data) == 1:
        row = data.iloc[0]
        cards = st.columns(3)
        cards[0].metric('Last measured WQI', format_number(row[observed], 1) if observed else '—')
        cards[1].metric('Predicted next WQI', format_number(row[pred], 1))
        cards[2].metric('Estimated category', row['Estimated category'])
        if forecast_date:
            st.caption(f"Estimated next sampling event: {str(row[forecast_date])[:10]} (not a daily forecast)")
    else:
        cards = st.columns(3)
        cards[0].metric('Stations forecasted', len(data))
        cards[1].metric('Mean predicted WQI', format_number(data[pred].mean(), 1))
        cards[2].metric('Estimated category of mean', wqi_category(data[pred].mean()))
    st.caption('Status colours: 🔵 Clean (WQI > 80)  •  🟡 Slightly Polluted (60–80)  •  🔴 Polluted (< 60). Categories here are calculated from predicted WQI, not predicted by XGBoost.')

    st.markdown('#### Next-sampling forecast by station')
    data['_label'] = data[station_key].astype(str) if station_key else [f'Station {i+1}' for i in range(len(data))]
    data['_colour'] = data['Estimated category'].map(STATUS_COLORS)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['_label'], y=data[pred], marker_color=data['_colour'], name='Predicted WQI',
                         customdata=data[['Estimated category']].to_numpy(),
                         hovertemplate='Station %{x}<br>Predicted WQI %{y:.1f}<br>%{customdata[0]}<extra></extra>'))
    fig.add_hline(y=80, line_dash='dot', line_color='#2788e8', opacity=.5)
    fig.add_hline(y=60, line_dash='dot', line_color='#e9b700', opacity=.5)
    fig.update_layout(height=310, margin=dict(t=12,b=10), xaxis_title='Monitoring station', yaxis_title='Predicted next WQI', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    if len(data) == 1 and observed and pd.notna(data.iloc[0][observed]):
        st.markdown('#### Latest observation versus next forecast')
        row = data.iloc[0]
        compare = pd.DataFrame({'Stage':['Last observation','Next sampling forecast'], 'WQI':[float(row[observed]),float(row[pred])]})
        fig2 = px.bar(compare, x='Stage', y='WQI', color='Stage', text='WQI', color_discrete_sequence=['#8c9baa','#458ed0'])
        fig2.update_layout(height=250, showlegend=False, margin=dict(t=8,b=8))
        st.plotly_chart(fig2, use_container_width=True)
    columns = [c for c in [station_key, observed, pred, 'Estimated category', forecast_date] if c]
    with st.expander('View exact station forecasts'):
        st.dataframe(data[columns].drop_duplicates(), hide_index=True, use_container_width=True)
    st.caption('These are next-sample estimates from saved model outputs. They are not live readings, seven-day forecasts, or a statement that river water is safe to use.')

def projection_until_today(frame, river, station):
    """Transparent carry-forward scenario, NOT repeated CEEMDAN-LSTM inference."""
    data = filter_station(filter_river(frame, river), station)
    pc = col(data, 'predicted_wqi', 'predicted_next_wqi')
    dc = col(data, 'forecast_date_estimate', 'forecast_date')
    lc = col(data, 'last_sample_date')
    sc = col(data, 'station_id', 'station')
    if data.empty or not pc or not sc:
        return
    data = data.copy()
    data[pc] = pd.to_numeric(data[pc], errors='coerce')
    data = data.dropna(subset=[pc])
    if data.empty:
        return
    today = pd.Timestamp.now().normalize()
    rows = []
    for _, record in data.iterrows():
        base = pd.to_datetime(record[dc], errors='coerce') if dc else pd.NaT
        if pd.isna(base) and lc:
            last = pd.to_datetime(record[lc], errors='coerce')
            base = last + pd.DateOffset(months=2) if pd.notna(last) else pd.NaT
        if pd.isna(base):
            continue
        base = pd.Timestamp(base).normalize()
        value = float(record[pc])
        if base > today:
            continue
        # Project at approximate 2-month intervals, matching the sampling cadence.
        date = base
        while date <= today:
            rows.append({'Station': str(record[sc]), 'Period': date, 'Projected WQI': value,
                         'Estimated status': wqi_category(value), 'Method': 'Constant carry-forward scenario'})
            date += pd.DateOffset(months=2)
        # Include today's date as an explicitly carried-forward estimate.
        if rows and rows[-1]['Period'] < today:
            rows.append({'Station': str(record[sc]), 'Period': today, 'Projected WQI': value,
                         'Estimated status': wqi_category(value), 'Method': 'Constant carry-forward scenario'})
    if not rows:
        st.info('No saved forecast dates fall before today for the current selection.')
        return
    result = pd.DataFrame(rows)
    st.markdown('#### Experimental projection up to today')
    st.warning('This is **not a validated CEEMDAN–LSTM forecast for 2025–2026**. '
               'It holds each station’s last saved next-sampling WQI prediction constant through today '
               '(a persistence scenario). No new water-quality observations or future ML predictions are available.')
    station_count = result['Station'].nunique()
    end_rows = result.sort_values('Period').groupby('Station', as_index=False).tail(1)
    a, b, c = st.columns(3)
    a.metric('Projection date', today.strftime('%d %b %Y'))
    b.metric('Stations', station_count)
    c.metric('Mean projected WQI', format_number(end_rows['Projected WQI'].mean(), 1))
    fig = px.line(result, x='Period', y='Projected WQI', color='Station', markers=True,
                  title='Carry-forward scenario by station (not a daily model forecast)')
    fig.update_layout(height=350, yaxis_title='WQI', xaxis_title='Approximate sampling period')
    st.plotly_chart(fig, use_container_width=True)
    status_counts = end_rows['Estimated status'].value_counts().rename_axis('Status').reset_index(name='Stations')
    fig_status = px.bar(status_counts, x='Status', y='Stations', color='Status',
                        color_discrete_map=STATUS_COLORS, title='Estimated status at projection endpoint')
    fig_status.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_status, use_container_width=True)
    with st.expander('View / download experimental projections'):
        st.dataframe(result, hide_index=True, use_container_width=True)
        st.download_button('Download scenario CSV', result.to_csv(index=False),
                           file_name=f'carry_forward_{river.replace(" ", "_")}.csv',
                           mime='text/csv', key=f'projection_download_{river}')

latest = prepare_river(csv_file(DASH / 'wqi_latest_forecasts.csv'))
wqi = prepare_river(csv_file(DASH / 'wqi_evaluation_predictions.csv'))
risk = prepare_river(csv_file(DASH / 'risk_evaluation_predictions.csv'))
metrics = prepare_river(csv_file(DASH / 'model_metrics.csv'))
confusions = prepare_river(csv_file(DASH / 'classification_confusion_matrices.csv'))
observations = prepare_river(csv_file(DASH / 'latest_station_observations.csv'))
integrated = prepare_river(csv_file(EVAL / 'integrated_pipeline_predictions.csv'))
raw = prepare_river(csv_file(ROOT / 'outputs' / 'combined_doe_dataset.csv'))

st.title('💧 River Water Quality Forecast')
st.caption('Sungai Klang and Sungai Skudai • Seven-day illustrative outlook • All saved models')
st.info('DOE samples are approximately two months apart. The seven-day display is a carry-forward scenario based on a saved next-sampling WQI estimate, not seven independently modelled daily forecasts.')
if all(df.empty for df in [latest, wqi, risk, metrics, raw]):
    st.error('No dashboard data found. Place app.py in the project root and run src/19_prepare_dashboard_data.py first.')
    st.code('python src/19_prepare_dashboard_data.py\npython -m streamlit run app.py', language='powershell')
    st.stop()

tab_prediction, tab_evaluation = st.tabs(['💧 Water Quality Forecast', '📊 Model Evaluation'])

def seven_day_scenario(frame, river, station, chosen_model):
    """Daily display ONLY: carry forward saved next-sampling estimate, no daily ML inference."""
    data = filter_station(filter_river(frame, river), station)
    pc = col(data, 'predicted_wqi', 'predicted_next_wqi')
    sc = col(data, 'station_id', 'station')
    oc = col(data, 'last_observed_wqi')
    lc = col(data, 'last_sample_date')
    if data.empty or not pc:
        st.info('No saved next-sampling WQI estimate available for this selection.')
        return
    data = data.copy()
    data[pc] = pd.to_numeric(data[pc], errors='coerce')
    data = data.dropna(subset=[pc])
    if data.empty:
        st.info('No numeric next-sampling WQI estimate available.')
        return
    today = pd.Timestamp.now().normalize()
    station_values = data[sc].astype(str) if sc else pd.Series(['Station'] * len(data), index=data.index)
    data['_station'] = station_values
    # Never call a carried-forward number a fresh 2026 CEEMDAN-LSTM forecast.
    seed = float(data[pc].mean())
    category = wqi_category(seed)
    st.markdown('#### Latest estimated status — today')
    c1, c2, c3 = st.columns(3)
    c1.metric('Today', today.strftime('%d %b %Y'))
    c2.metric('Scenario WQI', f'{seed:.1f}')
    c3.metric('Estimated status', category)
    st.caption('Today is the scenario display date, **not a DOE measurement date**. The latest available source observation is from 2024; this value is carried forward from a saved next-sampling forecast.')
    st.markdown('#### Seven-day outlook (today + next six days)')
    days = pd.date_range(today, periods=7, freq='D')
    rows = []
    for _, rec in data.iterrows():
        for day in days:
            rows.append({'Date':day, 'Station':rec['_station'], 'Scenario WQI':float(rec[pc]),
                         'Estimated status':wqi_category(rec[pc]), 'Method':'Unchanged carry-forward of saved next-sampling forecast'})
    outlook = pd.DataFrame(rows)
    daily = outlook.groupby('Date', as_index=False)['Scenario WQI'].mean()
    daily['Estimated status'] = daily['Scenario WQI'].map(wqi_category)
    card_cols = st.columns(7)
    for i, (_, record) in enumerate(daily.iterrows()):
        with card_cols[i]:
            st.markdown(f"**{record['Date']:%a}**")
            st.markdown(f"**{record['Scenario WQI']:.1f}**")
            st.caption(record['Estimated status'])
    st.markdown('#### Seven-day WQI outlook chart')
    if len(data) == 1:
        fig = px.bar(daily, x='Date', y='Scenario WQI', color='Estimated status',
                     color_discrete_map=STATUS_COLORS, text='Scenario WQI')
    else:
        fig = px.line(outlook, x='Date', y='Scenario WQI', color='Station', markers=True)
    fig.update_layout(height=340, yaxis_range=[0, 100], xaxis_title='Date', yaxis_title='Scenario WQI')
    st.plotly_chart(fig, use_container_width=True)
    st.warning('**Research limitation:** These seven daily values are a visualization-only persistence scenario. '
               'Your CEEMDAN–LSTM is trained on approximately two-monthly observations and has NOT generated or validated seven independent daily forecasts. '
               'The same estimate is repeated, not newly predicted each day. The dashboard does not provide live water quality.')
    if lc and lc in data:
        last_dates = pd.to_datetime(data[lc], errors='coerce').dropna()
        if not last_dates.empty:
            st.caption(f'Latest recorded sample in this selection: {last_dates.max():%d %b %Y}.')
    st.download_button('Download seven-day scenario CSV', outlook.to_csv(index=False),
                       file_name='seven_day_wqi_scenario.csv', mime='text/csv',
                       key=f'daily_{river}_{station}'.replace(' ','_'))

with tab_prediction:
    st.subheader('Seven-day water quality outlook')
    st.info('The display begins **today** and covers seven days. It is an explicitly labelled persistence scenario, not a validated daily ML forecast. All saved models remain accessible below.')
    river_options = ['Both Rivers'] + sorted({v for df in [latest, raw, risk] if '_river' in df for v in df['_river'].dropna().unique()})
    controls = st.columns([1,1,1.3])
    river = controls[0].selectbox('River', river_options, key='forecast_river')
    station_source = pd.concat([filter_river(df, river) for df in [latest, observations, raw] if not df.empty], ignore_index=True)
    sc = col(station_source, 'station_id', 'id stn baru', 'station')
    station_options = ['All Stations'] + (sorted(station_source[sc].dropna().astype(str).unique()) if sc else [])
    station = controls[1].selectbox('Station', station_options, key='forecast_station')
    available_models = sorted(set(model_names(latest) + model_names(wqi) + model_names(risk)))
    if not available_models: available_models = ['No saved model']
    preferred = next((m for m in available_models if 'CEEMDAN' in m.upper()), available_models[0])
    model = controls[2].selectbox('Model (all saved models)', available_models,
                                  index=available_models.index(preferred), key='forecast_model')
    latest_models = model_names(latest)
    if model in latest_models:
        selected = latest.copy()
        mc = col(selected, 'model', 'model_name')
        if mc: selected = selected[selected[mc].astype(str) == model]
        panels = [r for r in river_options[1:] if not filter_river(selected, r).empty] if river == 'Both Rivers' else [river]
        for current_river in panels:
            st.markdown(f'### {current_river}')
            seven_day_scenario(selected, current_river, station, model)
    else:
        st.warning(f'**{model}** has no saved future WQI forecast. A seven-day WQI prediction cannot be attributed to this model.')
        st.markdown('#### Available saved model results')
        wqi_subset = wqi.copy()
        wm = col(wqi_subset, 'model', 'model_name')
        if wm: wqi_subset = wqi_subset[wqi_subset[wm].astype(str) == model]
        risk_subset = risk.copy()
        rm = col(risk_subset, 'model', 'model_name')
        if rm: risk_subset = risk_subset[risk_subset[rm].astype(str) == model]
        display = wqi_subset if not wqi_subset.empty else risk_subset
        display = filter_station(filter_river(display, river), station)
        if not display.empty:
            st.dataframe(display.drop(columns=['_river'], errors='ignore'), hide_index=True, use_container_width=True)
            st.caption('Historical held-out evaluation predictions only — not forecasts for today or the next seven days.')
        else:
            st.info('No prediction rows for the selected model and filters.')
    st.markdown('### Selected model evaluation')
    chosen_metrics = metric_subset(metrics, model)
    if '_river' in chosen_metrics and river != 'Both Rivers':
        sub = filter_river(chosen_metrics, river)
        if not sub.empty: chosen_metrics = sub
    show_metric_cards(chosen_metrics)
    st.caption('Evaluation scores are from previously saved test sets; they do not validate the seven-day scenario.')

with tab_evaluation:
    st.subheader('Model evaluation')
    st.caption('Compare models using their saved test results. WQI regression and status classification use different metrics.')
    section = st.radio('Evaluation category', ['WQI Regression', 'Contamination Classification', 'Integrated Pipeline'], horizontal=True, key='eval_category')
    if section == 'WQI Regression':
        show_table(metrics, 'Saved model metrics (all evaluation runs)', 'model_metrics.csv')
        st.subheader('Actual versus predicted WQI')
        wqi_models = model_names(wqi)
        chosen = st.multiselect('Models to display', wqi_models, default=wqi_models, key='eval_wqi_models')
        plot_data = wqi.copy()
        mc = col(plot_data, 'model', 'model_name')
        actual = col(plot_data, 'actual_wqi', 'actual')
        predicted = col(plot_data, 'predicted_wqi', 'prediction')
        if mc and actual and predicted and chosen:
            plot_data = plot_data[plot_data[mc].astype(str).isin(chosen)].copy()
            plot_data[actual] = pd.to_numeric(plot_data[actual], errors='coerce')
            plot_data[predicted] = pd.to_numeric(plot_data[predicted], errors='coerce')
            plot_data = plot_data.dropna(subset=[actual, predicted])
            if not plot_data.empty:
                fig = px.scatter(plot_data, x=actual, y=predicted, color=mc, hover_data=['_river'] if '_river' in plot_data else None)
                low = min(plot_data[actual].min(), plot_data[predicted].min())
                high = max(plot_data[actual].max(), plot_data[predicted].max())
                fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode='lines', name='Perfect prediction', line=dict(dash='dash', color='gray')))
                st.plotly_chart(fig, use_container_width=True)
            else: st.info('No matching numeric WQI predictions.')
        else: st.info('The saved WQI evaluation file does not contain the required model, actual and predicted columns, or no model is selected.')
        with st.expander('WQI evaluation prediction records'):
            st.dataframe(wqi.drop(columns=['_river'], errors='ignore'), use_container_width=True, hide_index=True)
    elif section == 'Contamination Classification':
        st.markdown('**Saved classification evaluation records**')
        classification_metrics = metrics.copy()
        type_col = col(classification_metrics, 'task', 'task_type', 'metric_type', 'evaluation_type')
        if type_col:
            subset = classification_metrics[classification_metrics[type_col].astype(str).str.contains('class|risk|status', case=False, na=False)]
            if not subset.empty: classification_metrics = subset
        st.dataframe(classification_metrics.drop(columns=['_river'], errors='ignore'), use_container_width=True, hide_index=True)
        st.markdown('**Confusion matrices**')
        if not confusions.empty:
            st.dataframe(confusions.drop(columns=['_river'], errors='ignore'), use_container_width=True, hide_index=True)
        else: st.info('No saved confusion-matrix file found.')
        with st.expander('Classification prediction records'):
            st.dataframe(risk.drop(columns=['_river'], errors='ignore'), use_container_width=True, hide_index=True)
    else:
        st.markdown('**Integrated CEEMDAN–LSTM + XGBoost evaluation**')
        st.info('The integrated approach is presented as an evaluated model, not as a proven improvement in contamination prediction.')
        metrics_json = EVAL / 'integrated_pipeline_metrics.json'
        if metrics_json.exists():
            with metrics_json.open('r', encoding='utf-8') as f:
                st.json(json.load(f), expanded=2)
        else: st.info('Integrated pipeline metrics JSON not found.')
        show_table(integrated, 'Integrated pipeline saved predictions', 'integrated_pipeline_predictions.csv')

st.caption('Research prototype • Uses saved DOE observations and model outputs • Predictions are estimates and should not be treated as real-time water safety measurements.')