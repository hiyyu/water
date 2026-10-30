"""
src/overview_ui.py
Visual design system + HTML builders for the Overview tab (hero, forecast strip, parameter panel).
All functions return compact HTML strings (no indentation / blank lines, so Streamlit's markdown keeps them as HTML).
"""

from model_engine import STATUS_CLEAN, STATUS_SLIGHTLY_POLLUTED, STATUS_POLLUTED, classify_wqi, get_doe_class

BADGE = {  # status: (background, text, soft tint, border)
    STATUS_CLEAN: ("#2563EB", "#ffffff", "#e8f0fe", "#bfd3fb"),
    STATUS_SLIGHTLY_POLLUTED: ("#EAB308", "#1f2937", "#fef6d8", "#f3dd8a"),
    STATUS_POLLUTED: ("#EF4444", "#ffffff", "#fdeaea", "#f6b9b9"),
}
SHORT = {STATUS_CLEAN: "Clean", STATUS_SLIGHTLY_POLLUTED: "Slightly polluted", STATUS_POLLUTED: "Polluted"}

# Malaysia NWQS Class III reference limits (DOE). Verify against the official table before publishing.
CLASS3_LIMITS = {"BOD": 6.0, "COD": 50.0, "NH3N": 0.9, "SS": 150.0}

NWQS_USES = {
    "Class I": [("🚰", "Water supply", "Practically no treatment necessary"), ("🌿", "Aquatic life", "Suitable for very sensitive species")],
    "Class II": [("🚰", "Water supply", "Conventional treatment required"), ("🏊", "Recreation", "Body-contact recreation possible (Class IIB)")],
    "Class III": [("🚰", "Water supply", "Extensive treatment required"), ("🐟", "Aquatic life & livestock", "Common, tolerant species; livestock drinking")],
    "Class IV": [("🌾", "Irrigation", "Suitable for irrigation only"), ("⚠️", "Water supply & aquatic life", "Not a designated use at this class")],
    "Class V": [("⛔", "No NWQS beneficial use", "Worse than Class IV: heavily polluted")],
}

CSS = """
<style>
.stApp { background: linear-gradient(180deg,#e6f1ec 0px,#f1f5f7 340px,#f1f5f7 100%); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1,h2,h3 { letter-spacing:-0.01em; }
.crumb { font-size:.78rem; color:#64748b; margin-bottom:.35rem; }
.h1 { font-size:2.1rem; font-weight:800; color:#0f172a; line-height:1.15; margin:0 0 .35rem 0; }
.sub { color:#475569; font-size:.95rem; margin-bottom:.9rem; }
.card { background:#fff; border-radius:16px; padding:18px 20px; box-shadow:0 1px 2px rgba(15,23,42,.05),0 4px 14px rgba(15,23,42,.04); margin-bottom:14px; }
.card h4 { margin:0; font-size:1.02rem; font-weight:700; color:#0f172a; }
.card .cap { color:#64748b; font-size:.8rem; margin:2px 0 12px 0; }
.hero { border-radius:18px; padding:16px 18px; border:1px solid; }
.hero .top { display:flex; align-items:center; gap:14px; }
.hero .big { font-size:1.7rem; font-weight:800; min-width:64px; text-align:center; padding:8px 10px; border-radius:12px; line-height:1; }
.hero .big small { display:block; font-size:.6rem; font-weight:600; opacity:.85; margin-top:3px; letter-spacing:.04em; }
.hero .st { font-size:1.2rem; font-weight:700; color:#0f172a; }
.hero .cl { font-size:.8rem; color:#475569; }
.hero .row { display:flex; justify-content:space-between; margin-top:12px; padding-top:10px; border-top:1px solid rgba(15,23,42,.1); font-size:.86rem; color:#334155; }
.hero .row b { color:#0f172a; }
.hero .mini { display:flex; gap:14px; margin-top:10px; background:rgba(255,255,255,.7); border-radius:10px; padding:8px 12px; font-size:.8rem; color:#475569; justify-content:space-between; }
.badge { display:inline-block; min-width:42px; text-align:center; padding:4px 9px; border-radius:8px; font-weight:700; font-size:.92rem; }
.fresh { display:flex; gap:10px; align-items:flex-start; border-radius:12px; padding:11px 14px; font-size:.86rem; margin-bottom:14px; border:1px solid; }
.fresh.warn { background:#fff7e6; border-color:#f5d28a; color:#7a4b00; }
.fresh.ok { background:#eaf7ef; border-color:#b9e3c6; color:#14532d; }
.strip { display:flex; gap:8px; overflow-x:auto; padding-bottom:6px; }
.sc { flex:0 0 96px; text-align:center; border-radius:12px; padding:10px 6px; background:#f8fafc; border:1px solid #eef2f6; }
.sc.fc { background:#fff; border:1px solid #dbe3ec; box-shadow:0 1px 3px rgba(15,23,42,.06); }
.sc.past { opacity:.55; }
.sc .d { font-size:.74rem; color:#475569; font-weight:600; margin-bottom:8px; min-height:2.1em; }
.sc .s { font-size:.7rem; color:#475569; margin-top:7px; line-height:1.15; min-height:2.3em; }
.sc .r { font-size:.7rem; color:#94a3b8; margin-top:3px; }
.sc .tag { font-size:.62rem; text-transform:uppercase; letter-spacing:.06em; color:#7c3aed; font-weight:700; margin-top:4px; }
.lrow { display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid #f1f5f9; }
.lrow:last-child { border-bottom:0; }
.lrow .n { flex:1; font-weight:600; color:#0f172a; font-size:.92rem; }
.lrow .n small { display:block; font-weight:400; color:#64748b; font-size:.74rem; }
.lrow .v { font-weight:700; color:#0f172a; font-size:.95rem; text-align:right; }
.lrow .v small { display:block; font-weight:400; color:#94a3b8; font-size:.72rem; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.bar { height:8px; border-radius:6px; background:#eef2f6; position:relative; flex:1; min-width:70px; }
.bar i { position:absolute; top:0; height:8px; border-radius:6px; opacity:.85; }
.alert { display:flex; gap:10px; border-radius:12px; padding:12px 14px; font-size:.88rem; align-items:flex-start; }
.alert.bad { background:#fdecec; color:#7f1d1d; border:1px solid #f6c4c4; }
.alert.good { background:#eaf7ef; color:#14532d; border:1px solid #b9e3c6; }
.use { display:flex; gap:12px; align-items:center; padding:10px 0; border-bottom:1px solid #f1f5f9; }
.use:last-child { border-bottom:0; }
.use .ic { font-size:1.4rem; width:44px; height:44px; border-radius:12px; background:#eaf5ee; display:flex; align-items:center; justify-content:center; }
.use b { color:#0f172a; font-size:.92rem; } .use span { color:#475569; font-size:.84rem; display:block; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px !important; background:#fff; border-color:#e8edf2 !important; box-shadow:0 1px 2px rgba(15,23,42,.05),0 4px 14px rgba(15,23,42,.04); }
button[data-baseweb="tab"] { font-weight:600; }
section[data-testid="stSidebar"] { background:#fbfcfd; }
</style>
"""


def badge(wqi, status=None, size=None):
    status = status or classify_wqi(wqi)
    bg, fg, _, _ = BADGE[status]
    style = f"background:{bg};color:{fg};" + (f"font-size:{size};" if size else "")
    return f'<span class="badge" style="{style}">{wqi:.0f}</span>'


def hero(scope_title, crumb, model, wqi, forecast_date, last_wqi, last_date, rmse, main):
    status = classify_wqi(wqi)
    bg, fg, tint, border = BADGE[status]
    delta = wqi - last_wqi
    arrow = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "▬")
    left = (f'<div class="crumb">{crumb}</div><div class="h1">Water quality in {scope_title}</div>'
            f'<div class="sub">Water Quality Index (WQI) forecast · model <b>{model}</b> · forecast for {forecast_date:%d %b %Y}</div>')
    concern = (f'<div class="row"><span>Main concern</span><b>{main["label"]} · {main["value"]:.2f} {main["unit"]}</b></div>' if main else "")
    right = (f'<div class="hero" style="background:{tint};border-color:{border};">'
             f'<div class="top"><div class="big" style="background:{bg};color:{fg};">{wqi:.0f}<small>WQI</small></div>'
             f'<div><div class="st">{SHORT[status]}</div><div class="cl">DOE {get_doe_class(wqi)} · forecast {forecast_date:%d %b %Y}</div></div></div>'
             f'{concern}'
             f'<div class="mini"><span>Last sample <b>{last_wqi:.0f}</b> ({last_date:%d %b %Y})</span><span>{arrow} {delta:+.1f}</span><span>±{rmse:.1f} RMSE</span></div></div>')
    return left, right


def freshness(fr):
    if fr["data_age_days"] is None:
        return ""
    age = fr["data_age_days"]
    if age > 120:
        return (f'<div class="fresh warn"><div>⚠️</div><div><b>These forecasts are not for today.</b> The newest DOE sample in the data is '
                f'{fr["last_obs"]:%d %b %Y} ({age} days ago), and the models only forecast the next sampling date after it '
                f'(≈ {fr["forecast_date"]:%d %b %Y}). Add newer DOE data and re-run the pipeline to see current conditions.</div></div>')
    return f'<div class="fresh ok"><div>✅</div><div>Data is current: newest DOE sample {fr["last_obs"]:%d %b %Y} ({age} days ago).</div></div>'


def weekly_strip(agg, n_past=2):
    fc_idx = agg.index[agg["is_forecast"]]
    if len(fc_idx) == 0:
        return ""
    first = fc_idx.min()
    part = agg.iloc[max(0, first - n_past):]
    cards = []
    for _, r in part.iterrows():
        cls = "sc fc" if r["is_forecast"] else "sc past"
        lab = r["label"].replace("Wk ", "")
        rng = f'{r["lower"]:.0f}–{r["upper"]:.0f}' if r["is_forecast"] and r["lower"] == r["lower"] else "&nbsp;"
        tag = '<div class="tag">forecast</div>' if r["is_forecast"] else '<div class="tag" style="color:#94a3b8">past</div>'
        cards.append(f'<div class="{cls}"><div class="d">Week of<br>{lab}</div>{badge(r["wqi"], r["status"])}'
                     f'<div class="s">{SHORT[r["status"]]}</div><div class="r">{rng}</div>{tag}</div>')
    return '<div class="strip">' + "".join(cards) + "</div>"


def monthly_rows(agg, n_past=1):
    fc_idx = agg.index[agg["is_forecast"]]
    if len(fc_idx) == 0:
        return ""
    first = fc_idx.min()
    part = agg.iloc[max(0, first - n_past):]
    rows = []
    for _, r in part.iterrows():
        bg = BADGE[r["status"]][0]
        lo, hi = (r["lower"], r["upper"]) if r["is_forecast"] and r["lower"] == r["lower"] else (r["wqi"], r["wqi"])
        left, width = max(0, lo), max(2, hi - lo)
        note = ("forecast" if r["is_forecast"] else "past") + (" · partial" if r["partial"] else "")
        rows.append(f'<div class="lrow"><div class="n">{r["label"]}<small>{note}</small></div>'
                    f'<div class="bar"><i style="left:{left:.0f}%;width:{width:.0f}%;background:{bg}"></i></div>'
                    f'{badge(r["wqi"], r["status"])}<div class="v" style="min-width:92px;font-weight:600;font-size:.8rem;color:#475569">{SHORT[r["status"]]}</div></div>')
    return "".join(rows)


def parameter_rows(info):
    out = []
    for p in info["params"]:
        si = p["sub_index"]
        color = "#2563EB" if si >= 81 else ("#EAB308" if si >= 60 else "#EF4444")
        lim = ""
        if p["key"] in CLASS3_LIMITS:
            lim = f'Class III limit ≤ {CLASS3_LIMITS[p["key"]]:g}'
        elif p["key"] == "DO":
            lim = "Class III: 3–5"
        elif p["key"] == "pH":
            lim = "Class III: 5–9"
        unit = f' {p["unit"]}' if p["unit"] else ""
        out.append(f'<div class="lrow"><span class="dot" style="background:{color}"></span>'
                   f'<div class="n">{p["label"]}<small>{p["sub"]}</small></div>'
                   f'<div class="v">{p["value"]:.2f}{unit}<small>{lim}</small></div></div>')
    return "".join(out)


def parameter_alert(info):
    worst, ratio = None, 0.0
    for p in info["params"]:
        lim = CLASS3_LIMITS.get(p["key"])
        if lim and p["value"] / lim > ratio:
            worst, ratio = p, p["value"] / lim
    if worst and ratio > 1:
        return (f'<div class="alert bad"><div>⚠️</div><div><b>{worst["label"]}</b> averages <b>{ratio:.1f}×</b> the NWQS Class III limit '
                f'({worst["value"]:.2f} vs {CLASS3_LIMITS[worst["key"]]:g} {worst["unit"]}).</div></div>')
    return '<div class="alert good"><div>✅</div><div>BOD, COD, NH₃-N and SS are all within NWQS Class III limits.</div></div>'


def uses_rows(wqi):
    cls = get_doe_class(wqi)
    items = NWQS_USES.get(cls, NWQS_USES["Class V"])
    return "".join(f'<div class="use"><div class="ic">{ic}</div><div><b>{t}</b><span>{d}</span></div></div>' for ic, t, d in items), cls
