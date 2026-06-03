import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Configuration ─────────────────────────────────
st.set_page_config(
    page_title="VitaCheck | Health Risk Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ─────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #0f1117;
    color: #e8ecf4;
}
[data-testid="stHeader"] { background-color: #0f1117; }

[data-testid="stSidebar"] {
    background-color: #1a1f2e;
    border-right: 1px solid #2d3347;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35);
}
[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
}

[data-testid="stAppViewContainer"] h1 { color: #f0f4ff !important; }
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 { color: #dde3f5 !important; }
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div { color: #c9d1e0 !important; }

.vc-metric {
    background: #1a1f2e;
    border: 1px solid #2d3347;
    border-top: 3px solid #2563eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.vc-metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #6b7ca4 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.vc-metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e8ecf4 !important;
    line-height: 1.2;
}

.vc-section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8ecf4 !important;
    margin-bottom: 0.2rem;
    margin-top: 1.2rem;
    border-left: 3px solid #2563eb;
    padding-left: 0.7rem;
}
.vc-section-caption {
    font-size: 0.81rem;
    color: #6b7ca4 !important;
    margin-bottom: 1rem;
    padding-left: 1rem;
}

.vc-result-risk {
    background: #1f1215;
    border: 1.5px solid #f5222d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.vc-result-healthy {
    background: #101f13;
    border: 1.5px solid #52c41a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.vc-result-label-risk {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ff4d4f !important;
    letter-spacing: 0.04em;
}
.vc-result-label-healthy {
    font-size: 1.3rem;
    font-weight: 700;
    color: #73d13d !important;
    letter-spacing: 0.04em;
}
.vc-result-sub {
    font-size: 0.85rem;
    color: #8892aa !important;
    margin-top: 0.5rem;
}
.vc-prob {
    font-size: 0.9rem;
    color: #c9d1e0 !important;
    margin: 0.4rem 0 0;
}

.vc-risk-item {
    background: #1f1215;
    border-left: 3px solid #f5222d;
    border-radius: 5px;
    padding: 0.42rem 0.9rem;
    margin-bottom: 0.35rem;
    font-size: 0.86rem;
    color: #e8ecf4 !important;
}
.vc-healthy-item {
    background: #101f13;
    border-left: 3px solid #52c41a;
    border-radius: 5px;
    padding: 0.38rem 0.9rem;
    margin-bottom: 0.3rem;
    font-size: 0.84rem;
    color: #c9d1e0 !important;
}

.vc-table-header {
    display: flex;
    padding: 0.38rem 0.9rem;
    gap: 0.5rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7ca4 !important;
    border-bottom: 1px solid #2d3347;
    margin-bottom: 0.35rem;
}
.vc-row-risk {
    display: flex;
    background: #1f1215;
    border-left: 3px solid #f5222d;
    border-radius: 5px;
    padding: 0.42rem 0.9rem;
    margin-bottom: 0.28rem;
    gap: 0.5rem;
    font-size: 0.84rem;
    color: #e8ecf4 !important;
}
.vc-row-normal {
    display: flex;
    background: #101f13;
    border-left: 3px solid #52c41a;
    border-radius: 5px;
    padding: 0.42rem 0.9rem;
    margin-bottom: 0.28rem;
    gap: 0.5rem;
    font-size: 0.84rem;
    color: #c9d1e0 !important;
}
.vc-col-feat   { flex: 2.2; font-weight: 600; color: inherit !important; }
.vc-col-val    { flex: 1;   text-align: center; color: inherit !important; }
.vc-col-range  { flex: 2;   text-align: center; color: #8892aa !important; }
.vc-col-status { flex: 1.2; text-align: center; font-weight: 700; }
.vc-col-imp    { flex: 1;   text-align: right;  color: #8892aa !important; }
.vc-status-risk    { color: #ff4d4f !important; }
.vc-status-healthy { color: #73d13d !important; }

[data-testid="stAlert"] {
    background-color: #1a2035 !important;
    border: 1px solid #2d4080 !important;
    color: #a0b0d0 !important;
    border-radius: 8px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #73d13d, #faad14, #ff4d4f);
}
[data-testid="stExpander"] {
    background: #1a1f2e !important;
    border: 1px solid #2d3347 !important;
    border-radius: 8px !important;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load("model/vitacheck_model.pkl")
    scaler     = joblib.load("model/scaler.pkl")
    importance = pd.read_csv("model/feature_importance.csv")
    with open("model/threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]
    return model, scaler, importance, threshold

model, scaler, importance_df, THRESHOLD = load_artifacts()

# ── Chart Style ────────────────────────────────────────
CHART_BG   = "#0f1117"
CHART_FG   = "#e8ecf4"
CHART_GRID = "#2d3347"

def style_ax(ax, fig):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.xaxis.label.set_color(CHART_FG)
    ax.title.set_color(CHART_FG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHART_GRID)
    ax.spines["bottom"].set_color(CHART_GRID)
    ax.xaxis.grid(True, color=CHART_GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

# ── Cached Global Importance Chart ────────────────────
# KEY FIX: @st.cache_data prevents this chart from
# redrawing every time a sidebar slider moves.
# Without this, every slider interaction triggers a full
# matplotlib redraw → page flickers and shakes.
# With cache: chart is built ONCE, stored in memory,
# returned instantly on subsequent reruns.
@st.cache_data
def build_global_chart(_importance_df):
    n      = len(_importance_df)
    colors = [
        "#c0392b" if i >= n - 3
        else "#2563eb" if i >= n - 8
        else "#3d4a63"
        for i in range(n)
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    style_ax(ax, fig)
    ax.barh(
        _importance_df["Feature"][::-1],
        _importance_df["Importance"][::-1],
        color=colors,
        edgecolor=CHART_BG,
        height=0.62
    )
    for i, imp in enumerate(_importance_df["Importance"][::-1]):
        ax.text(imp + 0.001, i, f"{imp:.3f}",
                va="center", fontsize=8, color="#8892aa")
    critical  = mpatches.Patch(color="#c0392b", label="Critical  (Top 3)")
    important = mpatches.Patch(color="#2563eb", label="Important (Top 4-8)")
    minor     = mpatches.Patch(color="#3d4a63", label="Minor")
    ax.legend(handles=[critical, important, minor],
              loc="lower right", fontsize=9,
              facecolor="#1a1f2e", edgecolor="#2d3347",
              labelcolor=CHART_FG)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(
        "Feature Importance — Random Forest (200 estimators)",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    return fig

# ── Healthy Ranges ─────────────────────────────────────
HEALTHY_RANGES = {
    "Age":                   (0,    100),
    "BMI":                   (18.5, 24.9),
    "Blood_Pressure":        (90,   120),
    "Cholesterol":           (0,    200),
    "Glucose_Level":         (70,   100),
    "Heart_Rate":            (60,   100),
    "Sleep_Hours":           (7,    9),
    "Exercise_Hours":        (0.5,  2),
    "Water_Intake":          (2,    4),
    "Stress_Level":          (1,    4),
    "Smoking":               (0,    0),
    "Alcohol":               (0,    0),
    "Diet":                  (0,    2),
    "MentalHealth":          (0,    1),
    "PhysicalActivity":      (1,    3),
    "MedicalHistory":        (0,    0),
    "Allergies":             (0,    1),
    "Diet_Type__Vegan":      (0,    1),
    "Diet_Type__Vegetarian": (0,    1),
    "Blood_Group_AB":        (0,    1),
    "Blood_Group_B":         (0,    1),
    "Blood_Group_O":         (0,    1),
}

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
st.sidebar.title("VitaCheck")
st.sidebar.caption("AI-Powered Health Risk Assessment")
st.sidebar.markdown("---")

st.sidebar.subheader("Physical Measurements")
age            = st.sidebar.slider("Age (years)",           1,    100,  30)
bmi            = st.sidebar.slider("BMI",                   10.0, 50.0, 22.0, step=0.1)
blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 60,   200,  110)
heart_rate     = st.sidebar.slider("Heart Rate (bpm)",      40,   180,  72)

st.sidebar.subheader("Lab Results")
cholesterol   = st.sidebar.slider("Cholesterol (mg/dL)",   100,  400,  180)
glucose_level = st.sidebar.slider("Glucose Level (mg/dL)", 50,   300,  95)

st.sidebar.subheader("Lifestyle")
sleep_hours    = st.sidebar.slider("Sleep Hours / day",     0.0,  12.0, 7.0,  step=0.5)
exercise_hours = st.sidebar.slider("Exercise Hours / day",  0.0,  5.0,  1.0,  step=0.1)
water_intake   = st.sidebar.slider("Water Intake (litres)", 0.0,  8.0,  2.5,  step=0.1)
stress_level   = st.sidebar.slider("Stress Level (1-10)",   1,    10,   4)

st.sidebar.subheader("Habits")
smoking = st.sidebar.selectbox(
    "Smoking Status",
    options=[0, 1, 2],
    format_func=lambda x: ["Non-Smoker", "Smoker", "Ex-Smoker"][x]
)
alcohol = st.sidebar.selectbox(
    "Alcohol Consumption",
    options=[0, 1, 2],
    format_func=lambda x: ["None", "Occasional", "Regular"][x]
)

st.sidebar.subheader("Health History")
diet = st.sidebar.selectbox(
    "Diet Type",
    options=[0, 1, 2],
    format_func=lambda x: ["Standard", "Vegetarian", "Vegan"][x]
)
mental_health = st.sidebar.selectbox(
    "Mental Health Status",
    options=[0, 1, 2],
    format_func=lambda x: ["Good", "Moderate", "Poor"][x]
)
physical_act = st.sidebar.selectbox(
    "Physical Activity Level",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["Sedentary", "Light", "Moderate", "Active"][x]
)
med_history = st.sidebar.selectbox(
    "Medical History",
    options=[0, 1, 2],
    format_func=lambda x: ["None", "Minor", "Major"][x]
)
allergies = st.sidebar.selectbox(
    "Allergies",
    options=[0, 1],
    format_func=lambda x: ["No", "Yes"][x]
)

st.sidebar.subheader("Diet and Blood Type")
diet_vegan      = 1 if diet == 2 else 0
diet_vegetarian = 1 if diet == 1 else 0
blood_group = st.sidebar.selectbox(
    "Blood Group", options=["A", "AB", "B", "O"]
)
blood_ab = 1 if blood_group == "AB" else 0
blood_b  = 1 if blood_group == "B"  else 0
blood_o  = 1 if blood_group == "O"  else 0

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(
    "Run Health Risk Assessment",
    use_container_width=True,
    type="primary"
)

# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown(
    "<h1 style='margin-bottom:0.15rem;color:#f0f4ff;'>"
    "VitaCheck — AI Health Risk Classifier</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#6b7ca4;font-size:0.92rem;margin-bottom:1.4rem;'>"
    "Powered by <b style='color:#93c5fd;'>Random Forest</b> trained on 9,800+ patient records "
    "from a biomedical observational study. Classifies individuals as "
    "<b style='color:#73d13d;'>Healthy</b> or "
    "<b style='color:#ff4d4f;'>At Risk</b> "
    "based on 22 clinical and lifestyle features.</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<hr style='border-color:#2d3347;margin-bottom:1.2rem;'>",
    unsafe_allow_html=True
)

# ── Metric Cards ───────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Training Samples",   "9,549"),
    (c2, "Algorithm",          "Random Forest"),
    (c3, "Recall",             "99.30%"),
    (c4, "Decision Threshold", f"{THRESHOLD:.2f}"),
]:
    col.markdown(
        f'<div class="vc-metric">'
        f'<div class="vc-metric-label">{label}</div>'
        f'<div class="vc-metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════
if not predict_btn:
    st.info(
        "Complete your health profile in the sidebar and click "
        "Run Health Risk Assessment to receive your personalised report."
    )

    st.markdown(
        '<div class="vc-section-title">Global Feature Importance</div>'
        '<div class="vc-section-caption">Features ranked by contribution to the model\'s '
        'predictions across all 9,549 training records. '
        'Red = critical (top 3) &nbsp;|&nbsp; Blue = important (top 4-8) &nbsp;|&nbsp; '
        'Grey = minor.</div>',
        unsafe_allow_html=True
    )

    # Uses cached chart — no redraw on slider interactions
    fig = build_global_chart(importance_df)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════
# PREDICTION PAGE
# ══════════════════════════════════════════════════════
else:
    input_data = {
        "Age":                   age,
        "BMI":                   bmi,
        "Blood_Pressure":        blood_pressure,
        "Cholesterol":           cholesterol,
        "Glucose_Level":         glucose_level,
        "Heart_Rate":            heart_rate,
        "Sleep_Hours":           sleep_hours,
        "Exercise_Hours":        exercise_hours,
        "Water_Intake":          water_intake,
        "Stress_Level":          stress_level,
        "Target":                0,
        "Smoking":               smoking,
        "Alcohol":               alcohol,
        "Diet":                  diet,
        "MentalHealth":          mental_health,
        "PhysicalActivity":      physical_act,
        "MedicalHistory":        med_history,
        "Allergies":             allergies,
        "Diet_Type__Vegan":      diet_vegan,
        "Diet_Type__Vegetarian": diet_vegetarian,
        "Blood_Group_AB":        blood_ab,
        "Blood_Group_B":         blood_b,
        "Blood_Group_O":         blood_o,
    }

    input_df   = pd.DataFrame([input_data]).drop("Target", axis=1)
    proba      = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= THRESHOLD)

    user_values = {k: v for k, v in input_data.items() if k != "Target"}

    risk_flags    = []
    healthy_flags = []
    for feature, value in user_values.items():
        low, high = HEALTHY_RANGES[feature]
        if value < low or value > high:
            risk_flags.append((feature, value, low, high))
        else:
            healthy_flags.append((feature, value))
    risky_features = [f[0] for f in risk_flags]

    # ── Result + Personal Risk ─────────────────────────
    st.markdown(
        '<div class="vc-section-title" style="margin-top:0;">'
        'Health Risk Assessment Result</div>',
        unsafe_allow_html=True
    )

    col_result, col_risk = st.columns([1, 1.6])

    with col_result:
        if prediction == 1:
            st.markdown(
                f'<div class="vc-result-risk">'
                f'<div class="vc-result-label-risk">RESULT: AT RISK</div>'
                f'<div class="vc-prob"><b>Risk Probability:</b> {proba*100:.1f}%</div>'
                f'<div class="vc-result-sub">The model indicates elevated health risk. '
                f'Please consult a qualified healthcare professional.</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="vc-result-healthy">'
                f'<div class="vc-result-label-healthy">RESULT: HEALTHY</div>'
                f'<div class="vc-prob"><b>Healthy Probability:</b> {(1-proba)*100:.1f}%</div>'
                f'<div class="vc-result-sub">The model indicates a generally healthy profile. '
                f'Continue your current lifestyle habits.</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.82rem;font-weight:600;color:#8892aa;"
            "margin-bottom:0.3rem;'>Risk Probability Score</p>",
            unsafe_allow_html=True
        )
        st.progress(float(proba))
        st.markdown(
            f"<p style='font-size:0.78rem;color:#6b7ca4;margin-top:0.3rem;'>"
            f"At-risk probability: {proba:.3f} &nbsp;|&nbsp; "
            f"Decision threshold: {THRESHOLD:.2f}</p>",
            unsafe_allow_html=True
        )

    with col_risk:
        st.markdown(
            '<div class="vc-section-title" style="margin-top:0;">'
            'Personal Risk Factors</div>',
            unsafe_allow_html=True
        )
        if risk_flags:
            st.markdown(
                f"<p style='font-size:0.84rem;color:#ff7875;font-weight:600;"
                f"margin-bottom:0.6rem;'>"
                f"{len(risk_flags)} value(s) outside clinically healthy range</p>",
                unsafe_allow_html=True
            )
            for feat, val, low, high in risk_flags:
                st.markdown(
                    f'<div class="vc-risk-item">'
                    f'<b>{feat}</b> &nbsp;—&nbsp; '
                    f'Your value: <b>{val}</b>'
                    f'<span style="color:#8892aa;"> &nbsp;|&nbsp; '
                    f'Healthy range: {low} – {high}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="background:#101f13;border:1px solid #52c41a;'
                'border-radius:8px;padding:0.8rem 1rem;font-size:0.9rem;'
                'color:#73d13d;">All 22 values are within clinically healthy ranges.</div>',
                unsafe_allow_html=True
            )

        if healthy_flags:
            with st.expander(f"View {len(healthy_flags)} healthy values"):
                for feat, val in healthy_flags:
                    low, high = HEALTHY_RANGES[feat]
                    st.markdown(
                        f'<div class="vc-healthy-item">'
                        f'<b>{feat}</b> — {val}'
                        f'<span style="color:#6b7ca4;"> '
                        f'(range: {low} – {high})</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── Personalised Chart ─────────────────────────────
    st.markdown(
        "<hr style='border-color:#2d3347;margin:1.5rem 0;'>",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="vc-section-title">Your Risk Profile vs Global Feature Importance</div>'
        '<div class="vc-section-caption">'
        'Red bars = features where YOUR values are outside the healthy range. '
        'Green bars = values within healthy range. '
        'Bar length = global importance score.</div>',
        unsafe_allow_html=True
    )

    bar_colors = [
        "#c0392b" if feat in risky_features else "#27ae60"
        for feat in importance_df["Feature"][::-1]
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    style_ax(ax, fig)
    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color=bar_colors,
        edgecolor=CHART_BG,
        height=0.62
    )
    for i, imp in enumerate(importance_df["Importance"][::-1]):
        ax.text(imp + 0.001, i, f"{imp:.3f}",
                va="center", fontsize=8, color="#8892aa")

    risky_patch   = mpatches.Patch(color="#c0392b",
                                   label="Outside healthy range (your values)")
    healthy_patch = mpatches.Patch(color="#27ae60",
                                   label="Within healthy range (your values)")
    ax.legend(handles=[risky_patch, healthy_patch],
              loc="lower right", fontsize=9,
              facecolor="#1a1f2e", edgecolor="#2d3347",
              labelcolor=CHART_FG)
    ax.set_xlabel("Global Importance Score", fontsize=11)
    ax.set_title(
        "Personalised Risk Profile — Feature Importance vs Your Health Values",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Summary Table ──────────────────────────────────
    st.markdown(
        "<hr style='border-color:#2d3347;margin:1.5rem 0;'>",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="vc-section-title">Complete Health Summary</div>'
        '<div class="vc-section-caption">'
        'All 22 features — your values, healthy reference ranges, '
        'status, and model importance score.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="vc-table-header">'
        '<span class="vc-col-feat">Feature</span>'
        '<span class="vc-col-val">Your Value</span>'
        '<span class="vc-col-range">Healthy Range</span>'
        '<span class="vc-col-status">Status</span>'
        '<span class="vc-col-imp">Importance</span>'
        '</div>',
        unsafe_allow_html=True
    )

    for feature, value in user_values.items():
        low, high  = HEALTHY_RANGES[feature]
        concerning = (value < low or value > high)
        row_class  = "vc-row-risk"    if concerning else "vc-row-normal"
        status_cls = "vc-status-risk" if concerning else "vc-status-healthy"
        status_txt = "Concerning"     if concerning else "Normal"

        imp_arr = importance_df[
            importance_df["Feature"] == feature
        ]["Importance"].values
        imp_val = f"{imp_arr[0]:.3f}" if len(imp_arr) > 0 else "—"

        st.markdown(
            f'<div class="{row_class}">'
            f'<span class="vc-col-feat">{feature}</span>'
            f'<span class="vc-col-val">{value}</span>'
            f'<span class="vc-col-range">{low} – {high}</span>'
            f'<span class="vc-col-status {status_cls}">{status_txt}</span>'
            f'<span class="vc-col-imp">{imp_val}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Footer ─────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#2d3347;margin:2rem 0 0.5rem;'>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:0.75rem;color:#3d4a63;text-align:center;"
        "letter-spacing:0.03em;'>"
        "VitaCheck &nbsp;·&nbsp; Random Forest + Streamlit &nbsp;·&nbsp; "
        "For research purposes only. Not a substitute for clinical diagnosis."
        "</p>",
        unsafe_allow_html=True
    )