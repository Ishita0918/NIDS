import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings
warnings.filterwarnings("ignore")
from database import init_db, login_user, register_user, save_scan, get_user_scans, get_user_stats, get_all_users, get_all_scans, get_global_stats

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NIDS Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
init_db()

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
for key, val in {"logged_in": False, "user": None, "page": "login", "auth_tab": "login"}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════
# MEGA CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Rajdhani:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #020810 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
header[data-testid="stHeader"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── 3D Perspective Grid ── */
.bg-grid {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    overflow: hidden;
}
.bg-grid::before {
    content: '';
    position: absolute;
    width: 200%; height: 200%;
    top: -50%; left: -50%;
    background-image:
        linear-gradient(rgba(0,180,216,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,216,0.08) 1px, transparent 1px);
    background-size: 60px 60px;
    transform: perspective(500px) rotateX(60deg);
    transform-origin: center top;
    animation: gridPerspective 8s linear infinite;
}
@keyframes gridPerspective {
    0%   { background-position: 0 0, 0 0; }
    100% { background-position: 0 60px, 0 60px; }
}
.bg-grid::after {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 50% 0%, rgba(0,119,182,0.18) 0%, transparent 60%),
        radial-gradient(ellipse at 0% 50%, rgba(0,180,216,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 50%, rgba(0,180,216,0.08) 0%, transparent 50%);
}

/* ── Floating orbs ── */
.orb {
    position: fixed; border-radius: 50%;
    filter: blur(80px); pointer-events: none; z-index: 0;
    animation: orbFloat 10s ease-in-out infinite;
}
.orb-1 {
    width: 500px; height: 500px;
    background: rgba(0,119,182,0.13);
    top: -150px; left: -150px;
    animation-duration: 12s;
}
.orb-2 {
    width: 350px; height: 350px;
    background: rgba(0,180,216,0.09);
    bottom: -80px; right: -80px;
    animation-duration: 15s; animation-delay: -5s;
}
.orb-3 {
    width: 250px; height: 250px;
    background: rgba(255,183,3,0.07);
    top: 40%; left: 45%;
    animation-duration: 18s; animation-delay: -8s;
}
@keyframes orbFloat {
    0%,100% { transform: translate(0,0) scale(1); }
    33%     { transform: translate(30px,-30px) scale(1.05); }
    66%     { transform: translate(-20px,20px) scale(0.95); }
}

/* ── Main wrapper ── */
.main-wrap { position: relative; z-index: 2; padding: 0; }

/* ── Navbar ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 32px;
    background: rgba(2,8,16,0.92);
    border-bottom: 1px solid rgba(0,180,216,0.25);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
}
.nav-logo {
    font-family: 'Orbitron', monospace;
    color: #00B4D8; font-size: 20px; font-weight: 900;
    letter-spacing: 4px;
    text-shadow: 0 0 20px rgba(0,180,216,0.5);
}
.nav-logo span { color: #FFB703; }
.nav-right { display: flex; align-items: center; gap: 12px; }
.nav-badge {
    padding: 4px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
}
.nav-badge.user  { background: rgba(0,180,216,0.12); color: #00B4D8; border: 1px solid rgba(0,180,216,0.35); }
.nav-badge.admin { background: rgba(255,183,3,0.12);  color: #FFB703; border: 1px solid rgba(255,183,3,0.35); }
.nav-badge.live  {
    background: rgba(0,255,136,0.08); color: #00FF88;
    border: 1px solid rgba(0,255,136,0.3);
    animation: livePulse 2s infinite;
}
@keyframes livePulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

/* ── Hero ── */
.hero-section {
    text-align: center; padding: 60px 20px 40px;
    animation: fadeInDown 0.8s ease;
}
@keyframes fadeInDown {
    from { opacity:0; transform:translateY(-30px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 4rem; font-weight: 900; color: #00B4D8;
    text-shadow: 0 0 60px rgba(0,180,216,0.6), 0 0 120px rgba(0,180,216,0.2);
    line-height: 1.1; margin-bottom: 10px;
    animation: titleGlow 3s ease-in-out infinite;
}
@keyframes titleGlow {
    0%,100% { text-shadow: 0 0 40px rgba(0,180,216,0.5), 0 0 80px rgba(0,180,216,0.2); }
    50%     { text-shadow: 0 0 80px rgba(0,180,216,0.8), 0 0 160px rgba(0,180,216,0.4); }
}
.hero-title span { color: #FFB703; }
.hero-sub {
    color: #8ECAE6; font-size: 1rem;
    letter-spacing: 4px; margin-bottom: 30px;
}
.hero-divider {
    width: 150px; height: 2px; margin: 0 auto 30px;
    background: linear-gradient(90deg, transparent, #00B4D8, #FFB703, transparent);
    animation: dividerGlow 2s ease-in-out infinite;
}
@keyframes dividerGlow {
    0%,100% { opacity:0.6; width:150px; }
    50%     { opacity:1;   width:200px; }
}

/* ── Auth card ── */
.auth-card {
    background: rgba(10,22,40,0.85);
    border: 1px solid rgba(0,180,216,0.25);
    border-radius: 16px; padding: 32px;
    backdrop-filter: blur(16px);
    animation: fadeInUp 0.6s ease;
    box-shadow:
        0 0 40px rgba(0,180,216,0.06),
        inset 0 1px 0 rgba(0,180,216,0.1);
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
.auth-title {
    font-family: 'Orbitron', monospace;
    color: #00B4D8; font-size: 1.2rem;
    letter-spacing: 2px; margin-bottom: 6px;
}
.auth-sub { color: #8ECAE6; font-size: 0.9rem; margin-bottom: 24px; }

/* ── Metric cards ── */
.metric-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 12px; margin-bottom: 20px;
}
.metric-card {
    background: rgba(10,22,40,0.85);
    border: 1px solid rgba(0,180,216,0.2);
    border-radius: 14px; padding: 20px 14px;
    text-align: center;
    animation: fadeInUp 0.5s ease both;
    transition: border-color 0.3s, transform 0.2s;
    position: relative; overflow: hidden;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
}
.metric-card::before {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.metric-card::after {
    content: ''; position: absolute;
    top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,180,216,0.04), transparent);
    transition: left 0.5s;
}
.metric-card:hover::after { left: 100%; }
.metric-card.teal::before  { background: linear-gradient(90deg, transparent, #00B4D8, transparent); }
.metric-card.green::before { background: linear-gradient(90deg, transparent, #00FF88, transparent); }
.metric-card.red::before   { background: linear-gradient(90deg, transparent, #FF4444, transparent); }
.metric-card.gold::before  { background: linear-gradient(90deg, transparent, #FFB703, transparent); }
.metric-card:hover { transform: translateY(-3px); border-color: rgba(0,180,216,0.4); }
.metric-val {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem; font-weight: 900; margin-bottom: 6px;
}
.metric-val.teal  { color: #00B4D8; text-shadow: 0 0 20px rgba(0,180,216,0.5); }
.metric-val.green { color: #00FF88; text-shadow: 0 0 20px rgba(0,255,136,0.5); }
.metric-val.red   { color: #FF4444; text-shadow: 0 0 20px rgba(255,68,68,0.5); }
.metric-val.gold  { color: #FFB703; text-shadow: 0 0 20px rgba(255,183,3,0.5); }
.metric-lbl { color: #8ECAE6; font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; }

/* ── Section card ── */
.section-card {
    background: rgba(10,22,40,0.8);
    border: 1px solid rgba(0,180,216,0.15);
    border-radius: 14px; padding: 22px 24px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
}
.section-title {
    font-family: 'Orbitron', monospace;
    color: #00B4D8; font-size: 0.82rem;
    letter-spacing: 2px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,180,216,0.4), transparent);
}

/* ── Result boxes ── */
.result-attack {
    background: rgba(30,0,0,0.9);
    border: 2px solid #FF4444; border-radius: 16px;
    padding: 30px; text-align: center;
    animation: attackPulse 1.5s infinite, fadeInUp 0.4s ease;
    box-shadow: 0 0 60px rgba(255,68,68,0.15), inset 0 0 60px rgba(255,68,68,0.05);
}
@keyframes attackPulse {
    0%,100% { box-shadow: 0 0 30px rgba(255,68,68,0.3), inset 0 0 40px rgba(255,68,68,0.05); }
    50%     { box-shadow: 0 0 70px rgba(255,68,68,0.7), inset 0 0 80px rgba(255,68,68,0.1); }
}
.result-normal {
    background: rgba(0,30,12,0.9);
    border: 2px solid #00FF88; border-radius: 16px;
    padding: 30px; text-align: center;
    animation: normalGlow 2s infinite, fadeInUp 0.4s ease;
    box-shadow: 0 0 40px rgba(0,255,136,0.1), inset 0 0 40px rgba(0,255,136,0.04);
}
@keyframes normalGlow {
    0%,100% { box-shadow: 0 0 20px rgba(0,255,136,0.2), inset 0 0 30px rgba(0,255,136,0.04); }
    50%     { box-shadow: 0 0 50px rgba(0,255,136,0.5), inset 0 0 60px rgba(0,255,136,0.08); }
}
.result-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem; font-weight: 900; margin-bottom: 8px;
}
.result-attack .result-title { color: #FF4444; text-shadow: 0 0 30px rgba(255,68,68,0.8); }
.result-normal .result-title { color: #00FF88; text-shadow: 0 0 30px rgba(0,255,136,0.8); }
.result-conf { font-size: 1.1rem; margin-bottom: 6px; }
.result-attack .result-conf { color: #FF8888; }
.result-normal .result-conf { color: #88FFBB; }

/* ── History rows ── */
.hist-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;
    background: rgba(2,8,16,0.7);
    border: 1px solid rgba(0,180,216,0.08);
    transition: all 0.3s;
    animation: fadeInUp 0.4s ease both;
}
.hist-row:hover {
    border-color: rgba(0,180,216,0.25);
    background: rgba(13,27,42,0.8);
    transform: translateX(4px);
}
.hist-info { color: #CAF0F8; font-size: 0.88rem; font-weight: 600; }
.hist-time { color: #4A6FA5; font-size: 0.76rem; margin-top: 2px; }
.risk-badge {
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.76rem; font-weight: 600;
}
.risk-high { background: rgba(45,0,0,0.8);  color: #FF4444; border: 1px solid rgba(255,68,68,0.35); }
.risk-med  { background: rgba(45,30,0,0.8); color: #FFB703; border: 1px solid rgba(255,183,3,0.35); }
.risk-low  { background: rgba(0,45,18,0.8); color: #00FF88; border: 1px solid rgba(0,255,136,0.35); }

/* ── Admin table ── */
.admin-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.admin-table th {
    color: #8ECAE6; padding: 10px 14px;
    border-bottom: 1px solid rgba(0,180,216,0.2);
    font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; font-size: 0.72rem; text-align: left;
}
.admin-table td { color: #CAF0F8; padding: 11px 14px; border-bottom: 1px solid rgba(255,255,255,0.03); transition: background 0.2s; }
.admin-table tr:hover td { background: rgba(0,180,216,0.04); }
.role-pill { padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.role-admin { background: rgba(255,183,3,0.12);  color: #FFB703; border: 1px solid rgba(255,183,3,0.3); }
.role-user  { background: rgba(0,180,216,0.12); color: #00B4D8; border: 1px solid rgba(0,180,216,0.3); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #005F8E, #00B4D8) !important;
    color: #020810 !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    padding: 10px 22px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(0,180,216,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,180,216,0.45) !important;
    background: linear-gradient(135deg, #0077B6, #00D4F8) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(10,22,40,0.9) !important;
    border: 1px solid rgba(0,180,216,0.25) !important;
    color: #CAF0F8 !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00B4D8 !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.15) !important;
}
.stSelectbox > div > div {
    background: rgba(10,22,40,0.9) !important;
    border: 1px solid rgba(0,180,216,0.25) !important;
    color: #CAF0F8 !important;
    border-radius: 8px !important;
}
label { color: #8ECAE6 !important; font-size: 0.85rem !important; letter-spacing: 0.5px !important; }

/* ── Streamlit tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(10,22,40,0.8) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid rgba(0,180,216,0.1) !important;
}
[data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 9.5px !important;
    letter-spacing: 1px !important;
    color: #8ECAE6 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
[aria-selected="true"] {
    background: rgba(0,180,216,0.18) !important;
    color: #00B4D8 !important;
    box-shadow: 0 0 12px rgba(0,180,216,0.2) !important;
}

/* ── Scanning text ── */
.scanning-text {
    font-family: 'Orbitron', monospace;
    color: #00B4D8; font-size: 0.88rem;
    letter-spacing: 2px; text-align: center; padding: 16px;
    animation: blink 0.8s infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.15;} }

/* ── Footer ── */
.footer {
    text-align: center; padding: 24px;
    color: rgba(0,180,216,0.25);
    font-size: 0.72rem; letter-spacing: 2px;
    border-top: 1px solid rgba(0,180,216,0.08);
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# BACKGROUND
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="bg-grid"></div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    model             = keras.models.load_model("hybrid_metaheuristic_model.keras")
    selected_features = np.load("selected_features.npy")
    return model, selected_features

model, selected_features = load_assets()

col_names = [
    "duration","protocol_type","service","flag",
    "src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count",
    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]
feature_cols = col_names[:41]

def preprocess_input(df):
    categorical_cols = ["protocol_type","service","flag"]
    encoder  = LabelEncoder()
    train_df = pd.read_csv("KDDTrain+.txt", header=None, names=col_names)
    test_df  = pd.read_csv("KDDTest+.txt",  header=None, names=col_names)
    df = df.copy()
    for col in categorical_cols:
        combined = pd.concat([train_df[col], test_df[col], df[col]])
        encoder.fit(combined)
        df[col] = encoder.transform(df[col])
    train_full = train_df[feature_cols].copy()
    for col in categorical_cols:
        combined = pd.concat([train_df[col], test_df[col]])
        encoder.fit(combined)
        train_full[col] = encoder.transform(train_df[col])
    scaler   = MinMaxScaler()
    scaler.fit(train_full.values)
    X_scaled = scaler.transform(df[feature_cols].values)
    return X_scaled[:, selected_features]

def predict(df):
    X     = preprocess_input(df)
    probs = model.predict(X, verbose=0).flatten()
    results = []
    for i, prob in enumerate(probs):
        is_attack  = prob > 0.4
        label      = "🔴 ATTACK" if is_attack else "🟢 NORMAL"
        confidence = prob * 100  if is_attack else (1 - prob) * 100
        results.append({
            "Connection": i + 1,
            "Result"    : label,
            "Confidence": f"{confidence:.1f}%",
            "Risk Score": round(float(prob), 4)
        })
    return pd.DataFrame(results), probs

def plotly_dark(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(2,8,16,0.8)",
        font=dict(color="#8ECAE6", family="Rajdhani"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ══════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════
def render_navbar():
    user = st.session_state.user
    if user:
        role_badge = f'<span class="nav-badge {"admin" if user["role"]=="admin" else "user"}">{"⚡ ADMIN" if user["role"]=="admin" else "👤 " + user["username"].upper()}</span>'
        live_badge = '<span class="nav-badge live">● LIVE</span>'
    else:
        role_badge = live_badge = ""
    st.markdown(f"""
    <div class="navbar">
        <div class="nav-logo">NIDS<span>.</span>SHIELD</div>
        <div class="nav-right">{live_badge}{role_badge}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: LOGIN / REGISTER
# ══════════════════════════════════════════════════════════
def page_login():
    render_navbar()
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">NIDS<span>.</span>SHIELD</div>
        <div class="hero-sub">NETWORK INTRUSION DETECTION SYSTEM</div>
        <div class="hero-divider"></div>
        <div style="color:#4A6FA5; font-size:0.85rem; letter-spacing:2px;">
            GA FEATURE SELECTION &nbsp;+&nbsp; PSO ARCHITECTURE &nbsp;+&nbsp; DEEP LEARNING
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        tab = st.session_state.auth_tab
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ LOGIN", use_container_width=True):
                st.session_state.auth_tab = "login"
                st.rerun()
        with c2:
            if st.button("✦ REGISTER", use_container_width=True):
                st.session_state.auth_tab = "register"
                st.rerun()

        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        if tab == "login":
            st.markdown('<div class="auth-title">WELCOME BACK</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-sub">Sign in to your security dashboard</div>', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username", key="li_user")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ SIGN IN", use_container_width=True):
                if username and password:
                    ok, result = login_user(username, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user      = result
                        st.session_state.page      = "admin" if result["role"] == "admin" else "dashboard"
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("Please fill in all fields!")
            st.markdown("""
            <div style='text-align:center; margin-top:16px; color:#4A6FA5; font-size:0.8rem;'>
                Default admin → <span style='color:#FFB703;'>admin / admin123</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown('<div class="auth-title">CREATE ACCOUNT</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-sub">Join the security network</div>', unsafe_allow_html=True)
            new_user  = st.text_input("Username", placeholder="Choose a username",   key="reg_user")
            new_email = st.text_input("Email",    placeholder="your@email.com",      key="reg_email")
            new_pass  = st.text_input("Password", type="password", placeholder="Create password", key="reg_pass")
            new_pass2 = st.text_input("Confirm",  type="password", placeholder="Repeat password", key="reg_pass2")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✦ CREATE ACCOUNT", use_container_width=True):
                if all([new_user, new_email, new_pass, new_pass2]):
                    if new_pass != new_pass2:
                        st.error("❌ Passwords don't match!")
                    elif len(new_pass) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    else:
                        ok, msg = register_user(new_user, new_email, new_pass)
                        if ok:
                            st.success(f"✅ {msg} Please login!")
                            st.session_state.auth_tab = "login"
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                else:
                    st.warning("Please fill in all fields!")

        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom stats
    st.markdown("""
    <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px; max-width:680px; margin:36px auto 0; padding:0 20px;'>
        <div style='text-align:center; padding:20px; background:rgba(10,22,40,0.7); border:1px solid rgba(0,180,216,0.2); border-radius:12px; backdrop-filter:blur(8px);'>
            <div style='font-family:Orbitron,monospace; color:#00B4D8; font-size:1.8rem; text-shadow:0 0 20px rgba(0,180,216,0.5);'>82%</div>
            <div style='color:#8ECAE6; font-size:0.72rem; margin-top:6px; letter-spacing:1px;'>MODEL ACCURACY</div>
        </div>
        <div style='text-align:center; padding:20px; background:rgba(10,22,40,0.7); border:1px solid rgba(0,180,216,0.2); border-radius:12px; backdrop-filter:blur(8px);'>
            <div style='font-family:Orbitron,monospace; color:#00B4D8; font-size:1.8rem; text-shadow:0 0 20px rgba(0,180,216,0.5);'>18</div>
            <div style='color:#8ECAE6; font-size:0.72rem; margin-top:6px; letter-spacing:1px;'>FEATURES SELECTED</div>
        </div>
        <div style='text-align:center; padding:20px; background:rgba(10,22,40,0.7); border:1px solid rgba(0,180,216,0.2); border-radius:12px; backdrop-filter:blur(8px);'>
            <div style='font-family:Orbitron,monospace; color:#00B4D8; font-size:1.8rem; text-shadow:0 0 20px rgba(0,180,216,0.5);'>96%</div>
            <div style='color:#8ECAE6; font-size:0.72rem; margin-top:6px; letter-spacing:1px;'>ATTACK PRECISION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: USER DASHBOARD
# ══════════════════════════════════════════════════════════
def page_dashboard():
    render_navbar()
    user  = st.session_state.user
    stats = get_user_stats(user["id"])
    scans = get_user_scans(user["id"])

    st.markdown(f"""
    <div style='padding:26px 32px 0;'>
        <div style='font-family:Orbitron,monospace; color:#CAF0F8; font-size:1.5rem; font-weight:600;'>
            Welcome back, <span style='color:#00B4D8; text-shadow:0 0 20px rgba(0,180,216,0.5);'>{user['username'].upper()}</span>
        </div>
        <div style='color:#4A6FA5; font-size:0.82rem; margin-top:4px; letter-spacing:1.5px;'>YOUR SECURITY OPERATIONS CENTER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:16px 32px;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card teal">
            <div class="metric-val teal">{stats['total_scans']}</div>
            <div class="metric-lbl">Total Scans</div>
        </div>
        <div class="metric-card green">
            <div class="metric-val green">{stats['total_normal']:,}</div>
            <div class="metric-lbl">Normal Traffic</div>
        </div>
        <div class="metric-card red">
            <div class="metric-val red">{stats['total_attacks']:,}</div>
            <div class="metric-lbl">Attacks Found</div>
        </div>
        <div class="metric-card gold">
            <div class="metric-val gold">{stats['avg_risk']}%</div>
            <div class="metric-lbl">Avg Risk Level</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📁 UPLOAD SCAN", "✍️ MANUAL INPUT", "🎮 SIMULATION", "📋 MY HISTORY"])

    # ── TAB 1: UPLOAD ──
    with t1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">UPLOAD NETWORK TRAFFIC FILE</div>', unsafe_allow_html=True)
        st.info("📌 Upload NSL-KDD format file (.txt or .csv) — 41 features, no header row")
        uploaded = st.file_uploader("", type=["txt","csv"], key="upload_file")
        if uploaded:
            df = pd.read_csv(uploaded, header=None, names=col_names)
            df = df.drop(["label","difficulty"], axis=1, errors="ignore")
            st.success(f"✅ {len(df):,} connections loaded")
            if st.button("⚡ ANALYZE TRAFFIC", key="btn_analyze"):
                ph = st.empty()
                for i in range(4):
                    ph.markdown(f'<div class="scanning-text">{"⚡" * (i+1)} SCANNING NETWORK TRAFFIC...</div>', unsafe_allow_html=True)
                    time.sleep(0.35)
                ph.empty()
                results_df, probs = predict(df)
                total   = len(results_df)
                attacks = len(results_df[results_df["Result"] == "🔴 ATTACK"])
                normals = total - attacks
                pct     = (attacks / total) * 100 if total > 0 else 0
                save_scan(user["id"], user["username"], "File Upload", total, attacks, normals, pct)
                st.markdown(f"""
                <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:14px 0;'>
                    <div class="metric-card teal"><div class="metric-val teal" style="font-size:1.6rem;">{total:,}</div><div class="metric-lbl">Total</div></div>
                    <div class="metric-card green"><div class="metric-val green" style="font-size:1.6rem;">{normals:,}</div><div class="metric-lbl">Normal</div></div>
                    <div class="metric-card red"><div class="metric-val red" style="font-size:1.6rem;">{attacks:,}</div><div class="metric-lbl">Attacks</div></div>
                    <div class="metric-card gold"><div class="metric-val gold" style="font-size:1.6rem;">{pct:.1f}%</div><div class="metric-lbl">Risk</div></div>
                </div>
                """, unsafe_allow_html=True)
                if pct > 50: st.error(f"🚨 CRITICAL THREAT — {pct:.1f}% malicious!")
                elif pct > 20: st.warning(f"⚠️ MODERATE RISK — {pct:.1f}% malicious")
                else: st.success(f"✅ LOW RISK — {pct:.1f}% flagged")
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure(go.Pie(
                        labels=["Normal","Attack"], values=[normals, attacks],
                        hole=0.6, marker_colors=["#00FF88","#FF4444"],
                        textfont=dict(color="white")
                    ))
                    fig.update_layout(title=dict(text="Traffic Split", font=dict(color="#00B4D8")),
                                      legend=dict(font=dict(color="#8ECAE6")))
                    st.plotly_chart(plotly_dark(fig), use_container_width=True)
                with c2:
                    fig2 = go.Figure(go.Histogram(x=probs, nbinsx=20, marker_color="#00B4D8", opacity=0.8))
                    fig2.add_vline(x=0.4, line_dash="dash", line_color="#FF4444",
                                   annotation_text="Threshold", annotation_font_color="#FF4444")
                    fig2.update_layout(title=dict(text="Risk Distribution", font=dict(color="#00B4D8")),
                                       xaxis=dict(color="#8ECAE6", gridcolor="#0D1B2A"),
                                       yaxis=dict(color="#8ECAE6", gridcolor="#0D1B2A"))
                    st.plotly_chart(plotly_dark(fig2), use_container_width=True)
                st.dataframe(results_df, use_container_width=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: MANUAL INPUT ──
    with t2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">MANUAL CONNECTION ANALYSIS</div>', unsafe_allow_html=True)
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown("**Connection Info**")
            duration  = st.number_input("Duration",  0, 60000, 0,   key="m_dur")
            protocol  = st.selectbox("Protocol",     ["tcp","udp","icmp"], key="m_proto")
            service   = st.selectbox("Service",      ["http","ftp","smtp","ssh","dns","ftp_data","other"], key="m_svc")
            flag      = st.selectbox("Flag",         ["SF","S0","REJ","RSTO","RSTR","S1","S2","S3"], key="m_flag")
            src_bytes = st.number_input("Src Bytes", 0, 1000000, 0, key="m_src")
            dst_bytes = st.number_input("Dst Bytes", 0, 1000000, 0, key="m_dst")
        with cb:
            st.markdown("**Activity Metrics**")
            land       = st.selectbox("Land",        [0,1], key="m_land")
            hot        = st.number_input("Hot",      0, 100, 0, key="m_hot")
            logged_in  = st.selectbox("Logged In",   [0,1], key="m_log")
            root_shell = st.selectbox("Root Shell",  [0,1], key="m_root")
            count      = st.number_input("Count",    0, 512, 10, key="m_cnt")
            srv_count  = st.number_input("Srv Count",0, 512, 10, key="m_srv")
        with cc:
            st.markdown("**Host Statistics**")
            serror_rate        = st.slider("SError Rate",   0.0, 1.0, 0.0, key="m_ser")
            rerror_rate        = st.slider("RError Rate",   0.0, 1.0, 0.0, key="m_rer")
            same_srv_rate      = st.slider("Same Srv Rate", 0.0, 1.0, 1.0, key="m_ssr")
            diff_srv_rate      = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0, key="m_dsr")
            dst_host_count     = st.number_input("Dst Host Count",   0, 255, 100, key="m_dhc")
            dst_host_srv_count = st.number_input("Dst Host Srv Cnt", 0, 255, 100, key="m_dhs")

        if st.button("⚡ ANALYZE CONNECTION", key="btn_manual"):
            row = {
                "duration":duration,"protocol_type":protocol,"service":service,"flag":flag,
                "src_bytes":src_bytes,"dst_bytes":dst_bytes,"land":land,"wrong_fragment":0,
                "urgent":0,"hot":hot,"num_failed_logins":0,"logged_in":logged_in,
                "num_compromised":0,"root_shell":root_shell,"su_attempted":0,"num_root":0,
                "num_file_creations":0,"num_shells":0,"num_access_files":0,"num_outbound_cmds":0,
                "is_host_login":0,"is_guest_login":0,"count":count,"srv_count":srv_count,
                "serror_rate":serror_rate,"srv_serror_rate":serror_rate,
                "rerror_rate":rerror_rate,"srv_rerror_rate":rerror_rate,
                "same_srv_rate":same_srv_rate,"diff_srv_rate":diff_srv_rate,
                "srv_diff_host_rate":0.0,"dst_host_count":dst_host_count,
                "dst_host_srv_count":dst_host_srv_count,
                "dst_host_same_srv_rate":same_srv_rate,"dst_host_diff_srv_rate":diff_srv_rate,
                "dst_host_same_src_port_rate":0.0,"dst_host_srv_diff_host_rate":0.0,
                "dst_host_serror_rate":serror_rate,"dst_host_srv_serror_rate":serror_rate,
                "dst_host_rerror_rate":rerror_rate,"dst_host_srv_rerror_rate":rerror_rate
            }
            with st.spinner("Analyzing..."):
                time.sleep(0.6)
                _, probs = predict(pd.DataFrame([row]))
            prob      = float(probs[0])
            is_attack = prob > 0.4

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text":"THREAT LEVEL","font":{"color":"#00B4D8","size":16}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#8ECAE6"},
                    "bar":{"color":"#FF4444" if is_attack else "#00FF88"},
                    "bgcolor":"rgba(0,0,0,0)",
                    "steps":[
                        {"range":[0,40],  "color":"rgba(0,45,18,0.3)"},
                        {"range":[40,70], "color":"rgba(45,30,0,0.3)"},
                        {"range":[70,100],"color":"rgba(45,0,0,0.3)"},
                    ],
                    "threshold":{"line":{"color":"#FF4444","width":3},"value":40}
                },
                number={"suffix":"%","font":{"color":"#00B4D8","size":36}}
            ))
            st.plotly_chart(plotly_dark(fig), use_container_width=True)
            save_scan(user["id"], user["username"], "Manual Input", 1,
                      1 if is_attack else 0, 0 if is_attack else 1,
                      prob*100 if is_attack else (1-prob)*100)
            if is_attack:
                st.markdown(f'<div class="result-attack"><div class="result-title">⚠ ATTACK DETECTED</div><div class="result-conf">Confidence: {prob*100:.1f}%</div><div style="color:#FF8888;font-size:0.9rem;margin-top:6px;">This connection shows malicious patterns — block recommended</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-normal"><div class="result-title">✓ SAFE CONNECTION</div><div class="result-conf">Confidence: {(1-prob)*100:.1f}%</div><div style="color:#88FFBB;font-size:0.9rem;margin-top:6px;">This connection appears to be legitimate traffic</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: SIMULATION ──
    with t3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">REAL-TIME TRAFFIC SIMULATION</div>', unsafe_allow_html=True)
        st.info("🎮 Watch the model detect attacks in real time — connection by connection!")
        n = st.slider("Connections to simulate", 10, 100, 25, key="sim_n")
        if st.button("▶ START SIMULATION", key="btn_sim"):
            test_df  = pd.read_csv("KDDTest+.txt", header=None, names=col_names)
            test_df  = test_df.drop(["label","difficulty"], axis=1, errors="ignore")
            sample   = test_df.sample(n, random_state=42).reset_index(drop=True)
            _, probs = predict(sample)
            chart_ph = st.empty()
            stats_ph = st.empty()
            live_results = []
            attacks_live = 0
            for i, prob in enumerate(probs):
                is_attack = prob > 0.4
                if is_attack: attacks_live += 1
                live_results.append({"x": i+1, "y": float(prob), "attack": is_attack})
                xs       = [r["x"] for r in live_results]
                ys       = [r["y"] for r in live_results]
                colors_pt = ["#FF4444" if r["attack"] else "#00FF88" for r in live_results]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    line=dict(color="#00B4D8", width=2),
                    marker=dict(color=colors_pt, size=9, line=dict(color="#020810", width=1)),
                    fill="tozeroy", fillcolor="rgba(0,180,216,0.05)"
                ))
                fig.add_hline(y=0.4, line_dash="dash", line_color="#FF4444",
                              annotation_text="Attack Threshold", annotation_font_color="#FF4444")
                fig.update_layout(
                    title=dict(text=f"Live Scan — {i+1}/{n} connections processed",
                               font=dict(color="#00B4D8", size=13)),
                    xaxis=dict(title="Connection #", color="#8ECAE6", gridcolor="#0D1B2A"),
                    yaxis=dict(title="Risk Score", range=[-0.05,1.05], color="#8ECAE6", gridcolor="#0D1B2A"),
                    height=320
                )
                chart_ph.plotly_chart(plotly_dark(fig), use_container_width=True)
                pct_live = (attacks_live / (i+1)) * 100
                stats_ph.markdown(f"""
                <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:8px 0;'>
                    <div class="metric-card teal"><div class="metric-val teal" style="font-size:1.5rem;">{i+1}</div><div class="metric-lbl">Scanned</div></div>
                    <div class="metric-card green"><div class="metric-val green" style="font-size:1.5rem;">{i+1-attacks_live}</div><div class="metric-lbl">Normal</div></div>
                    <div class="metric-card red"><div class="metric-val red" style="font-size:1.5rem;">{attacks_live}</div><div class="metric-lbl">Attacks</div></div>
                    <div class="metric-card gold"><div class="metric-val gold" style="font-size:1.5rem;">{pct_live:.1f}%</div><div class="metric-lbl">Risk</div></div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.12)
            save_scan(user["id"], user["username"], "Simulation", n, attacks_live, n-attacks_live, (attacks_live/n)*100)
            st.success(f"✅ Done! {attacks_live} attacks detected out of {n} connections.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 4: HISTORY ──
    with t4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">YOUR SCAN HISTORY</div>', unsafe_allow_html=True)
        if scans:
            for s in scans:
                scan_type, total, attacks, normal, risk, ts = s
                rc = "risk-high" if risk > 30 else "risk-med" if risk > 10 else "risk-low"
                st.markdown(f"""
                <div class="hist-row">
                    <div>
                        <div class="hist-info">{scan_type} — {total:,} connections</div>
                        <div class="hist-time">{ts}</div>
                    </div>
                    <div style='display:flex; gap:10px; align-items:center;'>
                        <span style='color:#4A6FA5; font-size:0.8rem;'>A:{attacks} | N:{normal}</span>
                        <span class="risk-badge {rc}">{risk:.1f}% risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#4A6FA5; text-align:center; padding:40px; font-family:Orbitron,monospace; font-size:0.85rem; letter-spacing:1px;">NO SCAN HISTORY YET</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _, lc, _ = st.columns([3, 1, 3])
    with lc:
        if st.button("⏻ LOGOUT", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user      = None
            st.session_state.page      = "login"
            st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE: ADMIN PANEL
# ══════════════════════════════════════════════════════════
def page_admin():
    render_navbar()
    g = get_global_stats()

    st.markdown("""
    <div style='padding:22px 32px 0;'>
        <div style='font-family:Orbitron,monospace; color:#FFB703; font-size:1.5rem; font-weight:600; letter-spacing:3px; text-shadow:0 0 20px rgba(255,183,3,0.4);'>
            ⚡ ADMIN CONTROL CENTER
        </div>
        <div style='color:#4A6FA5; font-size:0.82rem; margin-top:4px; letter-spacing:1.5px;'>FULL SYSTEM VISIBILITY</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:16px 32px;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card teal"><div class="metric-val teal">{g['total_users']}</div><div class="metric-lbl">Total Users</div></div>
        <div class="metric-card green"><div class="metric-val green">{g['total_scans']}</div><div class="metric-lbl">Total Scans</div></div>
        <div class="metric-card red"><div class="metric-val red">{g['total_attacks']:,}</div><div class="metric-lbl">Attacks Found</div></div>
        <div class="metric-card gold"><div class="metric-val gold">{g['avg_risk']}%</div><div class="metric-lbl">Avg Risk</div></div>
    </div>
    """, unsafe_allow_html=True)

    at1, at2, at3 = st.tabs(["👥 ALL USERS", "🔍 ALL SCANS", "🤖 MODEL INFO"])

    with at1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">REGISTERED USERS</div>', unsafe_allow_html=True)
        users = get_all_users()
        rows  = ""
        for u in users:
            uid, uname, email, role, created = u
            rp = f'<span class="role-pill role-{role}">{role.upper()}</span>'
            rows += f"<tr><td>{uid}</td><td style='color:#00B4D8;font-weight:600;'>{uname}</td><td>{email}</td><td>{rp}</td><td style='color:#4A6FA5;font-size:0.8rem;'>{created[:10]}</td></tr>"
        st.markdown(f"""
        <table class="admin-table">
            <tr><th>ID</th><th>Username</th><th>Email</th><th>Role</th><th>Joined</th></tr>
            {rows}
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with at2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">ALL SCAN ACTIVITY</div>', unsafe_allow_html=True)
        all_scans = get_all_scans()
        if all_scans:
            rows2 = ""
            for s in all_scans:
                uname, stype, total, attacks, risk, ts = s
                rc = "risk-high" if risk > 30 else "risk-med" if risk > 10 else "risk-low"
                rows2 += f"<tr><td style='color:#00B4D8;font-weight:600;'>{uname}</td><td>{stype}</td><td>{total:,}</td><td style='color:#FF4444;'>{attacks:,}</td><td><span class='risk-badge {rc}'>{risk:.1f}%</span></td><td style='color:#4A6FA5;font-size:0.8rem;'>{ts[:16]}</td></tr>"
            st.markdown(f"""
            <table class="admin-table">
                <tr><th>User</th><th>Type</th><th>Total</th><th>Attacks</th><th>Risk</th><th>Time</th></tr>
                {rows2}
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#4A6FA5; text-align:center; padding:30px; font-family:Orbitron,monospace; font-size:0.82rem;">NO SCAN DATA YET</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with at3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">MODEL INFORMATION</div>', unsafe_allow_html=True)
        info = [
            ("Model Type",        "Hybrid Metaheuristic Deep Learning"),
            ("Dataset",           "NSL-KDD (125,973 train / 22,544 test)"),
            ("Feature Selection", "Genetic Algorithm — 18 of 41 features"),
            ("Architecture",      "PSO NAS — 3 layers: [67, 47, 71] neurons"),
            ("Hyperparameters",   "GA Optimized — Adam, LR=0.001, Batch=256"),
            ("Final Accuracy",    "82.00%"),
            ("F1 Score",          "0.82"),
            ("Attack Precision",  "0.96 (96%)"),
            ("Attack Recall",     "0.63 (63%)"),
            ("Threshold",         "0.4 (optimized via F1 scan)"),
        ]
        rows3 = "".join(f"<tr><td style='color:#8ECAE6;font-weight:600;letter-spacing:0.5px;'>{i[0]}</td><td style='color:#CAF0F8;'>{i[1]}</td></tr>" for i in info)
        st.markdown(f'<table class="admin-table"><tr><th>Property</th><th>Value</th></tr>{rows3}</table>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("👤 USER VIEW", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    with c2:
        if st.button("⏻ LOGOUT", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user      = None
            st.session_state.page      = "login"
            st.rerun()

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
def render_footer():
    st.markdown("""
    <div class="footer">
        NIDS.SHIELD &nbsp;●&nbsp; GA + PSO + DEEP LEARNING &nbsp;●&nbsp; ICETFC — CRYPTOGRAPHY & NETWORK SECURITY
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

if not st.session_state.logged_in:
    page_login()
elif st.session_state.page == "admin":
    page_admin()
else:
    page_dashboard()

render_footer()
st.markdown("</div>", unsafe_allow_html=True)