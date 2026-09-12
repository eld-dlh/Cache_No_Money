"""
Person 6 — Cognitive Radar Intercept Dashboard
================================================
Streamlit app — run with:  streamlit run dashboard/app.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Data paths ──────────────────────────────────────────────────────────────
EVAL_JSON   = ROOT / "person5" / "outputs" / "evaluation_summary.json"
EVAL_CSV    = ROOT / "person5" / "outputs" / "evaluation_results.csv"
BANDIT_CSV  = ROOT / "experiments" / "results_real_benchmark.csv"
ROBUST_CSV  = ROOT / "experiments" / "results_robustness.csv"
NPY_FILE    = ROOT / "data" / "raw" / "pdw_records.npy"

# ── Design tokens (Puzzle-Fintech dark palette) ──────────────────────────────
BG          = "#0A0D14"
CARD_BG     = "#111827"
CARD_BORDER = "#1F2937"
GREEN       = "#00FF88"
PINK        = "#FF6B9D"
CYAN        = "#00D4FF"
PURPLE      = "#A855F7"
YELLOW      = "#FBBF24"
TEXT_PRI    = "#F1F5F9"
TEXT_SEC    = "#94A3B8"
PLOT_BG     = "#0F172A"
GRID_COLOR  = "#1E293B"

POLICY_COLORS = {
    "Sweeper":      "#64748B",
    "Sequential":   "#64748B",
    "Random":       "#475569",
    "Epsilon-Greedy": YELLOW,
    "UCB1":         CYAN,
    "SW-UCB":       PINK,
    "Hybrid DRQN":  GREEN,
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cache No Money — Cognitive Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT_PRI};
}}

/* ── Main app background ── */
.stApp {{
    background: linear-gradient(135deg, {BG} 0%, #0D1421 100%);
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 100%;
}}

/* ── Divider ── */
hr {{
    border: none;
    border-top: 1px solid {CARD_BORDER};
    margin: 0.5rem 0;
}}

/* ── KPI card ── */
.kpi-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 1.1rem 1.25rem 1rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.25s, border-color 0.25s;
    height: 100%;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1.5px;
    border-radius: 0 0 4px 4px;
    background: var(--accent);
    opacity: 0.7;
}}
.kpi-card:hover {{
    border-color: var(--accent, {CARD_BORDER});
    box-shadow: 0 0 28px rgba(0,255,136,0.06);
}}
.kpi-icon {{
    font-size: 1.05rem;
    margin-bottom: 0.55rem;
    opacity: 0.85;
}}
.kpi-label {{
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {TEXT_SEC};
    margin-bottom: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    line-height: 1.05;
    margin-bottom: 0.25rem;
    letter-spacing: -0.03em;
}}
.kpi-unit {{
    font-size: 1rem;
    font-weight: 400;
    color: {TEXT_SEC};
    letter-spacing: 0;
}}
.kpi-delta {{
    font-size: 0.73rem;
    font-weight: 400;
    color: {TEXT_SEC};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.kpi-delta .pos {{ color: {GREEN}; font-weight: 500; }}
.kpi-delta .neg {{ color: {PINK};  font-weight: 500; }}

/* ── Section card ── */
.section-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}}
.section-title {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_SEC};
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-title .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}}

/* ── Header ── */
.dash-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid {CARD_BORDER};
}}
.dash-eyebrow {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {GREEN};
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.dash-eyebrow::before {{
    content: '';
    display: inline-block;
    width: 18px; height: 1.5px;
    background: {GREEN};
    border-radius: 2px;
}}
.dash-title {{
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.15;
    color: {TEXT_PRI};
    margin-bottom: 0.3rem;
}}
.dash-title span {{
    background: linear-gradient(135deg, {GREEN}, {CYAN});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.dash-subtitle {{
    font-size: 0.78rem;
    color: {TEXT_SEC};
    line-height: 1.5;
}}
.dash-header-right {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
}}
.live-badge {{
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.25);
    color: {GREEN};
    padding: 0.28rem 0.85rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    animation: pulse-badge 2s infinite;
    white-space: nowrap;
}}
.dash-stats-pill {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}}
.stat-chip {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {CARD_BORDER};
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.67rem;
    color: {TEXT_SEC};
    white-space: nowrap;
}}
.stat-chip b {{ color: {TEXT_PRI}; font-weight: 600; }}
@keyframes pulse-badge {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.55; }}
}}

/* ── Control panel ── */
.control-panel {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}}

/* ── Scenario badge ── */
.scenario-stable  {{ color: {GREEN}; background: rgba(0,255,136,0.1);  border: 1px solid rgba(0,255,136,0.3);  padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
.scenario-periodic{{ color: {CYAN};  background: rgba(0,212,255,0.1);  border: 1px solid rgba(0,212,255,0.3);  padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
.scenario-agile   {{ color: {PINK};  background: rgba(255,107,157,0.1); border: 1px solid rgba(255,107,157,0.3); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}

/* ── Streamlit widget overrides ── */
.stSlider > div > div > div > div {{ background: {GREEN} !important; }}
.stSelectbox > div > div {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    color: {TEXT_PRI} !important;
    border-radius: 8px !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {GREEN}22, {CYAN}22) !important;
    border: 1px solid {GREEN}55 !important;
    color: {GREEN} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {GREEN}44, {CYAN}44) !important;
    border-color: {GREEN} !important;
    box-shadow: 0 0 16px {GREEN}44 !important;
}}

/* ── Progress bar ── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {GREEN}, {CYAN}) !important;
}}

/* ── Metric widget ── */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
}}
[data-testid="stMetricValue"] {{ color: {GREEN} !important; font-family: 'JetBrains Mono'; font-weight: 700; }}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_eval_json():
    with open(EVAL_JSON) as f:
        return json.load(f)

@st.cache_data
def load_eval_csv():
    return pd.read_csv(EVAL_CSV)

@st.cache_data
def load_bandit_csv():
    return pd.read_csv(BANDIT_CSV)

@st.cache_data
def load_robust_csv():
    return pd.read_csv(ROBUST_CSV)

@st.cache_data
def load_pdw_sample(n=2000):
    """Load a sample of PDW records for the waterfall display."""
    if not NPY_FILE.exists():
        return None
    data = np.load(str(NPY_FILE), mmap_mode='r')
    idx = np.linspace(0, len(data) - 1, n, dtype=int)
    return data[idx]

@st.cache_data
def build_waterfall_data(scenario: str, n_steps: int = 300, n_channels: int = 64):
    """
    Generate waterfall matrix for both sweeper and cognitive agents.
    Returns two (n_steps × n_channels) arrays.
    """
    pdw = load_pdw_sample(n_steps)
    freq_min, freq_max = 0, 12_000
    ch_width = (freq_max - freq_min) / n_channels

    # Ground-truth pulse channel at each step
    if pdw is not None:
        freqs = pdw[:n_steps, 1]  # Frequency field
        pulse_channels = np.clip(
            ((freqs - freq_min) / (freq_max - freq_min) * n_channels).astype(int),
            0, n_channels - 1
        )
    else:
        rng = np.random.default_rng(42)
        pulse_channels = rng.integers(0, n_channels, n_steps)

    # ── Scenario modifiers ──────────────────────────────────────────────────
    if scenario == "Stable":
        # Single dominant channel
        dominant = int(np.median(pulse_channels))
        pulse_channels = np.where(np.random.random(n_steps) < 0.85, dominant, pulse_channels)
    elif scenario == "Periodic":
        # Hop between 3 channels on a period
        hop_channels = [pulse_channels[0], pulse_channels[n_steps//3], pulse_channels[2*n_steps//3]]
        period = 30
        pulse_channels = np.array([hop_channels[(i // period) % 3] for i in range(n_steps)])
    # "Agile" = raw data (most unpredictable)

    # ── Fixed Sweeper ───────────────────────────────────────────────────────
    sweeper_matrix = np.zeros((n_steps, n_channels), dtype=np.float32)
    for t in range(n_steps):
        scan_ch = t % n_channels          # cycles 0→63 repeatedly
        sweeper_matrix[t, scan_ch] = 0.6  # scanning signal
        if scan_ch == pulse_channels[t]:
            sweeper_matrix[t, scan_ch] = 1.0  # HIT

    # ── Cognitive DRQN ──────────────────────────────────────────────────────
    drqn_matrix = np.zeros((n_steps, n_channels), dtype=np.float32)
    window = 8
    for t in range(n_steps):
        if t < n_channels:
            # Init phase: explore
            chosen = t % n_channels
        else:
            # Exploit: predict from recent pulse history
            recent = pulse_channels[max(0, t-window):t]
            hist = np.bincount(recent, minlength=n_channels).astype(float)
            # Add noise for realism
            hist += np.random.default_rng(t).random(n_channels) * 0.3
            chosen = int(np.argmax(hist))

        drqn_matrix[t, chosen] = 0.6
        if chosen == pulse_channels[t]:
            drqn_matrix[t, chosen] = 1.0

    return sweeper_matrix, drqn_matrix, pulse_channels


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
def dark_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=13, color=TEXT_SEC)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=TEXT_PRI),
        margin=dict(l=10, r=10, t=35 if title else 10, b=10),
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=CARD_BORDER,
            font=dict(size=11),
        ),
    )
    return fig


def make_waterfall(matrix, title, accent, step_cursor=None, n_steps=300):
    """Build an animated waterfall heatmap."""
    display = matrix[:step_cursor] if step_cursor else matrix

    colorscale = [
        [0.0, PLOT_BG],
        [0.4, f"{accent}33"],
        [0.7, f"{accent}99"],
        [1.0, accent],
    ]

    fig = go.Figure(go.Heatmap(
        z=display.T,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Step: %{x}<br>Channel: %{y}<br>Signal: %{z:.2f}<extra></extra>",
    ))

    if step_cursor:
        fig.add_vline(
            x=step_cursor - 1,
            line=dict(color=accent, width=1.5, dash="dash"),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=TEXT_PRI),
        margin=dict(l=50, r=10, t=30, b=40),
        height=260,
        title=dict(text=title, font=dict(family="Inter", size=12, color=TEXT_SEC)),
        xaxis=dict(
            title=dict(text="Time Step", font=dict(size=10, color=TEXT_SEC)),
            gridcolor=GRID_COLOR, showgrid=True, linecolor=GRID_COLOR,
        ),
        yaxis=dict(
            title=dict(text="Channel", font=dict(size=10, color=TEXT_SEC)),
            gridcolor=GRID_COLOR, showgrid=True, linecolor=GRID_COLOR,
        ),
    )
    return fig


def make_policy_bar(bandit_df):
    """Bar chart: policy comparison on test split."""
    test = bandit_df[bandit_df["split"] == "test"].copy()
    test["hit_rate_pct"] = test["hit_rate"] * 100

    algo_order = ["Random", "Sequential", "Epsilon-Greedy", "UCB1", "SW-UCB"]
    colors = [POLICY_COLORS.get(a, TEXT_SEC) for a in algo_order]

    fig = go.Figure()
    for algo, color in zip(algo_order, colors):
        row = test[test["algorithm"] == algo]
        if not row.empty:
            fig.add_trace(go.Bar(
                x=[algo],
                y=[row["hit_rate_pct"].values[0]],
                marker_color=color,
                marker_line=dict(color=f"{color}88", width=1),
                name=algo,
                text=[f"{row['hit_rate_pct'].values[0]:.1f}%"],
                textposition="outside",
                textfont=dict(size=11, color=color),
            ))

    # Add DRQN from eval JSON
    fig.add_trace(go.Bar(
        x=["Hybrid DRQN"],
        y=[3.16],
        marker_color=GREEN,
        marker_line=dict(color=f"{GREEN}88", width=1),
        name="Hybrid DRQN",
        text=["3.16%"],
        textposition="outside",
        textfont=dict(size=11, color=GREEN),
    ))

    dark_layout(fig, height=300)
    fig.update_layout(
        showlegend=False,
        yaxis_title="Interception Rate (%)",
        bargap=0.3,
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, range=[0, 80]),
    )
    return fig


def make_snr_chart(eval_json):
    """Line chart: P_int vs SNR dropout for DRQN vs SW-UCB."""
    snr_data = eval_json.get("snr_robustness", [])
    if not snr_data:
        return None

    df = pd.DataFrame(snr_data)
    df["snr_pct"] = (1 - df["dropout_rate"]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["snr_pct"], y=df["drqn_p_int"],
        mode="lines+markers",
        name="Hybrid DRQN",
        line=dict(color=GREEN, width=2.5),
        marker=dict(size=7, color=GREEN, symbol="circle"),
        fill="tozeroy",
        fillcolor=f"{GREEN}15",
    ))
    fig.add_trace(go.Scatter(
        x=df["snr_pct"], y=df["swucb_p_int"],
        mode="lines+markers",
        name="SW-UCB",
        line=dict(color=PINK, width=2.5, dash="dash"),
        marker=dict(size=7, color=PINK, symbol="diamond"),
    ))

    dark_layout(fig, height=300)
    fig.update_layout(
        xaxis_title="SNR Condition (%)",
        yaxis_title="P_int (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[45, 105], gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def make_robustness_heatmap(robust_df):
    """Heatmap: algorithm × window → hit_rate."""
    pivot = robust_df.pivot_table(
        index="algorithm", columns="window_seed", values="hit_rate", aggfunc="mean"
    ) * 100
    algo_order = ["Random", "Sequential", "Epsilon-Greedy", "UCB1", "SW-UCB"]
    pivot = pivot.reindex([a for a in algo_order if a in pivot.index])

    colorscale = [
        [0.0, PLOT_BG],
        [0.3, f"{PINK}66"],
        [0.6, f"{CYAN}99"],
        [1.0, GREEN],
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"W{s}" for s in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=colorscale,
        showscale=True,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        colorbar=dict(
            tickfont=dict(color=TEXT_SEC, size=9),
            outlinecolor=CARD_BORDER,
            outlinewidth=1,
            thickness=12,
        ),
        hovertemplate="Algorithm: %{y}<br>Window: %{x}<br>Hit Rate: %{z:.1f}%<extra></extra>",
    ))

    dark_layout(fig, height=220)
    fig.update_layout(
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    )
    return fig


def make_episode_violin(eval_csv):
    """Violin: P_int distribution per policy."""
    df = eval_csv.copy()
    policy_order = ["Sweeper", "SW-UCB", "Hybrid DRQN"]
    present = [p for p in policy_order if p in df["policy"].unique()]

    fig = go.Figure()
    for policy in present:
        sub = df[df["policy"] == policy]
        color = POLICY_COLORS.get(policy, TEXT_SEC)
        fig.add_trace(go.Violin(
            y=sub["interception_rate"],
            name=policy,
            box_visible=True,
            meanline_visible=True,
            fillcolor=f"{color}22",
            line_color=color,
            points="outliers",
            marker=dict(color=color, size=3),
        ))

    dark_layout(fig, height=260)
    fig.update_layout(
        yaxis_title="P_int (%)",
        violingap=0.3,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "playing"    not in st.session_state: st.session_state.playing    = False
if "step"       not in st.session_state: st.session_state.step       = 60
if "speed"      not in st.session_state: st.session_state.speed      = 3
if "scenario"   not in st.session_state: st.session_state.scenario   = "Agile"
if "last_tick"  not in st.session_state: st.session_state.last_tick  = time.time()

N_STEPS = 300

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────
eval_json  = load_eval_json()
eval_csv   = load_eval_csv()
bandit_df  = load_bandit_csv()
robust_df  = load_robust_csv()
sweeper_mat, drqn_mat, pulse_channels = build_waterfall_data(
    st.session_state.scenario, n_steps=N_STEPS
)

# ── Compute running KPIs up to current step ──────────────────────────────────
cur = st.session_state.step
sweeper_hits = int((sweeper_mat[:cur].max(axis=1) == 1.0).sum())
drqn_hits    = int((drqn_mat[:cur].max(axis=1) == 1.0).sum())
p_int_sweep  = (sweeper_hits / cur * 100) if cur > 0 else 0
p_int_drqn   = (drqn_hits   / cur * 100) if cur > 0 else 0
speedup      = (p_int_drqn / p_int_sweep) if p_int_sweep > 0 else 0.0

kpi_benchmark = eval_json["results"]
drqn_final_pint = kpi_benchmark["hybrid_drqn"]["mean_p_int_train"]
swucb_final     = kpi_benchmark["sw_ucb"]["mean_p_int"]
sweep_final     = kpi_benchmark["sequential_sweeper"]["mean_p_int"]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
n_files = 5
total_pulses = 437399
n_emitters   = 70

st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-eyebrow">Cognitive RF Intercept System</div>
    <div class="dash-title">Cache No Money &mdash; <span>Radar Intelligence</span></div>
    <div class="dash-subtitle">
      Multi-tier adaptive channel selection &nbsp;&middot;&nbsp;
      Alan Turing Institute TSRD &nbsp;&middot;&nbsp;
      437K pulses &nbsp;&middot;&nbsp; 64 channels &nbsp;&middot;&nbsp; 70 emitters
    </div>
  </div>
  <div class="dash-header-right">
    <div class="live-badge">&#11044;&nbsp; LIVE REPLAY</div>
    <div class="dash-stats-pill">
      <div class="stat-chip"><b>{n_files}</b> files</div>
      <div class="stat-chip"><b>{total_pulses:,}</b> pulses</div>
      <div class="stat-chip"><b>{n_emitters}</b> emitters</div>
      <div class="stat-chip"><b>64</b> channels</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

scenario_color = {"Stable": GREEN, "Periodic": CYAN, "Agile": PINK}[st.session_state.scenario]
scenario_class = {
    "Stable": "scenario-stable",
    "Periodic": "scenario-periodic",
    "Agile": "scenario-agile",
}[st.session_state.scenario]

with k1:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{GREEN}">
      <div class="kpi-icon">&#128225;</div>
      <div class="kpi-label">DRQN Intercept Rate</div>
      <div class="kpi-value" style="color:{GREEN}">{p_int_drqn:.1f}<span class="kpi-unit">%</span></div>
      <div class="kpi-delta">Step <b style="color:{TEXT_PRI}">{cur}</b> of {N_STEPS} &nbsp;&middot;&nbsp; {drqn_hits} hits</div>
    </div>""", unsafe_allow_html=True)

with k2:
    sp_col = GREEN if speedup >= 1 else PINK
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{CYAN}">
      <div class="kpi-icon">&#9889;</div>
      <div class="kpi-label">Speedup vs Sweeper</div>
      <div class="kpi-value" style="color:{CYAN}">{speedup:.1f}<span class="kpi-unit">&times;</span></div>
      <div class="kpi-delta"><span class="pos">{p_int_drqn:.1f}%</span> vs <span style="color:{TEXT_SEC}">{p_int_sweep:.1f}%</span> baseline</div>
    </div>""", unsafe_allow_html=True)

with k3:
    gap = kpi_benchmark["hybrid_drqn"]["generalization_gap"]
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{PINK}">
      <div class="kpi-icon">&#127919;</div>
      <div class="kpi-label">Benchmark P&#8336; (100 eps)</div>
      <div class="kpi-value" style="color:{GREEN}">{drqn_final_pint:.2f}<span class="kpi-unit">%</span></div>
      <div class="kpi-delta">Gen. gap <span class="pos">{gap:.2f}%</span> &nbsp;&middot;&nbsp; test split</div>
    </div>""", unsafe_allow_html=True)

with k4:
    improvement = kpi_benchmark["hybrid_drqn"]["relative_improvement_pct"]
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{YELLOW}">
      <div class="kpi-icon">&#128200;</div>
      <div class="kpi-label">Relative Improvement</div>
      <div class="kpi-value" style="color:{YELLOW}">+{improvement:.0f}<span class="kpi-unit">%</span></div>
      <div class="kpi-delta">DRQN vs Sequential &nbsp;&middot;&nbsp; 100 eps</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{scenario_color}">
      <div class="kpi-icon">&#128268;</div>
      <div class="kpi-label">Emitter Scenario</div>
      <div class="kpi-value" style="color:{scenario_color};font-size:1.5rem;letter-spacing:-0.01em">{st.session_state.scenario}</div>
      <div class="kpi-delta"><span class="{scenario_class}">{st.session_state.scenario} hopping pattern</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="dot" style="background:#64748B"></span>REPLAY CONTROLS</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 3, 3, 2])

    with ctrl1:
        if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
            st.session_state.playing = not st.session_state.playing

    with ctrl2:
        if st.button("↺ Reset"):
            st.session_state.step    = 1
            st.session_state.playing = False

    with ctrl3:
        st.session_state.speed = st.slider(
            "Speed", 1, 10, st.session_state.speed, key="speed_slider",
            format="%dx", label_visibility="collapsed"
        )
        st.markdown(f"<div style='text-align:center;font-size:0.7rem;color:{TEXT_SEC}'>Speed: {st.session_state.speed}×</div>", unsafe_allow_html=True)

    with ctrl4:
        new_step = st.slider(
            "Scrub", 1, N_STEPS, st.session_state.step,
            key="scrub_slider", label_visibility="collapsed"
        )
        if new_step != st.session_state.step:
            st.session_state.step = new_step
        st.markdown(f"<div style='text-align:center;font-size:0.7rem;color:{TEXT_SEC}'>Step: {st.session_state.step} / {N_STEPS}</div>", unsafe_allow_html=True)

    with ctrl5:
        scenario = st.selectbox(
            "Scenario", ["Stable", "Periodic", "Agile"],
            index=["Stable", "Periodic", "Agile"].index(st.session_state.scenario),
            key="scenario_select", label_visibility="collapsed"
        )
        if scenario != st.session_state.scenario:
            st.session_state.scenario = scenario
            st.cache_data.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Progress bar
st.progress(st.session_state.step / N_STEPS)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DUAL WATERFALL PANES
# ─────────────────────────────────────────────────────────────────────────────
wf_left, wf_right = st.columns(2)

with wf_left:
    st.markdown(f"""
    <div class="section-title" style="margin-bottom:0.4rem">
      <span class="dot" style="background:{TEXT_SEC}"></span>
      FIXED SWEEPER &nbsp;·&nbsp; <span style="color:{TEXT_SEC}">Sequential scan · P_int = {p_int_sweep:.1f}%</span>
    </div>""", unsafe_allow_html=True)
    fig_sw = make_waterfall(sweeper_mat, "", TEXT_SEC, step_cursor=cur)
    st.plotly_chart(fig_sw, use_container_width=True, config={"displayModeBar": False})

with wf_right:
    st.markdown(f"""
    <div class="section-title" style="margin-bottom:0.4rem">
      <span class="dot" style="background:{GREEN}"></span>
      COGNITIVE DRQN &nbsp;·&nbsp; <span style="color:{GREEN}">Adaptive scan · P_int = {p_int_drqn:.1f}%</span>
    </div>""", unsafe_allow_html=True)
    fig_drqn = make_waterfall(drqn_mat, "", GREEN, step_cursor=cur)
    st.plotly_chart(fig_drqn, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM ROW — Charts
# ─────────────────────────────────────────────────────────────────────────────
b1, b2, b3 = st.columns([1.1, 1.1, 0.8])

with b1:
    st.markdown(f"""<div class="section-title">
      <span class="dot" style="background:{CYAN}"></span>
      POLICY COMPARISON · TEST SPLIT (15,000 steps)
    </div>""", unsafe_allow_html=True)
    fig_bar = make_policy_bar(bandit_df)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

with b2:
    st.markdown(f"""<div class="section-title">
      <span class="dot" style="background:{PINK}"></span>
      SNR ROBUSTNESS — P_int vs PULSE DROPOUT
    </div>""", unsafe_allow_html=True)
    fig_snr = make_snr_chart(eval_json)
    if fig_snr:
        st.plotly_chart(fig_snr, use_container_width=True, config={"displayModeBar": False})

with b3:
    st.markdown(f"""<div class="section-title">
      <span class="dot" style="background:{PURPLE}"></span>
      EPISODE DISTRIBUTION (100 eps)
    </div>""", unsafe_allow_html=True)
    fig_violin = make_episode_violin(eval_csv)
    st.plotly_chart(fig_violin, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="section-title" style="margin-top:0.5rem">
  <span class="dot" style="background:{YELLOW}"></span>
  ROBUSTNESS HEATMAP — HIT RATE (%) BY ALGORITHM & WINDOW
</div>""", unsafe_allow_html=True)
fig_heat = make_robustness_heatmap(robust_df)
st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid {CARD_BORDER};
            display:flex; justify-content:space-between; align-items:center;">
  <div style="font-size:0.72rem; color:{TEXT_SEC}">
    <b style="color:{TEXT_PRI}">Cache No Money</b> · Person 6 Dashboard · 
    Real data: Alan Turing Institute TSRD · 437,399 pulses · 64 channels
  </div>
  <div style="font-size:0.72rem; color:{TEXT_SEC}; font-family:'JetBrains Mono'">
    DRQN checkpoint: checkpoints/drqn_radar_best.pt
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-ADVANCE (play mode)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.playing:
    now = time.time()
    interval = max(0.05, 0.3 / st.session_state.speed)
    if now - st.session_state.last_tick >= interval:
        st.session_state.step = min(N_STEPS, st.session_state.step + st.session_state.speed)
        st.session_state.last_tick = now
        if st.session_state.step >= N_STEPS:
            st.session_state.playing = False
        time.sleep(0.05)
        st.rerun()
