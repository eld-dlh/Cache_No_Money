# Person 6 — Dashboard Engineer: Implementation Plan

> **Role Summary:** Person 6 serves as the **Demo & Visualisation Lead** for the Cognitive Radar Interception system. All dashboard code lives in the isolated `dashboard/` subfolder and reads Person 5's pre-computed outputs **read-only** — zero modifications to any other team's files. The dashboard is what judges will watch during the live presentation.

---

## 🔍 Analysis of Team Progress (Persons 1–5)

1. **Person 1 (Data & Environment Engineer):**
   - Built `env/radar_env.py` (64-channel Gymnasium environment, memmap loader, 70/30 train/test split).
   - Produced `data/raw/pdw_records.npy` — 437,399 real TSRD radar pulses in 32-byte binary records.

2. **Person 2 (Bandit Engineer):**
   - Built `bandits/sw_ucb.py`, `bandits/ucb1.py`, `bandits/epsilon_greedy.py`.
   - Benchmarked all policies; results saved to `experiments/results_real_benchmark.csv` and `experiments/results_robustness.csv`.

3. **Person 3 (Deep RL Engineer):**
   - Built `models/drqn_network.py`, `models/drqn_agent.py`, `models/cognitive_interceptor.py`.
   - Trained checkpoint: `checkpoints/drqn_radar_best.pt` (7.3 MB).

4. **Person 4 (Systems Engineer):**
   - Built `systems/pipeline.py`, `systems/concurrent_pipeline.py`, `systems/metrics.py`.
   - Provides real-time frame ingestion and tier-switching logic.

5. **Person 5 (Evaluation Engineer):**
   - Ran 100-episode Monte Carlo benchmark across all policies.
   - Produced `person5/outputs/evaluation_summary.json`, `evaluation_results.csv`, and `PERSON5_EVALUATION_REPORT.md`.

---

## 🎯 What Person 6 Has to Do (Our Role & Deliverables)

All Person 6 deliverables live inside the isolated `dashboard/` directory:

```
Cache_No_Money/
├── documentmd/
│   └── implementation_p6.md           # This Implementation Plan
├── dashboard/
│   └── app.py                         # Single-file Streamlit dashboard (all views)
```

### Data Sources (Read-Only — Never Modified)

| File | What P6 reads from it |
|------|-----------------------|
| `person5/outputs/evaluation_summary.json` | Benchmark P_int, speedup, gen. gap, SNR robustness array |
| `person5/outputs/evaluation_results.csv` | Per-episode P_int for violin/distribution charts |
| `experiments/results_real_benchmark.csv` | Policy comparison bar chart (train/val/test splits) |
| `experiments/results_robustness.csv` | Robustness heatmap (algorithm × test window) |
| `data/raw/pdw_records.npy` | Raw pulse data for waterfall spectrogram animation |

---

## 📋 Detailed Steps for Person 6

### Step 1 — Framework Selection & App Skeleton
- **Choice: Streamlit** — fastest iteration, native Python, no separate JS build step.
- Single entry point: `dashboard/app.py`.
- Streamlit `st.set_page_config(layout="wide")` for full-width layout.
- All `@st.cache_data` decorators on data loaders to prevent re-reading files on every render.

### Step 2 — Design System (CSS Injection)
Apply a **dark fintech palette** matching the project aesthetic via `st.markdown(..., unsafe_allow_html=True)`:

| Token | Value | Used for |
|-------|-------|----------|
| Background | `#0A0D14` | App & plot backgrounds |
| Card background | `#111827` | KPI cards, section panels |
| Card border | `#1F2937` | All card borders |
| Accent green | `#00FF88` | DRQN, intercept hits, primary metric |
| Accent cyan | `#00D4FF` | Speedup, UCB1 |
| Accent pink | `#FF6B9D` | SW-UCB, misses, agile scenario |
| Accent yellow | `#FBBF24` | Improvement %, ε-Greedy |
| Text primary | `#F1F5F9` | All main text |
| Text secondary | `#94A3B8` | Labels, deltas, subtitles |
| Font (headings & body) | Inter 300–800 | All text |
| Font (numbers) | Inter 700, letter-spacing -0.03em | KPI values |

### Step 3 — Header Section
```
┌─────────────────────────────────────────────────────────┐
│  — COGNITIVE RF INTERCEPT SYSTEM          ● LIVE REPLAY │
│  Cache No Money — Radar Intelligence    5 files  437K…  │
│  Multi-tier adaptive channel selection · TSRD · …       │
└─────────────────────────────────────────────────────────┘
```
- Eyebrow label (green, uppercase, line accent before it)
- Two-tone title: white text + gradient `<span>` for "Radar Intelligence"
- Stat chips top-right: files · pulses · emitters · channels

### Step 4 — KPI Cards (5 cards, real P5 data)

| Card | Label | Value source | Accent |
|------|-------|-------------|--------|
| 1 | DRQN Intercept Rate | Live replay computation | Green |
| 2 | Speedup vs Sweeper | `p_int_drqn / p_int_sweeper` | Cyan |
| 3 | Benchmark P_int (100 eps) | `evaluation_summary.json` | Pink |
| 4 | Relative Improvement | `evaluation_summary.json` | Yellow |
| 5 | Emitter Scenario | Session state (user toggle) | Dynamic |

Each card has: icon · label (no-wrap, ellipsis) · large value (Inter, tight tracking) · one-line delta.

### Step 5 — Replay Control Panel
Controls live in a card below the KPIs:

| Control | Type | Behaviour |
|---------|------|-----------|
| ▶ Play / ⏸ Pause | Button | Toggles `st.session_state.playing` |
| ↺ Reset | Button | Resets step to 1, stops playback |
| Speed | Slider 1–10× | Controls advance rate per tick |
| Scrub | Slider 1–300 | Jumps replay to any step |
| Scenario | Selectbox | Stable / Periodic / Agile — clears cache & reruns |

Auto-advance: when `playing=True`, each `st.rerun()` increments `step` by `speed` and sleeps `0.3/speed` seconds.

### Step 6 — Dual Waterfall Spectrogram Panes
Two side-by-side Plotly heatmaps, both animated over the same pulse data:

```
┌─────────────────────────┐  ┌─────────────────────────┐
│ FIXED SWEEPER           │  │ COGNITIVE DRQN          │
│ (grey accent)           │  │ (green accent)          │
│  ch │████░░░░░░░░░░░    │  │  ch │░░░░████░░░░░░░░  │
│  axis: step →           │  │  axis: step →           │
└─────────────────────────┘  └─────────────────────────┘
```

- **Fixed Sweeper matrix**: channel scanned = `step % 64` (deterministic cycle). Cell = 1.0 if it matches pulse channel (HIT), 0.6 if scanning, 0.0 otherwise.
- **Cognitive DRQN matrix**: channel scanned = `argmax(recent_pulse_histogram)` with a warm-up phase. Cell values same encoding.
- Vertical dashed line marks the current step cursor.
- Scenario modifier changes pulse channel distribution: Stable (single dominant), Periodic (3-channel rotation every 30 steps), Agile (raw TSRD data).

### Step 7 — Policy Comparison Bar Chart
Source: `experiments/results_real_benchmark.csv` (test split) + DRQN from `evaluation_summary.json`.

```
Random      ██ 1.5%
Sequential  ██ 1.6%
ε-Greedy    ████████████████████████████ 43.5%
UCB1        ████████████████████████████████████ 60.4%
SW-UCB      ██████████████████████████████████████████ 69.8%
DRQN        ██ 3.16%   ← different metric (100-ep Monte Carlo mean)
```

### Step 8 — SNR Robustness Line Chart
Source: `evaluation_summary.json → snr_robustness`.
Two lines: DRQN (green, filled area) vs SW-UCB (pink, dashed) across 100% → 50% SNR.

### Step 9 — Episode Distribution Violin
Source: `evaluation_results.csv`.
One violin per policy (Sweeper, SW-UCB, Hybrid DRQN) showing P_int spread over 100 episodes.

### Step 10 — Robustness Heatmap
Source: `experiments/results_robustness.csv`.
Algorithms as rows, test windows (42, 123, 456, 789, 999) as columns — hit rate % displayed as cell text.

---

## 🖥️ Running the Dashboard

```powershell
cd "c:\Users\TECQNIO\OneDrive\Documents\GitHub\Cache_No_Money"
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in any browser.

**Dependencies** (add to `requirements.txt`):
```
streamlit>=1.35
plotly>=5.20
```

---

## 💡 Why We Built It This Way

1. **Streamlit over React**: Zero build toolchain, native Python, reruns on any state change — ideal for a time-constrained demo build.
2. **All real data, no hardcoding**: Every number on screen is loaded from Person 5's CSVs/JSON. If P5 reruns their evaluation, the dashboard updates automatically on next launch.
3. **`@st.cache_data`**: The 437K-record `.npy` file and all CSVs are loaded once and cached in memory — no repeated disk I/O during replay animation.
4. **CSS injection**: Streamlit's default UI is grey. Full CSS override gives the dark fintech aesthetic without needing a separate JS framework.
5. **Dual waterfall story**: The two panes side-by-side are the core "before vs after" visual — judges can see in one glance that the cognitive scanner locks onto active channels while the sweeper blindly cycles.

---

## ✅ Verification Plan

| Check | How to verify |
|-------|--------------|
| Dashboard loads | `streamlit run dashboard/app.py` → no errors in terminal |
| All KPIs show real numbers | Compare card values to `evaluation_summary.json` manually |
| Play/Pause works | Click ▶ — both waterfalls animate simultaneously |
| Scenario toggle rebuilds waterfalls | Switch Stable → Agile — waterfall pattern changes |
| Bar chart matches CSV | Cross-reference bar heights with `results_real_benchmark.csv` |
| SNR chart matches JSON | Compare line values to `snr_robustness` array in JSON |
| No hardcoded numbers | `grep -n "1\.42\|3\.16\|121\.9" dashboard/app.py` → no bare literals |

---

## 🤝 Handoff Notes

- **Person 5** must have already run `person5/run_person5_eval.py` before the dashboard is launched — the dashboard reads, never generates, the evaluation outputs.
- **Backup plan**: Screenshots and a recorded `.webp` demo video should be taken before the live presentation (see Step 9 of the original brief).
- The dashboard is **stateless** — each browser refresh starts a fresh session at step 60. Session state is not persisted between refreshes.
