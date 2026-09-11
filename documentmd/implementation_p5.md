# Person 5 — Evaluation Engineer: Implementation Plan

> **Role Summary:** Person 5 serves as the **Independent Evaluator & Quality Assurance Lead** for the Cognitive Radar Interception system. All evaluation logic, Monte Carlo benchmarks, stress-test scripts, and results are isolated inside the `person5/` subfolder to ensure **zero disturbance to existing team files**.

---

## 🔍 Analysis of Team Progress (Persons 1–4)

1. **Person 1 (Data & Environment Engineer):**
   - Built `env/radar_env.py` (`RadarEnv` Gymnasium environment with 64 channels, PDW memmap batch reader, and 70/30 train/test split).
2. **Person 2 (Bandit Engineer):**
   - Built `bandits/sw_ucb.py` (Sliding-Window UCB baseline) and reactive bandit policies.
3. **Person 3 (Deep RL Tier 2 Engineer):**
   - Built `models/cognitive_interceptor.py` and `models/drqn_network.py` (DRQN Dueling LSTM + Handoff Controller), trained weights `checkpoints/drqn_radar_best.pt` (achieved 87% interception rate).
4. **Person 4 (Systems Engineer):**
   - Built concurrency and streaming integration wrappers.

---

## 🎯 What Person 5 Has to Do (Our Role & Deliverables)

All Person 5 deliverables live inside the isolated `person5/` directory:

```
Cache_No_Money/
├── documentmd/
│   ├── implementation_p5.md              # This Implementation Plan
├── person5/
│   ├── __init__.py
│   ├── eval_harness.py                   # Core MonteCarloEvaluator engine
│   ├── noise_models.py                   # Low-SNR & pulse dropout simulators
│   ├── run_person5_eval.py               # Master execution script (100 episodes)
│   ├── test_person5_eval.py              # Pytest validation test suite
│   └── outputs/                          # Hand-off directory for Person 6 (Dashboard)
│       ├── evaluation_results.csv        # Detailed per-episode metrics table
│       ├── evaluation_summary.json       # Clean KPI summary JSON
│       ├── eval_interception_comparison.png # Interception rate comparison plot
│       ├── eval_snr_robustness.png       # Low-SNR robustness curve
│       └── PERSON5_EVALUATION_REPORT.md  # Executive 2-3 sentence summary per test
```

### Detailed Steps for Person 5:
1. **Monte Carlo Evaluation Harness (`person5/eval_harness.py`):**
   - Runs 100 episodes per policy with fixed random seeds for strict reproducibility.
   - Policies tested: **Sequential Sweeper Baseline**, **Sliding-Window UCB (Tier 1)**, and **Hybrid DRQN (Tier 2)**.
2. **Metrics Engine:**
   - Computes Interception Rate $P_{int} = \frac{\text{Intercepted Pulses}}{\text{Total Pulses}}$.
   - Computes Relative Improvement:
     $$\text{Relative Improvement (\%)} = \frac{P_{int, \text{Adaptive}} - P_{int, \text{Baseline}}}{P_{int, \text{Baseline}}} \times 100$$
   - Computes Wasted Dwells, Retune Cost, and Average Reward.
3. **Multi-Condition & Held-Out Dataset Evaluation:**
   - Evaluates on 70/30 held-out test split (verifies DRQN sequence generalization).
   - Evaluates on Low-SNR conditions (pulse dropouts 0% → 50%).
4. **Edge-Case Stress Testing:**
   - Emitter Pattern Shift: Emitter suddenly changes hopping pattern mid-episode to confirm fallback trigger successfully reverts to Tier 1 and recovers $P_{int}$.
5. **Dashboard Deliverables Hand-off to Person 6:**
   - Generates formatted CSV, JSON, plots, and markdown report in `person5/outputs/`.

---

## 💡 Why We Have to Do It

1. **Unbiased Baseline:** Each persona created their algorithm. Person 5 provides an **independent, standardized benchmark** across all algorithms on identical seeds.
2. **Reproducibility:** 100 Monte Carlo episodes with fixed seeds ensure that SIH judges see real, trustworthy, reproducible numbers.
3. **Real-World EW Robustness:** Low-SNR dropouts and agile emitter shifts prove the system handles real-world RF interference and adversary countermeasures.
4. **Dashboard Integration (Person 6):** Person 6's dashboard requires actual pre-computed JSON/CSV data and high-resolution plots to populate live KPI panels during the presentation demo.
