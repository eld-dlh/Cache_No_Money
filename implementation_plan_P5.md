# Implementation Plan — Person 5 (Evaluation Engineer)

Design and execute a comprehensive, reproducible evaluation suite for the Cognitive Radar Interception system. Evaluate all policies (**Sequential Sweeper Baseline**, **Sliding-Window UCB**, and **Hybrid DRQN Cognitive Interceptor**) across standard, low-SNR (noise-injected), and held-out data splits.

## User Review Required

> [!IMPORTANT]
> - All evaluations will be strictly non-destructive to existing Person 1–4 modules.
> - Evaluation scripts will produce structured JSON, CSV, plots, and markdown reports directly consumable by **Person 6 (Dashboard Engineer)**.
> - Monte Carlo simulations will use fixed random seeds ($N=100$ episodes per policy) to ensure 100% reproducible metrics.

## Proposed Changes

### Evaluation Engine & Experiments Component

#### [NEW] [eval_harness.py](file:///c:/SIH/eval_harness.py)
Core Monte Carlo evaluation harness class (`MonteCarloEvaluator`):
- Runs specified policies for $N=100$ episodes with deterministic seeds.
- Collects per-step & per-episode telemetry: Interception rate ($P_{int}$), Wasted Dwells, Retune Costs, and Cumulative Reward.
- Supports pluggable noise models (pulse dropouts, Gaussian frequency jitter, low SNR).

#### [NEW] [experiments/run_person5_eval_suite.py](file:///c:/SIH/experiments/run_person5_eval_suite.py)
Master evaluation runner script:
1. Runs 100-episode Monte Carlo evaluation across Sweeper, SW-UCB, and DRQN.
2. Performs 70/30 held-out dataset split evaluation for DRQN generalization check.
3. Evaluates low-SNR / noise-injected robustness (0% to 50% pulse dropout).
4. Evaluates agile emitter pattern shift stress test (fallback trigger recovery).
5. Calculates relative improvement percentage:
   $$\text{Relative Improvement (\%)} = \frac{P_{int, \text{Adaptive}} - P_{int, \text{Baseline}}}{P_{int, \text{Baseline}}} \times 100$$
6. Exports `evaluation_results.csv`, `evaluation_summary.json`, `eval_comparison.png`, and `eval_robustness.png`.

#### [NEW] [experiments/PERSON5_EVALUATION_REPORT.md](file:///c:/SIH/experiments/PERSON5_EVALUATION_REPORT.md)
Executive evaluation summary with concise numbers and 2-3 sentence interpretations per experiment for Person 6's dashboard & report.

---

### Verification & Testing Component

#### [NEW] [tests/test_evaluation_harness.py](file:///c:/SIH/tests/test_evaluation_harness.py)
Pytest test suite validating:
- Deterministic reproducible seed execution.
- Metrics calculation accuracy ($P_{int}$, relative improvement %).
- JSON/CSV export formatting.
- Correct integration with `RadarEnv`, `SW-UCB`, and `CognitiveInterceptor`.

## Verification Plan

### Automated Tests
- Run `pytest tests/test_evaluation_harness.py -v` to ensure evaluation harness logic is bug-free.
- Run `python experiments/run_person5_eval_suite.py` to execute full Monte Carlo benchmarks.

### Manual Verification
- Verify generated `evaluation_results.csv` and `evaluation_summary.json` contain accurate, complete numbers.
- Confirm high-resolution comparison plots are saved and formatted clearly.
