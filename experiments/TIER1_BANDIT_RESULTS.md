# Tier-1 Adaptive Bandit Results

## 1. Objective
The primary objective of this Tier-1 benchmark is to evaluate **adaptive RF channel selection under limited sequential scanning**. The algorithms must operate as a cognitive radar receiver making sequential choices among multiple discrete frequency channels to maximize pulse interceptions in a highly dense and non-stationary environment.

## 2. Algorithms
The following five discrete multi-armed bandit (MAB) algorithms were evaluated:
- **Random:** A non-learning baseline that selects a channel uniformly at random.
- **Sequential:** A non-learning baseline that cycles through channels `0, 1, 2, ..., 63` repeatedly.
- **Epsilon-Greedy (ε=0.1):** A learning bandit that exploits the best-known channel 90% of the time and explores uniformly 10% of the time.
- **UCB1:** An upper-confidence-bound algorithm that deterministically balances exploration and exploitation using historical averages.
- **Sliding-Window UCB (window=50):** A non-stationary UCB variant that forgets historical observations older than 50 steps, allowing rapid adaptation to shifting spectrum usage.

## 3. Environment
- **Channels:** 64 discrete frequency bins.
- **Spectrum Range:** 0 – 18,000 MHz.
- **Channel Width:** 281.25 MHz per channel.
- **State Space:** PDW observation window = `10 × 5` fields (10 past pulses).
- **Reward Signal:** Strictly binary interception reward (`1.0` if `intercepted` else `0.0`). 
- **Temporal Structure:** Pulses are strictly chronologically ordered by Time of Arrival (ToA).

*Note: The shaped environmental reward from RadarEnv and the ground-truth `pulse_channel` were explicitly withheld from the bandit update functions to prevent information leakage.*

## 4. Dataset
The benchmark uses a representative sample of the **real Alan Turing Institute Synthetic Radar Dataset**. 
- **Characterization Sample Size:** 12 HDF5 files (`scan` mode).
- **Files per Partition:** 5 `train_scan` files, 3 `val_scan` files, 4 `test_scan` files.
- **Total Pulses in Characterization:** Approximately 950,000 pulses.
- **Emitter Diversity:** 88 unique emitter IDs across the 12-file sample.

*Important:* The 12-file dataset represents the global characterization pool. The benchmarks below operate on focused 15,000-step continuous execution windows sampled from these isolated pools.

## 5. Initial Benchmark
The initial benchmark evaluated a single 15,000-step window from the start of each dataset partition.

**Test Split Results:**
| Algorithm            | Steps    | Hits   | Hit Rate   | Avg Reward |
|----------------------|----------|--------|------------|------------|
| Random               | 15,000   | 235    | 1.567%     | 0.0157     |
| Sequential           | 15,000   | 235    | 1.567%     | 0.0157     |
| Epsilon-Greedy       | 15,000   | 6,529  | 43.527%    | 0.4353     |
| UCB1                 | 15,000   | 9,063  | 60.420%    | 0.6042     |
| SW-UCB               | 15,000   | 10,473 | 69.820%    | 0.6982     |

## 6. Robustness Evaluation
To ensure the ~70% SW-UCB result was not a local artifact, the test split was evaluated across 5 different randomized starting windows (15,000 steps each).

**Results by Window (Hits %):**
- **Window 42:** Random: 1.62%, Sequential: 1.57%, ε-Greedy: 43.53%, UCB1: 60.42%, SW-UCB: 69.82%
- **Window 123:** Random: 1.52%, Sequential: 1.56%, ε-Greedy: 47.11%, UCB1: 57.12%, SW-UCB: 61.74%
- **Window 456:** Random: 1.57%, Sequential: 1.43%, ε-Greedy: 13.71%, UCB1: 36.25%, SW-UCB: 46.18%
- **Window 789:** Random: 1.50%, Sequential: 1.68%, ε-Greedy: 16.11%, UCB1: 30.13%, SW-UCB: 46.98%
- **Window 999:** Random: 1.51%, Sequential: 1.54%, ε-Greedy: 22.34%, UCB1: 40.61%, SW-UCB: 55.16%

**Aggregate Performance (Mean ± Standard Deviation):**
- **SW-UCB:** `55.98% ± 10.03%` (Range: 46.2% - 69.8%)
- **UCB1:** `44.90% ± 13.25%` (Range: 30.1% - 60.4%)
- **Epsilon-Greedy:** `28.56% ± 15.67%` (Range: 13.7% - 47.1%)
- **Sequential:** `1.56% ± 0.09%` (Range: 1.4% - 1.7%)
- **Random:** `1.54% ± 0.05%` (Range: 1.5% - 1.6%)

## 7. Non-Stationarity
We examined ground-truth temporal channel distributions across sequential windows to verify non-stationarity. In the 15,000-step training slice (divided into four 3,750-step chunks), there were 3 unique dominant channels (Channel 14, Channel 30, and Channel 33). This confirms that temporal channel distributions fundamentally shift across sequential windows as emitters turn on and off, changing the mathematically optimal channel over time.

## 8. Interpretation
SW-UCB drastically outperforms UCB1 and Epsilon-Greedy because of its recent-history weighting. While standard UCB1 drags the dead weight of historical channel averages, SW-UCB forgets stale observations older than 50 steps. This gives it the unique ability to successfully exploit persistent active channels, rapidly abandon them when they turn off, and adapt to changing channel distributions.

*Note: The extraordinarily high absolute interception rate (up to 70%) is specific to this simulation/environment constraint, because `RadarEnv` artificially models exactly one active pulse per discrete decision step.*

## 9. Limitations
- **Simulation Density:** `RadarEnv` assumes one pulse per environment step, which eliminates empty-air periods and theoretically bounds the maximum hit rate to 100%. 
- **Sample Size:** The characterization was constrained to a limited 12-file sample rather than the full multi-billion pulse dataset.
- **Development Focus:** This benchmark is a software development experiment for algorithmic integration, not a real RF hardware measurement. The results should not be presented as real-world SDR interception performance.
- **Frequency Calibrations:** The 18 GHz bound is a calibrated design boundary based on the observed sample and project handoff requirements; it is not a claim about the absolute theoretical maximum of the entire Turing dataset.

## 10. Person-2 Deliverable
The following Tier-1 discrete bandit components have been fully implemented, validated, and integrated with the environment:
- `Sequential` (Environment cycling baseline)
- `EpsilonGreedy` 
- `UCB1`
- `SlidingWindowUCB` 

**Clean Interface Configuration:**
The bandit algorithms strictly conform to the expected interface required for `RadarEnv`'s 64-channel action space. The current classes explicitly provide a clean `select_action()` and `update(action, reward)` interface. This satisfies the integration requirements for Person 4 without requiring architectural redesigns or wrapper abstractions.
