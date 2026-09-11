"""
experiments/run_bandit_comparison.py
=====================================
Tier-1 Bandit Evaluation — Comparing all bandit algorithms on RadarEnv.

Evaluates five policies:
    1. Random         (non-learning baseline)
    2. Sequential     (non-learning baseline)
    3. Epsilon-Greedy (epsilon = 0.1)
    4. UCB1
    5. Sliding-Window UCB (window_size = 50)

Reward used for bandit learning:
    bandit_reward = 1.0 if info["intercepted"] else 0.0

The shaped env reward and info["pulse_channel"] are NEVER passed into
agent.update(). They are used only for evaluation reporting.

Usage:
    python experiments/run_bandit_comparison.py
    python experiments/run_bandit_comparison.py --episodes 5
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

# ── Ensure stdout handles Unicode (emoji in RadarEnv print statements)
#    on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Project root on sys.path (supports running from any directory)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Pre-register env sub-modules to avoid env/__init__.py
#    env/__init__.py imports PDWDataset -> torch (not installed in bandit env).
#    By loading memmap_loader and radar_env directly via importlib and injecting
#    them into sys.modules under their fully-qualified names, Python's import
#    machinery never falls back to executing env/__init__.py.
def _load_env_module(name: str, rel_path: str):
    """Load a single .py file and register it under sys.modules[name]."""
    full_path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register BEFORE exec so internal imports resolve
    sys.modules["env"] = sys.modules.get("env") or mod   # satisfy 'env' package lookup
    spec.loader.exec_module(mod)
    return mod

_memmap_mod   = _load_env_module("env.memmap_loader", "env/memmap_loader.py")
_radar_env_mod = _load_env_module("env.radar_env",    "env/radar_env.py")

RadarEnv   = _radar_env_mod.RadarEnv
N_CHANNELS = _radar_env_mod.N_CHANNELS

from bandits import EpsilonGreedy, UCB1, SlidingWindowUCB



# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Statistics for a single episode."""
    steps:         int   = 0
    interceptions: int   = 0
    bandit_reward: float = 0.0

    @property
    def interception_rate(self) -> float:
        return self.interceptions / self.steps if self.steps > 0 else 0.0

    @property
    def avg_bandit_reward(self) -> float:
        return self.bandit_reward / self.steps if self.steps > 0 else 0.0


@dataclass
class AlgorithmResult:
    """Aggregated statistics across all episodes for one algorithm."""
    name:            str
    episodes:        list[EpisodeResult] = field(default_factory=list)
    elapsed_sec:     float = 0.0

    @property
    def total_steps(self) -> int:
        return sum(e.steps for e in self.episodes)

    @property
    def total_interceptions(self) -> int:
        return sum(e.interceptions for e in self.episodes)

    @property
    def overall_interception_rate(self) -> float:
        return self.total_interceptions / self.total_steps if self.total_steps > 0 else 0.0

    @property
    def avg_bandit_reward(self) -> float:
        total_r = sum(e.bandit_reward for e in self.episodes)
        return total_r / self.total_steps if self.total_steps > 0 else 0.0

    @property
    def avg_episode_interception_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.interception_rate for e in self.episodes) / len(self.episodes)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: run a single algorithm
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm(
    name:          str,
    agent_factory: Callable | None,   # returns a fresh agent; None for no-learning baselines
    n_episodes:    int,
    npy_path:      Path,
    seed:          int,
    algo_index:    int = 0,           # unique index per algorithm (used to derive seeds)
) -> AlgorithmResult:
    """
    Run one algorithm for n_episodes on a fresh RadarEnv and collect results.

    Parameters
    ----------
    name          : Display name of the algorithm.
    agent_factory : Callable that returns a fresh agent (k) -> agent.
                    Pass None for Random and Sequential (no agent needed).
    n_episodes    : Number of episodes to run.
    npy_path      : Path to the pdw_records.npy binary.
    seed          : Base RNG seed for the environment.
    """
    # ── Seed Python's global random module
    #    All three bandit classes (EpsilonGreedy, UCB1, SlidingWindowUCB) use
    #    Python's global `random` module directly and have no seed parameter.
    #    Seeding here — before the agent and env are created — makes every
    #    exploration decision fully reproducible between runs.
    #    We derive a per-algorithm seed so algorithms don't share the same
    #    random sequence despite starting from the same base seed.
    import random as _random
    _random.seed(seed * 31 + algo_index)

    result = AlgorithmResult(name=name)
    t_start = time.perf_counter()

    # Each algorithm gets its own fresh environment, seeded identically each run
    env = RadarEnv(npy_path, split="train", seed=seed + algo_index)
    K   = env.n_channels                        # 64 (or whatever is configured)

    # Fresh agent instance — created once per algorithm (NOT per episode)
    # Not resetting between episodes intentionally: the agent accumulates
    # channel knowledge across episodes (desirable for a non-stationary spectrum)
    agent = agent_factory(K) if agent_factory is not None else None

    # Track episode-level sequential step counter (for Sequential baseline)
    global_step = 0

    for ep_idx in range(n_episodes):
        # Pass a deterministic per-episode seed to env.reset() so the starting
        # cursor is reproducible across runs (not just within a single run).
        ep_seed = seed + algo_index * 1000 + ep_idx
        obs, _reset_info = env.reset(seed=ep_seed)
        terminated = False
        truncated  = False
        ep = EpisodeResult()

        while not (terminated or truncated):
            # ── Action selection ───────────────────────────────────────────
            if agent is not None:
                # Learning algorithm
                action = agent.select_action()
            elif name == "Random":
                action = env.action_space.sample()
            elif name == "Sequential":
                action = global_step % K
            else:
                raise ValueError(f"Unknown non-learning algorithm: {name!r}")

            # ── Environment step ───────────────────────────────────────────
            obs, _shaped_reward, terminated, truncated, info = env.step(action)

            # ── Bandit training reward — ONLY binary intercepted/not ───────
            bandit_reward = 1.0 if info["intercepted"] else 0.0

            # ── Agent update (learning algorithms only) ────────────────────
            if agent is not None:
                agent.update(action, bandit_reward)

            # ── Statistics ────────────────────────────────────────────────
            # pulse_channel is collected ONLY for evaluation, never for training
            ep.steps         += 1
            ep.interceptions += int(info["intercepted"])
            ep.bandit_reward += bandit_reward
            global_step      += 1

        result.episodes.append(ep)

    env.close()
    result.elapsed_sec = time.perf_counter() - t_start
    return result


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION: verify an AlgorithmResult before printing
# ─────────────────────────────────────────────────────────────────────────────

def validate_result(result: AlgorithmResult, K: int, n_episodes: int):
    """Lightweight sanity checks on a completed run."""
    assert len(result.episodes) == n_episodes, \
        f"{result.name}: expected {n_episodes} episodes, got {len(result.episodes)}"
    assert result.total_steps > 0, f"{result.name}: ran 0 steps"
    assert 0.0 <= result.overall_interception_rate <= 1.0, \
        f"{result.name}: interception rate out of [0,1]"


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[AlgorithmResult], K: int, n_episodes: int):
    """Print a clean, aligned comparison table."""
    random_theoretical = 1.0 / K

    sep  = "=" * 78
    dash = "-" * 78

    print(f"\n{sep}")
    print(f"  Bandit Comparison on RadarEnv")
    print(f"{sep}")
    print(f"  K (channels) = {K}")
    print(f"  Episodes     = {n_episodes}")
    print(f"  Theoretical random hit rate = 1/{K} = {random_theoretical*100:.4f}%")
    print(f"{dash}")
    print(f"  {'Algorithm':<22} {'Steps':>8} {'Hits':>7} {'Hit Rate':>10} {'Avg Reward':>12} {'Time(s)':>8}")
    print(f"{dash}")

    best_rate = -1.0
    best_name = ""

    for r in results:
        rate = r.overall_interception_rate
        print(f"  {r.name:<22} {r.total_steps:>8,} {r.total_interceptions:>7,} "
              f"{rate*100:>9.3f}% {r.avg_bandit_reward:>12.4f} {r.elapsed_sec:>7.1f}s")
        if rate > best_rate:
            best_rate = rate
            best_name = r.name

    print(f"{dash}")
    print(f"\n  Random theoretical hit rate : {random_theoretical*100:.4f}%")
    print(f"  Best algorithm              : {best_name}  ({best_rate*100:.3f}%)")

    # Compare each learner against random baseline
    print(f"\n  Lift over theoretical random:")
    for r in results:
        lift = r.overall_interception_rate - random_theoretical
        symbol = "+" if lift >= 0 else ""
        print(f"    {r.name:<22}  {symbol}{lift*100:.3f}%")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier-1 Bandit Algorithm Comparison on RadarEnv")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run per algorithm (default: 3)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Base RNG seed (default: 42)")
    parser.add_argument("--npy",      type=str, default=None,
                        help="Path to pdw_records.npy (default: data/raw/pdw_records.npy)")
    args = parser.parse_args()

    npy_path = Path(args.npy) if args.npy else PROJECT_ROOT / "data" / "raw" / "pdw_records.npy"

    # ── Pre-flight check ─────────────────────────────────────────────────────
    if not npy_path.exists():
        print(f"\n[ERROR] Binary dataset not found: {npy_path}")
        print(f"\nPlease run the data pipeline first:")
        print(f"  1. python data/download_dataset.py")
        print(f"  2. python data/parse_pdw.py")
        print(f"  3. python data/convert_to_binary.py")
        print(f"\nThen re-run this experiment.")
        sys.exit(1)

    n_episodes = args.episodes
    seed       = args.seed

    # ── Algorithm registry ───────────────────────────────────────────────────
    # Format: (display_name, agent_factory or None)
    # agent_factory receives K (int) and returns a fresh agent
    algorithms = [
        ("Random",         None),
        ("Sequential",     None),
        ("Epsilon-Greedy", lambda k: EpsilonGreedy(k=k, epsilon=0.1)),
        ("UCB1",           lambda k: UCB1(k=k)),
        ("SW-UCB",         lambda k: SlidingWindowUCB(k=k, window_size=50)),
    ]

    print(f"\nStarting Tier-1 Bandit Comparison")
    print(f"  Binary   : {npy_path}")
    print(f"  Episodes : {n_episodes} per algorithm")
    print(f"  Seed     : {seed}")
    print(f"  Algorithms: {[name for name, _ in algorithms]}\n")

    # ── Run all algorithms ───────────────────────────────────────────────────
    results = []
    K = None   # will be confirmed from the first env created

    for algo_idx, (name, factory) in enumerate(algorithms):
        print(f"  Running {name}...", flush=True)
        result = run_algorithm(
            name          = name,
            agent_factory = factory,
            n_episodes    = n_episodes,
            npy_path      = npy_path,
            seed          = seed,
            algo_index    = algo_idx,
        )
        # Confirm K from N_CHANNELS constant (same across all runs)
        if K is None:
            K = N_CHANNELS

        validate_result(result, K, n_episodes)
        results.append(result)
        print(f"    Done — {result.total_steps:,} steps, "
              f"hit rate {result.overall_interception_rate*100:.3f}%")

    # ── Print final comparison table ─────────────────────────────────────────
    print_comparison_table(results, K=K, n_episodes=n_episodes)


if __name__ == "__main__":
    main()
