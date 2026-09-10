"""
experiments/run_real_bandit_benchmark.py
========================================
Tier-1 Bandit Benchmark on the Real Alan Turing Dataset Sample.

This script:
1. Parses train, val, and test HDF5 files into isolated temporary binaries.
2. Evaluates the bandits (Random, Sequential, Epsilon-Greedy, UCB1, SW-UCB)
   on each continuous dataset sequence to evaluate non-stationary tracking.
3. Generates performance metrics, CSV reports, and plots.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure stdout handles Unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import Project Modules ───────────────────────────────────────────────────
from data.parse_pdw import parse_h5_file, PDW_COLS
from data.convert_to_binary import convert_to_binary

def _load_env_module(name: str, rel_path: str):
    full_path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.modules["env"] = sys.modules.get("env") or mod
    spec.loader.exec_module(mod)
    return mod

_memmap_mod   = _load_env_module("env.memmap_loader", "env/memmap_loader.py")
_radar_env_mod = _load_env_module("env.radar_env",    "env/radar_env.py")

RadarEnv   = _radar_env_mod.RadarEnv
N_CHANNELS = _radar_env_mod.N_CHANNELS

from bandits import EpsilonGreedy, UCB1, SlidingWindowUCB

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS_PER_SPLIT = 15000  # Cap execution steps per split to save time
ROLLING_WINDOW = 500         # For temporal analysis

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def prepare_split_binary(split_name: str, h5_files: list[Path]) -> Path | None:
    """Combines HDF5 files into a temporary Parquet, then to a temporary NPY."""
    if not h5_files:
        return None
        
    print(f"\n[Data] Preparing {split_name} binary from {len(h5_files)} files...")
    all_dfs = []
    
    # Process files sequentially based on filename to maintain temporal consistency 
    # across pulse trains (within files, pulses are strictly sequential).
    for i, h5_path in enumerate(sorted(h5_files)):
        df = parse_h5_file(h5_path, train_id=i)
        if df is not None:
            all_dfs.append(df)
            
    if not all_dfs:
        return None
        
    combined = pd.concat(all_dfs, ignore_index=True)
    parquet_path = PROJECT_ROOT / "data" / "raw" / f"temp_{split_name}.parquet"
    npy_path = PROJECT_ROOT / "data" / "raw" / f"temp_{split_name}.npy"
    
    combined.to_parquet(parquet_path, index=False)
    # Convert to binary
    convert_to_binary(pdw_parquet=parquet_path, output_path=npy_path)
    
    print(f"       -> Created {npy_path.name} with {len(combined)} pulses.")
    return npy_path

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def run_algorithm(
    name: str,
    agent_factory,
    npy_path: Path,
    seed: int,
    algo_index: int = 0
) -> dict:
    
    import random
    random.seed(seed * 31 + algo_index)
    
    # Initialize environment with max_steps
    # split="train" tells it to use 100% of the file (since it's an isolated binary)
    # Actually wait: split="train" will use first 70%. But wait!
    # Because there are very few train_ids, `RadarEnv` uses TRAIN_RATIO=0.7 by default.
    # We must patch TRAIN_RATIO internally to 1.0 just for this isolated run, or rely on index fallback.
    # Actually, the simplest fix is to patch it just for the execution.
    _radar_env_mod.TRAIN_RATIO = 1.0 
    
    env = RadarEnv(npy_path, split="train", seed=seed + algo_index, max_steps=MAX_STEPS_PER_SPLIT)
    K = env.n_channels
    agent = agent_factory(K) if agent_factory is not None else None
    
    global_step = 0
    interceptions = 0
    bandit_reward_total = 0.0
    
    # Trackers for analysis
    rolling_hits = []
    pulse_channel_history = []
    chosen_channel_history = []
    hit_history = []
    
    obs, info = env.reset(seed=seed)
    term, trunc = False, False
    
    t_start = time.perf_counter()
    
    while not (term or trunc):
        # Action selection
        if agent is not None:
            action = agent.select_action()
        elif name == "Random":
            action = env.action_space.sample()
        elif name == "Sequential":
            action = global_step % K
            
        # Step
        obs, _reward, term, trunc, info = env.step(action)
        
        # Bandit Reward
        b_reward = 1.0 if info["intercepted"] else 0.0
        
        # Update agent
        if agent is not None:
            agent.update(action, b_reward)
            
        # Track metrics
        global_step += 1
        interceptions += int(info["intercepted"])
        bandit_reward_total += b_reward
        
        pulse_channel_history.append(info["pulse_channel"])
        chosen_channel_history.append(info["chosen_channel"])
        hit_history.append(int(info["intercepted"]))
        
        if global_step % ROLLING_WINDOW == 0:
            window_hits = sum(hit_history[-ROLLING_WINDOW:])
            rolling_hits.append(window_hits / ROLLING_WINDOW)

    t_end = time.perf_counter()
    env.close()
    
    return {
        "name": name,
        "steps": global_step,
        "interceptions": interceptions,
        "hit_rate": interceptions / global_step if global_step > 0 else 0.0,
        "avg_reward": bandit_reward_total / global_step if global_step > 0 else 0.0,
        "time_sec": t_end - t_start,
        "rolling_hits": rolling_hits,
        "pulse_history": np.array(pulse_channel_history),
        "chosen_history": np.array(chosen_channel_history),
        "hit_history": np.array(hit_history)
    }

# ─────────────────────────────────────────────────────────────────────────────
# NON-STATIONARITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_non_stationarity(pulse_history: np.ndarray, split_name: str, K: int=64) -> dict:
    """Analyze the ground truth pulse channels across time windows to detect non-stationarity."""
    n_windows = 4
    window_size = len(pulse_history) // n_windows
    
    if window_size == 0:
        return {"stationary": True, "reason": "Not enough data"}
        
    window_dists = []
    top_channels = []
    
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else len(pulse_history)
        chunk = pulse_history[start:end]
        
        dist = np.bincount(chunk, minlength=K)
        window_dists.append(dist)
        top_ch = np.argmax(dist)
        top_channels.append(int(top_ch))
        
    unique_top = len(set(top_channels))
    is_non_stationary = unique_top > 1
    
    return {
        "is_non_stationary": is_non_stationary,
        "unique_top_channels_across_windows": unique_top,
        "top_channels_sequence": top_channels,
        "reason": f"Top channel changed {unique_top} times across {n_windows} windows: {top_channels}" if is_non_stationary else f"Top channel {top_channels[0]} remained dominant across all {n_windows} windows."
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    base_dir = PROJECT_ROOT / "data" / "raw" / "scan"
    
    h5_train = list((base_dir / "train_scan").glob("*.h5"))
    h5_val   = list((base_dir / "val_scan").glob("*.h5"))
    h5_test  = list((base_dir / "test_scan").glob("*.h5"))
    
    print(f"Discovered {len(h5_train)} train, {len(h5_val)} val, {len(h5_test)} test HDF5 files.")
    
    binaries = {}
    for name, files in [("train", h5_train), ("val", h5_val), ("test", h5_test)]:
        if files:
            binaries[name] = prepare_split_binary(name, files)
            
    algorithms = [
        ("Random",         None),
        ("Sequential",     None),
        ("Epsilon-Greedy", lambda k: EpsilonGreedy(k=k, epsilon=0.1)),
        ("UCB1",           lambda k: UCB1(k=k)),
        ("SW-UCB",         lambda k: SlidingWindowUCB(k=k, window_size=50)),
    ]
    
    all_results = []
    
    for split_name, npy_path in binaries.items():
        if not npy_path:
            continue
            
        print(f"\n{'='*60}")
        print(f"BENCHMARKING SPLIT: {split_name.upper()}")
        print(f"{'='*60}")
        
        split_results = []
        ground_truth_history = None
        
        for algo_idx, (algo_name, factory) in enumerate(algorithms):
            print(f"  Running {algo_name}...")
            res = run_algorithm(algo_name, factory, npy_path, args.seed, algo_idx)
            split_results.append(res)
            
            # Save ground truth (it's the same for all algos on the same split/seed)
            if ground_truth_history is None:
                ground_truth_history = res["pulse_history"]
                
        # Non-stationarity check
        ns_stats = analyze_non_stationarity(ground_truth_history, split_name)
        print(f"\n  Non-stationarity Check:")
        print(f"    {ns_stats['reason']}")
        
        # Table
        print(f"\n  Results for {split_name.upper()}:")
        print(f"  {'Algorithm':<20} | {'Steps':>8} | {'Hits':>8} | {'Hit Rate':>10}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
        for r in split_results:
            print(f"  {r['name']:<20} | {r['steps']:>8,} | {r['interceptions']:>8,} | {r['hit_rate']*100:>9.3f}%")
            
            # Save tabular results
            all_results.append({
                "split": split_name,
                "algorithm": r['name'],
                "steps": r['steps'],
                "hits": r['interceptions'],
                "hit_rate": r['hit_rate'],
                "avg_reward": r['avg_reward'],
                "time_sec": r['time_sec']
            })
            

        
    # Save CSV
    df_res = pd.DataFrame(all_results)
    csv_path = PROJECT_ROOT / "experiments" / "results_real_benchmark.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n✅ Benchmark complete. Saved full results to {csv_path.name}")
    
    # Cleanup temp files
    print("Cleaning up temporary binary files...")
    for split_name in ["train", "val", "test"]:
        pq = PROJECT_ROOT / "data" / "raw" / f"temp_{split_name}.parquet"
        npy = PROJECT_ROOT / "data" / "raw" / f"temp_{split_name}.npy"
        if pq.exists(): pq.unlink()
        if npy.exists(): npy.unlink()

if __name__ == "__main__":
    main()
