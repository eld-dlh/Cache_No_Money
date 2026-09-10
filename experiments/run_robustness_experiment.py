"""
experiments/run_robustness_experiment.py
========================================
Robustness evaluation of Tier-1 Bandit Algorithms.
Tests whether the high SW-UCB performance holds across 
multiple random windows in the TEST dataset.
"""

from __future__ import annotations

import sys
import time
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

# ── Ensure stdout handles Unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.parse_pdw import parse_h5_file
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
_radar_env_mod.TRAIN_RATIO = 1.0  # Force it to use 100% of the given temp NPY

from bandits import EpsilonGreedy, UCB1, SlidingWindowUCB

MAX_STEPS = 15000
SEEDS = [42, 123, 456, 789, 999]  # 5 different starting windows

def run_algorithm(name, agent_factory, npy_path: Path, seed: int, algo_index: int) -> dict:
    import random
    random.seed(seed * 31 + algo_index)
    
    env = RadarEnv(npy_path, split="train", seed=seed + algo_index, max_steps=MAX_STEPS)
    K = env.n_channels
    agent = agent_factory(K) if agent_factory is not None else None
    
    global_step = 0
    interceptions = 0
    pulse_channels = []
    
    obs, info = env.reset(seed=seed)
    term, trunc = False, False
    
    t_start = time.perf_counter()
    
    while not (term or trunc):
        if agent is not None:
            action = agent.select_action()
        elif name == "Random":
            action = env.action_space.sample()
        elif name == "Sequential":
            action = global_step % K
            
        obs, _reward, term, trunc, info = env.step(action)
        
        b_reward = 1.0 if info["intercepted"] else 0.0
        
        if agent is not None:
            agent.update(action, b_reward)
            
        global_step += 1
        interceptions += int(info["intercepted"])
        pulse_channels.append(info["pulse_channel"])

    t_end = time.perf_counter()
    env.close()
    
    # Ground truth analysis
    dist = np.bincount(pulse_channels, minlength=K)
    top_ch = int(np.argmax(dist))
    top_ch_prop = float(dist[top_ch]) / global_step if global_step > 0 else 0.0
    
    return {
        "window_seed": seed,
        "algorithm": name,
        "steps": global_step,
        "hits": interceptions,
        "hit_rate": interceptions / global_step if global_step > 0 else 0.0,
        "time_sec": t_end - t_start,
        "gt_top_channel": top_ch,
        "gt_top_prop": top_ch_prop
    }

def prepare_test_binary():
    base_dir = PROJECT_ROOT / "data" / "raw" / "scan" / "test_scan"
    h5_files = list(base_dir.glob("*.h5"))
    
    all_dfs = []
    for i, h5_path in enumerate(sorted(h5_files)):
        df = parse_h5_file(h5_path, train_id=i)
        if df is not None:
            all_dfs.append(df)
            
    combined = pd.concat(all_dfs, ignore_index=True)
    parquet_path = PROJECT_ROOT / "data" / "raw" / "temp_test.parquet"
    npy_path = PROJECT_ROOT / "data" / "raw" / "temp_test.npy"
    
    combined.to_parquet(parquet_path, index=False)
    convert_to_binary(pdw_parquet=parquet_path, output_path=npy_path)
    return npy_path

def main():
    npy_test = prepare_test_binary()
        
    algorithms = [
        ("Random",         None),
        ("Sequential",     None),
        ("Epsilon-Greedy", lambda k: EpsilonGreedy(k=k, epsilon=0.1)),
        ("UCB1",           lambda k: UCB1(k=k)),
        ("SW-UCB",         lambda k: SlidingWindowUCB(k=k, window_size=50)),
    ]
    
    print("=" * 60)
    print("ROBUSTNESS EXPERIMENT")
    print("=" * 60)
    print(f"Evaluating {len(algorithms)} algorithms across {len(SEEDS)} different random windows in the TEST set.")
    
    all_results = []
    
    for seed in SEEDS:
        print(f"\n--- Running Window (Seed: {seed}) ---")
        for algo_idx, (algo_name, factory) in enumerate(algorithms):
            res = run_algorithm(algo_name, factory, npy_test, seed, algo_idx)
            all_results.append(res)
            print(f"  {algo_name:<15}: {res['hit_rate']*100:>6.2f}% hits (Top True Ch {res['gt_top_channel']} @ {res['gt_top_prop']*100:.1f}%)")
            
    df = pd.DataFrame(all_results)
    
    # Calculate aggregation
    agg = df.groupby("algorithm")["hit_rate"].agg(['mean', 'std', 'min', 'max']).reset_index()
    # Sort by mean descending
    agg = agg.sort_values("mean", ascending=False)
    
    # Save CSV
    csv_path = PROJECT_ROOT / "experiments" / "results_robustness.csv"
    df.to_csv(csv_path, index=False)
    
    # Save text report
    report_path = PROJECT_ROOT / "experiments" / "robustness_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ROBUSTNESS EXPERIMENT RESULTS\n")
        f.write("=============================\n\n")
        f.write("1. Results by Window\n")
        f.write("--------------------\n")
        for seed in SEEDS:
            f.write(f"\nWindow Seed: {seed}\n")
            seed_df = df[df["window_seed"] == seed]
            gt_ch = seed_df.iloc[0]["gt_top_channel"]
            gt_prop = seed_df.iloc[0]["gt_top_prop"]
            f.write(f"Ground Truth: Top Channel {gt_ch} contained {gt_prop*100:.1f}% of pulses.\n")
            for _, row in seed_df.iterrows():
                f.write(f"  {row['algorithm']:<15}: {row['hit_rate']*100:>6.2f}% hits\n")
                
        f.write("\n\n2. Aggregate Performance (Mean ± Std Dev)\n")
        f.write("-----------------------------------------\n")
        for _, row in agg.iterrows():
            f.write(f"  {row['algorithm']:<15}: {row['mean']*100:>6.2f}% ± {row['std']*100:>5.2f}%  (Range: {row['min']*100:>5.1f}% - {row['max']*100:>5.1f}%)\n")
            
    print(f"\n✅ Experiment complete. Saved to {csv_path.name} and {report_path.name}.")

if __name__ == "__main__":
    main()
