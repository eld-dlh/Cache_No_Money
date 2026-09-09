"""
Step 10 — Random Agent Test: 100-Step Sanity Check
====================================================

This script runs a completely random agent through the RadarEnv for 100 steps.

What we're checking:
  ✅ Environment doesn't crash
  ✅ Observations have correct shape and dtype
  ✅ Rewards are finite (not NaN, not ±inf)
  ✅ Reward range makes sense (not all 0.0, not implausibly large)
  ✅ Episode resets work correctly
  ✅ Train/test splits are disjoint (no data leakage)
  ✅ Info dict contains expected keys

A "random agent" just calls env.action_space.sample() — it picks a
completely random frequency channel every step. This is our baseline:
anything smarter (bandit, DRL) should beat this.
"""

import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.radar_env import RadarEnv, WINDOW_SIZE, N_CHANNELS

DEFAULT_NPY = Path(__file__).parent.parent / "data" / "raw" / "pdw_records.npy"

# ─────────────────────────────────────────────────────────────────────────────

def run_random_agent(
    npy_path:  Path = DEFAULT_NPY,
    n_steps:   int  = 100,
    seed:      int  = 42,
    verbose:   bool = True,
) -> dict:
    """
    Run a random-action agent for n_steps and collect statistics.

    Returns:
        dict with keys: rewards, intercepts, misses, episodes, etc.
    """
    print(f"\n{'='*60}")
    print(f"RANDOM AGENT — {n_steps} STEPS")
    print(f"{'='*60}")

    env = RadarEnv(npy_path, split="train", seed=seed)
    obs, info = env.reset(seed=seed)

    stats = {
        "rewards":         [],
        "intercepted":     [],
        "pulse_channels":  [],
        "chosen_channels": [],
        "n_episodes":      1,
        "n_resets":        1,
    }

    print(f"\nRunning {n_steps} steps...")
    for step_i in range(n_steps):
        # Random action — choose a random channel
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # ── Validate each step
        assert obs.shape == (WINDOW_SIZE, 5), \
            f"Step {step_i}: obs shape {obs.shape} != ({WINDOW_SIZE}, 5)"
        assert obs.dtype == np.float32, \
            f"Step {step_i}: obs dtype {obs.dtype} != float32"
        assert np.isfinite(reward), \
            f"Step {step_i}: reward is not finite: {reward}"
        assert "intercepted"     in info, f"Step {step_i}: missing 'intercepted' in info"
        assert "pulse_channel"   in info, f"Step {step_i}: missing 'pulse_channel' in info"
        assert "chosen_channel"  in info, f"Step {step_i}: missing 'chosen_channel' in info"
        assert 0 <= info["pulse_channel"]  < N_CHANNELS
        assert 0 <= info["chosen_channel"] < N_CHANNELS

        # Store stats
        stats["rewards"].append(reward)
        stats["intercepted"].append(int(info["intercepted"]))
        stats["pulse_channels"].append(info["pulse_channel"])
        stats["chosen_channels"].append(info["chosen_channel"])

        if verbose and (step_i + 1) % 10 == 0:
            recent_rewards = stats["rewards"][-10:]
            recent_ints    = stats["intercepted"][-10:]
            print(f"  Step {step_i+1:3d}: "
                  f"avg_reward={np.mean(recent_rewards):+.4f}  "
                  f"intercepts_last10={sum(recent_ints)}/10  "
                  f"chose_ch={action}  pulse_ch={info['pulse_channel']}")

        # Handle episode end
        if terminated or truncated:
            stats["n_episodes"] += 1
            stats["n_resets"]   += 1
            obs, info = env.reset()

    env.close()
    return stats


def analyse_results(stats: dict, n_steps: int):
    """Print a clean summary of the random agent's performance."""
    rewards   = np.array(stats["rewards"])
    intercepts = np.array(stats["intercepted"])

    total_intercepts = intercepts.sum()
    total_pulses     = n_steps  # one pulse per step in this dataset
    p_int            = total_intercepts / total_pulses

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Steps run:         {n_steps}")
    print(f"  Episodes:          {stats['n_episodes']}")
    print(f"  Total intercepts:  {total_intercepts} / {total_pulses}")
    print(f"  P_int (random):    {p_int:.3f}  ({p_int*100:.1f}%)")
    print(f"")
    print(f"  Reward stats:")
    print(f"    Mean:    {rewards.mean():+.4f}")
    print(f"    Std:     {rewards.std():.4f}")
    print(f"    Min:     {rewards.min():+.4f}")
    print(f"    Max:     {rewards.max():+.4f}")
    print(f"    Total:   {rewards.sum():+.2f}")

    # Sanity checks
    print(f"\n{'─'*60}")
    print(f"SANITY CHECKS")
    print(f"{'─'*60}")

    checks = []
    checks.append(("All rewards finite",  np.all(np.isfinite(rewards))))
    checks.append(("Rewards not all zero", not np.all(rewards == 0.0)))
    checks.append(("Max reward <= R_INTERCEPT + 1.0",
                   rewards.max() <= 2.0))        # generous upper bound
    checks.append(("Min reward >= -5.0",
                   rewards.min() >= -5.0))       # generous lower bound
    checks.append(("P_int in [0, 1]",
                   0.0 <= p_int <= 1.0))
    checks.append(("Random P_int reasonable (> 0.001)",
                   p_int > 0.001))               # at least rare intercepts

    all_passed = True
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"✅ ALL 100 STEPS COMPLETED — ENVIRONMENT IS WORKING CORRECTLY")
        print(f"   Random baseline P_int = {p_int:.3f}")
        print(f"   (Persons 2 & 3: beat this with bandit/DRQN!)")
    else:
        print(f"❌ SOME CHECKS FAILED — Review the output above.")
    print(f"{'='*60}")

    return all_passed


def test_train_test_split_disjoint(npy_path: Path = DEFAULT_NPY):
    """
    Verify that the train and test splits have no overlapping record indices.
    This is critical — overlapping splits would mean data leakage.
    """
    print(f"\n{'─'*60}")
    print(f"TRAIN/TEST SPLIT DISJOINTNESS CHECK")
    print(f"{'─'*60}")

    train_env = RadarEnv(npy_path, split="train")
    test_env  = RadarEnv(npy_path, split="test")

    train_range = set(range(train_env.start_idx, train_env.end_idx))
    test_range  = set(range(test_env.start_idx,  test_env.end_idx))
    overlap     = train_range & test_range

    print(f"  Train range: [{train_env.start_idx:,} – {train_env.end_idx:,}]  "
          f"({len(train_range):,} records)")
    print(f"  Test range:  [{test_env.start_idx:,} – {test_env.end_idx:,}]   "
          f"({len(test_range):,} records)")
    print(f"  Overlap:     {len(overlap)} records")

    if len(overlap) == 0:
        print(f"  ✅ Splits are fully disjoint — no data leakage")
    else:
        print(f"  ❌ WARNING: Splits overlap! This means data leakage.")

    train_env.close()
    test_env.close()

    return len(overlap) == 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PERSON 1 — FULL ENVIRONMENT VALIDATION")
    print("=" * 60)

    npy = DEFAULT_NPY
    if not npy.exists():
        print(f"\n❌ Binary file not found: {npy}")
        print(f"   Please run the pipeline in order:")
        print(f"   1. python data/download_dataset.py")
        print(f"   2. python data/parse_pdw.py")
        print(f"   3. python data/convert_to_binary.py")
        print(f"   4. python tests/test_random_agent.py  ← (this script)")
        sys.exit(1)

    N_STEPS = 100

    # Run the random agent
    stats = run_random_agent(npy, n_steps=N_STEPS, seed=42, verbose=True)

    # Analyse results
    passed = analyse_results(stats, N_STEPS)

    # Check split disjointness
    split_ok = test_train_test_split_disjoint(npy)

    # Final verdict
    print(f"\n{'='*60}")
    if passed and split_ok:
        print(f"🎉 ALL TESTS PASSED — PERSON 1 HANDOFF READY")
        print(f"")
        print(f"   Persons 2 & 3 can now import:")
        print(f"   >>> from env.radar_env import RadarEnv")
        print(f"   >>> env = RadarEnv(split='train')")
        print(f"   >>> obs, info = env.reset()")
    else:
        print(f"❌ SOME TESTS FAILED — Fix before handoff")
    print(f"{'='*60}")
