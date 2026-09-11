"""
Demo: SlidingWindowUCB on a non-stationary ToyBanditEnv.

Experiment design:
  Phase 1 (steps   1-200): Channel 3 is the best arm (win rate = 0.85)
  Phase 2 (steps 201-400): Channel 1 becomes the new best arm (win rate = 0.90)

We expect SW-UCB to adapt to the shift and converge on the new best arm.

Run with: python bandits/demo_sw_ucb.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bandits import ToyBanditEnv, SlidingWindowUCB


def run_phase(env, agent, num_steps: int, phase_label: str):
    """Run a single phase of the experiment and return the total reward."""
    total_reward = 0.0
    for _ in range(num_steps):
        action = agent.select_action()
        reward, _ = env.step(action)
        agent.update(action, reward)
        total_reward += reward
    return total_reward


def print_arm_stats(agent, true_probs, label: str):
    """Print per-arm statistics after a phase."""
    print(f"\n  [{label}] Per-Arm Statistics:")
    print(f"  {'Arm':>4} | {'Pulled (window)':>15} | {'Est. Win Rate':>13} | {'True Win Rate':>13}")
    print(f"  {'-'*4}-+-{'-'*15}-+-{'-'*13}-+-{'-'*13}")
    for i in range(agent.k):
        recent_count = agent._recent_count(i)
        recent_mean  = agent._recent_mean(i)
        true_p       = true_probs[i]
        print(f"  {i:>4} | {recent_count:>15} | {recent_mean:>13.3f} | {true_p:>13.2f}")

    best_arm = max(range(agent.k), key=lambda i: agent._recent_mean(i))
    print(f"\n  Estimated Best Arm (by recent mean): Arm {best_arm}")


def main():
    # 5 channels, window_size=50 (last 50 observations per arm are remembered)
    K           = 5
    WINDOW_SIZE = 50
    PHASE_STEPS = 200

    # -----------------------------------------------------------------------
    # Phase 1 probabilities: Arm 3 is the clear winner
    # -----------------------------------------------------------------------
    phase1_probs = [0.15, 0.25, 0.30, 0.85, 0.20]

    # -----------------------------------------------------------------------
    # Phase 2 probabilities: Arm 1 becomes the new winner, Arm 3 drops sharply
    # -----------------------------------------------------------------------
    phase2_probs = [0.15, 0.90, 0.30, 0.20, 0.20]

    env   = ToyBanditEnv(k=K, p_dist=phase1_probs)
    agent = SlidingWindowUCB(k=K, window_size=WINDOW_SIZE)

    print("=" * 60)
    print("      Sliding-Window UCB - Non-Stationary Demo             ")
    print("=" * 60)
    print(f"Arms (channels)  : {K}")
    print(f"Window size      : {WINDOW_SIZE}")
    print(f"Steps per phase  : {PHASE_STEPS}")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"PHASE 1 (steps 1-{PHASE_STEPS})")
    print(f"  True probabilities: {phase1_probs}")
    print(f"  Optimal arm: 3  (win rate = 0.85)")

    reward1 = run_phase(env, agent, PHASE_STEPS, "Phase 1")

    print(f"\n  Total Reward: {reward1:.0f} / {PHASE_STEPS}  "
          f"(Average: {reward1/PHASE_STEPS:.2f})")
    print_arm_stats(agent, phase1_probs, "After Phase 1")

    # ── Environment shift ────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("ENVIRONMENT SHIFT - channel qualities have changed!")
    print(f"  New probabilities: {phase2_probs}")
    print(f"  New optimal arm: 1  (win rate = 0.90)")
    env.update_probabilities(phase2_probs)

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"PHASE 2 (steps {PHASE_STEPS+1}-{2*PHASE_STEPS})")
    print(f"  Agent continues with window_size={WINDOW_SIZE}")

    reward2 = run_phase(env, agent, PHASE_STEPS, "Phase 2")

    print(f"\n  Total Reward: {reward2:.0f} / {PHASE_STEPS}  "
          f"(Average: {reward2/PHASE_STEPS:.2f})")
    print_arm_stats(agent, phase2_probs, "After Phase 2")

    # ── Overall summary ──────────────────────────────────────────────────────
    total = reward1 + reward2
    total_steps = 2 * PHASE_STEPS
    print(f"\n{'=' * 60}")
    print(f"  Overall Total Reward : {total:.0f} / {total_steps}  "
          f"(Average: {total/total_steps:.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
