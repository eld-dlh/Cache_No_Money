"""
Demo script to test UCB1 agent with ToyBanditEnv.
Run with: python bandits/demo_ucb1.py
"""

import sys
import os

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bandits import ToyBanditEnv, UCB1


def main():
    # 1. Create a 5-channel environment with custom reward probabilities
    # Channel 3 (0.85) is the optimal channel!
    true_probabilities = [0.15, 0.35, 0.50, 0.85, 0.20]
    env = ToyBanditEnv(k=5, p_dist=true_probabilities)

    # 2. Create a UCB1 agent for k=5 arms
    agent = UCB1(k=5)

    print("==================================================")
    print("           UCB1 Multi-Armed Bandit Test           ")
    print("==================================================")
    print(f"True Channel Probabilities: {true_probabilities}")
    print(f"Optimal Channel: Index 3 (85% win rate)\n")

    # 3. Run interaction loop for 300 steps
    num_steps = 300
    total_reward = 0.0

    for step in range(1, num_steps + 1):
        action = agent.select_action()
        reward, info = env.step(action)
        agent.update(action, reward)
        total_reward += reward

    # 4. Print Summary Results
    print(f"--- Results after {num_steps} Steps ---")
    print(f"Total Reward Collected: {total_reward} / {num_steps} (Average: {total_reward/num_steps:.2f})")
    print("\nPer-Arm Memory Breakdown:")
    import math
    log_t = math.log(agent.t)
    for i in range(env.k):
        exploration_term = math.sqrt((2.0 * log_t) / agent.counts[i])
        ucb_score = agent.q_values[i] + exploration_term
        print(f" Arm {i}: Pulled {agent.counts[i]:3d} times | "
              f"Est. Win Rate: {agent.q_values[i]:.3f} | "
              f"UCB Score: {ucb_score:.3f} | "
              f"True Win Rate: {true_probabilities[i]:.2f}")

    print("\nLearned Best Arm:", agent.q_values.index(max(agent.q_values)))


if __name__ == "__main__":
    main()
