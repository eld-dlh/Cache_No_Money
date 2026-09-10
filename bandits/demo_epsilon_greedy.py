"""
Demo script to test EpsilonGreedy agent with ToyBanditEnv.
Run with: python bandits/demo_epsilon_greedy.py
"""

import sys
import os

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bandits import ToyBanditEnv, EpsilonGreedy


def main():
    # 1. Create a 5-channel environment with fixed probabilities
    # Channel 2 (0.90) is the optimal channel!
    true_probabilities = [0.10, 0.30, 0.90, 0.40, 0.20]
    env = ToyBanditEnv(k=5, p_dist=true_probabilities)

    # 2. Create an Epsilon-Greedy agent with epsilon = 0.1 (10% exploration)
    agent = EpsilonGreedy(k=5, epsilon=0.1)

    print("==================================================")
    print("      Epsilon-Greedy Multi-Armed Bandit Test      ")
    print("==================================================")
    print(f"True Channel Probabilities: {true_probabilities}")
    print(f"Optimal Channel: Index 2 (90% success rate)\n")

    # 3. Run the interaction loop for 300 steps
    num_steps = 300
    total_reward = 0.0

    for step in range(1, num_steps + 1):
        # Step A: Agent chooses an action
        action = agent.select_action()

        # Step B: Environment returns reward and info
        reward, info = env.step(action)

        # Step C: Agent updates its memory with (action, reward)
        agent.update(action, reward)

        total_reward += reward

    # 4. Print Summary Results
    print(f"--- Results after {num_steps} Steps ---")
    print(f"Total Reward Collected: {total_reward} / {num_steps} (Average: {total_reward/num_steps:.2f})")
    print("\nPer-Arm Memory Breakdown:")
    for i in range(env.k):
        print(f" Arm {i}: Pulled {agent.counts[i]:3d} times | "
              f"Est. Win Rate: {agent.q_values[i]:.3f} | "
              f"True Win Rate: {true_probabilities[i]:.2f}")

    print("\nLearned Best Arm:", agent.q_values.index(max(agent.q_values)))


if __name__ == "__main__":
    main()
