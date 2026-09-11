import sys
import os

# Add parent directory to sys.path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bandits import ToyBanditEnv


def main():
    # 1. Create a 3-channel environment with custom reward probabilities
    # Channel 0: 10% win rate
    # Channel 1: 80% win rate (best channel!)
    # Channel 2: 30% win rate
    env = ToyBanditEnv(k=3, p_dist=[0.1, 0.8, 0.3])
    print(f"Environment initialized with 3 channels.")
    print(f"True Channel Probabilities: {env.p_dist}\n")

    # 2. Take 5 actions on Channel 1 (the best channel)
    print("--- Pulling Channel 1 (80% probability) 5 times ---")
    for _ in range(5):
        reward, info = env.step(action=1)
        print(f"Step {info['step']} | Action: {info['chosen_action']} | Reward: {reward}")

    # 3. Take 5 actions on Channel 0 (the poor channel)
    print("\n--- Pulling Channel 0 (10% probability) 5 times ---")
    for _ in range(5):
        reward, info = env.step(action=0)
        print(f"Step {info['step']} | Action: {info['chosen_action']} | Reward: {reward}")

    # 4. Demonstrate updating channel quality (Non-Stationary shift)
    print("\n--- Simulating Radar Frequency Shift (Non-Stationary) ---")
    print("Updating probabilities: Channel 0 is now 90%, Channel 1 is now 10%")
    env.update_probabilities(new_p_dist=[0.9, 0.1, 0.3])

    print("--- Pulling Channel 0 (now 90% probability) 3 times ---")
    for _ in range(3):
        reward, info = env.step(action=0)
        print(f"Step {info['step']} | Action: {info['chosen_action']} | Reward: {reward}")

if __name__ == "__main__":
    main()
