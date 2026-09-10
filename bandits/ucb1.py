import math
import random


class UCB1:
    """
    Upper Confidence Bound 1 (UCB1) Multi-Armed Bandit algorithm.

    Formula:
        UCB_i = Q_i + sqrt((2 * ln(t)) / N_i)

    Parameters:
    -----------
    k : int
        Number of arms/channels.
    """

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("Number of arms 'k' must be a positive integer.")

        self.k = k
        self.t = 0                        # Total number of selections made across all arms
        self.counts = [0] * k             # Selection count per arm (N_i)
        self.total_rewards = [0.0] * k    # Total reward accumulated per arm
        self.q_values = [0.0] * k         # Estimated average reward per arm (Q_i)

    def select_action(self) -> int:
        """
        Selects an arm based on the UCB1 algorithm.

        Phase 1: Selects untried arms first (if any arm has count == 0).
        Phase 2: Computes the UCB score for all arms and picks the arm with the highest score.

        Returns:
        --------
        action : int
            The selected arm index (0 to k-1).
        """
        # Step 1: Force exploration of untried arms first
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # Step 2: Once every arm has been pulled at least once, compute UCB scores
        ucb_values = [0.0] * self.k
        log_t = math.log(self.t)

        for i in range(self.k):
            exploitation = self.q_values[i]
            exploration = math.sqrt((2.0 * log_t) / self.counts[i])
            ucb_values[i] = exploitation + exploration

        # Step 3: Pick the arm with the maximum UCB score (break ties randomly)
        max_ucb = max(ucb_values)
        best_arms = [i for i in range(self.k) if ucb_values[i] == max_ucb]
        return random.choice(best_arms)

    def update(self, action: int, reward: float):
        """
        Updates the agent's statistics after receiving a reward.

        Parameters:
        -----------
        action : int
            The arm index that was pulled.
        reward : float
            The reward received from the environment.
        """
        if not (0 <= action < self.k):
            raise ValueError(f"Action {action} is out of bounds for {self.k} arms.")

        self.t += 1
        self.counts[action] += 1
        self.total_rewards[action] += reward
        self.q_values[action] = self.total_rewards[action] / self.counts[action]
