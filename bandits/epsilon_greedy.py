import random


class EpsilonGreedy:
    """
    Epsilon-Greedy algorithm for Multi-Armed Bandits.

    Parameters:
    -----------
    k : int
        Number of arms/channels.
    epsilon : float, optional (default=0.1)
        Probability of selecting a random action (exploration).
        1 - epsilon is the probability of selecting the best-known action (exploitation).
    """

    def __init__(self, k: int, epsilon: float = 0.1):
        if k <= 0:
            raise ValueError("Number of arms 'k' must be a positive integer.")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("Epsilon must be between 0.0 and 1.0.")

        self.k = k
        self.epsilon = epsilon

        # Statistics maintained for each arm
        self.counts = [0] * k            # How many times each arm was pulled
        self.total_rewards = [0.0] * k   # Total accumulated reward per arm
        self.q_values = [0.0] * k        # Estimated average reward per arm (Q-values)

    def select_action(self) -> int:
        """
        Selects an arm to pull based on the Epsilon-Greedy policy.

        Returns:
        --------
        action : int
            The selected arm index (0 to k-1).
        """
        # Explore: pick a random arm with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.k)

        # Exploit: find the maximum Q-value and pick from arms tied for max
        max_q = max(self.q_values)
        best_arms = [i for i in range(self.k) if self.q_values[i] == max_q]
        return random.choice(best_arms)

    def update(self, action: int, reward: float):
        """
        Updates the agent's memory after receiving a reward from taking an action.

        Parameters:
        -----------
        action : int
            The arm index that was pulled.
        reward : float
            The reward received from the environment.
        """
        if not (0 <= action < self.k):
            raise ValueError(f"Action {action} is out of bounds for {self.k} arms.")

        self.counts[action] += 1
        self.total_rewards[action] += reward

        # Update average reward estimate (Q-value)
        self.q_values[action] = self.total_rewards[action] / self.counts[action]
