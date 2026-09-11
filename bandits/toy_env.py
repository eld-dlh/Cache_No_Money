import random


class ToyBanditEnv:
    """
    A simple Multi-Armed Bandit Environment for RF spectrum / channel selection.

    Parameters:
    -----------
    k : int
        The number of channels/arms available.
    p_dist : list of float, optional
        Success probabilities for each channel. If None, random probabilities
        between 0.0 and 1.0 are generated.
    """

    def __init__(self, k: int = 5, p_dist=None):
        if k <= 0:
            raise ValueError("Number of arms 'k' must be a positive integer.")

        self.k = k
        self.current_step = 0

        if p_dist is not None:
            self._set_p_dist(p_dist)
        else:
            # Generate random win probabilities for k arms using standard random
            self.p_dist = [random.random() for _ in range(self.k)]

    def _set_p_dist(self, p_dist):
        """Helper method to validate and set channel reward probabilities."""
        p_dist_list = [float(p) for p in p_dist]
        if len(p_dist_list) != self.k:
            raise ValueError(f"Length of p_dist ({len(p_dist_list)}) must match k ({self.k}).")
        if any(p < 0.0 or p > 1.0 for p in p_dist_list):
            raise ValueError("All probabilities in p_dist must be between 0.0 and 1.0.")
        self.p_dist = p_dist_list

    def step(self, action: int):
        """
        Takes an action by pulling arm/channel 'action'.

        Parameters:
        -----------
        action : int
            Index of the channel to pull (0 <= action < k).

        Returns:
        --------
        reward : float
            1.0 if pulse/signal intercepted, 0.0 otherwise.
        info : dict
            Diagnostic information (e.g. step count and current arm probability).
        """
        if not (0 <= action < self.k):
            raise ValueError(f"Action {action} is out of bounds for {self.k} arms.")

        # Simulate Bernoulli trial: draw random float in [0.0, 1.0)
        # If random value < channel probability, reward is 1.0, else 0.0
        success_prob = self.p_dist[action]
        reward = 1.0 if random.random() < success_prob else 0.0

        self.current_step += 1

        info = {
            "step": self.current_step,
            "chosen_action": action,
            "channel_probability": success_prob,
        }

        return reward, info

    def update_probabilities(self, new_p_dist):
        """
        Updates channel reward probabilities to simulate non-stationary environments.
        """
        self._set_p_dist(new_p_dist)

    def reset(self):
        """Resets the environment step counter."""
        self.current_step = 0
        return self.current_step

