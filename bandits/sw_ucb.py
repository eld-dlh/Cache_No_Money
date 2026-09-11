import math
import random
from collections import deque


class SlidingWindowUCB:
    """
    Sliding-Window UCB (SW-UCB) for non-stationary Multi-Armed Bandits.

    Unlike UCB1, which uses ALL past observations to estimate an arm's value,
    SW-UCB only uses the most recent `window_size` observations per arm.
    This allows the agent to adapt when channel quality changes over time.

    Formula (applied only to recent observations within the window):
        SW_UCB_i = recent_mean_i + sqrt((2 * ln(t)) / recent_count_i)

    Parameters:
    -----------
    k : int
        Number of arms/channels.
    window_size : int, optional (default=50)
        Maximum number of recent observations to remember per arm.
        Observations older than this window are discarded.
    """

    def __init__(self, k: int, window_size: int = 50):
        if k <= 0:
            raise ValueError("Number of arms 'k' must be a positive integer.")
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        self.k = k
        self.window_size = window_size
        self.t = 0  # Total number of pulls made (across all arms, all time)

        # For each arm, store its recent rewards in a deque.
        # A deque with maxlen automatically drops the oldest entry when full.
        # This is the core mechanism of the sliding window.
        self.windows = [deque(maxlen=window_size) for _ in range(k)]

    def _recent_count(self, arm: int) -> int:
        """Returns the number of recent observations for the given arm."""
        return len(self.windows[arm])

    def _recent_mean(self, arm: int) -> float:
        """Returns the average reward from recent observations for the given arm.
        Returns 0.0 if the arm has never been pulled.
        """
        if len(self.windows[arm]) == 0:
            return 0.0
        return sum(self.windows[arm]) / len(self.windows[arm])

    def select_action(self) -> int:
        """
        Selects an arm to pull based on the SW-UCB algorithm.

        Phase 1: Pull every arm at least once (avoids division-by-zero).
        Phase 2: Use the SW-UCB formula, considering only recent observations.

        Returns:
        --------
        action : int
            The selected arm index (0 to k-1).
        """
        # Phase 1: Force each arm to be tried at least once before scoring
        for i in range(self.k):
            if self._recent_count(i) == 0:
                return i

        # Phase 2: Compute SW-UCB scores using only recent observations
        ucb_scores = [0.0] * self.k
        log_t = math.log(min(self.t, self.window_size))

        for i in range(self.k):
            exploitation = self._recent_mean(i)
            exploration = math.sqrt((2.0 * log_t) / self._recent_count(i))
            ucb_scores[i] = exploitation + exploration

        # Pick the arm with the highest SW-UCB score; break ties randomly
        max_score = max(ucb_scores)
        best_arms = [i for i in range(self.k) if ucb_scores[i] == max_score]
        return random.choice(best_arms)

    def update(self, action: int, reward: float):
        """
        Records the latest observation and advances the global step counter.
        Old observations beyond window_size are automatically discarded by the deque.

        Parameters:
        -----------
        action : int
            The arm index that was just pulled.
        reward : float
            The reward received from the environment.
        """
        if not (0 <= action < self.k):
            raise ValueError(f"Action {action} is out of bounds for {self.k} arms.")

        self.t += 1
        # Append the new reward. The deque automatically drops the oldest
        # entry if it is already at maxlen (= window_size).
        self.windows[action].append(reward)
