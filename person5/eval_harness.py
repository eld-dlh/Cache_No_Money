"""
person5/eval_harness.py — Monte Carlo Evaluation Harness
=========================================================

Implements Person 5's core Monte Carlo evaluation engine to evaluate:
  1. Sequential Sweeper Baseline
  2. Sliding-Window UCB (Tier 1 Bandit)
  3. Hybrid DRQN Cognitive Interceptor (Tier 1 + Tier 2)

Calculates deterministic, reproducible metrics across 100 episodes with fixed random seeds.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enforce project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from bandits.sw_ucb import SlidingWindowUCB
from models.cognitive_interceptor import CognitiveInterceptor


# ─────────────────────────────────────────────────────────────────────────────
# POLICIES
# ─────────────────────────────────────────────────────────────────────────────

class BasePolicy:
    """Base policy interface for evaluation."""
    def name(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError

    def update_feedback(self, reward: float, intercepted: bool, pulse_channel: int) -> None:
        pass


class SequentialSweeperPolicy(BasePolicy):
    """
    Sequential Sweeper Baseline.
    Loops through channels sequentially: 0 -> 1 -> 2 -> ... -> 63 -> 0.
    """
    def __init__(self, n_channels: int = 64):
        self.n_channels = n_channels
        self.current_channel = 0

    def name(self) -> str:
        return "Sequential Sweeper (Baseline)"

    def reset(self) -> None:
        self.current_channel = 0

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        action = self.current_channel
        self.current_channel = (self.current_channel + 1) % self.n_channels
        return action


class SWUCBPolicyWrapper(BasePolicy):
    """
    Sliding-Window UCB (Tier 1 Bandit).
    Reacts to active pulse channels using a rolling window of past observations.
    """
    def __init__(self, n_channels: int = 64, window_size: int = 50):
        self.n_channels = n_channels
        self.window_size = window_size
        self.bandit = SlidingWindowUCB(k=n_channels, window_size=window_size)
        self.last_action: Optional[int] = None

    def name(self) -> str:
        return f"Sliding-Window UCB (W={self.window_size})"

    def reset(self) -> None:
        self.bandit = SlidingWindowUCB(k=self.n_channels, window_size=self.window_size)
        self.last_action = None

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        action = self.bandit.select_action()
        self.last_action = action
        return action

    def update_feedback(self, reward: float, intercepted: bool, pulse_channel: int) -> None:
        if self.last_action is not None:
            # Reward is 1.0 if intercepted, 0.0 otherwise for bandit update
            b_reward = 1.0 if intercepted else 0.0
            self.bandit.update(self.last_action, b_reward)


class HybridDRQNPolicyWrapper(BasePolicy):
    """
    Hybrid DRQN Cognitive Interceptor (Tier 1 + Tier 2).
    Combines SW-UCB reactive bandit with DRQN dueling LSTM pattern predictor
    and pattern-lock/fallback handoff controller.
    """
    def __init__(
        self,
        weights_path: str | Path | None = "checkpoints/drqn_radar_best.pt",
        n_channels: int = 64,
        max_steps: int = 500,
        device: str = "cpu",
    ):
        self.n_channels = n_channels
        self.max_steps = max_steps
        self.weights_path = weights_path
        self.device = device
        
        # Instantiate fallback bandit
        self.fallback_bandit = SlidingWindowUCB(k=n_channels, window_size=50)
        
        # Instantiate unified interceptor
        self.interceptor = CognitiveInterceptor(
            weights_path=weights_path,
            n_channels=n_channels,
            max_steps=max_steps,
            fallback_bandit=self.fallback_bandit,
            device=device,
        )

    def name(self) -> str:
        return "Hybrid DRQN (Tier 2 + Fallback)"

    def reset(self) -> None:
        self.interceptor.reset()

    def select_action(self, obs: np.ndarray, info: dict) -> int:
        last_step_info = info.copy() if info else {}
        action = self.interceptor.predict_channel(obs, last_step_info)
        return action

    def update_feedback(self, reward: float, intercepted: bool, pulse_channel: int) -> None:
        self.interceptor.update_feedback(reward=reward, intercepted=intercepted, pulse_channel=pulse_channel)


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO EVALUATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloEvaluator:
    """
    Person 5 Monte Carlo Evaluation Harness.
    Executes reproducible, multi-episode policy benchmarks.
    """

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    @staticmethod
    def set_seed(seed: int) -> None:
        """Enforces deterministic reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def evaluate_episode(
        self,
        env: Any,
        policy: BasePolicy,
        seed: int,
        max_steps: int = 500,
    ) -> Dict[str, Any]:
        """
        Runs a single episode of policy evaluation on the environment.
        """
        self.set_seed(seed)
        policy.reset()

        obs, info = env.reset(seed=seed)
        
        total_pulses = 0
        intercepted_count = 0
        wasted_dwells = 0
        total_retuning_penalty = 0.0
        total_reward = 0.0
        
        tier1_steps = 0
        tier2_steps = 0

        last_action = 0

        for step in range(max_steps):
            # Select action
            action = policy.select_action(obs, info)

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)

            # Extract metrics from step_info
            pulse_present = step_info.get("pulse_present", False)
            intercepted = step_info.get("intercepted", step_info.get("pulse_intercepted", False))
            pulse_channel = step_info.get("pulse_channel", -1)

            if pulse_present:
                total_pulses += 1
                if intercepted:
                    intercepted_count += 1
            else:
                wasted_dwells += 1

            # Retuning distance penalty
            retuning_dist = abs(action - last_action)
            total_retuning_penalty += retuning_dist
            last_action = action

            total_reward += float(reward)

            # Track tier telemetry if available
            if hasattr(policy, "interceptor"):
                if policy.interceptor.current_tier == 2:
                    tier2_steps += 1
                else:
                    tier1_steps += 1

            # Inform policy of outcome
            policy.update_feedback(reward=float(reward), intercepted=intercepted, pulse_channel=pulse_channel)

            obs = next_obs
            info = step_info

            if terminated or truncated:
                break

        interception_rate = (intercepted_count / total_pulses * 100.0) if total_pulses > 0 else 0.0
        dwell_waste_ratio = (wasted_dwells / (step + 1) * 100.0)

        return {
            "episode_seed": seed,
            "total_steps": step + 1,
            "total_pulses": total_pulses,
            "intercepted_count": intercepted_count,
            "interception_rate": interception_rate,
            "wasted_dwells": wasted_dwells,
            "dwell_waste_ratio": dwell_waste_ratio,
            "retuning_distance": total_retuning_penalty,
            "total_reward": total_reward,
            "tier1_steps": tier1_steps,
            "tier2_steps": tier2_steps,
        }

    def run_benchmark(
        self,
        env: Any,
        policy: BasePolicy,
        n_episodes: int = 100,
        max_steps: int = 500,
    ) -> Dict[str, Any]:
        """
        Executes Monte Carlo evaluation across N episodes with deterministic seeds.
        """
        episodes_data = []
        interception_rates = []
        rewards = []

        for ep in range(n_episodes):
            seed = self.base_seed + ep
            ep_metrics = self.evaluate_episode(env=env, policy=policy, seed=seed, max_steps=max_steps)
            episodes_data.append(ep_metrics)
            interception_rates.append(ep_metrics["interception_rate"])
            rewards.append(ep_metrics["total_reward"])

        rates_arr = np.array(interception_rates)
        rewards_arr = np.array(rewards)

        summary = {
            "policy_name": policy.name(),
            "n_episodes": n_episodes,
            "mean_interception_rate": float(np.mean(rates_arr)),
            "std_interception_rate": float(np.std(rates_arr)),
            "min_interception_rate": float(np.min(rates_arr)),
            "max_interception_rate": float(np.max(rates_arr)),
            "p95_interception_rate": float(np.percentile(rates_arr, 95)),
            "mean_total_reward": float(np.mean(rewards_arr)),
            "std_total_reward": float(np.std(rewards_arr)),
            "episodes": episodes_data,
        }

        return summary


def compute_relative_improvement(p_adaptive: float, p_baseline: float) -> float:
    """
    Computes relative improvement formula:
        ((P_int,Adaptive - P_int,Baseline) / P_int,Baseline) * 100
    """
    if p_baseline <= 0:
        return 0.0
    return ((p_adaptive - p_baseline) / p_baseline) * 100.0
