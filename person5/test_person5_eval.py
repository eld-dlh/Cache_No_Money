"""
person5/test_person5_eval.py — Unit & Integration Test Suite for Person 5
==========================================================================

Validates deterministic seed reproducibility, metrics calculation, noise injection,
and output artifact generation.
"""

import json
import sys
from pathlib import Path

# Enforce project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from env.radar_env import RadarEnv
from person5.eval_harness import (
    MonteCarloEvaluator,
    SequentialSweeperPolicy,
    SWUCBPolicyWrapper,
    compute_relative_improvement,
)
from person5.noise_models import CorruptedRadarEnv


def test_relative_improvement_formula():
    # Test formula: ((P_adaptive - P_baseline) / P_baseline) * 100
    imp = compute_relative_improvement(p_adaptive=50.0, p_baseline=10.0)
    assert imp == 400.0, f"Expected 400.0%, got {imp}"

    imp_zero = compute_relative_improvement(p_adaptive=50.0, p_baseline=0.0)
    assert imp_zero == 0.0, "Expected 0.0 for 0 baseline"


def test_sequential_sweeper_reproducibility():
    env = RadarEnv(split="train", max_steps=50)
    policy = SequentialSweeperPolicy(n_channels=64)
    evaluator = MonteCarloEvaluator(base_seed=123)

    res1 = evaluator.evaluate_episode(env, policy, seed=100, max_steps=50)
    res2 = evaluator.evaluate_episode(env, policy, seed=100, max_steps=50)

    assert res1["interception_rate"] == res2["interception_rate"]
    assert res1["total_reward"] == res2["total_reward"]


def test_corrupted_radar_env_wrapper():
    base_env = RadarEnv(split="train", max_steps=50)
    noisy_env = CorruptedRadarEnv(base_env, dropout_rate=0.5, channel_jitter=0.2)

    obs, info = noisy_env.reset(seed=42)
    assert obs.shape == (10, 5)

    obs, reward, terminated, truncated, step_info = noisy_env.step(action=0)
    assert isinstance(reward, float)


def test_evaluator_benchmark_run():
    env = RadarEnv(split="train", max_steps=20)
    policy = SequentialSweeperPolicy(n_channels=64)
    evaluator = MonteCarloEvaluator(base_seed=42)

    summary = evaluator.run_benchmark(env, policy, n_episodes=5, max_steps=20)
    assert summary["n_episodes"] == 5
    assert "mean_interception_rate" in summary
    assert "mean_total_reward" in summary
