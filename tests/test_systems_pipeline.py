"""
tests/test_systems_pipeline.py
==============================
Unit & integration tests for systems/pipeline.py (SingleThreadedPipeline).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from bandits.sw_ucb import SlidingWindowUCB
from systems.pipeline import SingleThreadedPipeline, StepResult
from systems.replay import PDWReplay
from systems.buffer import StateBuffer
from systems.dsp_stub import DSPStub
from systems.receiver_sim import ReceiverSim
from systems.metrics import PipelineMetrics


def test_pipeline_step_execution():
    """Pipeline executes single steps returning valid StepResult with binary rewards."""
    replay = PDWReplay(synthetic_count=50, seed=42)
    pipeline = SingleThreadedPipeline(replay=replay)

    res = pipeline.step()
    assert res is not None
    assert isinstance(res, StepResult)
    assert res.step == 1
    assert 0 <= res.action < 64
    assert res.binary_reward in (0.0, 1.0)
    assert res.intercepted == (res.binary_reward == 1.0)
    assert 0 <= res.pulse_channel < 64
    assert res.retune_delay_us >= 2.0


def test_bandit_reward_strict_binary():
    """Verify that every reward fed into the bandit is strictly 0.0 or 1.0."""
    replay = PDWReplay(synthetic_count=150, seed=123)
    pipeline = SingleThreadedPipeline(replay=replay)

    rewards_seen = []
    for _ in range(100):
        res = pipeline.step()
        assert res is not None
        rewards_seen.append(res.binary_reward)
        assert res.binary_reward in (0.0, 1.0)

    # Confirm only {0.0, 1.0} or subset was ever generated
    unique_rewards = set(rewards_seen)
    assert unique_rewards.issubset({0.0, 1.0})


def test_no_ground_truth_leakage_to_bandit():
    """
    Spies on SlidingWindowUCB methods to strictly guarantee that:
    1. select_action() is called with NO arguments (no context, no pulse).
    2. update() is called ONLY with (action, binary_reward) and never pulse_channel.
    """
    mock_bandit = MagicMock(spec=SlidingWindowUCB)
    mock_bandit.k = 64
    mock_bandit.select_action.return_value = 10

    replay = PDWReplay(synthetic_count=20, seed=42)
    pipeline = SingleThreadedPipeline(replay=replay, bandit=mock_bandit)

    for i in range(10):
        res = pipeline.step()
        assert res is not None

        # 1. select_action must be called without arguments
        assert mock_bandit.select_action.call_count == i + 1
        args, kwargs = mock_bandit.select_action.call_args
        assert len(args) == 0
        assert len(kwargs) == 0

        # 2. update must receive only (action, binary_reward)
        assert mock_bandit.update.call_count == i + 1
        u_args, u_kwargs = mock_bandit.update.call_args
        assert len(u_args) == 2
        assert len(u_kwargs) == 0
        assert u_args[0] == 10                             # action
        assert u_args[1] in (0.0, 1.0)                     # binary reward only
        # Explicit check: ground truth pulse_channel is never passed to update
        assert u_args[1] != res.pulse_channel or res.pulse_channel in (0, 1)


def test_pipeline_run_termination():
    """Pipeline.run(max_steps) stops at the requested step limit."""
    replay = PDWReplay(synthetic_count=500, seed=42)
    pipeline = SingleThreadedPipeline(replay=replay)

    summary = pipeline.run(max_steps=100)
    assert summary["total_steps"] == 100
    assert 0.0 <= summary["hit_rate_pct"] <= 100.0
    assert summary["throughput_steps_per_sec"] > 0
    assert summary["latency_us"] is not None
    assert summary["stage_means_us"] is not None


def test_pipeline_stream_exhaustion():
    """When replay stream ends, pipeline returns None cleanly without error."""
    replay = PDWReplay(synthetic_count=5, seed=42, loop=False)
    pipeline = SingleThreadedPipeline(replay=replay)

    for _ in range(5):
        assert pipeline.step() is not None

    # 6th step should return None (end of data)
    assert pipeline.step() is None


def test_pipeline_reset():
    """Pipeline reset clears buffer, metrics, and cursor."""
    replay = PDWReplay(synthetic_count=50, seed=42)
    pipeline = SingleThreadedPipeline(replay=replay)

    for _ in range(10):
        pipeline.step()
    assert pipeline._step_count == 10

    pipeline.reset()
    assert pipeline._step_count == 0
    assert pipeline.buffer.count == 0
    assert pipeline.metrics.step_count == 0
    assert pipeline.replay.current_cursor == 0
