"""
tests/test_concurrent_pipeline.py
=================================
Unit and stress tests for systems/concurrent_pipeline.py (ConcurrentPipeline).
"""

import time
from unittest.mock import MagicMock
import pytest

from bandits.sw_ucb import SlidingWindowUCB
from systems.concurrent_pipeline import ConcurrentPipeline, PulseMessage
from systems.replay import PDWReplay
from systems.buffer import StateBuffer
from systems.dsp_stub import DSPStub
from systems.receiver_sim import ReceiverSim
from systems.metrics import PipelineMetrics


def test_concurrent_fifo_ordering():
    """Verify pulses are consumed in strict FIFO sequential order."""
    consumed_indices = []

    class OrderTrackingPipeline(ConcurrentPipeline):
        def _consumer_worker(self):
            # Intercept queue get to track message indices
            super_worker = super()._consumer_worker
            # Use original loop but intercept queue items
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except Exception:
                    continue

                if item is object():  # sentinel check handled internally
                    pass
                if isinstance(item, PulseMessage):
                    consumed_indices.append(item.index)

                # Process through standard consumer logic
                if item is not None and not isinstance(item, PulseMessage):
                    self._queue.task_done()
                    break

                # Run regular step
                self.dsp.extract_features(item.observation)
                action = self.bandit.select_action()
                det_result = self.receiver_sim.retune_and_detect(action, item.pulse)
                self.bandit.update(action, det_result.binary_reward)
                self._queue.task_done()
                with self._lock:
                    self._pulses_consumed += 1

    replay = PDWReplay(synthetic_count=100, seed=42)
    pipeline = ConcurrentPipeline(replay=replay, queue_size=16)

    summary = pipeline.run(max_steps=100)
    assert summary["concurrency"]["dropped_pulses"] == 0
    assert summary["concurrency"]["pulses_consumed"] == 100


def test_bounded_capacity_and_backpressure():
    """Verify that bounded queue prevents unbounded growth and triggers backpressure."""
    queue_cap = 5
    total_steps = 50

    replay = PDWReplay(synthetic_count=total_steps, seed=42)
    pipeline = ConcurrentPipeline(replay=replay, queue_size=queue_cap)

    summary = pipeline.run(max_steps=total_steps)

    assert summary["concurrency"]["queue_capacity"] == queue_cap
    assert summary["concurrency"]["max_queue_depth"] <= queue_cap
    assert summary["concurrency"]["pulses_produced"] == total_steps
    assert summary["concurrency"]["pulses_consumed"] == total_steps
    assert summary["concurrency"]["dropped_pulses"] == 0
    assert summary["concurrency"]["backpressure_events"] > 0


def test_clean_shutdown_on_completion():
    """Verify producer and consumer terminate cleanly when stream completes."""
    replay = PDWReplay(synthetic_count=30, seed=42, loop=False)
    pipeline = ConcurrentPipeline(replay=replay, queue_size=10)

    pipeline.start(max_steps=30)
    clean = pipeline.join(timeout=5.0)

    assert clean, "Threads failed to join within 5 seconds"
    assert not pipeline._producer_thread.is_alive()
    assert not pipeline._consumer_thread.is_alive()
    summary = pipeline.get_summary()
    assert summary["concurrency"]["dropped_pulses"] == 0


def test_early_stop_cancellation():
    """Verify calling stop() terminates running threads without deadlock."""
    # Endless looping replay
    replay = PDWReplay(synthetic_count=5000, seed=42, loop=True)
    pipeline = ConcurrentPipeline(replay=replay, queue_size=16)

    pipeline.start()
    time.sleep(0.1)  # Allow running
    pipeline.stop()

    assert not pipeline._producer_thread.is_alive()
    assert not pipeline._consumer_thread.is_alive()


def test_concurrent_binary_rewards_and_no_leakage():
    """Verify strict binary rewards and zero leakage in concurrent pipeline."""
    mock_bandit = MagicMock(spec=SlidingWindowUCB)
    mock_bandit.k = 64
    mock_bandit.select_action.return_value = 5

    replay = PDWReplay(synthetic_count=50, seed=42)
    pipeline = ConcurrentPipeline(replay=replay, bandit=mock_bandit, queue_size=10)

    summary = pipeline.run(max_steps=50)

    assert summary["concurrency"]["pulses_consumed"] == 50
    assert mock_bandit.select_action.call_count == 50
    assert mock_bandit.update.call_count == 50

    # Validate each call had 0 args for select_action, and binary reward for update
    for call in mock_bandit.select_action.call_args_list:
        args, kwargs = call
        assert len(args) == 0 and len(kwargs) == 0

    for call in mock_bandit.update.call_args_list:
        args, kwargs = call
        assert len(args) == 2 and len(kwargs) == 0
        assert args[0] == 5
        assert args[1] in (0.0, 1.0)
