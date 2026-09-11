"""
systems/concurrent_pipeline.py
==============================
Decoupled Concurrent Pipeline for the Person 4 Systems Layer.

Architecture:
  ┌────────────────────────────────────────────────────────┐
  │ PRODUCER THREAD (Ingest & State Window)               │
  │   PDWReplay -> StateBuffer.push() -> Queue.put()       │
  └──────────────────────────┬─────────────────────────────┘
                             │  Bounded Thread-Safe FIFO Queue
                             ▼  (Backpressure & FIFO preserved)
  ┌────────────────────────────────────────────────────────┐
  │ CONSUMER THREAD (DSP & Algorithmic Decision)          │
  │   Queue.get() -> DSPStub -> SW-UCB.select_action() -> │
  │   ReceiverSim.retune_and_detect() ->                  │
  │   SW-UCB.update(action, binary_reward) -> Metrics     │
  └────────────────────────────────────────────────────────┘

CONSTRAINTS:
  1. Strict FIFO ordering of pulses is preserved.
  2. Bounded thread-safe queue prevents memory bloat / applies backpressure.
  3. Bandit receives strictly binary reward (1.0 or 0.0).
  4. Ground-truth pulse_channel is NEVER exposed to the bandit.
  5. Zero dropped pulses during clean shutdown or stream completion.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bandits.sw_ucb import SlidingWindowUCB
from systems.replay import PDWReplay
from systems.buffer import StateBuffer
from systems.dsp_stub import DSPStub, N_CHANNELS
from systems.receiver_sim import ReceiverSim, DetectionResult
from systems.metrics import PipelineMetrics, LatencyBreakdown


# Sentinel object signaling clean producer-to-consumer termination
_SENTINEL = object()


@dataclass(frozen=True)
class PulseMessage:
    """Immutable packet passed through the thread-safe queue."""
    index: int
    pulse: np.ndarray          # shape (8,) float32
    observation: np.ndarray    # shape (10, 5) float32
    enqueue_time_ns: int


class ConcurrentPipeline:
    """
    Decoupled two-stage concurrent pipeline with bounded backpressure.
    """

    def __init__(
        self,
        replay: PDWReplay | None = None,
        buffer: StateBuffer | None = None,
        dsp: DSPStub | None = None,
        bandit: SlidingWindowUCB | None = None,
        receiver_sim: ReceiverSim | None = None,
        metrics: PipelineMetrics | None = None,
        queue_size: int = 128,
        backpressure_timeout: float = 2.0,
    ):
        """
        Args:
            replay: PDWReplay streamer instance.
            buffer: StateBuffer instance.
            dsp: DSPStub instance.
            bandit: SlidingWindowUCB bandit instance.
            receiver_sim: ReceiverSim instance.
            metrics: PipelineMetrics instance.
            queue_size: Maximum capacity of the bounded queue (backpressure threshold).
            backpressure_timeout: Max seconds to wait when queue is full before checking stop_event.
        """
        if queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {queue_size}")

        self.dsp = dsp if dsp is not None else DSPStub(n_channels=N_CHANNELS)
        self.replay = replay if replay is not None else PDWReplay()
        self.buffer = buffer if buffer is not None else StateBuffer(window_size=10)
        self.bandit = bandit if bandit is not None else SlidingWindowUCB(k=self.dsp.n_channels, window_size=50)
        self.receiver_sim = receiver_sim if receiver_sim is not None else ReceiverSim(dsp=self.dsp)
        self.metrics = metrics if metrics is not None else PipelineMetrics()

        self.queue_size = queue_size
        self.backpressure_timeout = backpressure_timeout

        # Bounded FIFO queue
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()

        self._producer_thread: threading.Thread | None = None
        self._consumer_thread: threading.Thread | None = None

        # Concurrency & telemetry counters
        self._lock = threading.Lock()
        self._pulses_produced: int = 0
        self._pulses_consumed: int = 0
        self._max_queue_depth: int = 0
        self._backpressure_events: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Producer Worker
    # ──────────────────────────────────────────────────────────────────────────

    def _producer_worker(self, max_steps: int | None) -> None:
        """Producer stage: sequentially fetches pulses, updates buffer, enqueues."""
        step = 0
        while not self._stop_event.is_set():
            if max_steps is not None and step >= max_steps:
                break

            t0 = time.perf_counter_ns()
            pulse = self.replay.next_pulse()
            if pulse is None:
                # End of replay stream reached
                break

            self.buffer.push(pulse)
            obs = self.buffer.get_observation()

            msg = PulseMessage(
                index=step,
                pulse=pulse,
                observation=obs,
                enqueue_time_ns=t0,
            )

            # Check queue depth for telemetry
            qdepth = self._queue.qsize()
            with self._lock:
                if qdepth > self._max_queue_depth:
                    self._max_queue_depth = qdepth
                if self._queue.full():
                    self._backpressure_events += 1

            # Put with timeout to allow reactive check on _stop_event
            enqueued = False
            while not self._stop_event.is_set():
                try:
                    self._queue.put(msg, timeout=self.backpressure_timeout)
                    enqueued = True
                    break
                except queue.Full:
                    with self._lock:
                        self._backpressure_events += 1

            if not enqueued:
                break

            with self._lock:
                self._pulses_produced += 1
            step += 1

        # Signal completion to consumer
        try:
            self._queue.put(_SENTINEL, timeout=self.backpressure_timeout)
        except queue.Full:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Consumer Worker
    # ──────────────────────────────────────────────────────────────────────────

    def _consumer_worker(self) -> None:
        """Consumer stage: pulls messages, runs DSP, selects action, detects, updates."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                self._queue.task_done()
                break

            assert isinstance(item, PulseMessage)
            t_dequeue = time.perf_counter_ns()

            # 1. DSP extraction
            t_dsp_start = time.perf_counter_ns()
            _features = self.dsp.extract_features(item.observation)
            t_dsp_end = time.perf_counter_ns()

            # 2. SW-UCB decision (strictly 0 arguments)
            t_dec_start = time.perf_counter_ns()
            action = self.bandit.select_action()
            t_dec_end = time.perf_counter_ns()

            # 3. Retune simulation & passband detection
            t_sim_start = time.perf_counter_ns()
            det_result: DetectionResult = self.receiver_sim.retune_and_detect(
                target_channel=action,
                pulse_record=item.pulse,
            )
            t_sim_end = time.perf_counter_ns()

            # 4. Strict binary reward constraint (1.0 or 0.0)
            binary_reward = det_result.binary_reward
            assert binary_reward in (0.0, 1.0), f"Reward must be binary, got {binary_reward}"

            # 5. Bandit update (only action and binary_reward)
            t_upd_start = time.perf_counter_ns()
            self.bandit.update(action, binary_reward)
            t_upd_end = time.perf_counter_ns()

            t_consumer_end = time.perf_counter_ns()

            # Compute queue dwell and stage latencies
            queue_dwell_ns = max(0, t_dequeue - item.enqueue_time_ns)
            total_latency_ns = t_consumer_end - item.enqueue_time_ns

            breakdown = LatencyBreakdown(
                replay_ns=0,  # Producer stage (decoupled)
                buffer_ns=queue_dwell_ns,  # queue wait time
                dsp_ns=t_dsp_end - t_dsp_start,
                decision_ns=t_dec_end - t_dec_start,
                sim_ns=t_sim_end - t_sim_start,
                update_ns=t_upd_end - t_upd_start,
                total_step_ns=total_latency_ns,
            )

            self.metrics.record_step(
                intercepted=det_result.intercepted,
                binary_reward=binary_reward,
                breakdown=breakdown,
            )

            with self._lock:
                self._pulses_consumed += 1
            self._queue.task_done()

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle Control
    # ──────────────────────────────────────────────────────────────────────────

    def start(self, max_steps: int | None = None) -> None:
        """Start producer and consumer threads."""
        self._stop_event.clear()
        self._producer_thread = threading.Thread(
            target=self._producer_worker,
            args=(max_steps,),
            name="EWProducerThread",
            daemon=True,
        )
        self._consumer_thread = threading.Thread(
            target=self._consumer_worker,
            name="EWConsumerThread",
            daemon=True,
        )
        self._consumer_thread.start()
        self._producer_thread.start()

    def join(self, timeout: float | None = 10.0) -> bool:
        """
        Wait for producer and consumer threads to terminate.

        Returns True if both joined cleanly, False if timed out.
        """
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=timeout)
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=timeout)

        producer_alive = self._producer_thread is not None and self._producer_thread.is_alive()
        consumer_alive = self._consumer_thread is not None and self._consumer_thread.is_alive()
        return not (producer_alive or consumer_alive)

    def stop(self) -> None:
        """Signal threads to stop and drain sentinel."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(_SENTINEL)
        except (queue.Full, Exception):
            pass
        self.join(timeout=2.0)

    def run(self, max_steps: int | None = None, timeout: float | None = 30.0) -> dict[str, Any]:
        """
        Run the concurrent pipeline to completion or max_steps.

        Returns:
            Comprehensive performance and queue telemetry dictionary.
        """
        self.start(max_steps=max_steps)
        clean_join = self.join(timeout=timeout)
        if not clean_join:
            self.stop()

        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        """Return combined pipeline metrics and queue telemetry."""
        summary = self.metrics.get_summary()
        with self._lock:
            summary["concurrency"] = {
                "queue_capacity": self.queue_size,
                "pulses_produced": self._pulses_produced,
                "pulses_consumed": self._pulses_consumed,
                "dropped_pulses": max(0, self._pulses_produced - self._pulses_consumed),
                "max_queue_depth": self._max_queue_depth,
                "backpressure_events": self._backpressure_events,
            }
        return summary
