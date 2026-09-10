"""
systems/metrics.py
==================
Empirical Metrics & Latency Profiler for the Person 4 Systems Pipeline.

Measures actual wall-clock execution latency per stage using high-resolution
hardware timers (time.perf_counter_ns).

CONSTRAINT: Strictly reports empirically measured latencies without claiming
unverified latency targets.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LatencyBreakdown:
    """Latency breakdown for a single pipeline step in nanoseconds."""
    replay_ns: int = 0
    buffer_ns: int = 0
    dsp_ns: int = 0
    decision_ns: int = 0
    sim_ns: int = 0
    update_ns: int = 0
    total_step_ns: int = 0


class PipelineMetrics:
    """
    Thread-safe latency and performance metrics collector.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.step_count: int = 0
        self.interceptions: int = 0
        self.total_reward: float = 0.0

        # High-resolution step latencies in nanoseconds
        self._step_latencies_ns: list[int] = []
        self._breakdowns: list[LatencyBreakdown] = []

        self._start_time: float = time.perf_counter()

    def record_step(
        self,
        intercepted: bool,
        binary_reward: float,
        breakdown: LatencyBreakdown | None = None,
    ) -> None:
        """
        Record the outcome and latency profile of a single pipeline step.
        """
        with self._lock:
            self.step_count += 1
            if intercepted:
                self.interceptions += 1
            self.total_reward += binary_reward

            if breakdown is not None:
                self._step_latencies_ns.append(breakdown.total_step_ns)
                self._breakdowns.append(breakdown)

    @property
    def hit_rate(self) -> float:
        """Overall pulse interception rate in [0.0, 1.0]."""
        with self._lock:
            return self.interceptions / self.step_count if self.step_count > 0 else 0.0

    def get_summary(self) -> dict:
        """
        Compute empirical performance statistics.

        Returns a dictionary with hit rate, throughput, and latency percentiles
        converted to microseconds (µs).
        """
        with self._lock:
            elapsed_sec = max(time.perf_counter() - self._start_time, 1e-9)
            throughput = self.step_count / elapsed_sec

            summary = {
                "total_steps": self.step_count,
                "total_intercepts": self.interceptions,
                "hit_rate_pct": (self.interceptions / self.step_count * 100.0) if self.step_count > 0 else 0.0,
                "avg_reward": (self.total_reward / self.step_count) if self.step_count > 0 else 0.0,
                "elapsed_sec": elapsed_sec,
                "throughput_steps_per_sec": throughput,
            }

            if self._step_latencies_ns:
                arr = np.array(self._step_latencies_ns, dtype=np.float64) / 1000.0  # convert ns to µs
                summary["latency_us"] = {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                }

                if self._breakdowns:
                    # Per-stage mean latencies in µs
                    summary["stage_means_us"] = {
                        "replay": float(np.mean([b.replay_ns for b in self._breakdowns])) / 1000.0,
                        "buffer": float(np.mean([b.buffer_ns for b in self._breakdowns])) / 1000.0,
                        "dsp": float(np.mean([b.dsp_ns for b in self._breakdowns])) / 1000.0,
                        "decision": float(np.mean([b.decision_ns for b in self._breakdowns])) / 1000.0,
                        "sim": float(np.mean([b.sim_ns for b in self._breakdowns])) / 1000.0,
                        "update": float(np.mean([b.update_ns for b in self._breakdowns])) / 1000.0,
                    }
            else:
                summary["latency_us"] = None
                summary["stage_means_us"] = None

            return summary

    def reset(self) -> None:
        """Reset all recorded metrics."""
        with self._lock:
            self.step_count = 0
            self.interceptions = 0
            self.total_reward = 0.0
            self._step_latencies_ns.clear()
            self._breakdowns.clear()
            self._start_time = time.perf_counter()
