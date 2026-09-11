"""
systems/pipeline.py
===================
Single-Threaded Pipeline Orchestration for the Person 4 Systems Layer.

Wires the sequential end-to-end cognitive radar interception pipeline:
  PDWReplay
      ↓
  StateBuffer
      ↓
  DSPStub
      ↓
  SlidingWindowUCB.select_action()
      ↓
  ReceiverSim.retune_and_detect()
      ↓
  SlidingWindowUCB.update(action, binary_reward)

CONSTRAINTS:
  1. The bandit receives ONLY 1.0 or 0.0 as reward.
  2. Ground-truth pulse_channel is NEVER exposed or passed to the bandit.
  3. Uses existing bandits/ interfaces and Step 2 systems/ components.
  4. Records empirical latency and throughput using PipelineMetrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bandits.sw_ucb import SlidingWindowUCB
from systems.replay import PDWReplay
from systems.buffer import StateBuffer
from systems.dsp_stub import DSPStub, N_CHANNELS
from systems.receiver_sim import ReceiverSim, DetectionResult
from systems.metrics import PipelineMetrics, LatencyBreakdown


@dataclass(frozen=True)
class StepResult:
    """Outcome and telemetry of a single pipeline step."""
    step: int
    action: int
    intercepted: bool
    binary_reward: float
    retune_delay_us: float
    pulse_channel: int          # For evaluation logging only; never fed to bandit
    pulse_freq_mhz: float


class SingleThreadedPipeline:
    """
    Synchronous single-threaded orchestrator for the EW cognitive scan pipeline.
    """

    def __init__(
        self,
        replay: PDWReplay | None = None,
        buffer: StateBuffer | None = None,
        dsp: DSPStub | None = None,
        bandit: SlidingWindowUCB | None = None,
        receiver_sim: ReceiverSim | None = None,
        metrics: PipelineMetrics | None = None,
    ):
        self.dsp = dsp if dsp is not None else DSPStub(n_channels=N_CHANNELS)
        self.replay = replay if replay is not None else PDWReplay()
        self.buffer = buffer if buffer is not None else StateBuffer(window_size=10)
        self.bandit = bandit if bandit is not None else SlidingWindowUCB(k=self.dsp.n_channels, window_size=50)
        self.receiver_sim = receiver_sim if receiver_sim is not None else ReceiverSim(dsp=self.dsp)
        self.metrics = metrics if metrics is not None else PipelineMetrics()

        self._step_count: int = 0

    def step(self) -> StepResult | None:
        """
        Execute one complete pipeline cycle.

        Order:
          1. Replay: fetch next incoming pulse
          2. Buffer: push pulse into sliding observation window
          3. DSP: extract features & validate channels
          4. Bandit: select channel action
          5. Sim: retune receiver and check passband detection
          6. Bandit: update with binary reward (1.0 or 0.0)
          7. Metrics: record stage latencies and hit status

        Returns:
          StepResult containing step telemetry, or None if replay stream is exhausted.
        """
        t0 = time.perf_counter_ns()

        # ── 1. Dataset Replay ────────────────────────────────────────────────
        t_rep_start = time.perf_counter_ns()
        pulse = self.replay.next_pulse()
        if pulse is None:
            return None
        t_rep_end = time.perf_counter_ns()

        # ── 2. State Buffer ──────────────────────────────────────────────────
        t_buf_start = time.perf_counter_ns()
        self.buffer.push(pulse)
        obs = self.buffer.get_observation()
        t_buf_end = time.perf_counter_ns()

        # ── 3. DSP / Feature Stub ────────────────────────────────────────────
        t_dsp_start = time.perf_counter_ns()
        _features = self.dsp.extract_features(obs)
        t_dsp_end = time.perf_counter_ns()

        # ── 4. Tier-1 SW-UCB Action Selection ────────────────────────────────
        # Note: select_action() takes no arguments; no ground truth is passed
        t_dec_start = time.perf_counter_ns()
        action = self.bandit.select_action()
        t_dec_end = time.perf_counter_ns()

        # ── 5. Retune Simulation & Detection ─────────────────────────────────
        t_sim_start = time.perf_counter_ns()
        det_result: DetectionResult = self.receiver_sim.retune_and_detect(
            target_channel=action,
            pulse_record=pulse,
        )
        t_sim_end = time.perf_counter_ns()

        # ── 6. Bandit Update (Strictly Binary Reward) ────────────────────────
        # Constraint: reward MUST be 1.0 if intercepted else 0.0
        binary_reward = det_result.binary_reward
        assert binary_reward in (0.0, 1.0), f"Reward must be binary, got {binary_reward}"

        t_upd_start = time.perf_counter_ns()
        self.bandit.update(action, binary_reward)
        t_upd_end = time.perf_counter_ns()

        t_total = time.perf_counter_ns() - t0
        self._step_count += 1

        # ── 7. Empirical Latency Profiling ───────────────────────────────────
        breakdown = LatencyBreakdown(
            replay_ns=t_rep_end - t_rep_start,
            buffer_ns=t_buf_end - t_buf_start,
            dsp_ns=t_dsp_end - t_dsp_start,
            decision_ns=t_dec_end - t_dec_start,
            sim_ns=t_sim_end - t_sim_start,
            update_ns=t_upd_end - t_upd_start,
            total_step_ns=t_total,
        )
        self.metrics.record_step(
            intercepted=det_result.intercepted,
            binary_reward=binary_reward,
            breakdown=breakdown,
        )

        return StepResult(
            step=self._step_count,
            action=action,
            intercepted=det_result.intercepted,
            binary_reward=binary_reward,
            retune_delay_us=det_result.retune_delay_us,
            pulse_channel=det_result.pulse_channel,
            pulse_freq_mhz=det_result.pulse_freq_mhz,
        )

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """
        Run the pipeline continuously until stream exhaustion or max_steps.

        Args:
            max_steps: Maximum number of steps to execute (None runs to exhaustion).

        Returns:
            Dictionary containing metrics summary (throughput, hit rate, empirical latencies).
        """
        steps_taken = 0
        while True:
            if max_steps is not None and steps_taken >= max_steps:
                break
            result = self.step()
            if result is None:
                break
            steps_taken += 1

        return self.metrics.get_summary()

    def reset(self, start_idx: int = 0) -> None:
        """Reset internal pipeline state."""
        self.replay.reset(start_idx=start_idx)
        self.buffer.reset()
        self.receiver_sim.reset()
        self.metrics.reset()
        self._step_count = 0
