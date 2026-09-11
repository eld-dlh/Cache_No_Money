"""
systems/receiver_sim.py
=======================
RF Receiver Retune Simulation and Detection Component.

Models the physical behavior of a cognitive electronic warfare receiver:
  - Local Oscillator (LO) / Phase Locked Loop (PLL) retuning settling delay
  - Passband channel filtering
  - Pulse interception determination
  - Strictly binary reward generation (1.0 for intercepted, 0.0 for miss)

IMPORTANT: The ground-truth pulse_channel is recorded only for evaluation
diagnostics and telemetry. The bandit must only receive binary_reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from systems.dsp_stub import DSPStub, N_CHANNELS


@dataclass(frozen=True)
class DetectionResult:
    """Telemetry and outcome of a single scan/step."""
    intercepted: bool
    binary_reward: float       # Strictly 1.0 if intercepted else 0.0
    chosen_channel: int
    pulse_channel: int         # Ground truth for logging/eval only
    pulse_freq_mhz: float
    retune_delay_us: float


class ReceiverSim:
    """
    Simulates RF receiver tuning delay and pulse interception.
    """

    def __init__(
        self,
        dsp: DSPStub | None = None,
        base_retune_us: float = 2.0,
        per_channel_retune_us: float = 0.1,
        initial_channel: int = 0,
    ):
        """
        Args:
            dsp: DSPStub instance for channel mapping (defaults to standard 64 channels).
            base_retune_us: Minimum PLL settling latency in microseconds.
            per_channel_retune_us: Additional PLL settling latency per channel distance.
            initial_channel: Starting tuned channel.
        """
        self.dsp = dsp if dsp is not None else DSPStub()
        self.base_retune_us = base_retune_us
        self.per_channel_retune_us = per_channel_retune_us
        self.current_channel = initial_channel

    def retune_and_detect(
        self,
        target_channel: int,
        pulse_record: np.ndarray,
    ) -> DetectionResult:
        """
        Retune receiver to target_channel and evaluate interception of pulse_record.

        Args:
            target_channel: Selected channel index [0, n_channels - 1].
            pulse_record: PDW array where field [1] is Frequency (MHz).

        Returns:
            DetectionResult dataclass containing hit status, binary reward, and telemetry.
        """
        if not (0 <= target_channel < self.dsp.n_channels):
            raise ValueError(
                f"Target channel {target_channel} out of range [0, {self.dsp.n_channels})"
            )

        # 1. Simulate retuning latency (LO / PLL settling)
        channel_hop_distance = abs(target_channel - self.current_channel)
        retune_delay_us = self.base_retune_us + self.per_channel_retune_us * channel_hop_distance
        self.current_channel = target_channel

        # 2. Extract incoming pulse frequency
        pulse_freq_mhz = float(pulse_record[1])
        pulse_channel = int(self.dsp.freq_to_channel(pulse_freq_mhz))

        # 3. Determine interception
        intercepted = (target_channel == pulse_channel)

        # 4. Strict binary reward constraint: 1.0 if intercepted else 0.0
        binary_reward = 1.0 if intercepted else 0.0

        return DetectionResult(
            intercepted=intercepted,
            binary_reward=binary_reward,
            chosen_channel=target_channel,
            pulse_channel=pulse_channel,
            pulse_freq_mhz=pulse_freq_mhz,
            retune_delay_us=retune_delay_us,
        )

    def reset(self, initial_channel: int = 0) -> None:
        """Reset receiver to initial channel."""
        self.current_channel = initial_channel
