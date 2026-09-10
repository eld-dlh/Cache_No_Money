"""
systems/dsp_stub.py
===================
DSP / Feature Extraction Stub for the Person 4 Systems Pipeline.

Provides RF channel discretization, frequency conversion utilities,
and observation feature formatting.

Adheres strictly to the project channel specification:
  - 64 discrete channels
  - Frequency range: 0 MHz to 18,000 MHz (281.25 MHz per bin)

IMPORTANT: Does NOT access, compute, or expose future or ground-truth
pulse channels to the bandit decision layer.
"""

from __future__ import annotations

import numpy as np

# Project standard constants (matching env/radar_env.py)
N_CHANNELS = 64
FREQ_MIN = 0.0       # MHz
FREQ_MAX = 18_000.0  # MHz


class DSPStub:
    """
    DSP / Feature processing layer.

    Bridges the raw PDW state buffer and the algorithmic decision layer.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        freq_min: float = FREQ_MIN,
        freq_max: float = FREQ_MAX,
    ):
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        if freq_max <= freq_min:
            raise ValueError(f"freq_max ({freq_max}) must be greater than freq_min ({freq_min})")

        self.n_channels = n_channels
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.channel_width = (self.freq_max - self.freq_min) / self.n_channels

    def freq_to_channel(self, freq_mhz: float | np.ndarray) -> int | np.ndarray:
        """
        Map frequency in MHz to a discrete channel index in [0, n_channels - 1].
        """
        norm = (freq_mhz - self.freq_min) / (self.freq_max - self.freq_min)
        if isinstance(freq_mhz, np.ndarray):
            ch = (norm * self.n_channels).astype(int)
        else:
            ch = int(norm * self.n_channels)
        return np.clip(ch, 0, self.n_channels - 1)

    def channel_to_freq_centre(self, channel: int) -> float:
        """Return the center frequency in MHz for a given channel index."""
        if not (0 <= channel < self.n_channels):
            raise ValueError(f"Channel index {channel} out of range [0, {self.n_channels})")
        return self.freq_min + (channel + 0.5) * self.channel_width

    def channel_to_freq_range(self, channel: int) -> tuple[float, float]:
        """Return the (low_MHz, high_MHz) passband bounds for a given channel index."""
        if not (0 <= channel < self.n_channels):
            raise ValueError(f"Channel index {channel} out of range [0, {self.n_channels})")
        low = self.freq_min + channel * self.channel_width
        high = low + self.channel_width
        return low, high

    def extract_features(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract formatted feature representation from the state buffer observation.

        Args:
            obs: np.ndarray of shape (W, 5) representing past pulses.

        Returns:
            np.ndarray of shape (W * 5,) float32 flattened features.
        """
        return np.asarray(obs, dtype=np.float32).flatten()
