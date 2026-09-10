"""
models/state_builder.py — Observation Normalisation & Context Feature Engineering
==================================================================================

This module sits between Person 1's raw RadarEnv observations and the DRQN
network. Raw PDW observations have wildly different scales:

    ToA:        0 – millions of µs
    Frequency:  0 – 18,000 MHz
    PulseWidth: 0.1 – 100 µs
    AoA:        0 – 360 degrees
    Amplitude:  -100 – 0 dBm

Feeding these raw into an LSTM causes numerical instability. This module:

  1. Normalises the (10, 5) pulse window to [0, 1] range per field.
  2. Converts absolute ToA → ΔToA (inter-pulse intervals) for temporal patterns.
  3. Engineers 5 operational context features that give the network situational
     awareness beyond the raw pulse data.
  4. Outputs a combined state ready for the DRQN.

Usage:
    from models.state_builder import StateBuilder

    builder = StateBuilder(n_channels=64, max_steps=500)

    # At each env step:
    state = builder.build_state(obs, info)
    # state['sequence']  → tensor (10, 5) normalised pulse window
    # state['context']   → tensor (5,)   engineered context features

    # On episode reset:
    builder.reset()
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT NORMALISATION RANGES (from Person 1's dataset inspection)
# ─────────────────────────────────────────────────────────────────────────────
# These are safe upper bounds. Values outside are clamped to [0, 1].

NORM_RANGES = {
    "delta_toa":    (0.0, 1000.0),      # µs — typical inter-pulse interval
    "frequency":    (0.0, 18_000.0),    # MHz — full radar band
    "pulse_width":  (0.0, 100.0),       # µs
    "aoa":          (0.0, 360.0),       # degrees
    "amplitude":    (-120.0, 0.0),      # dBm (note: negative range)
}


class StateBuilder:
    """
    Transforms raw RadarEnv observations into normalised, feature-enriched
    states suitable for DRQN input.

    Parameters
    ----------
    n_channels : int
        Number of discrete frequency channels (default: 64).
    max_steps : int
        Maximum steps per episode (default: 500).
    hit_ema_alpha : float
        Exponential moving average decay for recent_hit_rate (default: 0.2).
        Higher = more responsive to recent hits/misses.
    miss_cap : int
        Maximum consecutive misses before the count saturates (default: 10).
        Used to normalise consecutive_misses to [0, 1].
    n_quadrants : int
        Number of frequency quadrants for channel_occupancy_hist (default: 4).
        Splits the 64 channels into 4 groups of 16 for occupancy tracking.
    """

    def __init__(
        self,
        n_channels: int = 64,
        max_steps: int = 500,
        hit_ema_alpha: float = 0.2,
        miss_cap: int = 10,
        n_quadrants: int = 4,
    ):
        self.n_channels = n_channels
        self.max_steps = max_steps
        self.hit_ema_alpha = hit_ema_alpha
        self.miss_cap = miss_cap
        self.n_quadrants = n_quadrants

        # ── Internal tracking state (reset each episode)
        self._recent_hit_rate: float = 0.0
        self._consecutive_misses: int = 0
        self._last_action: int = 0
        self._current_step: int = 0
        self._prev_toa: float = 0.0
        self._first_obs: bool = True

        # Track which quadrants had recent pulse activity
        # (rolling window of the last 10 observed pulse channels)
        self._recent_channels: deque = deque(maxlen=10)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all internal state at the start of a new episode."""
        self._recent_hit_rate = 0.0
        self._consecutive_misses = 0
        self._last_action = 0
        self._current_step = 0
        self._prev_toa = 0.0
        self._first_obs = True
        self._recent_channels.clear()

    def build_state(
        self,
        obs: np.ndarray,
        info: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Build the full state representation from a raw RadarEnv observation.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation from RadarEnv, shape (window_size, 5).
            Columns: [ToA, Frequency, PulseWidth, AoA, Amplitude].
        info : dict
            Info dict from RadarEnv.step() containing at minimum:
            - 'intercepted' (bool): whether the last action was a hit
            - 'chosen_channel' (int): the channel the agent selected
            - 'step' (int): current step number in the episode

        Returns
        -------
        dict with keys:
            'sequence' : torch.Tensor of shape (window_size, 5)
                Normalised pulse window (ΔToA, Freq, PW, AoA, Amp).
            'context'  : torch.Tensor of shape (5,)
                Engineered context features.
        """
        # ── 1. Normalise the pulse window
        norm_seq = self._normalise_window(obs)

        # ── 2. Update internal tracking from info
        self._update_tracking(info)

        # ── 3. Build context features
        context = self._build_context()

        return {
            "sequence": torch.from_numpy(norm_seq).float(),
            "context": torch.tensor(context, dtype=torch.float32),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # NORMALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_window(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalise the raw (window_size, 5) observation to [0, 1].

        Key transformation: Column 0 (ToA) is converted from absolute
        timestamps to inter-pulse intervals (ΔToA), which captures the
        temporal rhythm of emitter transmissions.

        Parameters
        ----------
        obs : np.ndarray, shape (W, 5)

        Returns
        -------
        np.ndarray, shape (W, 5), values in [0, 1]
        """
        result = obs.copy().astype(np.float32)
        W = result.shape[0]

        # ── Convert absolute ToA → ΔToA (inter-pulse interval)
        # For the first pulse in the window, ΔToA = 0
        toa_col = result[:, 0].copy()

        # Find which rows are non-zero (i.e., not zero-padded)
        active_mask = np.any(result != 0.0, axis=1)

        # Compute deltas only for active (non-padded) rows
        delta_toa = np.zeros(W, dtype=np.float32)
        for i in range(1, W):
            if active_mask[i] and active_mask[i - 1]:
                delta_toa[i] = max(0.0, toa_col[i] - toa_col[i - 1])

        result[:, 0] = delta_toa

        # ── Normalise each column to [0, 1]
        ranges = [
            NORM_RANGES["delta_toa"],
            NORM_RANGES["frequency"],
            NORM_RANGES["pulse_width"],
            NORM_RANGES["aoa"],
            NORM_RANGES["amplitude"],
        ]

        for col_idx, (lo, hi) in enumerate(ranges):
            span = hi - lo
            if span > 0:
                result[:, col_idx] = (result[:, col_idx] - lo) / span
            else:
                result[:, col_idx] = 0.0

        # ── Clamp to [0, 1] for safety
        np.clip(result, 0.0, 1.0, out=result)

        # ── Zero out padded rows (keep them as clean zeros)
        result[~active_mask] = 0.0

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # CONTEXT FEATURES
    # ──────────────────────────────────────────────────────────────────────────

    def _update_tracking(self, info: dict) -> None:
        """
        Update internal tracking variables from the latest step info.

        This must be called BEFORE _build_context() so the context
        reflects the current state.
        """
        intercepted = info.get("intercepted", False)
        chosen_channel = info.get("chosen_channel", 0)
        pulse_channel = info.get("pulse_channel", 0)
        step = info.get("step", self._current_step + 1)

        # ── Update hit rate EMA
        hit_val = 1.0 if intercepted else 0.0
        self._recent_hit_rate = (
            self.hit_ema_alpha * hit_val
            + (1.0 - self.hit_ema_alpha) * self._recent_hit_rate
        )

        # ── Update consecutive misses
        if intercepted:
            self._consecutive_misses = 0
        else:
            self._consecutive_misses += 1

        # ── Track last action and step
        self._last_action = chosen_channel
        self._current_step = step

        # ── Track recent pulse channels (only channels we successfully
        #    intercepted — in real hardware we can't observe missed channels)
        if intercepted:
            self._recent_channels.append(pulse_channel)

    def _build_context(self) -> list[float]:
        """
        Build the 5-dimensional context feature vector.

        Returns
        -------
        list of 5 floats, each in [0, 1]:
            [0] recent_hit_rate        — EMA of intercept success
            [1] consecutive_misses     — normalised miss streak
            [2] last_action_channel    — normalised channel index
            [3] dwell_budget_remaining — fraction of episode remaining
            [4] channel_occupancy_hist — fraction of quadrants with recent activity
        """
        # Feature 1: Recent hit rate (already in [0, 1] from EMA)
        f_hit_rate = float(np.clip(self._recent_hit_rate, 0.0, 1.0))

        # Feature 2: Consecutive misses, normalised by miss_cap
        f_misses = float(min(self._consecutive_misses, self.miss_cap)) / self.miss_cap

        # Feature 3: Last action channel, normalised to [0, 1]
        f_last_action = float(self._last_action) / max(self.n_channels - 1, 1)

        # Feature 4: Dwell budget remaining (1.0 at start, 0.0 at end)
        f_budget = 1.0 - (float(self._current_step) / max(self.max_steps, 1))
        f_budget = float(np.clip(f_budget, 0.0, 1.0))

        # Feature 5: Channel occupancy — what fraction of frequency quadrants
        # have had recent pulse activity
        f_occupancy = self._compute_quadrant_occupancy()

        return [f_hit_rate, f_misses, f_last_action, f_budget, f_occupancy]

    def _compute_quadrant_occupancy(self) -> float:
        """
        Compute what fraction of frequency quadrants have recent activity.

        Divides the N_CHANNELS into n_quadrants equal groups and checks
        which quadrants have been observed in _recent_channels.

        Returns
        -------
        float in [0, 1]: fraction of quadrants with recent activity.
        """
        if len(self._recent_channels) == 0:
            return 0.0

        channels_per_quad = max(self.n_channels // self.n_quadrants, 1)
        active_quadrants = set()

        for ch in self._recent_channels:
            quad_idx = min(int(ch) // channels_per_quad, self.n_quadrants - 1)
            active_quadrants.add(quad_idx)

        return len(active_quadrants) / self.n_quadrants


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("StateBuilder — Standalone Smoke Test")
    print("=" * 60)

    builder = StateBuilder(n_channels=64, max_steps=500)
    builder.reset()

    # Simulate a fake observation (10 pulses × 5 fields)
    fake_obs = np.zeros((10, 5), dtype=np.float32)
    # Fill last 3 rows with realistic-ish values
    fake_obs[7] = [100.0, 5000.0, 2.5, 45.0, -30.0]
    fake_obs[8] = [150.0, 5000.0, 2.5, 46.0, -31.0]
    fake_obs[9] = [200.0, 9000.0, 3.0, 90.0, -25.0]

    fake_info = {
        "intercepted": True,
        "chosen_channel": 17,
        "pulse_channel": 17,
        "step": 1,
    }

    state = builder.build_state(fake_obs, fake_info)

    print(f"\n  Sequence shape: {state['sequence'].shape}")
    print(f"  Sequence dtype: {state['sequence'].dtype}")
    print(f"  Sequence range: [{state['sequence'].min():.4f}, {state['sequence'].max():.4f}]")
    print(f"\n  Context shape:  {state['context'].shape}")
    print(f"  Context dtype:  {state['context'].dtype}")
    print(f"  Context values: {state['context'].tolist()}")

    # Verify all values are in [0, 1]
    assert state["sequence"].min() >= 0.0, "Sequence has negative values!"
    assert state["sequence"].max() <= 1.0, "Sequence has values > 1!"
    assert state["context"].min() >= 0.0, "Context has negative values!"
    assert state["context"].max() <= 1.0, "Context has values > 1!"

    print(f"\n  ✅ All values in [0, 1] — normalisation working correctly.")
    print("=" * 60)
