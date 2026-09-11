"""
person5/noise_models.py — Noise Injection & Edge-Case Stress Testing
====================================================================

Wraps Person 1's RadarEnv to inject synthetic electronic warfare (EW) noise:
  - Low-SNR pulse dropout rate (0.0 to 0.5)
  - Frequency channel jitter / noise
  - Mid-episode emitter pattern shift (agile emitter countermeasure)
"""

from __future__ import annotations

import random
from typing import Any, Tuple

import gymnasium as gym
import numpy as np


class CorruptedRadarEnv(gym.Wrapper):
    """
    Gymnasium environment wrapper for noise injection and edge-case stress testing.
    
    Parameters
    ----------
    env : gym.Env
        Instance of Person 1's RadarEnv.
    dropout_rate : float
        Probability [0.0, 1.0] of dropping a pulse (simulating Low-SNR signal fade).
    channel_jitter : float
        Probability [0.0, 1.0] of scrambling channel indices (simulating frequency noise).
    pattern_shift_step : int or None
        Step number at which emitter abruptly shifts hopping sequence mid-episode.
    """

    def __init__(
        self,
        env: gym.Env,
        dropout_rate: float = 0.0,
        channel_jitter: float = 0.0,
        pattern_shift_step: int | None = None,
    ):
        super().__init__(env)
        self.dropout_rate = dropout_rate
        self.channel_jitter = channel_jitter
        self.pattern_shift_step = pattern_shift_step
        self.current_step = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.current_step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1

        # Execute step on underlying RadarEnv
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Apply Pulse Dropout (Low-SNR simulation)
        if self.dropout_rate > 0.0 and random.random() < self.dropout_rate:
            # Drop pulse presence in info telemetry
            info["pulse_present"] = False
            info["pulse_intercepted"] = False
            # Reduce reward to dwell cost penalty
            reward = -0.05

        # 2. Apply Channel Jitter (Frequency noise simulation)
        if self.channel_jitter > 0.0 and random.random() < self.channel_jitter:
            if info.get("pulse_present", False):
                # Randomly shift pulse channel by +/- 1 to 3 bins
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                orig_ch = info.get("pulse_channel", 0)
                info["pulse_channel"] = int(np.clip(orig_ch + shift, 0, 63))
                # Re-evaluate interception condition
                info["pulse_intercepted"] = (action == info["pulse_channel"])

        # 3. Apply Mid-Episode Agile Emitter Pattern Shift
        if self.pattern_shift_step is not None and self.current_step == self.pattern_shift_step:
            info["pattern_shifted"] = True
            # Force pulse channel shift to break Tier 2 pattern-lock
            if info.get("pulse_present", False):
                info["pulse_channel"] = (info.get("pulse_channel", 0) + 32) % 64
                info["pulse_intercepted"] = (action == info["pulse_channel"])

        return obs, reward, terminated, truncated, info
