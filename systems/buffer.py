"""
systems/buffer.py
=================
State Buffer Component for the Person 4 Systems Pipeline.

Maintains a thread-safe sliding window of the most recent PDW pulses,
matching the Gymnasium environment observation space Box(shape=(10, 5)).

Fields in observation:
  [0] ToA (µs)
  [1] Frequency (MHz)
  [2] PulseWidth (µs)
  [3] AoA (degrees)
  [4] Amplitude (dBm)
"""

from __future__ import annotations

import threading
import numpy as np

# Standard window dimensions from project specification
DEFAULT_WINDOW_SIZE = 10
PDW_NUM_FIELDS = 5


class StateBuffer:
    """
    Sliding observation buffer storing the latest W pulses.

    Thread-safe implementation using an internal mutex lock, allowing
    seamless integration into both single-threaded and concurrent pipelines.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.window_size = window_size
        self._lock = threading.Lock()
        self._buffer = np.zeros((self.window_size, PDW_NUM_FIELDS), dtype=np.float32)
        self._count = 0

    def push(self, pulse: np.ndarray) -> None:
        """
        Push a new PDW record into the sliding buffer.

        Args:
            pulse: 1D array of at least 5 elements [ToA, Freq, PW, AoA, Amp, ...]
        """
        if len(pulse) < PDW_NUM_FIELDS:
            raise ValueError(
                f"Expected pulse record with at least {PDW_NUM_FIELDS} fields, got shape {pulse.shape}"
            )

        features = np.asarray(pulse[:PDW_NUM_FIELDS], dtype=np.float32)

        with self._lock:
            # Shift buffer left by 1 and place the latest pulse at the end
            self._buffer = np.roll(self._buffer, shift=-1, axis=0)
            self._buffer[-1] = features
            self._count += 1

    def get_observation(self) -> np.ndarray:
        """
        Get the current observation window.

        Returns:
            np.ndarray of shape (window_size, 5), dtype=float32
        """
        with self._lock:
            return self._buffer.copy()

    def reset(self) -> None:
        """Reset the buffer to zeros and zero the pulse counter."""
        with self._lock:
            self._buffer.fill(0.0)
            self._count = 0

    @property
    def is_full(self) -> bool:
        """True if at least window_size pulses have been pushed."""
        with self._lock:
            return self._count >= self.window_size

    @property
    def count(self) -> int:
        """Total number of pulses pushed into the buffer since last reset."""
        with self._lock:
            return self._count

    def __len__(self) -> int:
        """Current number of valid pulses stored, capped at window_size."""
        with self._lock:
            return min(self._count, self.window_size)
