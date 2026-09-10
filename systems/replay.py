"""
systems/replay.py
=================
Dataset Replay Component for the Person 4 Systems Pipeline.

Responsible for sequential pulse playback from the memory-mapped
32-byte PDW binary file (env/memmap_loader.py), or generating a
deterministic synthetic pulse stream when binary files are absent.

Each pulse record conforms to the project specification (8 x float32):
  [0] ToA (µs)
  [1] Frequency (MHz)
  [2] PulseWidth (µs)
  [3] AoA (degrees)
  [4] Amplitude (dBm)
  [5] reserved_0 (0.0)
  [6] train_id
  [7] emitter_id
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
import numpy as np

# Default dataset binary path (matches Person 1 / Person 2 convention)
DEFAULT_NPY = Path(__file__).parent.parent / "data" / "raw" / "pdw_records.npy"


class PDWReplay:
    """
    Sequential pulse stream reader.

    Supports reading directly from `PDWMemmap` with zero-copy disk mapping,
    or falling back to a deterministic synthetic radar emitter generator
    for development, testing, and CI environments without raw files.
    """

    def __init__(
        self,
        npy_path: str | Path | None = None,
        start_idx: int = 0,
        end_idx: int | None = None,
        loop: bool = False,
        synthetic_fallback: bool = True,
        synthetic_count: int = 10_000,
        seed: int | None = 42,
    ):
        """
        Args:
            npy_path: Path to the pdw_records.npy file.
            start_idx: First pulse index to replay (default 0).
            end_idx: Upper boundary index for replay (exclusive).
            loop: If True, loops back to start_idx when end_idx is reached.
            synthetic_fallback: If True and npy_path does not exist, generates synthetic pulses.
            synthetic_count: Number of synthetic pulses to generate if fallback is used.
            seed: RNG seed for synthetic pulse generation.
        """
        self.npy_path = Path(npy_path) if npy_path is not None else DEFAULT_NPY
        self.start_idx = max(0, start_idx)
        self.loop = loop
        self._cursor = self.start_idx
        self.is_synthetic = False
        self._data: np.ndarray | None = None

        if self.npy_path.exists():
            # Real dataset via memmap
            from env.memmap_loader import PDWMemmap
            self._loader = PDWMemmap(self.npy_path)
            total = len(self._loader)
            self.end_idx = min(total, end_idx) if end_idx is not None else total
            self.total_records = self.end_idx - self.start_idx
        elif synthetic_fallback:
            # Deterministic synthetic radar pulse generator
            self.is_synthetic = True
            self._loader = None
            self._data = self._generate_synthetic_pulses(synthetic_count, seed=seed)
            total = len(self._data)
            self.end_idx = min(total, end_idx) if end_idx is not None else total
            self.total_records = self.end_idx - self.start_idx
        else:
            raise FileNotFoundError(
                f"Binary dataset not found at {self.npy_path} and synthetic_fallback is False."
            )

    @staticmethod
    def _generate_synthetic_pulses(count: int, seed: int | None = 42) -> np.ndarray:
        """
        Generate deterministic multi-emitter synthetic PDW records (count x 8).
        Simulates frequency agility, fixed emitters, and pulse repetition intervals.
        """
        rng = np.random.default_rng(seed)
        records = np.zeros((count, 8), dtype=np.float32)

        # 4 simulated emitters with distinct center frequencies (MHz) and PRIs (µs)
        # Channels: e.g. ~2000 MHz (ch 7), ~5600 MHz (ch 19), ~9200 MHz (ch 32), ~14000 MHz (ch 49)
        emitters = [
            {"id": 1, "center_freq": 2000.0, "freq_jitter": 50.0, "pri": 100.0, "pw": 1.2, "aoa": 45.0, "amp": -40.0},
            {"id": 2, "center_freq": 5625.0, "freq_jitter": 80.0, "pri": 150.0, "pw": 2.5, "aoa": 120.0, "amp": -35.0},
            {"id": 3, "center_freq": 9281.0, "freq_jitter": 30.0, "pri": 80.0,  "pw": 0.8, "aoa": -60.0, "amp": -50.0},
            {"id": 4, "center_freq": 14062.0, "freq_jitter": 120.0, "pri": 200.0, "pw": 3.0, "aoa": 15.0, "amp": -45.0},
        ]

        curr_toa = 0.0
        for i in range(count):
            # Pick active emitter (non-stationary: distribution shifts over time)
            phase = (i // 1000) % 4
            weights = [0.1, 0.1, 0.1, 0.1]
            weights[phase] = 0.7  # Dominant active emitter shifts every 1000 pulses
            em_idx = rng.choice(len(emitters), p=weights)
            em = emitters[em_idx]

            # Ingest fields
            dt = max(1.0, em["pri"] + rng.normal(0, 5.0))
            curr_toa += dt
            freq = float(np.clip(em["center_freq"] + rng.normal(0, em["freq_jitter"]), 10.0, 17990.0))
            pw = float(max(0.1, em["pw"] + rng.normal(0, 0.1)))
            aoa = float(np.clip(em["aoa"] + rng.normal(0, 2.0), -180.0, 180.0))
            amp = float(em["amp"] + rng.normal(0, 1.5))

            records[i, 0] = curr_toa
            records[i, 1] = freq
            records[i, 2] = pw
            records[i, 3] = aoa
            records[i, 4] = amp
            records[i, 5] = 0.0                   # reserved_0
            records[i, 6] = float(phase + 1)      # train_id
            records[i, 7] = float(em["id"])       # emitter_id

        return records

    def reset(self, start_idx: int | None = None) -> None:
        """Reset replay cursor to start position."""
        if start_idx is not None:
            self.start_idx = max(0, start_idx)
        self._cursor = self.start_idx

    def has_next(self) -> bool:
        """Check if more pulses are available in the current pass."""
        if self.loop:
            return True
        return self._cursor < self.end_idx

    def next_pulse(self) -> np.ndarray | None:
        """
        Fetch the next PDW pulse.

        Returns:
            np.ndarray of shape (8,), dtype=float32, or None if end of data reached.
        """
        if self._cursor >= self.end_idx:
            if self.loop and self.total_records > 0:
                self._cursor = self.start_idx
            else:
                return None

        if self._loader is not None:
            record = self._loader.get_record(self._cursor)
        else:
            assert self._data is not None
            record = self._data[self._cursor].copy()

        self._cursor += 1
        return record

    @property
    def current_cursor(self) -> int:
        """Current zero-based index in the dataset."""
        return self._cursor

    @property
    def remaining(self) -> int:
        """Number of pulses remaining in this pass."""
        return max(0, self.end_idx - self._cursor)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        pulse = self.next_pulse()
        if pulse is None:
            raise StopIteration
        return pulse

    def __len__(self) -> int:
        return self.total_records
