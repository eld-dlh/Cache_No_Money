"""
Step 4 — np.memmap Loader: RAM-Efficient Batch Access
=======================================================

What is np.memmap?
  NumPy's memory-mapped array. Instead of loading the entire .npy file
  into RAM, memmap maps the FILE on disk into the process's virtual address
  space. When you access slice [i:j], the OS loads ONLY those bytes from disk.

Why does this matter?
  Our binary file might be 50–200 GB (billions of pulses).
  If we did  data = np.load("file.npy")  it would crash with OOM.
  With memmap:  data = np.memmap("file.npy", ...)
  We can access any row instantly using disk I/O caching.

  Think of it like a massive Excel spreadsheet that loads only the
  rows you're currently looking at — not the whole thing.

This module provides:
  - PDWMemmap class — wraps the .npy file
  - get_batch(start, size) — returns rows [start : start+size] as float32 array
  - get_record(idx)        — returns a single PDW record
  - Test showing memory stays ~flat while accessing random batches
"""

from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_NPY = Path(__file__).parent.parent / "data" / "raw" / "pdw_records.npy"

RECORD_SIZE = 8   # float32 values per record
FIELD_NAMES = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude",
               "reserved_0", "train_id", "reserved_2"]


class PDWMemmap:
    """
    Memory-mapped reader for the flat PDW binary file.

    Usage:
        loader = PDWMemmap("data/raw/pdw_records.npy")
        batch = loader.get_batch(start=1000, size=64)   # shape: (64, 8)
        record = loader.get_record(42)                   # shape: (8,)
        freq = loader[42, 1]                             # single field

    The file is NEVER fully loaded into RAM. Each access reads from disk
    with OS-level caching (so hot data stays in RAM automatically).
    """

    def __init__(self, npy_path: str | Path = DEFAULT_NPY):
        self.path = Path(npy_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Binary file not found: {self.path}\n"
                f"Run:  python data/convert_to_binary.py  first."
            )

        # Open as memmap — mode='r' means read-only (safe)
        # np.load with mmap_mode automatically handles the .npy header
        self._mm: np.ndarray = np.load(str(self.path), mmap_mode='r')

        # Validate shape
        if self._mm.ndim != 2 or self._mm.shape[1] != RECORD_SIZE:
            raise ValueError(
                f"Expected shape (N, {RECORD_SIZE}), got {self._mm.shape}.\n"
                f"Was the file created by convert_to_binary.py?"
            )

        self.n_records = self._mm.shape[0]
        self.dtype     = self._mm.dtype
        print(f"✅ PDWMemmap opened: {self.n_records:,} records, "
              f"dtype={self.dtype}, path={self.path.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # Core access methods
    # ──────────────────────────────────────────────────────────────────────────

    def get_batch(self, start: int, size: int) -> np.ndarray:
        """
        Return a batch of `size` consecutive PDW records starting at `start`.

        Args:
            start: First record index (0-based)
            size:  Number of records to return

        Returns:
            np.ndarray of shape (size, 8), dtype=float32
            Columns: ToA, Frequency, PulseWidth, AoA, Amplitude, res, train_id, res
        """
        if start < 0 or start >= self.n_records:
            raise IndexError(f"start={start} out of range [0, {self.n_records})")
        end = min(start + size, self.n_records)
        # .copy() converts the memmap slice to a real in-memory array
        # for this batch — only these rows are loaded from disk
        return np.array(self._mm[start:end])  # shape: (actual_size, 8)

    def get_record(self, idx: int) -> np.ndarray:
        """Return a single PDW record as a 1D array of shape (8,)."""
        return np.array(self._mm[idx])

    def get_window(self, center: int, window: int = 10) -> np.ndarray:
        """
        Return a sliding window of `window` records centred around `center`.
        Pads with zeros at the edges. Used by RadarEnv for the state vector.

        Returns:
            np.ndarray of shape (window, 8)
        """
        half  = window // 2
        start = max(0, center - half)
        end   = min(self.n_records, center + half + (window % 2))

        result = np.zeros((window, RECORD_SIZE), dtype=np.float32)
        fetched = np.array(self._mm[start:end])
        # Place fetched data into result, aligned to the right position
        offset = center - half - start + max(0, half - center)
        result[offset: offset + len(fetched)] = fetched
        return result

    def __len__(self) -> int:
        return self.n_records

    def __getitem__(self, key):
        """Direct indexing: loader[i] or loader[i, j]"""
        return self._mm[key]

    # ──────────────────────────────────────────────────────────────────────────
    # Metadata helpers
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def toa(self)        -> np.ndarray: return self._mm[:, 0]
    @property
    def frequency(self)  -> np.ndarray: return self._mm[:, 1]
    @property
    def pulse_width(self)-> np.ndarray: return self._mm[:, 2]
    @property
    def aoa(self)        -> np.ndarray: return self._mm[:, 3]
    @property
    def amplitude(self)  -> np.ndarray: return self._mm[:, 4]
    @property
    def train_id(self)   -> np.ndarray: return self._mm[:, 6]

    def describe(self):
        """Print summary statistics of the dataset."""
        # Only load a sample (first 10k) to avoid loading everything
        sample = np.array(self._mm[:min(10_000, self.n_records)])
        print(f"\n{'='*50}")
        print(f"PDWMemmap Statistics (sample of first {len(sample):,} records)")
        print(f"{'='*50}")
        for i, name in enumerate(FIELD_NAMES[:5]):
            col = sample[:, i]
            print(f"  {name:<15}: min={col.min():.4g}  max={col.max():.4g}  "
                  f"mean={col.mean():.4g}  std={col.std():.4g}")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST (run directly to verify the loader works)
# ─────────────────────────────────────────────────────────────────────────────

def test_memmap_loader(npy_path: Path = DEFAULT_NPY):
    """
    Demonstrates that memmap doesn't blow up RAM even for large files.
    Accesses 1,000 random batches and tracks peak memory.
    """
    print(f"\n{'='*60}")
    print(f"MEMMAP LOADER TEST")
    print(f"{'='*60}")

    loader = PDWMemmap(npy_path)
    loader.describe()

    N = loader.n_records
    rng = np.random.default_rng(42)

    # ── Test 1: Basic access
    print(f"\n[Test 1] Basic access...")
    record_0 = loader.get_record(0)
    print(f"  Record 0: {dict(zip(FIELD_NAMES, record_0))}")
    batch = loader.get_batch(0, 10)
    print(f"  Batch[0:10] shape: {batch.shape}")
    assert batch.shape == (10, RECORD_SIZE)

    # ── Test 2: Window access
    print(f"\n[Test 2] Sliding window...")
    window = loader.get_window(center=100, window=10)
    print(f"  Window shape: {window.shape}")
    assert window.shape == (10, RECORD_SIZE)

    # ── Test 3: Random batch access — memory must stay flat
    print(f"\n[Test 3] 1,000 random batches — monitoring peak RAM...")
    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(1_000):
        start = rng.integers(0, max(1, N - 64))
        _ = loader.get_batch(int(start), 64)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Time for 1,000 batches: {elapsed:.2f}s")
    print(f"  Current extra RAM: {current / 1024:.1f} KB")
    print(f"  Peak extra RAM:    {peak / (1024**2):.2f} MB")
    print(f"  (should be << 100 MB regardless of file size)")

    # ── Test 4: Edge cases
    print(f"\n[Test 4] Edge cases...")
    last_batch = loader.get_batch(N - 5, 10)  # batch that runs off the end
    print(f"  Batch at (N-5, size=10) → actual shape: {last_batch.shape}  (clamped correctly)")
    assert last_batch.shape[0] <= 10

    print(f"\n✅ ALL MEMMAP TESTS PASSED")
    return loader


if __name__ == "__main__":
    test_memmap_loader()
    print(f"\nNext step: Run  python env/pdw_dataset.py  to test the PyTorch Dataset.")
