"""
Step 5 — PyTorch Dataset Wrapper
=================================

What this does:
  Wraps the PDWMemmap loader in a standard PyTorch Dataset.

Why do we need this?
  PyTorch's DataLoader expects objects with __len__ and __getitem__.
  Once you have a Dataset, you can do:

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    for batch in loader:
        # batch is a tensor of shape (64, window_size, 8)
        ...

  This is the standard interface that Person 3 (DRQN engineer) will use
  to train the neural network.

What does each "sample" look like?
  A sample is a SLIDING WINDOW of consecutive PDW records.
  window_size=10 → shape (10, 5) — last 10 pulses, 5 fields each.
  This is the "observation" fed to the neural network.

  Why a window? A single PDW record has no temporal context.
  The network needs to see SEQUENCE to detect patterns like:
  "this emitter hops every 50 µs between 2.4 GHz and 3.1 GHz".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from env.memmap_loader import PDWMemmap

DEFAULT_NPY = Path(__file__).parent.parent / "data" / "raw" / "pdw_records.npy"

# Only these 5 PDW fields are fed to the model (no reserved fields)
PDW_FIELD_INDICES = [0, 1, 2, 3, 4]  # ToA, Freq, PW, AoA, Amp


class PDWDataset(Dataset):
    """
    PyTorch Dataset backed by a memory-mapped PDW binary file.

    Each sample is a sliding window of `window_size` consecutive PDW records,
    returned as a float32 tensor of shape (window_size, 5).

    Args:
        npy_path:    Path to the .npy binary (from convert_to_binary.py)
        window_size: Number of consecutive pulses per sample (default: 10)
        start_idx:   First valid record index (for train/test splits)
        end_idx:     Last valid record index (exclusive)
        normalise:   If True, apply per-field normalisation (recommended for DRQN)

    Example:
        dataset = PDWDataset("data/raw/pdw_records.npy", window_size=10,
                             start_idx=0, end_idx=int(0.7 * N))
        sample = dataset[0]       # tensor shape: (10, 5)
        sample = dataset[100]     # different window
    """

    # Rough normalisation stats (will be updated when we inspect the real dataset)
    # Format: (mean, std) for each of the 5 PDW fields
    # These get refined in compute_norm_stats() below
    _DEFAULT_STATS = {
        "mean": np.array([0.0, 10_000.0, 50.0, 0.0,  -50.0], dtype=np.float32),
        "std":  np.array([1e6,  5_000.0, 50.0, 90.0,  20.0], dtype=np.float32),
    }

    def __init__(
        self,
        npy_path:    str | Path   = DEFAULT_NPY,
        window_size: int          = 10,
        start_idx:   int          = 0,
        end_idx:     int | None   = None,
        normalise:   bool         = False,
    ):
        self.loader      = PDWMemmap(npy_path)
        self.window_size = window_size
        self.start_idx   = start_idx
        self.end_idx     = end_idx if end_idx is not None else len(self.loader)
        self.normalise   = normalise

        # Clamp to valid range
        self.start_idx = max(0,                   self.start_idx)
        self.end_idx   = min(len(self.loader),    self.end_idx)

        # Total number of VALID windows we can produce.
        # We need window_size records starting at each index,
        # so the last valid start is (end_idx - window_size).
        self._n_samples = max(0, self.end_idx - self.start_idx - self.window_size + 1)

        # Normalisation stats (computed lazily or set manually)
        self._norm_mean: np.ndarray | None = None
        self._norm_std:  np.ndarray | None = None

        print(f"✅ PDWDataset: {self._n_samples:,} windows of size {window_size} "
              f"from records [{self.start_idx:,} – {self.end_idx:,}]")

    # ──────────────────────────────────────────────────────────────────────────
    # Required Dataset interface
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Total number of samples this dataset provides."""
        return self._n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return the idx-th sample as a float32 tensor.

        Args:
            idx: Sample index (0 to len(self)-1)

        Returns:
            torch.Tensor of shape (window_size, 5)
            Columns in order: ToA, Frequency, PulseWidth, AoA, Amplitude
        """
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._n_samples})")

        # Map dataset index → absolute record index in the memmap
        record_start = self.start_idx + idx

        # Fetch window from memmap (only reads from disk for these rows)
        window = self.loader.get_batch(record_start, self.window_size)  # (W, 8)

        # Keep only the 5 PDW fields (drop reserved columns)
        window = window[:, PDW_FIELD_INDICES]  # (W, 5)

        # Optional normalisation
        if self.normalise and self._norm_mean is not None:
            window = (window - self._norm_mean) / (self._norm_std + 1e-8)

        return torch.from_numpy(window.copy())   # (window_size, 5), float32

    # ──────────────────────────────────────────────────────────────────────────
    # Normalisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def compute_norm_stats(self, n_samples: int = 10_000) -> dict:
        """
        Estimate mean/std from a random sample of records.
        Call this once after creating the dataset, then enable normalise=True.

        Args:
            n_samples: How many records to sample for estimation

        Returns:
            dict with keys 'mean' and 'std' (np.ndarray of shape (5,))
        """
        rng = np.random.default_rng(42)
        n   = min(n_samples, self._n_samples)
        idxs = rng.integers(0, self._n_samples, size=n)

        # Collect a flat array of PDW records
        all_records = []
        for i in idxs:
            rec = self.loader.get_record(self.start_idx + i)
            all_records.append(rec[PDW_FIELD_INDICES])

        arr = np.stack(all_records, axis=0)  # (n, 5)
        self._norm_mean = arr.mean(axis=0).astype(np.float32)
        self._norm_std  = arr.std(axis=0).astype(np.float32)

        print(f"\n📐 Normalisation stats (from {n:,} samples):")
        for i, name in enumerate(["ToA", "Freq", "PW", "AoA", "Amp"]):
            print(f"   {name}: mean={self._norm_mean[i]:.4g}  std={self._norm_std[i]:.4g}")

        return {"mean": self._norm_mean, "std": self._norm_std}

    def set_norm_stats(self, mean: np.ndarray, std: np.ndarray):
        """Manually set normalisation stats (e.g. load from a saved file)."""
        self._norm_mean = np.array(mean, dtype=np.float32)
        self._norm_std  = np.array(std,  dtype=np.float32)
        self.normalise  = True


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_pdw_dataset(npy_path: Path = DEFAULT_NPY):
    """Test the Dataset and DataLoader interface."""
    print(f"\n{'='*60}")
    print(f"PYTORCH DATASET TEST")
    print(f"{'='*60}")

    # 1. Basic dataset
    ds = PDWDataset(npy_path, window_size=10)

    print(f"\n[Test 1] __len__ and __getitem__...")
    print(f"  Dataset length: {len(ds):,}")
    sample = ds[0]
    print(f"  ds[0] shape: {sample.shape}")
    assert sample.shape == (10, 5), f"Expected (10,5) got {sample.shape}"
    assert sample.dtype == torch.float32

    print(f"\n[Test 2] Sample values (first window):")
    field_names = ["ToA", "Freq", "PW", "AoA", "Amp"]
    for i, name in enumerate(field_names):
        print(f"  {name}: {sample[:, i].numpy()[:3]} ...")

    # 2. DataLoader
    print(f"\n[Test 3] DataLoader (batch_size=32)...")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    first_batch = next(iter(loader))
    print(f"  First batch shape: {first_batch.shape}")
    assert first_batch.shape == (32, 10, 5), f"Expected (32,10,5) got {first_batch.shape}"

    # 3. Normalisation
    print(f"\n[Test 4] Normalisation stats computation...")
    stats = ds.compute_norm_stats(n_samples=1_000)
    ds.set_norm_stats(stats["mean"], stats["std"])
    sample_norm = ds[0]
    print(f"  Normalised ds[0] mean: {sample_norm.mean():.4g}  (should be ~0)")

    print(f"\n✅ ALL DATASET TESTS PASSED")
    return ds


if __name__ == "__main__":
    test_pdw_dataset()
    print(f"\nNext step: Run  python env/radar_env.py  to build the Gymnasium environment.")
