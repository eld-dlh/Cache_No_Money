"""
Step 2 — Parse HDF5 Files into PDW Records
===========================================

ACTUAL dataset format (discovered by inspection):
  Each .h5 file contains one "pulse train" with:
    data/             shape=(N, 5)  float32
                      columns: [ToA_us, Frequency_MHz, PulseWidth_us, AoA_deg, Amplitude_dBm]
    labels/           shape=(N, 1)  int  — emitter ID for each pulse
    metadata/feature_names  → [b'ToA', b'Frequency', b'PulseWidth', b'AoA', b'Amplitude']
    metadata/transmitters/  — per-emitter config (frequencies, PRIs, etc.)

  One .h5 file = one pulse train (a simulated scenario with multiple emitters)
  The dataset has ~6000 such files across scan/ and stare/ folders.

What this script does:
  1. Reads ALL .h5 files in data/raw/
  2. Extracts data (N×5) and labels (N×1)
  3. Adds a train_id column (derived from filename, e.g. "config_1" → 1)
  4. Concatenates into one big pandas DataFrame
  5. Saves as data/raw/pdw_parsed.parquet

Frequency range in the dataset: ~100 MHz – 18000 MHz (based on dataset docs)
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR  = Path(__file__).parent / "raw"
OUT_FILE = RAW_DIR / "pdw_parsed.parquet"

# Confirmed column order from metadata/feature_names in the .h5 files
PDW_COLS = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]


def parse_h5_file(h5_path: Path, train_id: int) -> pd.DataFrame | None:
    """
    Parse one .h5 pulse-train file into a DataFrame.

    Args:
        h5_path:  Path to the .h5 file
        train_id: Integer ID to assign to this pulse train

    Returns:
        DataFrame with columns [ToA, Frequency, PulseWidth, AoA, Amplitude,
                                 emitter_id, train_id]
        or None if the file is malformed / empty.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                return None

            # Main PDW array: shape (N, 5)
            data = np.array(f["data"], dtype=np.float32)
            if data.ndim != 2 or data.shape[1] != 5 or len(data) == 0:
                return None

            # Emitter labels: shape (N, 1) or (N,)
            emitter_ids = None
            if "labels" in f:
                emitter_ids = np.array(f["labels"]).flatten().astype(np.int32)

    except Exception as e:
        print(f"  Warning: could not read {h5_path.name}: {e}")
        return None

    # Build DataFrame
    df = pd.DataFrame(data, columns=PDW_COLS)
    df["train_id"]   = np.int32(train_id)
    df["emitter_id"] = emitter_ids if emitter_ids is not None else np.int32(-1)

    return df


def parse_all_h5_files(
    raw_dir: Path = RAW_DIR,
    save:    bool = True,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Find and parse all .h5 files under raw_dir.

    Args:
        raw_dir:   Directory containing .h5 files (scanned recursively)
        save:      If True, save result to pdw_parsed.parquet
        max_files: Limit number of files (useful for quick testing)

    Returns:
        Combined DataFrame with all pulses.
    """
    h5_files = sorted(raw_dir.rglob("*.h5"))
    if not h5_files:
        print(f"ERROR: No .h5 files found in {raw_dir}")
        print("Run: python data/download_dataset.py  first.")
        sys.exit(1)

    if max_files:
        h5_files = h5_files[:max_files]

    print(f"\n{'='*60}")
    print(f"PARSING HDF5 FILES")
    print(f"{'='*60}")
    print(f"Found {len(h5_files)} .h5 file(s) to parse")
    print(f"Output: {OUT_FILE}\n")

    all_dfs = []
    total_pulses = 0
    failed = 0

    for i, h5_path in enumerate(tqdm(h5_files, desc="Parsing")):
        df = parse_h5_file(h5_path, train_id=i)
        if df is None:
            failed += 1
            continue
        all_dfs.append(df)
        total_pulses += len(df)

    if not all_dfs:
        print("ERROR: All files failed to parse.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"PARSE COMPLETE")
    print(f"{'='*60}")
    print(f"  Files parsed:    {len(all_dfs)} / {len(h5_files)}")
    print(f"  Failed:          {failed}")
    print(f"  Total pulses:    {total_pulses:,}")
    print(f"  Unique trains:   {combined['train_id'].nunique()}")
    print(f"  Unique emitters: {combined['emitter_id'].nunique()}")
    print(f"\n  Field ranges:")
    for col in PDW_COLS:
        c = combined[col]
        print(f"    {col:<14}: min={c.min():.4g}  max={c.max():.4g}  mean={c.mean():.4g}")

    if save:
        combined.to_parquet(OUT_FILE, index=False)
        size_mb = OUT_FILE.stat().st_size / (1024**2)
        print(f"\n  Saved: {OUT_FILE}  ({size_mb:.1f} MB)")

    return combined


if __name__ == "__main__":
    df = parse_all_h5_files(save=True)
    print(f"\nNext step: Run  python data/convert_to_binary.py")
