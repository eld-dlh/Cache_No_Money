"""
Step 3 — Convert Parsed PDW Data into a Flat Binary .npy File
==============================================================

What this script does:
  Takes the cleaned parquet from Step 2 and writes it as a flat NumPy
  binary file with FIXED-SIZE 32-byte records.

Why 32 bytes per record?
  Each PDW has 5 fields (ToA, Frequency, PulseWidth, AoA, Amplitude).
  5 × float32 = 20 bytes. We pad to 32 bytes (8 × float32) with 3 reserved
  zeros for future use and memory alignment.

  The KEY advantage: with fixed-size records, record #i starts at exactly
  byte offset (i × 32). This makes random access O(1) — no scanning needed.

  Compare to CSV/JSON/parquet:
    - CSV:    O(n) — must scan from the start
    - Parquet: chunked, good for sequential; random access is complex
    - .npy:   O(1) — np.memmap[i] is instantaneous

Record layout (32 bytes = 8 × float32):
  [0]  ToA          (µs)
  [1]  Frequency    (MHz)
  [2]  PulseWidth   (µs)
  [3]  AoA          (degrees)
  [4]  Amplitude    (dBm)
  [5]  reserved_0   (zero)
  [6]  reserved_1   (zero)  — could store emitter_id later
  [7]  reserved_2   (zero)  — could store train_id later
"""

import sys
from pathlib import Path

# Fix Windows terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR  = Path(__file__).parent / "raw"
OUT_FILE = RAW_DIR / "pdw_records.npy"

# Number of float32 values per record = 32 bytes / 4 bytes = 8
RECORD_SIZE = 8   # float32 values
BYTE_SIZE   = RECORD_SIZE * 4  # = 32 bytes

PDW_COLS = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]


def convert_to_binary(
    pdw_parquet: Path = RAW_DIR / "pdw_parsed.parquet",
    meta_parquet: Path | None = None,
    output_path: Path = OUT_FILE,
    chunk_size: int = 100_000,
) -> Path:
    """
    Read the parsed PDW parquet and write a flat float32 .npy binary.

    We write in chunks to keep memory usage low, even for huge files.

    Args:
        pdw_parquet:  Path to the parsed PDW parquet (from Step 2)
        meta_parquet: Optional path to metadata parquet (for emitter_id in field [6])
        output_path:  Where to write the .npy file
        chunk_size:   How many rows to process at a time

    Returns:
        Path to the written .npy file
    """
    if not pdw_parquet.exists():
        raise FileNotFoundError(
            f"Parsed parquet not found at {pdw_parquet}.\n"
            f"Run:  python data/parse_pdw.py  first."
        )

    print(f"\n{'='*60}")
    print(f"CONVERTING TO BINARY")
    print(f"{'='*60}")

    # Load everything (or stream if huge — adjust chunk_size)
    print(f"📂 Reading: {pdw_parquet}")
    pdw_df = pd.read_parquet(pdw_parquet)
    N = len(pdw_df)
    print(f"   {N:,} pulses to convert")

    # Optionally load metadata
    train_ids = None
    if meta_parquet and Path(meta_parquet).exists():
        meta_df   = pd.read_parquet(meta_parquet)
        if "train_id" in meta_df.columns:
            train_ids = meta_df["train_id"].values.astype(np.float32)
            print(f"   Including train_id in reserved field [6]")
    elif "train_id" in pdw_df.columns:
        train_ids = pdw_df["train_id"].values.astype(np.float32)
        print(f"   Including train_id from pdw_parsed in reserved field [6]")

    # Allocate the full output array in memory
    # Shape: (N, 8) — N records, 8 float32 values each
    print(f"\n🗄️  Allocating array: ({N:,}, {RECORD_SIZE}) float32 "
          f"= {N * RECORD_SIZE * 4 / (1024**3):.2f} GB")

    data = np.zeros((N, RECORD_SIZE), dtype=np.float32)

    # Fill the 5 PDW columns
    print(f"📝 Filling PDW fields...")
    for i, col in enumerate(tqdm(PDW_COLS, desc="   Columns")):
        data[:, i] = pdw_df[col].values.astype(np.float32)

    # Fill reserved field [6] with train_id if available
    if train_ids is not None:
        data[:, 6] = train_ids[:N]

    # Validate — check for NaN/Inf that would corrupt the binary
    bad_mask = ~np.isfinite(data[:, :5])
    bad_count = bad_mask.sum()
    if bad_count > 0:
        print(f"\n⚠️  Found {bad_count} NaN/Inf values — replacing with 0.0")
        data[:, :5] = np.where(bad_mask, 0.0, data[:, :5])
    else:
        print(f"✅ No NaN/Inf values in PDW fields")

    # Save as .npy
    print(f"\n💾 Saving to: {output_path}")
    np.save(str(output_path), data)

    # Verify the file
    loaded = np.load(str(output_path), mmap_mode='r')
    assert loaded.shape == (N, RECORD_SIZE), f"Shape mismatch! {loaded.shape} vs {(N, RECORD_SIZE)}"
    expected_bytes = N * RECORD_SIZE * 4 + 128  # +128 for .npy header
    actual_bytes   = output_path.stat().st_size

    print(f"\n{'='*60}")
    print(f"✅ BINARY FILE CREATED")
    print(f"{'='*60}")
    print(f"   Path:    {output_path}")
    print(f"   Shape:   {loaded.shape}  ({N:,} records × {RECORD_SIZE} float32)")
    print(f"   Size:    {actual_bytes / (1024**2):.1f} MB")
    print(f"   Record:  {BYTE_SIZE} bytes each")
    print(f"\n   Field layout per record:")
    print(f"   [0] ToA          = {loaded[0,0]:.4g} µs")
    print(f"   [1] Frequency    = {loaded[0,1]:.4g} MHz")
    print(f"   [2] PulseWidth   = {loaded[0,2]:.4g} µs")
    print(f"   [3] AoA          = {loaded[0,3]:.4g}°")
    print(f"   [4] Amplitude    = {loaded[0,4]:.4g} dBm")
    print(f"   [5-7] reserved   = {loaded[0,5:8]}")

    return output_path


if __name__ == "__main__":
    path = convert_to_binary()
    print(f"\nNext step: Run  python env/memmap_loader.py  to test the batch loader.")
