"""
data/generate_synthetic_pdw.py — Synthetic Radar PDW Generator
===============================================================

Generates a realistic synthetic 32-byte PDW dataset file (data/raw/pdw_records.npy)
when raw Turing Synthetic Radar Dataset files are not locally downloaded.
Allows running RadarEnv and evaluation benchmarks out of the box.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

RAW_DIR = Path(__file__).parent / "raw"
OUT_FILE = RAW_DIR / "pdw_records.npy"


def generate_synthetic_dataset(
    output_path: Path = OUT_FILE,
    n_pulse_trains: int = 100,
    pulses_per_train: int = 1000,
    n_channels: int = 64,
    freq_min: float = 1000.0,
    freq_max: float = 18000.0,
    seed: int = 42,
) -> Path:
    """
    Generates n_pulse_trains * pulses_per_train synthetic PDW records
    with realistic frequency-hopping patterns.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    total_pulses = n_pulse_trains * pulses_per_train
    records = np.zeros((total_pulses, 8), dtype=np.float32)

    current_toa = 0.0
    record_idx = 0

    for train_id in range(n_pulse_trains):
        # Pick a period P between 3 and 8 for this pulse train
        period = rng.integers(3, 9)
        hopping_pattern = rng.integers(0, n_channels, size=period)

        for p_i in range(pulses_per_train):
            # Inter-pulse interval (10 to 50 us)
            current_toa += float(rng.uniform(10.0, 50.0))
            
            # Determine channel from hopping pattern + occasional noise
            pattern_ch = hopping_pattern[p_i % period]
            if rng.random() < 0.05:  # 5% random hop noise
                pattern_ch = int(rng.integers(0, n_channels))

            # Map channel to MHz frequency cleanly matching channel center
            freq_MHz = (pattern_ch + 0.5) / n_channels * 18000.0

            pw = float(rng.uniform(1.0, 10.0))
            aoa = float(rng.uniform(0.0, 360.0))
            amp = float(rng.uniform(-80.0, -10.0))

            records[record_idx, 0] = current_toa     # ToA (us)
            records[record_idx, 1] = freq_MHz       # Frequency (MHz)
            records[record_idx, 2] = pw             # PW (us)
            records[record_idx, 3] = aoa            # AoA (deg)
            records[record_idx, 4] = amp            # Amp (dBm)
            records[record_idx, 5] = 0.0            # Reserved 0
            records[record_idx, 6] = 0.0            # Reserved 1
            records[record_idx, 7] = float(train_id)# train_id

            record_idx += 1

    np.save(output_path, records)
    print(f"✅ Generated synthetic PDW dataset: {output_path} ({total_pulses:,} records, {output_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_path


if __name__ == "__main__":
    generate_synthetic_dataset()
