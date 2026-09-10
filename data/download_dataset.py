"""
Step 1 — Download the Alan Turing Institute Synthetic Radar Dataset
===================================================================

Downloads .h5 files from the Turing Synthetic Radar Dataset on HuggingFace.
Dataset repo: alan-turing-institute/turing-synthetic-radar-dataset

Files in this repository are structured as HDF5 (.h5) files:
  scan/train_scan/config_*.h5
  scan/val_scan/config_*.h5
  stare/...

If the Hugging Face dataset is gated or inaccessible, this script automatically
provides a synthetic generator fallback so that the entire downstream pipeline
(parsing -> binary conversion -> DRQN training) runs without getting stuck.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fix Windows terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
REPO_TYPE = "dataset"
SAVE_DIR = Path(__file__).parent / "raw"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_FOLDERS = ["scan/train_scan", "scan", "stare", "archive"]


def authenticate(token: str | None = None) -> bool:
    """Attempt Hugging Face authentication if a token is present."""
    from huggingface_hub import login

    token = token or os.environ.get("HF_TOKEN", "").strip()
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("✅ Authenticated with HuggingFace.")
            return True
        except Exception as e:
            print(f"⚠️  HF login with provided token failed: {e}")
            return False
    return False


def generate_synthetic_h5_shards(
    output_dir: Path = SAVE_DIR / "scan" / "train_scan",
    n_shards: int = 10,
    pulses_per_shard: int = 5000,
) -> list[Path]:
    """
    Fallback generator: Creates realistic multi-emitter radar .h5 files
    with frequency-hopping patterns. Conforms exactly to the Turing dataset schema:
      - /data    : shape (N, 5) float32 [ToA, Frequency, PulseWidth, AoA, Amplitude]
      - /labels  : shape (N, 1) int32 [emitter_id]
    """
    print(f"\n📡 Generating {n_shards} synthetic radar .h5 shards in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    rng = np.random.default_rng(42)

    for shard_idx in range(n_shards):
        h5_path = output_dir / f"config_{shard_idx}.h5"

        # Simulate 3-5 distinct emitters in this environment
        n_emitters = rng.integers(3, 6)
        all_toas = []
        all_freqs = []
        all_pws = []
        all_aoas = []
        all_amps = []
        all_labels = []

        for e_id in range(n_emitters):
            # Base parameters for this emitter
            pri = rng.uniform(80.0, 350.0)       # Pulse repetition interval (us)
            pw = rng.uniform(1.0, 15.0)          # Pulse width (us)
            aoa = rng.uniform(20.0, 340.0)       # Angle of arrival (degrees)
            amp = rng.uniform(-65.0, -25.0)      # Amplitude (dBm)

            # Hopping channel pattern (e.g., period of 3-6 channels)
            hop_period = rng.integers(3, 7)
            hop_channels = rng.choice(64, size=hop_period, replace=False)
            # Map channel indices to frequencies in [500, 11500] MHz
            freq_step = 12000.0 / 64.0
            hop_freqs = hop_channels * freq_step + (freq_step / 2.0)

            n_pulses_emitter = pulses_per_shard // n_emitters
            emitter_toas = np.cumsum(rng.normal(pri, pri * 0.02, size=n_pulses_emitter))
            emitter_freqs = np.array([hop_freqs[i % hop_period] for i in range(n_pulses_emitter)])
            emitter_pws = rng.normal(pw, pw * 0.05, size=n_pulses_emitter)
            emitter_aoas = rng.normal(aoa, 1.5, size=n_pulses_emitter) % 360.0
            emitter_amps = rng.normal(amp, 2.0, size=n_pulses_emitter)

            all_toas.append(emitter_toas)
            all_freqs.append(emitter_freqs)
            all_pws.append(emitter_pws)
            all_aoas.append(emitter_aoas)
            all_amps.append(emitter_amps)
            all_labels.append(np.full(n_pulses_emitter, e_id, dtype=np.int32))

        toas = np.concatenate(all_toas)
        freqs = np.clip(np.concatenate(all_freqs), 50.0, 11950.0)
        pws = np.clip(np.concatenate(all_pws), 0.5, 50.0)
        aoas = np.concatenate(all_aoas)
        amps = np.concatenate(all_amps)
        labels = np.concatenate(all_labels)

        # Sort all interleaved pulses chronologically by ToA
        sort_order = np.argsort(toas)
        data_matrix = np.column_stack([
            toas[sort_order],
            freqs[sort_order],
            pws[sort_order],
            aoas[sort_order],
            amps[sort_order],
        ]).astype(np.float32)

        label_matrix = labels[sort_order].reshape(-1, 1).astype(np.int32)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data_matrix)
            f.create_dataset("labels", data=label_matrix)

        generated.append(h5_path)

    print(f"✅ Generated {len(generated)} .h5 files (~{n_shards * pulses_per_shard:,} total pulses).")
    return generated


def download_sample(
    num_files: int = 15,
    force_synthetic: bool = False,
    token: str | None = None,
) -> list[Path]:
    """
    Downloads .h5 shards from HuggingFace, or falls back to generating
    high-fidelity synthetic radar files if HF is gated or restricted.
    """
    if force_synthetic:
        return generate_synthetic_h5_shards(n_shards=num_files)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        authenticate(token=token)

        print(f"\n📂 Fetching file list from HuggingFace repo: {REPO_ID}...")
        all_files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))

        h5_files = [f for f in all_files if f.endswith(".h5")]
        if not h5_files:
            raise ValueError("No .h5 files found in repository.")

        print(f"📊 Found {len(h5_files)} .h5 files in repository.")

        # Sort by preferred scan mode files
        def folder_priority(fname: str) -> int:
            for i, folder in enumerate(PREFERRED_FOLDERS):
                if folder in fname:
                    return i
            return len(PREFERRED_FOLDERS)

        h5_files.sort(key=folder_priority)
        targets = h5_files[:num_files]

        print(f"\n📥 Downloading {len(targets)} .h5 shard(s) into: {SAVE_DIR}")
        downloaded = []
        for i, fname in enumerate(targets):
            print(f"   [{i+1}/{len(targets)}] Downloading {fname}...")
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=fname,
                local_dir=str(SAVE_DIR),
            )
            downloaded.append(Path(local_path))

        print(f"\n🎉 Downloaded {len(downloaded)} .h5 files from HuggingFace.")
        return downloaded

    except Exception as e:
        print(f"\n⚠️  HuggingFace download could not proceed: {e}")
        print("ℹ️  The Turing Radar Dataset is gated on Hugging Face and requires approved access.")
        print("🔄 Automatically activating high-fidelity synthetic radar generator fallback...")
        return generate_synthetic_h5_shards(n_shards=num_files)


def main():
    parser = argparse.ArgumentParser(description="Download or generate radar dataset shards.")
    parser.add_argument("--num-files", type=int, default=15, help="Number of .h5 files to obtain.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic radar data directly.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace access token.")
    args = parser.parse_args()

    download_sample(
        num_files=args.num_files,
        force_synthetic=args.synthetic,
        token=args.token,
    )
    print("\nNext step: Run  python data/parse_pdw.py")


if __name__ == "__main__":
    main()
