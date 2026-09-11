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

    Generates a coherent EW scenario where a persistent pool of emitters
    transmit continuously across all shards, mirroring real radar missions.
    """
    print(f"\n📡 Generating {n_shards} synthetic radar .h5 shards in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    rng = np.random.default_rng(42)

    # Global pool of 8 distinct radar emitters with realistic radar parameters
    n_global_emitters = 8
    emitters = []
    freq_step = 12000.0 / 64.0

    for e_id in range(n_global_emitters):
        pri = float(rng.uniform(80.0, 350.0))       # Pulse repetition interval (us)
        pw = float(rng.uniform(1.0, 15.0))          # Pulse width (us)
        aoa = float(rng.uniform(20.0, 340.0))       # Angle of arrival (degrees)
        amp = float(rng.uniform(-65.0, -25.0))      # Amplitude (dBm)

        # Cyclic frequency-hopping pattern (period 3-6 channels)
        hop_period = int(rng.integers(3, 7))
        hop_channels = rng.choice(64, size=hop_period, replace=False)
        hop_freqs = hop_channels * freq_step + (freq_step / 2.0)

        emitters.append({
            "e_id": e_id,
            "pri": pri,
            "pw": pw,
            "aoa": aoa,
            "amp": amp,
            "hop_period": hop_period,
            "hop_channels": hop_channels,
            "hop_freqs": hop_freqs,
            "cur_toa": 0.0,
            "hop_idx": 0,
        })

    for shard_idx in range(n_shards):
        h5_path = output_dir / f"config_{shard_idx}.h5"

        all_toas = []
        all_freqs = []
        all_pws = []
        all_aoas = []
        all_amps = []
        all_labels = []

        pulses_per_emitter = pulses_per_shard // n_global_emitters

        for e in emitters:
            e_toas = []
            e_freqs = []
            t = e["cur_toa"]
            h_idx = e["hop_idx"]

            for _ in range(pulses_per_emitter):
                t += max(10.0, float(rng.normal(e["pri"], e["pri"] * 0.02)))
                e_toas.append(t)
                e_freqs.append(e["hop_freqs"][h_idx % e["hop_period"]])
                h_idx += 1

            e["cur_toa"] = t
            e["hop_idx"] = h_idx

            e_pws = rng.normal(e["pw"], e["pw"] * 0.05, size=pulses_per_emitter)
            e_aoas = (rng.normal(e["aoa"], 1.5, size=pulses_per_emitter)) % 360.0
            e_amps = rng.normal(e["amp"], 2.0, size=pulses_per_emitter)
            e_labels = np.full(pulses_per_emitter, e["e_id"], dtype=np.int32)

            all_toas.extend(e_toas)
            all_freqs.extend(e_freqs)
            all_pws.extend(e_pws)
            all_aoas.extend(e_aoas)
            all_amps.extend(e_amps)
            all_labels.extend(e_labels)

        # Sort all interleaved pulses chronologically by ToA
        sort_order = np.argsort(all_toas)
        data_matrix = np.column_stack([
            np.array(all_toas)[sort_order],
            np.clip(np.array(all_freqs)[sort_order], 50.0, 11950.0),
            np.clip(np.array(all_pws)[sort_order], 0.5, 50.0),
            np.array(all_aoas)[sort_order],
            np.array(all_amps)[sort_order],
        ]).astype(np.float32)

        label_matrix = np.array(all_labels)[sort_order].reshape(-1, 1).astype(np.int32)

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
