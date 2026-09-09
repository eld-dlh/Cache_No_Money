"""
Step 1 — Download the Alan Turing Institute Synthetic Radar Dataset
===================================================================

Dataset structure (visible on HuggingFace Files tab):
  scan/    — scan-mode radar pulse trains  ← we use this first
  stare/   — stare-mode pulse trains
  archive/ — older archived data

How to get your HuggingFace token:
  1. Go to https://huggingface.co/settings/tokens
  2. Click "New token" → Read access → Generate → Copy it
  3. Either set  HF_TOKEN=<your_token>  as an environment variable
     OR just run this script and paste it when prompted.

Why only one shard to start?
  The full dataset is ~4 billion pulses. We grab one parquet shard
  (typically a few hundred MB) to validate the pipeline end-to-end,
  then scale up once everything works.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, login

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
REPO_ID     = "alan-turing-institute/turing-synthetic-radar-dataset"
REPO_TYPE   = "dataset"
SAVE_DIR    = Path(__file__).parent / "raw"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Dataset folder priority: scan > stare > archive
# (We want scan-mode data for the cognitive radar task)
PREFERRED_FOLDERS = ["scan", "stare", "archive"]


def authenticate():
    """Log in to HuggingFace. Reads HF_TOKEN env var or prompts."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("=" * 60)
        print("HuggingFace login required.")
        print("Get your token at: https://huggingface.co/settings/tokens")
        print("=" * 60)
        token = input("Paste your HF_TOKEN here: ").strip()
    if not token:
        print("ERROR: No token provided. Cannot download dataset.")
        sys.exit(1)
    login(token=token, add_to_git_credential=False)
    print("✅ Authenticated with HuggingFace")


def list_available_files() -> list[str]:
    """Print all files in the repo and return the list."""
    print("\n📂 Files available in the dataset repo:")
    files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    for f in sorted(files):
        print(f"   {f}")
    return files


def download_shard(filename: str) -> Path:
    """
    Download a single file from the HuggingFace dataset repo.

    Args:
        filename: Relative path inside the repo (e.g. 'scan/train-0000.parquet')

    Returns:
        Local path where the file was saved.
    """
    print(f"\n⬇️  Downloading: {filename}")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(SAVE_DIR),
    )
    file_size_mb = Path(local_path).stat().st_size / (1024 ** 2)
    print(f"✅ Saved to: {local_path}  ({file_size_mb:.1f} MB)")
    return Path(local_path)


def download_sample(download_full: bool = False) -> list[Path]:
    """
    Main download entry point.

    Args:
        download_full: If True, downloads ALL shards. If False, just the first shard.

    Returns:
        List of local paths to downloaded parquet files.
    """
    authenticate()
    all_files = list_available_files()

    # Filter to only parquet files
    parquet_files = [f for f in all_files if f.endswith(".parquet")]

    if not parquet_files:
        print("\n⚠️  No parquet files found. All files listed above — check manually.")
        sys.exit(1)

    print(f"\n📊 Found {len(parquet_files)} parquet file(s) total.")

    # Sort by preferred folder priority: scan > stare > archive
    def folder_priority(fname: str) -> int:
        for i, folder in enumerate(PREFERRED_FOLDERS):
            if fname.startswith(folder + "/") or f"/{folder}/" in fname:
                return i
        return len(PREFERRED_FOLDERS)

    parquet_files.sort(key=folder_priority)
    print(f"   Priority order: {PREFERRED_FOLDERS}")
    print(f"   Best candidate: {parquet_files[0]}")

    if download_full:
        targets = parquet_files
        print(f"\n⚠️  Downloading ALL {len(targets)} shards — may take a long time!")
    else:
        targets = parquet_files[:1]
        print(f"\n📥 Downloading just the first shard from '{parquet_files[0].split('/')[0]}/'")
        print(f"   (Change download_full=True below to grab everything)")

    downloaded = []
    for fname in targets:
        path = download_shard(fname)
        downloaded.append(path)

    print(f"\n🎉 Download complete! Files are in: {SAVE_DIR}")
    return downloaded


if __name__ == "__main__":
    # Change download_full=True when you're ready for the full dataset
    downloaded_files = download_sample(download_full=False)
    print("\nNext step: Run  python data/parse_pdw.py  to inspect and parse the data.")
