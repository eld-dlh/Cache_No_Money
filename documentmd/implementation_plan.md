# Person 1 — Data & Environment Engineer: Implementation Plan

## Overview

You are building the **foundational data pipeline and Gymnasium environment** for a cognitive radar interception system. Everything Persons 2–6 depend on flows from your work. Here's what we'll build, step by step.

---

## What is the Dataset?

The **Turing Synthetic Radar Dataset (TSRD)** is hosted on HuggingFace:
- `alan-turing-institute/turing-synthetic-radar-dataset`
- ~4 billion pulses across 6,000 pulse trains
- Each pulse is a **PDW (Pulse Descriptor Word)** with 5 fields:
  - **ToA** — Time of Arrival (µs): when the pulse was received
  - **Frequency** — Centre frequency (MHz): which channel
  - **PW** — Pulse Width (µs): how long the pulse lasted
  - **AoA** — Angle of Arrival (degrees): direction
  - **Amplitude** — Signal strength (dBm)

> [!IMPORTANT]
> The dataset requires a HuggingFace account and accepting dataset terms. You will need to:
> 1. Create/log in at huggingface.co
> 2. Accept the dataset's terms on the dataset page
> 3. Generate an access token (Settings → Access Tokens)
> 4. Use `huggingface_hub` Python library to download

---

## Project Structure

All code goes in:
```
c:\Users\TECQNIO\OneDrive\Documents\GitHub\Cache_No_Money\
├── data/
│   ├── download_dataset.py      # Step 1: Download
│   ├── parse_pdw.py             # Step 2: Inspect + parse to PDW records
│   ├── convert_to_binary.py     # Step 3: Save as flat .npy binary
│   └── raw/                     # Raw downloaded files go here (gitignored)
├── env/
│   ├── memmap_loader.py         # Step 4: np.memmap batch loader
│   ├── pdw_dataset.py           # Step 5: PyTorch Dataset wrapper
│   └── radar_env.py             # Steps 6-9: Gymnasium environment
├── tests/
│   └── test_random_agent.py     # Step 10: 100-step sanity check
├── requirements.txt
└── README.md
```

---

## Step-by-Step Plan

### Step 1 — Download the Dataset
- Use `huggingface_hub` (`hf_hub_download` or `snapshot_download`)
- Download only a **small split first** (the dataset is huge — start with 1 parquet shard)
- Save to `data/raw/`

### Step 2 — Inspect & Parse to PDW Records
- Inspect the parquet schema to find exact column names
- Map columns → `[ToA, Frequency, PulseWidth, AoA, Amplitude]`
- Script produces a clean pandas DataFrame with exactly 5 float64 columns

### Step 3 — Convert to Flat Binary `.npy`
- Each record = 5 × float32 values = 20 bytes
- We pad to **32 bytes** (4 extra float32 zeros as reserved fields) for alignment
- Save with `np.save()` → `.npy` file

> [!NOTE]
> Why 32 bytes? Fixed-size records allow O(1) random access via byte offset.
> `record_i = file[i * 32 : (i+1) * 32]` without scanning the whole file.

### Step 4 — np.memmap Loader
- Open file with `np.memmap(path, dtype='float32', mode='r', shape=(N, 8))`
- Expose a `get_batch(start, size)` function that slices rows without loading all data
- Write a test showing memory stays flat while batching

### Step 5 — PyTorch Dataset
```python
class PDWDataset(Dataset):
    def __init__(self, npy_path, window_size=10):
        ...
    def __len__(self): ...
    def __getitem__(self, idx):
        # Returns a window of `window_size` consecutive PDW records as a tensor
        ...
```

### Step 6 — Channel/State Representation Design
- **Decision: 64 discrete frequency channels**
- Frequency range from dataset (will be confirmed after inspection, expected ~1–18 GHz)
- Each channel maps to a frequency bin: `channel = int((freq - f_min) / (f_max - f_min) * 64)`
- State vector for the environment = last N=10 observed pulses (ToA, channel, PW, AoA, Amp)

### Step 7 — Gymnasium Environment (`RadarEnv`)
```python
class RadarEnv(gymnasium.Env):
    reset() → initial_state
    step(action: int) → (next_state, reward, terminated, truncated, info)
```
- **Action space**: `Discrete(64)` — pick one of 64 channels to scan
- **Observation space**: `Box` of shape `(10, 5)` — sliding window of last 10 PDW records
- `reset()`: pick a random start index in the pulse train (train split only)
- `step(action)`: advance time by one step, check if a pulse arrived on the chosen channel

### Step 8 — Reward Function
```
reward = (
    +R_intercept   if pulse detected on chosen channel,
    -P_miss        if pulse was present but NOT on chosen channel,
    -C_dwell       always (cost of using this time slot),
    -α * |f_target - f_current|  retuning penalty
)
```
- Default values: `R_intercept=1.0`, `P_miss=0.5`, `C_dwell=0.1`, `α=0.001`

### Step 9 — 70/30 Train/Test Split
- Split **by pulse train ID** (not by individual pulses)
- First 70% of pulse train IDs → train, last 30% → test
- `env = RadarEnv(split='train')` vs `RadarEnv(split='test')`

### Step 10 — Random Agent Test
```python
env = RadarEnv(split='train')
obs, _ = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
print("✅ 100 steps completed without crash")
```

---

## Required Python Packages

```
numpy
pandas
pyarrow          # for reading parquet
gymnasium
torch
huggingface_hub
tqdm
```

---

## Open Questions

> [!IMPORTANT]
> **Do you already have a HuggingFace account?** You'll need one to download the dataset.
> The dataset may require accepting terms of use on the HuggingFace website before downloading.

> [!NOTE]
> **Dataset size**: The full dataset is very large (~4B pulses). We'll start with 1–2 parquet shards
> for development, then scale. Confirm if you want to download the full dataset or just a sample.

---

## Verification Plan

- Run `test_random_agent.py` — must complete 100 steps without crash
- Confirm reward values are in a reasonable range (not all zeros, not ±infinity)
- Confirm train/test split indices are disjoint (no data leakage)
- Confirm the `.npy` binary file has exactly `N × 32` bytes

---

## Handoff Artifacts

Once complete, Persons 2 and 3 will import:
```python
from env.radar_env import RadarEnv
env = RadarEnv(split='train')  # or 'test'
```
