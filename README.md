# Cache No Money — Cognitive Radar Interception

## Person 1 (Data & Environment) — Handoff Package

---

## Quick Start for Persons 2, 3, and 4

```python
from env.radar_env import RadarEnv

# Training environment
env = RadarEnv(split='train')
obs, info = env.reset()

# One step
action = env.action_space.sample()   # random; replace with your policy
obs, reward, terminated, truncated, info = env.step(action)

# Test environment (held-out pulse trains)
env_test = RadarEnv(split='test')
```

---

## File Structure

```
Cache_No_Money/
├── data/
│   ├── download_dataset.py   # Step 1: Download from HuggingFace
│   ├── parse_pdw.py          # Step 2: Inspect + normalise to 5-field PDW
│   ├── convert_to_binary.py  # Step 3: Write flat 32-byte .npy
│   └── raw/
│       ├── *.parquet          # downloaded shards (gitignored)
│       ├── pdw_parsed.parquet # cleaned PDW data
│       └── pdw_records.npy    # flat binary (the main artefact)
├── env/
│   ├── memmap_loader.py      # Step 4: np.memmap batch reader
│   ├── pdw_dataset.py        # Step 5: PyTorch Dataset
│   └── radar_env.py          # Steps 6–9: Gymnasium RadarEnv
├── tests/
│   └── test_random_agent.py  # Step 10: 100-step sanity check
└── requirements.txt
```

---

## Run Order (once you have a HuggingFace token)

```bash
pip install -r requirements.txt

# 1. Download one shard from HuggingFace (needs HF account + token)
python data/download_dataset.py

# 2. Inspect + parse → 5-field PDW records
python data/parse_pdw.py

# 3. Convert to flat binary .npy
python data/convert_to_binary.py

# 4. Test memmap loader
python env/memmap_loader.py

# 5. Test PyTorch dataset
python env/pdw_dataset.py

# 6. Full 100-step random agent test
python tests/test_random_agent.py
```

---

## Environment API

| Attribute | Value |
|-----------|-------|
| `observation_space` | `Box(shape=(10, 5), dtype=float32)` |
| `action_space` | `Discrete(64)` |
| `n_channels` | 64 |
| `freq_range` | 1,000 – 18,000 MHz (adjust to dataset) |
| `window_size` | 10 pulses |
| `max_steps` | 500 per episode |

### Reward Function

```
reward = + R_INTERCEPT  (1.0)    if pulse on chosen channel
         - P_MISS       (0.5)    if pulse on a different channel  
         - C_DWELL      (0.05)   always
         - alpha * |ch_chosen - ch_pulse|   (0.01 per channel distance)
```

### Train / Test Split

- Split is done at the **pulse-train level** (by `train_id`).
- First 70% of unique train IDs → `split='train'`
- Last 30% of unique train IDs → `split='test'`
- The splits are disjoint — verified by `test_random_agent.py`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Channels | **64** | Fine enough for emitter separation, tractable action space |
| Binary format | **32-byte records (.npy)** | O(1) random access via np.memmap |
| Window size | **10 pulses** | Enough context for pattern detection |
| Split strategy | **By pulse train ID** | Prevents data leakage from train to test |
| Reward alpha | **0.01 per channel** | Gentle retuning penalty; tune if needed |

---

## Tuning Parameters

Edit the constants at the top of `env/radar_env.py`:

```python
N_CHANNELS  = 64      # Change to 32 or 128 if desired
FREQ_MIN    = 1_000   # MHz — update after inspecting your dataset shard
FREQ_MAX    = 18_000  # MHz
WINDOW_SIZE = 10
R_INTERCEPT = 1.0
P_MISS      = 0.5
C_DWELL     = 0.05
ALPHA       = 0.01
MAX_STEPS   = 500
TRAIN_RATIO = 0.70
```

---

## Notes for Person 2 (Bandit)

- Import `RadarEnv` and use `env.action_space.sample()` as your random baseline.
- `info['intercepted']` tells you if the last action was a hit.
- `info['pulse_channel']` gives the ground-truth channel (for oracle comparisons).
- The `get_channel_info()` method gives you the frequency range of each channel.

## Notes for Person 3 (DRQN)

- Use `PDWDataset` with a `DataLoader` for offline pre-training.
- The observation tensor shape is `(10, 5)` → flatten to `(50,)` or use as-is for LSTM input.
- `env.window_size = 10` matches the dataset window.

## Notes for Person 4 (Systems)

- `PDWMemmap` is already thread-safe for read-only access (multiple threads can call `get_batch()` concurrently).
- The `.npy` binary is the **320-byte sliding-window buffer** — 10 records × 32 bytes each.
