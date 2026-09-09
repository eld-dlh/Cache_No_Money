"""
Steps 6–9 — Cognitive Radar Interception Gymnasium Environment
==============================================================

This is the CORE environment that Persons 2 and 3 will use to train
and evaluate their AI agents.

────────────────────────────────────────────────────────────────────────────
STEP 6: Channel / State Design
────────────────────────────────────────────────────────────────────────────
We use 64 discrete frequency channels.

  Why 64?
  • Fine enough to distinguish most radar emitters (typical hop step ~10 MHz)
  • Coarse enough to keep the action space tractable for bandit/DRL methods
  • Powers-of-two make indexing clean (and channels map nicely to a 8×8 grid)

  Frequency range: determined from the dataset (set FREQ_MIN/FREQ_MAX below).
  Typical radar bands: 1–18 GHz. Adjust after running parse_pdw.py.

  Channel formula:
    channel_idx = int( (freq_MHz - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) * N_CHANNELS )
    channel_idx = clamp(channel_idx, 0, N_CHANNELS - 1)

  State vector: sliding window of the last WINDOW_SIZE PDW records.
  Shape: (WINDOW_SIZE, 5) = (10, 5) float32

────────────────────────────────────────────────────────────────────────────
STEP 7: Gymnasium Environment (reset + step)
────────────────────────────────────────────────────────────────────────────
  Observation space: Box(shape=(10, 5), dtype=float32)
    - Last 10 pulses seen (or zeros if fewer)
    - Columns: ToA, Frequency, PulseWidth, AoA, Amplitude

  Action space: Discrete(64)
    - Choose one of 64 frequency channels to scan this time step

  reset():
    - Picks a random start index in the TRAINING split (or TEST split)
    - Returns the initial observation window (zeros + first pulse)
    - Resets internal state (current position, last channel, etc.)

  step(action):
    - Advances the pulse-train cursor by 1 step
    - Checks if a pulse was present at the chosen channel
    - Computes reward (see Step 8)
    - Returns (observation, reward, terminated, truncated, info)

────────────────────────────────────────────────────────────────────────────
STEP 8: Reward Function
────────────────────────────────────────────────────────────────────────────
  reward = (
    + R_INTERCEPT   if a pulse was on the chosen channel,
    - P_MISS        if a pulse was present but on a DIFFERENT channel,
    - C_DWELL       always (cost of occupying this time slot),
    - alpha * |channel_chosen - channel_of_pulse|  (retuning penalty)
  )

  Note: if NO pulse was present at this time step, only C_DWELL applies.

  Default values (tunable):
    R_INTERCEPT = 1.0    (reward for catching a pulse)
    P_MISS      = 0.5    (penalty for missing a pulse that was there)
    C_DWELL     = 0.05   (small cost per time step)
    ALPHA       = 0.01   (retuning penalty scale per channel distance)

────────────────────────────────────────────────────────────────────────────
STEP 9: 70/30 Train/Test Split
────────────────────────────────────────────────────────────────────────────
  Split is done at the PULSE TRAIN level (not individual pulses).
  This prevents data leakage — the model can't memorise specific pulse trains.

  If train_id is available in the binary:
    - Sort unique train IDs
    - First 70% → train, last 30% → test
  Otherwise:
    - Split by record index (first 70% of all records → train)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.memmap_loader import PDWMemmap

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Channel design
N_CHANNELS = 64      # Number of discrete frequency channels (Step 6)
FREQ_MIN   = 0       # MHz — actual dataset min ~4.7 MHz (use 0 for clean bucket edges)
FREQ_MAX   = 12_000  # MHz — actual dataset max ~11,990 MHz

# Observation window
WINDOW_SIZE = 10     # How many past pulses the agent sees as its state

# Reward shaping (Step 8)
R_INTERCEPT = 1.0    # +reward for catching a pulse
P_MISS      = 0.5    # -penalty for missing a pulse that was there
C_DWELL     = 0.05   # -cost per time step (always applied)
ALPHA       = 0.01   # -penalty per channel of retuning distance

# Episode length
MAX_STEPS   = 500    # Steps before episode forcibly ends (truncation)

# Train/test split ratio
TRAIN_RATIO = 0.70

DEFAULT_NPY = Path(__file__).parent.parent / "data" / "raw" / "pdw_records.npy"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def freq_to_channel(freq_mhz: float | np.ndarray) -> int | np.ndarray:
    """
    Map a frequency (MHz) to a discrete channel index [0, N_CHANNELS).

    Example:
      freq_to_channel(5000)  →  channel 15 (if range is 1–18 GHz, 64 channels)
    """
    norm = (freq_mhz - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
    ch   = (norm * N_CHANNELS).astype(int) if isinstance(freq_mhz, np.ndarray) \
           else int(norm * N_CHANNELS)
    return np.clip(ch, 0, N_CHANNELS - 1)


def channel_to_freq_centre(channel: int) -> float:
    """Return the centre frequency (MHz) of a given channel index."""
    step = (FREQ_MAX - FREQ_MIN) / N_CHANNELS
    return FREQ_MIN + (channel + 0.5) * step


def compute_split_indices(
    loader: PDWMemmap,
    split: str = "train",
) -> tuple[int, int]:
    """
    Compute the (start, end) record indices for the requested split.

    Strategy:
      1. If train_id field is populated (field [6] != 0), split by unique train IDs.
      2. Otherwise, split by raw record index.

    Args:
        loader: PDWMemmap instance
        split:  'train' or 'test'

    Returns:
        (start_idx, end_idx) — inclusive start, exclusive end
    """
    N = len(loader)

    # Try train-ID-based split first
    train_ids = loader.train_id  # memmap slice — free to read
    unique_ids = np.unique(train_ids)
    unique_ids = unique_ids[unique_ids != 0]  # filter out zeros (no id set)

    if len(unique_ids) > 10:
        # We have real train IDs — split by them
        unique_ids_sorted = np.sort(unique_ids)
        split_at = int(len(unique_ids_sorted) * TRAIN_RATIO)
        train_ids_set = set(unique_ids_sorted[:split_at].tolist())
        test_ids_set  = set(unique_ids_sorted[split_at:].tolist())

        # Find index ranges for each set
        # (This is a linear scan — only needed once at env initialisation)
        all_ids = train_ids[:]
        if split == "train":
            mask   = np.isin(all_ids, list(train_ids_set))
        else:
            mask   = np.isin(all_ids, list(test_ids_set))

        indices = np.where(mask)[0]
        if len(indices) == 0:
            warnings.warn(f"No records found for split='{split}'. Falling back to index split.")
        else:
            return int(indices[0]), int(indices[-1] + 1)

    # Fallback: split by record index
    split_at = int(N * TRAIN_RATIO)
    if split == "train":
        return 0, split_at
    else:
        return split_at, N


# ─────────────────────────────────────────────────────────────────────────────
# THE ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class RadarEnv(gym.Env):
    """
    Cognitive Radar Interception Environment.

    The agent controls a cognitive radar receiver. At each time step, it
    chooses one of N_CHANNELS frequency channels to scan. If a radar pulse
    is present on that channel, it gets an intercept reward. If it picks
    the wrong channel while a pulse is present elsewhere, it gets a miss
    penalty plus a retuning cost proportional to how far it had to retune.

    Observation: The last WINDOW_SIZE PDW records (shape: WINDOW_SIZE × 5)
    Action:      Integer channel index (0 to N_CHANNELS-1)

    Usage:
        env = RadarEnv()                    # training environment
        env = RadarEnv(split='test')        # held-out test environment
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    Persons 2 and 3 import this class directly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        npy_path: str | Path = DEFAULT_NPY,
        split:    str         = "train",
        window_size: int      = WINDOW_SIZE,
        n_channels:  int      = N_CHANNELS,
        freq_min:    float    = FREQ_MIN,
        freq_max:    float    = FREQ_MAX,
        # Reward shaping parameters
        r_intercept: float    = R_INTERCEPT,
        p_miss:      float    = P_MISS,
        c_dwell:     float    = C_DWELL,
        alpha:       float    = ALPHA,
        max_steps:   int      = MAX_STEPS,
        seed:        int | None = None,
    ):
        super().__init__()

        # ── Store config
        self.split        = split
        self.window_size  = window_size
        self.n_channels   = n_channels
        self.freq_min     = freq_min
        self.freq_max     = freq_max
        self.r_intercept  = r_intercept
        self.p_miss       = p_miss
        self.c_dwell      = c_dwell
        self.alpha        = alpha
        self.max_steps    = max_steps

        # ── Load data
        self.loader = PDWMemmap(npy_path)

        # ── Compute split boundaries (Step 9)
        self.start_idx, self.end_idx = compute_split_indices(self.loader, split)
        print(f"✅ RadarEnv [{split}]: records [{self.start_idx:,} – {self.end_idx:,}] "
              f"({self.end_idx - self.start_idx:,} pulses), "
              f"{n_channels} channels, window={window_size}")

        # ── Gymnasium spaces
        # Observation: last `window_size` PDW records, 5 fields each
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (window_size, 5),
            dtype = np.float32,
        )
        # Action: choose one of N_CHANNELS frequency channels
        self.action_space = spaces.Discrete(n_channels)

        # ── RNG
        self.np_random = np.random.default_rng(seed)

        # ── Internal state (reset on each episode)
        self._cursor:       int = self.start_idx   # current position in pulse train
        self._last_channel: int = 0                # last channel we tuned to
        self._step_count:   int = 0                # steps taken this episode
        self._obs_buffer:   np.ndarray = np.zeros((window_size, 5), dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # reset() — Step 7a
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        Picks a random starting position in the pulse train (within the
        split's valid range). Resets internal state. Returns the initial
        observation window (mostly zeros + first few pulses).

        Returns:
            obs:  np.ndarray of shape (window_size, 5)
            info: dict with metadata (start index, split, etc.)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Pick a random start point — ensure there's room for at least one episode
        safe_end = self.end_idx - self.max_steps - self.window_size
        if safe_end <= self.start_idx:
            # Edge case: very small split — just start at the beginning
            self._cursor = self.start_idx
        else:
            self._cursor = int(self.np_random.integers(self.start_idx, safe_end))

        self._step_count   = 0
        self._last_channel = 0
        self._obs_buffer   = np.zeros((self.window_size, 5), dtype=np.float32)

        obs = self._get_obs()
        info = {
            "start_index": self._cursor,
            "split":       self.split,
            "n_channels":  self.n_channels,
        }
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # step() — Step 7b
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take one action (choose a frequency channel to scan).

        The environment advances to the next pulse in the data, checks
        whether the chosen channel matches, and computes the reward.

        Args:
            action: int in [0, N_CHANNELS) — which channel to scan

        Returns:
            obs:        New observation window (window_size, 5)
            reward:     Float — computed according to the reward function
            terminated: True if we've passed the end of valid data
            truncated:  True if MAX_STEPS reached
            info:       Dict with debugging info
        """
        assert self.action_space.contains(action), \
            f"Invalid action {action} — must be in [0, {self.n_channels})"

        # ── Advance cursor — fetch the current pulse
        if self._cursor >= self.end_idx:
            # No more pulses — terminate
            obs = self._get_obs()
            return obs, 0.0, True, False, {"reason": "end_of_data"}

        current_record = self.loader.get_record(self._cursor)
        freq_mhz = float(current_record[1])   # field [1] = Frequency

        # ── Map the actual pulse to its channel
        pulse_channel = freq_to_channel(freq_mhz)

        # ── Determine if a pulse was actually present
        # (In the dataset, every record IS a pulse, so pulse_present = True always.
        # In a real SDR scenario, some time slots might be empty. For now: always present.)
        pulse_present = True

        # ── Step 8: Compute reward
        reward = self._compute_reward(action, pulse_channel, pulse_present)

        # ── Update state
        self._obs_buffer = np.roll(self._obs_buffer, shift=-1, axis=0)
        self._obs_buffer[-1] = current_record[:5]  # only keep 5 PDW fields

        self._last_channel = action
        self._cursor      += 1
        self._step_count  += 1

        # ── Build next observation
        obs = self._get_obs()

        # ── Termination conditions
        terminated = self._cursor >= self.end_idx
        truncated  = self._step_count >= self.max_steps

        info = {
            "step":           self._step_count,
            "cursor":         self._cursor,
            "pulse_channel":  int(pulse_channel),
            "chosen_channel": int(action),
            "intercepted":    (action == pulse_channel) and pulse_present,
            "pulse_freq_mhz": freq_mhz,
            "pulse_present":  pulse_present,
        }

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: Reward Function
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        chosen_channel: int,
        pulse_channel:  int,
        pulse_present:  bool,
    ) -> float:
        """
        Reward = +R_intercept   if pulse on chosen channel
               - P_miss         if pulse present but on different channel
               - C_dwell        always (time cost)
               - alpha * |chosen - pulse_channel|   (retuning penalty)

        Args:
            chosen_channel: The channel the agent chose to scan
            pulse_channel:  The channel the actual pulse was on
            pulse_present:  Whether any pulse occurred at this time step

        Returns:
            float reward value
        """
        reward = 0.0

        # Always pay the dwell cost
        reward -= self.c_dwell

        if pulse_present:
            if chosen_channel == pulse_channel:
                # ✅ INTERCEPT: we scanned the right channel
                reward += self.r_intercept
            else:
                # ❌ MISS: pulse was present but we were on the wrong channel
                reward -= self.p_miss

                # Retuning penalty proportional to frequency distance
                channel_distance = abs(chosen_channel - pulse_channel)
                reward -= self.alpha * channel_distance

        return float(reward)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return current observation: the rolling window buffer (window_size, 5)."""
        return self._obs_buffer.copy()

    def get_channel_info(self) -> dict:
        """
        Return the channel frequency mapping for debugging/visualisation.
        Maps channel index → (low_MHz, centre_MHz, high_MHz).
        """
        step = (self.freq_max - self.freq_min) / self.n_channels
        return {
            ch: {
                "low_mhz":    self.freq_min + ch * step,
                "centre_mhz": self.freq_min + (ch + 0.5) * step,
                "high_mhz":   self.freq_min + (ch + 1) * step,
            }
            for ch in range(self.n_channels)
        }

    def seed(self, seed: int | None = None):
        """Set the RNG seed."""
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def render(self):
        """Minimal render: print current state."""
        print(f"Step={self._step_count}  Cursor={self._cursor}  "
              f"LastChannel={self._last_channel}  "
              f"LastFreq={self._obs_buffer[-1, 1]:.0f} MHz")

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience factories (used in tests)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def make_train(cls, **kwargs) -> "RadarEnv":
        return cls(split="train", **kwargs)

    @classmethod
    def make_test(cls, **kwargs) -> "RadarEnv":
        return cls(split="test", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST (quick verification without a random agent)
# ─────────────────────────────────────────────────────────────────────────────

def test_env_basic(npy_path: Path = DEFAULT_NPY):
    """Basic smoke test for the environment."""
    print(f"\n{'='*60}")
    print(f"RADAR ENV BASIC TEST")
    print(f"{'='*60}")

    env = RadarEnv(npy_path, split="train")

    print(f"\n[Test 1] Observation space: {env.observation_space}")
    print(f"[Test 2] Action space:      {env.action_space}")

    print(f"\n[Test 3] reset()...")
    obs, info = env.reset(seed=42)
    print(f"  obs shape:  {obs.shape}   (expected ({WINDOW_SIZE}, 5))")
    print(f"  obs dtype:  {obs.dtype}")
    print(f"  info:       {info}")
    assert obs.shape == (WINDOW_SIZE, 5)
    assert obs.dtype == np.float32

    print(f"\n[Test 4] 10 steps with action=0...")
    for i in range(10):
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1}: reward={reward:+.4f}  intercepted={info['intercepted']}  "
              f"pulse_ch={info['pulse_channel']}  chosen_ch={info['chosen_channel']}")
        assert obs.shape == (WINDOW_SIZE, 5)
        assert isinstance(reward, float)

    print(f"\n[Test 5] channel_info sample (first 5 channels):")
    ch_info = env.get_channel_info()
    for ch in range(5):
        ci = ch_info[ch]
        print(f"  ch{ch}: {ci['low_mhz']:.0f}–{ci['high_mhz']:.0f} MHz "
              f"(centre={ci['centre_mhz']:.0f} MHz)")

    print(f"\n✅ ENV BASIC TEST PASSED")
    return env


if __name__ == "__main__":
    test_env_basic()
    print(f"\nNext step: Run  python tests/test_random_agent.py  for the 100-step test.")
