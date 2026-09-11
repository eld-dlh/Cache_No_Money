"""
models/replay_buffer.py — Recurrent Experience Replay Buffer
==============================================================

Standard DQN replay buffers store individual transitions (s, a, r, s').
This breaks LSTM hidden state consistency — if you randomly sample a
transition from step 47, the LSTM has no idea what happened in steps 1–46.

This recurrent replay buffer solves this by storing COMPLETE EPISODES
and sampling contiguous SEQUENCE CHUNKS of length T for training.

How it works:
    1. During rollout, transitions are added one-by-one via add_transition().
    2. When an episode ends, end_episode() is called to store the full episode.
    3. During training, sample() returns a batch of random contiguous chunks
       from random episodes. The LSTM can then properly unroll across each chunk.

Memory layout:
    Each stored episode is a dict of numpy arrays:
        obs_seq:  (episode_len, window_size, 5)   — normalised pulse windows
        context:  (episode_len, context_dim)       — context features
        actions:  (episode_len,)                   — chosen channels
        rewards:  (episode_len,)                   — scalar rewards
        dones:    (episode_len,)                   — episode termination flags

Usage:
    from models.replay_buffer import RecurrentReplayBuffer

    buffer = RecurrentReplayBuffer(capacity=10000, chunk_len=16)

    # During rollout:
    buffer.add_transition(obs_seq, context, action, reward, done)
    if done:
        buffer.end_episode()

    # During training:
    if buffer.can_sample(batch_size=32):
        batch = buffer.sample(batch_size=32)
        # batch['obs_seq']  → (32, 16, 10, 5)
        # batch['context']  → (32, 16, 5)
        # batch['actions']  → (32, 16)
        # batch['rewards']  → (32, 16)
        # batch['dones']    → (32, 16)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


class RecurrentReplayBuffer:
    """
    Episode-based replay buffer that samples contiguous sequence chunks
    for recurrent (LSTM) training.

    Parameters
    ----------
    capacity : int
        Maximum number of episodes to store. When full, the oldest
        episode is overwritten (FIFO circular buffer).
    chunk_len : int
        Length of contiguous sequence chunks to sample for training.
        Recommended: 16–20 steps. Too short = LSTM can't build context.
        Too long = high memory usage and slower training.
    window_size : int
        Observation window size from RadarEnv (default: 10).
    context_dim : int
        Number of engineered context features (default: 5).
    """

    def __init__(
        self,
        capacity: int = 10_000,
        chunk_len: int = 16,
        window_size: int = 10,
        context_dim: int = 5,
    ):
        self.capacity = capacity
        self.chunk_len = chunk_len
        self.window_size = window_size
        self.context_dim = context_dim

        # ── Episode storage (circular buffer)
        self._episodes: list[dict[str, np.ndarray]] = []
        self._write_pos: int = 0  # next write position in circular buffer

        # ── Current episode being collected (not yet stored)
        self._current_obs: list[np.ndarray] = []
        self._current_ctx: list[np.ndarray] = []
        self._current_act: list[int] = []
        self._current_rew: list[float] = []
        self._current_done: list[bool] = []

        # ── Statistics
        self._total_transitions: int = 0
        self._total_episodes: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # COLLECTING EXPERIENCE
    # ──────────────────────────────────────────────────────────────────────────

    def add_transition(
        self,
        obs_seq: np.ndarray,
        context: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """
        Add a single transition to the current episode being collected.

        Parameters
        ----------
        obs_seq : np.ndarray, shape (window_size, 5)
            Normalised pulse observation window.
        context : np.ndarray, shape (context_dim,)
            Engineered context features.
        action : int
            Chosen channel index (0 to n_actions-1).
        reward : float
            Scalar reward from the environment.
        done : bool
            True if the episode terminated or was truncated.
        """
        self._current_obs.append(obs_seq.copy())
        self._current_ctx.append(context.copy())
        self._current_act.append(action)
        self._current_rew.append(reward)
        self._current_done.append(done)

        self._total_transitions += 1

    def end_episode(self) -> None:
        """
        Finalise and store the current episode into the replay buffer.

        Call this when the environment returns terminated=True or truncated=True.
        Only stores episodes that are at least chunk_len steps long
        (shorter episodes don't provide enough context for LSTM training).
        """
        ep_len = len(self._current_obs)

        if ep_len < self.chunk_len:
            # Episode too short — discard it (can't extract a full chunk)
            self._clear_current()
            return

        # Pack current episode into a dict of numpy arrays
        episode = {
            "obs_seq": np.stack(self._current_obs, axis=0),     # (ep_len, W, 5)
            "context": np.stack(self._current_ctx, axis=0),     # (ep_len, C)
            "actions": np.array(self._current_act, dtype=np.int64),   # (ep_len,)
            "rewards": np.array(self._current_rew, dtype=np.float32), # (ep_len,)
            "dones":   np.array(self._current_done, dtype=np.bool_),  # (ep_len,)
        }

        # Store in circular buffer
        if len(self._episodes) < self.capacity:
            self._episodes.append(episode)
        else:
            self._episodes[self._write_pos] = episode

        self._write_pos = (self._write_pos + 1) % self.capacity
        self._total_episodes += 1

        self._clear_current()

    def _clear_current(self) -> None:
        """Reset the current episode collector."""
        self._current_obs.clear()
        self._current_ctx.clear()
        self._current_act.clear()
        self._current_rew.clear()
        self._current_done.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # SAMPLING
    # ──────────────────────────────────────────────────────────────────────────

    def can_sample(self, batch_size: int) -> bool:
        """
        Check if there are enough stored episodes to sample a batch.

        We need at least batch_size episodes, and each must be
        long enough for a chunk_len slice.
        """
        return len(self._episodes) >= batch_size

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Sample a batch of contiguous sequence chunks for DRQN training.

        For each sample in the batch:
          1. Pick a random episode from the buffer.
          2. Pick a random starting index within that episode.
          3. Extract chunk_len consecutive transitions.

        Parameters
        ----------
        batch_size : int
            Number of sequence chunks to sample.
        device : torch.device or str
            Device to place the output tensors on.

        Returns
        -------
        dict with keys:
            'obs_seq'  : torch.Tensor, shape (batch, chunk_len, window_size, 5)
            'context'  : torch.Tensor, shape (batch, chunk_len, context_dim)
            'actions'  : torch.LongTensor, shape (batch, chunk_len)
            'rewards'  : torch.Tensor, shape (batch, chunk_len)
            'dones'    : torch.BoolTensor, shape (batch, chunk_len)
        """
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample batch of {batch_size} — only "
                f"{len(self._episodes)} episodes in buffer."
            )

        # Pre-allocate batch arrays
        batch_obs = np.zeros(
            (batch_size, self.chunk_len, self.window_size, 5),
            dtype=np.float32,
        )
        batch_ctx = np.zeros(
            (batch_size, self.chunk_len, self.context_dim),
            dtype=np.float32,
        )
        batch_act = np.zeros((batch_size, self.chunk_len), dtype=np.int64)
        batch_rew = np.zeros((batch_size, self.chunk_len), dtype=np.float32)
        batch_done = np.zeros((batch_size, self.chunk_len), dtype=np.bool_)

        # Sample random episodes and random chunks within them
        sampled_episodes = random.choices(self._episodes, k=batch_size)

        for i, episode in enumerate(sampled_episodes):
            ep_len = len(episode["actions"])

            # Pick a random start index such that we can fit chunk_len steps
            max_start = ep_len - self.chunk_len
            start = random.randint(0, max_start)
            end = start + self.chunk_len

            batch_obs[i] = episode["obs_seq"][start:end]
            batch_ctx[i] = episode["context"][start:end]
            batch_act[i] = episode["actions"][start:end]
            batch_rew[i] = episode["rewards"][start:end]
            batch_done[i] = episode["dones"][start:end]

        return {
            "obs_seq": torch.from_numpy(batch_obs).to(device),
            "context": torch.from_numpy(batch_ctx).to(device),
            "actions": torch.from_numpy(batch_act).to(device),
            "rewards": torch.from_numpy(batch_rew).to(device),
            "dones":   torch.from_numpy(batch_done).to(device),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # STATISTICS & UTILITIES
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of complete episodes currently stored."""
        return len(self._episodes)

    @property
    def total_transitions(self) -> int:
        """Total number of transitions added (across all episodes ever)."""
        return self._total_transitions

    @property
    def total_episodes(self) -> int:
        """Total number of episodes stored (including overwritten ones)."""
        return self._total_episodes

    def stats(self) -> dict[str, Any]:
        """Return a summary of buffer statistics."""
        ep_lengths = [len(ep["actions"]) for ep in self._episodes]
        return {
            "stored_episodes": len(self._episodes),
            "capacity": self.capacity,
            "total_transitions_added": self._total_transitions,
            "total_episodes_added": self._total_episodes,
            "chunk_len": self.chunk_len,
            "avg_episode_len": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "min_episode_len": int(np.min(ep_lengths)) if ep_lengths else 0,
            "max_episode_len": int(np.max(ep_lengths)) if ep_lengths else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RecurrentReplayBuffer — Standalone Smoke Test")
    print("=" * 60)

    buffer = RecurrentReplayBuffer(
        capacity=100,
        chunk_len=8,
        window_size=10,
        context_dim=5,
    )

    # ── Simulate collecting 5 episodes of varying lengths
    print("\n── Collecting episodes...")
    for ep_idx in range(5):
        ep_len = random.randint(20, 50)  # random episode length
        for step in range(ep_len):
            obs = np.random.randn(10, 5).astype(np.float32)
            ctx = np.random.randn(5).astype(np.float32)
            act = random.randint(0, 63)
            rew = random.uniform(-1.0, 1.0)
            done = (step == ep_len - 1)

            buffer.add_transition(obs, ctx, act, rew, done)

        buffer.end_episode()
        print(f"   Episode {ep_idx + 1}: {ep_len} steps")

    print(f"\n── Buffer stats:")
    stats = buffer.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # ── Test sampling
    print("\n── Sampling a batch of 4 chunks...")
    assert buffer.can_sample(4), "Should be able to sample!"

    batch = buffer.sample(batch_size=4, device="cpu")

    print(f"   obs_seq shape:  {batch['obs_seq'].shape}")    # (4, 8, 10, 5)
    print(f"   context shape:  {batch['context'].shape}")    # (4, 8, 5)
    print(f"   actions shape:  {batch['actions'].shape}")    # (4, 8)
    print(f"   rewards shape:  {batch['rewards'].shape}")    # (4, 8)
    print(f"   dones shape:    {batch['dones'].shape}")      # (4, 8)

    assert batch["obs_seq"].shape == (4, 8, 10, 5)
    assert batch["context"].shape == (4, 8, 5)
    assert batch["actions"].shape == (4, 8)
    assert batch["rewards"].shape == (4, 8)
    assert batch["dones"].shape == (4, 8)

    # Verify action values are valid channel indices
    assert batch["actions"].min() >= 0
    assert batch["actions"].max() < 64

    print("\n   ✅ All shapes and values correct!")

    # ── Test that short episodes are discarded
    print("\n── Testing short episode discard...")
    n_before = len(buffer)
    for step in range(3):  # only 3 steps < chunk_len=8
        buffer.add_transition(
            np.zeros((10, 5), dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            0, 0.0, (step == 2),
        )
    buffer.end_episode()
    n_after = len(buffer)
    assert n_after == n_before, "Short episode should not be stored!"
    print(f"   Episodes before: {n_before}, after: {n_after}")
    print("   ✅ Short episode correctly discarded")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")
