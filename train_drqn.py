"""
train_drqn.py — Full Training Pipeline for Tier 2 DRQN Agent
================================================================

This script implements the two-stage training strategy from the
Person 3 implementation plan:

  Stage 8A: Offline Supervised Pre-training
    - Uses Person 1's PDWDataset to train the LSTM on next-channel
      prediction (cross-entropy over 64 channels).
    - Gives the network a strong understanding of pulse sequences
      before RL begins.

  Stage 8B: Online Reinforcement Learning
    - Runs the DRQN agent in RadarEnv(split='train').
    - Collects rollouts into the RecurrentReplayBuffer.
    - Trains with Double-DQN every few steps.
    - Evaluates on RadarEnv(split='test') periodically.
    - Saves the best checkpoint to checkpoints/drqn_radar_best.pt.

Usage (local):
    python train_drqn.py --device cpu --rl-steps 5000

Usage (Google Colab with GPU):
    !python train_drqn.py --device cuda --rl-steps 20000

Usage (skip pre-training, RL only):
    python train_drqn.py --skip-pretrain --rl-steps 10000

Usage (resume from checkpoint):
    python train_drqn.py --resume checkpoints/drqn_radar_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.radar_env import RadarEnv, N_CHANNELS, FREQ_MIN, FREQ_MAX
from env.memmap_loader import PDWMemmap
from models.state_builder import StateBuilder
from models.drqn_network import DRQNNetwork
from models.replay_buffer import RecurrentReplayBuffer
from models.drqn_agent import DRQNAgent

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_NPY = PROJECT_ROOT / "data" / "raw" / "pdw_records.npy"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "drqn_radar_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8A: OFFLINE SUPERVISED PRE-TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def freq_to_channel(freq_mhz: float, n_channels: int = N_CHANNELS) -> int:
    """Convert a frequency in MHz to a discrete channel index."""
    ratio = (freq_mhz - FREQ_MIN) / max(FREQ_MAX - FREQ_MIN, 1.0)
    ch = int(ratio * n_channels)
    return max(0, min(ch, n_channels - 1))


class PretrainDataset(torch.utils.data.Dataset):
    """
    Dataset for supervised pre-training: predict the next pulse's channel
    from a window of consecutive pulses.

    Each sample:
        input:  (window_size, 5) — normalised pulse window
        target: int — channel index of the (window_size + 1)-th pulse
    """

    def __init__(
        self,
        npy_path: Path,
        window_size: int = 10,
        n_channels: int = N_CHANNELS,
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        self.loader = PDWMemmap(npy_path)
        self.window_size = window_size
        self.n_channels = n_channels
        self.start_idx = start_idx
        self.end_idx = end_idx or len(self.loader)

        # Valid indices: we need window_size pulses + 1 for the target
        self._valid_len = max(0, (self.end_idx - self.start_idx) - window_size)

        # Pre-compute normalisation bounds from StateBuilder
        self._state_builder = StateBuilder(n_channels=n_channels)

    def __len__(self) -> int:
        return self._valid_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        actual_idx = self.start_idx + idx

        # Get the observation window
        window = self.loader.get_batch(actual_idx, self.window_size)  # (W, 8)
        obs = window[:, :5].astype(np.float32)  # Only 5 PDW fields

        # Normalise the window
        norm_obs = self._state_builder._normalise_window(obs)

        # Get the NEXT pulse (the target to predict)
        next_pulse = self.loader.get_record(actual_idx + self.window_size)
        next_freq = float(next_pulse[1])  # Frequency field
        target_channel = freq_to_channel(next_freq, self.n_channels)

        return torch.from_numpy(norm_obs), target_channel


def run_pretrain(
    agent: DRQNAgent,
    npy_path: Path,
    n_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """
    Stage 8A: Supervised pre-training on next-channel prediction.

    Trains the DRQN's shared layers (feature extractor + LSTM) to
    predict which channel the next pulse arrives on. This gives the
    network a strong prior on pulse sequence dynamics before RL.

    Parameters
    ----------
    agent : DRQNAgent
    npy_path : Path to the .npy binary
    n_epochs : int, number of training epochs
    batch_size : int
    lr : float, learning rate for pre-training (higher than RL)
    device : str

    Returns
    -------
    list of average losses per epoch
    """
    print("\n" + "=" * 60)
    print("STAGE 8A: OFFLINE SUPERVISED PRE-TRAINING")
    print("=" * 60)

    if not npy_path.exists():
        print(f"  ⚠️  Data file not found: {npy_path}")
        print("  Skipping pre-training.")
        return []

    # Create dataset (use train split only — first 70% of records)
    loader = PDWMemmap(npy_path)
    split_at = int(len(loader) * 0.7)

    dataset = PretrainDataset(
        npy_path, window_size=10, n_channels=N_CHANNELS,
        start_idx=0, end_idx=split_at,
    )

    if len(dataset) == 0:
        print("  ⚠️  Dataset is empty. Skipping pre-training.")
        return []

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device != "cpu"),
    )

    print(f"  Dataset size: {len(dataset):,} samples")
    print(f"  Batch size:   {batch_size}")
    print(f"  Epochs:       {n_epochs}")
    print(f"  Device:       {device}")

    # Use cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()

    # Use a separate optimiser with higher LR for pre-training
    pretrain_optimiser = torch.optim.Adam(
        agent.q_network.parameters(), lr=lr,
    )

    network = agent.q_network.to(device)
    network.train()

    epoch_losses = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        t_start = time.time()

        for batch_idx, (obs_window, target_ch) in enumerate(dataloader):
            obs_window = obs_window.to(device)       # (B, 10, 5)
            target_ch = target_ch.to(device).long()   # (B,)

            B = obs_window.size(0)

            # Create dummy context features (zeros for pre-training)
            dummy_ctx = torch.zeros(B, 5, device=device)

            # Forward pass — Q-values used as logits
            q_values, _ = network(obs_window, dummy_ctx)  # (B, 64)

            # Cross-entropy loss: treat Q-values as class logits
            loss = criterion(q_values, target_ch)

            # Backward pass
            pretrain_optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            pretrain_optimiser.step()

            # Track accuracy
            predicted = q_values.argmax(dim=-1)
            total_correct += (predicted == target_ch).sum().item()
            total_loss += loss.item() * B
            total_samples += B

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        elapsed = time.time() - t_start

        epoch_losses.append(avg_loss)
        print(
            f"  Epoch {epoch + 1}/{n_epochs}: "
            f"loss={avg_loss:.4f}  acc={accuracy:.3f}  "
            f"({elapsed:.1f}s)"
        )

    # Sync target network after pre-training
    agent.sync_target()
    print("  ✅ Pre-training complete. Target network synced.")

    return epoch_losses


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8B: ONLINE REINFORCEMENT LEARNING
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    agent: DRQNAgent,
    npy_path: Path,
    n_episodes: int = 5,
    max_steps: int = 500,
) -> dict[str, float]:
    """
    Evaluate the agent on the test split (no training, ε=0).

    Returns dict with avg_reward, hit_rate, avg_episode_reward.
    """
    agent.set_eval_mode()
    state_builder = StateBuilder(n_channels=agent.n_actions, max_steps=max_steps)

    all_rewards = []
    all_hits = []
    episode_rewards = []

    for ep in range(n_episodes):
        env = RadarEnv(npy_path, split="test", max_steps=max_steps)
        obs, info = env.reset()
        state_builder.reset()
        hidden = agent.q_network.init_hidden(1, device=agent.device)

        ep_reward = 0.0

        # Build initial state with default info
        init_info = {
            "intercepted": False, "chosen_channel": 0,
            "pulse_channel": 0, "step": 0,
        }
        state = state_builder.build_state(obs, init_info)

        for step in range(max_steps):
            action, hidden = agent.select_action(state, hidden, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)

            state = state_builder.build_state(obs, info)

            all_rewards.append(reward)
            all_hits.append(int(info.get("intercepted", False)))
            ep_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        env.close()

    agent.set_train_mode()

    return {
        "avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "hit_rate": float(np.mean(all_hits)) if all_hits else 0.0,
        "avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "n_steps": len(all_rewards),
    }


def run_rl_training(
    agent: DRQNAgent,
    npy_path: Path,
    total_steps: int = 20_000,
    max_steps_per_episode: int = 500,
    batch_size: int = 32,
    buffer_capacity: int = 5_000,
    chunk_len: int = 16,
    train_every: int = 4,
    min_buffer_episodes: int = 10,
    eval_every_steps: int = 2_000,
    eval_episodes: int = 5,
    log_every_steps: int = 200,
    device: str = "cpu",
) -> dict[str, list]:
    """
    Stage 8B: Online Double-DQN training in RadarEnv.

    Parameters
    ----------
    agent : DRQNAgent
    npy_path : Path to the .npy binary
    total_steps : int, total environment steps to train
    max_steps_per_episode : int
    batch_size : int, replay buffer sample size
    buffer_capacity : int, max episodes in replay buffer
    chunk_len : int, sequence chunk length for training
    train_every : int, train every N env steps
    min_buffer_episodes : int, minimum episodes before training starts
    eval_every_steps : int, evaluate on test split every N steps
    eval_episodes : int, number of evaluation episodes
    log_every_steps : int, print metrics every N steps
    device : str

    Returns
    -------
    dict with training history lists (losses, rewards, hit_rates, etc.)
    """
    print("\n" + "=" * 60)
    print("STAGE 8B: ONLINE REINFORCEMENT LEARNING")
    print("=" * 60)

    if not npy_path.exists():
        print(f"  ⚠️  Data file not found: {npy_path}")
        return {"losses": [], "rewards": [], "hit_rates": []}

    # ── Initialise components
    buffer = RecurrentReplayBuffer(
        capacity=buffer_capacity, chunk_len=chunk_len,
    )
    state_builder = StateBuilder(
        n_channels=agent.n_actions, max_steps=max_steps_per_episode,
    )

    # ── Training history
    history = {
        "losses": [],
        "rewards": [],
        "hit_rates": [],
        "epsilons": [],
        "eval_rewards": [],
        "eval_hit_rates": [],
        "eval_steps": [],
    }

    # ── Training loop
    global_step = 0
    episode_count = 0
    best_eval_reward = float("-inf")

    recent_rewards = []
    recent_hits = []

    print(f"  Total steps:  {total_steps:,}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Chunk length: {chunk_len}")
    print(f"  Train every:  {train_every} steps")
    print(f"  Eval every:   {eval_every_steps} steps")
    print(f"  Device:       {device}")
    print()

    agent.set_train_mode()
    t_start = time.time()

    while global_step < total_steps:
        # ── Start a new episode
        env = RadarEnv(
            npy_path, split="train",
            max_steps=max_steps_per_episode,
        )
        obs, info = env.reset()
        state_builder.reset()
        hidden = agent.q_network.init_hidden(1, device=agent.device)

        ep_reward = 0.0
        ep_hits = 0
        ep_steps = 0

        # Initial state (no prior action info)
        init_info = {
            "intercepted": False, "chosen_channel": 0,
            "pulse_channel": 0, "step": 0,
        }
        state = state_builder.build_state(obs, init_info)

        for step in range(max_steps_per_episode):
            if global_step >= total_steps:
                break

            # ── Select action
            action, hidden = agent.select_action(state, hidden)

            # ── Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # ── Build next state
            next_state = state_builder.build_state(obs, info)

            # ── Store transition in replay buffer
            done = terminated or truncated
            buffer.add_transition(
                state["sequence"].numpy(),
                state["context"].numpy(),
                action, reward, done,
            )

            # ── Track metrics
            ep_reward += reward
            ep_hits += int(info.get("intercepted", False))
            ep_steps += 1
            global_step += 1
            agent.increment_step()

            recent_rewards.append(reward)
            recent_hits.append(int(info.get("intercepted", False)))

            # ── Train from replay buffer
            if (
                global_step % train_every == 0
                and buffer.can_sample(batch_size)
                and len(buffer) >= min_buffer_episodes
            ):
                batch = buffer.sample(batch_size, device=agent.device)
                loss = agent.train_step(batch)
                history["losses"].append(loss)

            # ── Log progress
            if global_step % log_every_steps == 0 and len(recent_rewards) > 0:
                avg_r = np.mean(recent_rewards[-log_every_steps:])
                avg_h = np.mean(recent_hits[-log_every_steps:])
                eps = agent.epsilon

                history["rewards"].append(avg_r)
                history["hit_rates"].append(avg_h)
                history["epsilons"].append(eps)

                elapsed = time.time() - t_start
                steps_per_sec = global_step / max(elapsed, 0.1)

                print(
                    f"  Step {global_step:>6,}/{total_steps:,} | "
                    f"ε={eps:.3f} | "
                    f"reward={avg_r:+.3f} | "
                    f"hit_rate={avg_h:.3f} | "
                    f"loss={history['losses'][-1]:.4f} | "
                    f"buf={len(buffer)} eps | "
                    f"{steps_per_sec:.0f} steps/s"
                ) if history["losses"] else print(
                    f"  Step {global_step:>6,}/{total_steps:,} | "
                    f"ε={eps:.3f} | "
                    f"reward={avg_r:+.3f} | "
                    f"hit_rate={avg_h:.3f} | "
                    f"collecting..."
                )

            # ── Periodic evaluation
            if global_step % eval_every_steps == 0 and global_step > 0:
                print(f"\n  ── Evaluating on test split...")
                eval_result = evaluate_agent(
                    agent, npy_path,
                    n_episodes=eval_episodes,
                    max_steps=max_steps_per_episode,
                )

                history["eval_rewards"].append(eval_result["avg_episode_reward"])
                history["eval_hit_rates"].append(eval_result["hit_rate"])
                history["eval_steps"].append(global_step)

                print(
                    f"     Test hit_rate={eval_result['hit_rate']:.3f} | "
                    f"avg_ep_reward={eval_result['avg_episode_reward']:.2f}"
                )

                # Save best checkpoint
                if eval_result["avg_episode_reward"] > best_eval_reward:
                    best_eval_reward = eval_result["avg_episode_reward"]
                    agent.save(BEST_CKPT, eval_reward=best_eval_reward)
                    print(f"     💾 New best! Saved to {BEST_CKPT}")

                print()
                agent.set_train_mode()

            # ── Update state
            state = next_state

            if done:
                break

        # ── End of episode
        buffer.end_episode()
        episode_count += 1

        env.close()

    # ── Final evaluation
    print("\n" + "-" * 60)
    print("FINAL EVALUATION")
    print("-" * 60)

    final_eval = evaluate_agent(
        agent, npy_path,
        n_episodes=eval_episodes * 2,
        max_steps=max_steps_per_episode,
    )

    print(f"  Final test hit_rate:       {final_eval['hit_rate']:.3f}")
    print(f"  Final avg_episode_reward:  {final_eval['avg_episode_reward']:.2f}")
    print(f"  Total episodes trained:    {episode_count}")
    print(f"  Total env steps:           {global_step:,}")
    print(f"  Total training updates:    {agent._train_step_count:,}")
    print(f"  Final epsilon:             {agent.epsilon:.4f}")

    elapsed = time.time() - t_start
    print(f"  Wall time:                 {elapsed:.1f}s")

    # Save final checkpoint
    agent.save(BEST_CKPT, eval_reward=final_eval["avg_episode_reward"])
    print(f"\n  💾 Final checkpoint saved to {BEST_CKPT}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the Tier 2 DRQN agent for cognitive radar interception.",
    )

    # Paths
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_NPY),
        help="Path to the pdw_records.npy binary file.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from.",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to train on (default: auto-detect).",
    )

    # Pre-training (Stage 8A)
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Skip Stage 8A supervised pre-training.",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=5,
        help="Number of pre-training epochs (default: 5).",
    )
    parser.add_argument(
        "--pretrain-lr", type=float, default=1e-3,
        help="Learning rate for pre-training (default: 1e-3).",
    )
    parser.add_argument(
        "--pretrain-batch", type=int, default=64,
        help="Batch size for pre-training (default: 64).",
    )

    # RL Training (Stage 8B)
    parser.add_argument(
        "--rl-steps", type=int, default=20_000,
        help="Total environment steps for RL training (default: 20000).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Replay buffer batch size (default: 32).",
    )
    parser.add_argument(
        "--chunk-len", type=int, default=16,
        help="Sequence chunk length for LSTM training (default: 16).",
    )
    parser.add_argument(
        "--eval-every", type=int, default=2_000,
        help="Evaluate on test split every N steps (default: 2000).",
    )
    parser.add_argument(
        "--log-every", type=int, default=200,
        help="Log metrics every N steps (default: 200).",
    )

    # Agent hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="RL learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument("--eps-decay", type=int, default=20_000, help="Epsilon decay steps.")

    args = parser.parse_args()

    # ── Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    npy_path = Path(args.data)

    print("=" * 60)
    print("TIER 2 DRQN TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Data:         {npy_path}")
    print(f"  Device:       {device}")
    print(f"  Pre-train:    {'skip' if args.skip_pretrain else f'{args.pretrain_epochs} epochs'}")
    print(f"  RL steps:     {args.rl_steps:,}")
    print(f"  Resume from:  {args.resume or 'scratch'}")

    # ── Check data file
    if not npy_path.exists():
        print(f"\n❌ Data file not found: {npy_path}")
        print("   Run the data pipeline first:")
        print("     python data/download_dataset.py")
        print("     python data/parse_pdw.py")
        print("     python data/convert_to_binary.py")
        sys.exit(1)

    # ── Create agent
    agent = DRQNAgent(
        n_actions=N_CHANNELS,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay,
        device=device,
    )

    print(f"\n  Agent: {agent.q_network.count_parameters():,} parameters")

    # ── Resume from checkpoint if provided
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = agent.load(resume_path)
            print(f"  ✅ Resumed from {resume_path}")
            print(f"     Global step: {ckpt['training_state']['global_step']:,}")
        else:
            print(f"  ⚠️  Checkpoint not found: {resume_path}. Starting fresh.")

    # ── Stage 8A: Pre-training
    if not args.skip_pretrain and args.resume is None:
        pretrain_losses = run_pretrain(
            agent, npy_path,
            n_epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch,
            lr=args.pretrain_lr,
            device=device,
        )

    # ── Stage 8B: Online RL
    history = run_rl_training(
        agent, npy_path,
        total_steps=args.rl_steps,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
        eval_every_steps=args.eval_every,
        log_every_steps=args.log_every,
        device=device,
    )

    # ── Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Checkpoint: {BEST_CKPT}")
    print(f"  Agent status: {agent.status()}")

    if history["eval_hit_rates"]:
        print(f"  Best eval hit rate: {max(history['eval_hit_rates']):.3f}")
        print(f"  Final eval hit rate: {history['eval_hit_rates'][-1]:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
