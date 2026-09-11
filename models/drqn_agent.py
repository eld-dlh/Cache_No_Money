"""
models/drqn_agent.py — Double-DQN Agent with Epsilon-Greedy Exploration
=========================================================================

This is the decision-making agent that ties together:
  - Main Q-network (θ): updated every training step
  - Target Q-network (θ⁻): frozen copy, synced periodically
  - Epsilon-greedy exploration schedule
  - Double-DQN training logic (eliminates Q-value overestimation)

Double-DQN (DDQN) formula:
    Y_t = r_t + γ · Q(s_{t+1}, argmax_a Q(s_{t+1}, a; θ); θ⁻)

    The MAIN network selects the best action (argmax),
    the TARGET network evaluates how good that action is.
    This prevents the same network from both proposing and
    judging actions, which causes systematic overestimation.

Epsilon Schedule:
    ε decays linearly from 1.0 → 0.05 over 20,000 steps.
    - At ε=1.0: 100% random exploration (start of training)
    - At ε=0.05: 95% exploitation, 5% random (converged)
    - In evaluation mode: ε=0.0 (pure exploitation)

Usage:
    from models.drqn_agent import DRQNAgent

    agent = DRQNAgent(n_actions=64, device="cuda")

    # Select action:
    action, hidden = agent.select_action(state, hidden)

    # Train from replay buffer batch:
    loss = agent.train_step(batch)

    # Sync target network:
    agent.sync_target()

    # Save/Load:
    agent.save("checkpoints/drqn_radar_best.pt")
    agent.load("checkpoints/drqn_radar_best.pt")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.drqn_network import DRQNNetwork


class DRQNAgent:
    """
    Double-DQN agent with recurrent (LSTM) Q-network and epsilon-greedy
    exploration.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions / frequency channels (default: 64).
    input_dim : int
        Per-timestep feature dimension (default: 5).
    context_dim : int
        Number of context features from StateBuilder (default: 5).
    feature_dim : int
        Feature extraction layer output dim (default: 128).
    hidden_dim : int
        LSTM hidden state dimension (default: 256).
    n_layers : int
        Number of LSTM layers (default: 2).
    lr : float
        Learning rate for AdamW optimiser (default: 1e-4).
    weight_decay : float
        L2 regularisation for AdamW (default: 1e-5).
    gamma : float
        Discount factor for future rewards (default: 0.99).
    epsilon_start : float
        Initial exploration rate (default: 1.0).
    epsilon_end : float
        Final exploration rate (default: 0.05).
    epsilon_decay_steps : int
        Number of steps over which epsilon decays linearly (default: 20000).
    target_sync_freq : int
        Sync target network every N training steps (default: 500).
    max_grad_norm : float
        Maximum gradient norm for clipping (default: 5.0).
    device : str or torch.device
        Device to run on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        n_actions: int = 64,
        input_dim: int = 5,
        context_dim: int = 5,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        target_sync_freq: int = 500,
        max_grad_norm: float = 5.0,
        device: str | torch.device = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_freq = target_sync_freq
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # ── Build main Q-network and target Q-network
        net_kwargs = dict(
            input_dim=input_dim,
            context_dim=context_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_actions=n_actions,
        )
        self.q_network = DRQNNetwork(**net_kwargs).to(self.device)
        self.target_network = DRQNNetwork(**net_kwargs).to(self.device)

        # Initialise target with same weights as main
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target is never trained directly

        # ── Optimiser: AdamW (Adam with decoupled weight decay)
        self.optimiser = optim.AdamW(
            self.q_network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,
        )

        # ── Loss: Smooth L1 (Huber loss) — less sensitive to outliers
        self.loss_fn = nn.SmoothL1Loss()

        # ── Step counters
        self._global_step: int = 0      # total env steps taken
        self._train_step_count: int = 0  # total training updates done
        self._best_eval_reward: float = float("-inf")

    # ──────────────────────────────────────────────────────────────────────────
    # EPSILON SCHEDULE
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """Current exploration rate based on linear decay schedule."""
        progress = min(self._global_step / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def increment_step(self) -> None:
        """Increment the global step counter (call once per env step)."""
        self._global_step += 1

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION SELECTION
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        state: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        evaluate: bool = False,
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : dict
            Output from StateBuilder.build_state(). Keys:
            - 'sequence': tensor (window_size, 5)
            - 'context':  tensor (context_dim,)
        hidden : tuple of (h, c) or None
            Previous LSTM hidden state.
        evaluate : bool
            If True, use ε=0 (pure exploitation, no exploration).

        Returns
        -------
        action : int
            Selected channel index (0 to n_actions-1).
        hidden : tuple of (h, c)
            Updated LSTM hidden state.
        """
        eps = 0.0 if evaluate else self.epsilon

        # Epsilon-greedy: explore with probability ε
        if np.random.random() < eps:
            # Random exploration
            action = np.random.randint(0, self.n_actions)

            # Still run forward pass to update hidden state (important!)
            seq = state["sequence"].unsqueeze(0).to(self.device)  # (1, W, 5)
            ctx = state["context"].unsqueeze(0).to(self.device)   # (1, C)
            _, hidden = self.q_network(seq, ctx, hidden)

            return action, hidden

        # Exploitation: pick action with highest Q-value
        seq = state["sequence"].unsqueeze(0).to(self.device)  # (1, W, 5)
        ctx = state["context"].unsqueeze(0).to(self.device)   # (1, C)

        # In evaluation mode on full observation windows, process the window cleanly
        h_in = None if evaluate else hidden
        q_values, hidden = self.q_network(seq, ctx, h_in)
        action = q_values.argmax(dim=-1).item()

        return action, hidden

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Perform one Double-DQN training update from a replay buffer batch.

        The batch contains sequence chunks of length T. We:
          1. Unroll the main network across all T steps → Q-values at each step.
          2. Unroll the target network across all T steps → target Q-values.
          3. Compute Double-DQN targets for steps 0..T-2:
               Y_t = r_t + γ · Q_target(s_{t+1}, argmax Q_main(s_{t+1})) · (1-done)
          4. Compute Huber loss between predicted Q(s_t, a_t) and Y_t.
          5. Backpropagate and clip gradients.

        Parameters
        ----------
        batch : dict
            From RecurrentReplayBuffer.sample(). Keys:
            - 'obs_seq':  (batch, chunk_len, window_size, 5)
            - 'context':  (batch, chunk_len, context_dim)
            - 'actions':  (batch, chunk_len)
            - 'rewards':  (batch, chunk_len)
            - 'dones':    (batch, chunk_len)

        Returns
        -------
        loss_value : float
            The scalar loss for logging.
        """
        self.q_network.train()

        obs_seq = batch["obs_seq"]    # (B, T, W, 5)
        context = batch["context"]    # (B, T, C)
        actions = batch["actions"]    # (B, T)
        rewards = batch["rewards"]    # (B, T)
        dones = batch["dones"].float()  # (B, T) — convert bool → float

        batch_size = obs_seq.size(0)
        chunk_len = obs_seq.size(1)

        # ── Step 1: Unroll main network across all T steps
        main_q_all = self._unroll_network(
            self.q_network, obs_seq, context, batch_size, chunk_len
        )  # (B, T, n_actions)

        # ── Step 2: Unroll target network across all T steps
        with torch.no_grad():
            target_q_all = self._unroll_network(
                self.target_network, obs_seq, context, batch_size, chunk_len
            )  # (B, T, n_actions)

        # ── Step 3: Compute Double-DQN targets for steps 0..T-2
        # Current Q-values: Q_main(s_t, a_t) for t in [0, T-2]
        current_actions = actions[:, :-1].unsqueeze(-1)  # (B, T-1, 1)
        current_q = main_q_all[:, :-1, :].gather(2, current_actions).squeeze(-1)
        # → (B, T-1)

        # Next-step Q-values using Double-DQN:
        # Action selection: argmax of MAIN network at t+1
        next_actions = main_q_all[:, 1:, :].argmax(dim=-1, keepdim=True)  # (B, T-1, 1)

        # Action evaluation: Q-value from TARGET network at t+1
        next_q = target_q_all[:, 1:, :].gather(2, next_actions).squeeze(-1)
        # → (B, T-1)

        # Compute targets: Y_t = r_t + γ · next_q · (1 - done_t)
        targets = rewards[:, :-1] + self.gamma * next_q * (1.0 - dones[:, :-1])

        # ── Step 4: Compute loss (Smooth L1 / Huber)
        loss = self.loss_fn(current_q, targets.detach())

        # ── Step 5: Backpropagate with gradient clipping
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.max_grad_norm
        )
        self.optimiser.step()

        # ── Update counters and auto-sync target if needed
        self._train_step_count += 1
        if self._train_step_count % self.target_sync_freq == 0:
            self.sync_target()

        return loss.item()

    def _unroll_network(
        self,
        network: DRQNNetwork,
        obs_seq: torch.Tensor,
        context: torch.Tensor,
        batch_size: int,
        chunk_len: int,
    ) -> torch.Tensor:
        """
        Unroll a DRQN network step-by-step across a sequence chunk.

        At each step t, the full observation window (10 pulses) is fed
        through the network, and the LSTM hidden state carries forward
        to the next step.

        Parameters
        ----------
        network : DRQNNetwork
        obs_seq : (batch, chunk_len, window_size, 5)
        context : (batch, chunk_len, context_dim)
        batch_size : int
        chunk_len : int

        Returns
        -------
        torch.Tensor of shape (batch, chunk_len, n_actions)
        """
        hidden = network.init_hidden(batch_size, device=obs_seq.device)
        all_q_values = []

        for t in range(chunk_len):
            # obs at step t: (batch, window_size, 5) — the sequence input
            step_obs = obs_seq[:, t, :, :]   # (B, W, 5)
            step_ctx = context[:, t, :]      # (B, C)

            q_values, hidden = network(step_obs, step_ctx, hidden)
            all_q_values.append(q_values)

            # Detach hidden state to prevent backprop through entire chunk
            # (we still get gradients within each step, but not across steps
            # — this is a standard DRQN trade-off for memory efficiency)
            hidden = network.detach_hidden(hidden)

        return torch.stack(all_q_values, dim=1)  # (B, T, n_actions)

    # ──────────────────────────────────────────────────────────────────────────
    # TARGET NETWORK SYNC
    # ──────────────────────────────────────────────────────────────────────────

    def sync_target(self) -> None:
        """
        Hard-copy main network weights to the target network.

        Called automatically every target_sync_freq training steps.
        Can also be called manually.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_sync_target(self, tau: float = 0.005) -> None:
        """
        Soft (Polyak) update of target network weights.

        θ⁻ ← τ·θ + (1-τ)·θ⁻

        Smoother than hard sync but slower convergence. Use one or the other.
        """
        for target_param, main_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters(),
        ):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )

    # ──────────────────────────────────────────────────────────────────────────
    # CHECKPOINTING
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path, eval_reward: float | None = None) -> None:
        """
        Save a full checkpoint for resuming training or deployment.

        Parameters
        ----------
        path : str or Path
            File path for the checkpoint (e.g., 'checkpoints/drqn_radar_best.pt').
        eval_reward : float or None
            If provided, updates the best eval reward tracker.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if eval_reward is not None:
            self._best_eval_reward = max(self._best_eval_reward, eval_reward)

        checkpoint = {
            "model_state_dict": self.q_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "config": {
                "n_actions": self.n_actions,
                "input_dim": self.q_network.input_dim,
                "context_dim": self.q_network.context_dim,
                "feature_dim": self.q_network.feature_dim,
                "hidden_dim": self.q_network.hidden_dim,
                "n_layers": self.q_network.n_layers,
                "gamma": self.gamma,
            },
            "training_state": {
                "global_step": self._global_step,
                "train_step_count": self._train_step_count,
                "epsilon": self.epsilon,
                "best_eval_reward": self._best_eval_reward,
            },
        }

        torch.save(checkpoint, path)

    def load(self, path: str | Path, load_optimiser: bool = True) -> dict:
        """
        Load a checkpoint.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        load_optimiser : bool
            If True, restore optimiser state (for resuming training).
            If False, only load model weights (for inference/deployment).

        Returns
        -------
        dict : The full checkpoint dict for inspection.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["model_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])

        if load_optimiser and "optimizer_state_dict" in checkpoint:
            self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

        if "training_state" in checkpoint:
            ts = checkpoint["training_state"]
            self._global_step = ts.get("global_step", 0)
            self._train_step_count = ts.get("train_step_count", 0)
            self._best_eval_reward = ts.get("best_eval_reward", float("-inf"))

        return checkpoint

    # ──────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return a summary of the agent's current state."""
        return {
            "global_step": self._global_step,
            "train_steps": self._train_step_count,
            "epsilon": round(self.epsilon, 4),
            "best_eval_reward": self._best_eval_reward,
            "device": str(self.device),
            "total_params": self.q_network.count_parameters(),
        }

    def set_eval_mode(self) -> None:
        """Set both networks to evaluation mode (disables dropout)."""
        self.q_network.eval()
        self.target_network.eval()

    def set_train_mode(self) -> None:
        """Set main network to training mode (enables dropout)."""
        self.q_network.train()
        self.target_network.eval()  # Target is always in eval mode


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DRQNAgent — Standalone Smoke Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    agent = DRQNAgent(n_actions=64, device=device)
    print(f"\n  Agent status: {agent.status()}")

    # ── Test 1: Action selection
    print("\n── Test 1: Action selection (epsilon-greedy)")
    fake_state = {
        "sequence": torch.randn(10, 5),
        "context": torch.randn(5),
    }
    hidden = None

    actions_taken = []
    for i in range(20):
        action, hidden = agent.select_action(fake_state, hidden)
        actions_taken.append(action)
        agent.increment_step()

    print(f"   Actions taken: {actions_taken[:10]}...")
    print(f"   All in [0, 63]: {all(0 <= a < 64 for a in actions_taken)}")
    print(f"   Current epsilon: {agent.epsilon:.4f}")
    print("   ✅ Action selection OK")

    # ── Test 2: Training step
    print("\n── Test 2: Training step (Double-DQN)")
    batch_size = 4
    chunk_len = 8
    fake_batch = {
        "obs_seq": torch.randn(batch_size, chunk_len, 10, 5, device=device),
        "context": torch.randn(batch_size, chunk_len, 5, device=device),
        "actions": torch.randint(0, 64, (batch_size, chunk_len), device=device),
        "rewards": torch.randn(batch_size, chunk_len, device=device),
        "dones":   torch.zeros(batch_size, chunk_len, dtype=torch.bool, device=device),
    }

    loss = agent.train_step(fake_batch)
    print(f"   Loss: {loss:.4f}")
    print(f"   Train steps: {agent._train_step_count}")
    print("   ✅ Training step OK")

    # ── Test 3: Target sync
    print("\n── Test 3: Target network sync")
    # Modify main network slightly
    for p in agent.q_network.parameters():
        p.data += 0.01
    # Check they differ
    main_p = list(agent.q_network.parameters())[0].data.flatten()[:5]
    target_p = list(agent.target_network.parameters())[0].data.flatten()[:5]
    differ_before = not torch.allclose(main_p, target_p)
    print(f"   Networks differ before sync: {differ_before}")

    agent.sync_target()
    main_p = list(agent.q_network.parameters())[0].data.flatten()[:5]
    target_p = list(agent.target_network.parameters())[0].data.flatten()[:5]
    same_after = torch.allclose(main_p, target_p)
    print(f"   Networks same after sync: {same_after}")
    assert same_after, "Target should match main after sync!"
    print("   ✅ Target sync OK")

    # ── Test 4: Save and load
    print("\n── Test 4: Checkpoint save/load")
    import tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), "test_drqn_ckpt.pt")
    agent.save(tmp_path, eval_reward=0.42)
    print(f"   Saved to: {tmp_path}")

    # Create a fresh agent and load
    agent2 = DRQNAgent(n_actions=64, device=device)
    ckpt = agent2.load(tmp_path)
    print(f"   Loaded config: {ckpt['config']}")
    print(f"   Loaded training state: {ckpt['training_state']}")

    # Verify weights match
    for p1, p2 in zip(agent.q_network.parameters(), agent2.q_network.parameters()):
        assert torch.allclose(p1, p2), "Loaded weights don't match!"
    print("   ✅ Save/Load OK")

    # Clean up
    os.remove(tmp_path)

    # ── Test 5: Epsilon decay
    print("\n── Test 5: Epsilon decay schedule")
    agent3 = DRQNAgent(epsilon_decay_steps=100)
    eps_values = []
    for i in range(110):
        eps_values.append(agent3.epsilon)
        agent3.increment_step()
    print(f"   ε at step 0:   {eps_values[0]:.4f}")
    print(f"   ε at step 50:  {eps_values[50]:.4f}")
    print(f"   ε at step 100: {eps_values[100]:.4f}")
    assert abs(eps_values[0] - 1.0) < 1e-6, "Initial epsilon should be 1.0"
    assert abs(eps_values[100] - 0.05) < 1e-6, "Final epsilon should be 0.05"
    print("   ✅ Epsilon decay OK")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")
