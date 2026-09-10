"""
models/drqn_network.py — Deep Recurrent Q-Network with Dueling Architecture
=============================================================================

This is the core neural network for Tier 2 cognitive radar interception.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────┐
    │  Input: normalised pulse window (batch, seq_len, 5)     │
    │         + context features (batch, 5)                   │
    └────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Feature Extract │  Linear(5 → 128) + LayerNorm + ReLU
                    │ (per time step) │  Output: (batch, seq_len, 128)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   2-Layer LSTM  │  LSTM(128, 256, num_layers=2)
                    │  (sequence mem) │  Output: (batch, seq_len, 256)
                    └────────┬────────┘
                             │ take last time step output
                             │
                    ┌────────▼────────┐
                    │ Concat Context  │  cat(lstm_out[-1], context) → (batch, 261)
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
        ┌────────▼────────┐     ┌────────▼────────┐
        │  Value Stream   │     │ Advantage Stream│
        │ 261→64→ReLU→1   │     │ 261→64→ReLU→64  │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 └───────────┬───────────┘
                             │
                    ┌────────▼────────┐
                    │   Q(s,a) = V(s) │
                    │ + A(s,a) - μ(A) │  Dueling combination
                    └────────┬────────┘
                             │
                    Output: (batch, 64) Q-values

Why Dueling?
    Standard DQN outputs Q(s,a) directly. Dueling separates the value of
    being in state s from the advantage of each action. This helps when
    many actions have similar values — the network learns "this is a good
    state" without needing to evaluate every single channel.

Why LSTM (not GRU or Transformer)?
    - LSTM's gating mechanism (forget/input/output gates) is well-suited
      for detecting periodic patterns in radar pulse sequences.
    - 2 layers allow hierarchical feature extraction: Layer 1 detects
      single-hop transitions, Layer 2 detects multi-hop sequences.
    - Computationally efficient for Person 4's ≤2ms latency budget.

Usage:
    from models.drqn_network import DRQNNetwork

    net = DRQNNetwork(n_actions=64)

    # Training mode (full sequence):
    q_values, hidden = net(sequence, context)

    # Inference mode (single step, maintaining hidden state):
    q_values, hidden = net(sequence, context, hidden=prev_hidden)

    # Reset hidden state:
    hidden = net.init_hidden(batch_size=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DRQNNetwork(nn.Module):
    """
    Deep Recurrent Q-Network with Dueling architecture.

    Parameters
    ----------
    input_dim : int
        Number of features per time step in the pulse window (default: 5).
    context_dim : int
        Number of engineered context features (default: 5).
    feature_dim : int
        Output dimension of the feature extraction layer (default: 128).
    hidden_dim : int
        LSTM hidden state dimension (default: 256).
    n_layers : int
        Number of stacked LSTM layers (default: 2).
    n_actions : int
        Number of discrete actions / frequency channels (default: 64).
    dropout : float
        Dropout between LSTM layers (default: 0.1). Only active during training.
    """

    def __init__(
        self,
        input_dim: int = 5,
        context_dim: int = 5,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_actions: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_actions = n_actions

        # ── Layer 1: Feature Extraction (applied per time step)
        # Transforms raw 5-dim pulse features into a richer 128-dim representation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

        # ── Layer 2: Recurrent Memory
        # 2-layer LSTM that processes the sequence and maintains hidden state
        # across time steps for pattern detection
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # ── Layer 3: Dueling Q-Value Heads
        # Combined dimension: LSTM output (256) + context features (5) = 261
        combined_dim = hidden_dim + context_dim

        # Value stream: "How good is this state overall?"
        self.value_stream = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Advantage stream: "How much better is each channel vs. average?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

        # ── Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise network weights using best practices for ReLU + LSTM.

        - Linear layers: Kaiming (He) initialisation for ReLU activations.
        - LSTM: Orthogonal initialisation for recurrent weights (prevents
          vanishing/exploding gradients in long unrolls).
        - Biases: Zero initialised, except LSTM forget gate bias set to 1.0
          (encourages remembering by default — critical for sequence learning).
        """
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.kaiming_normal_(param, nonlinearity="relu")
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1.0 (LSTM bias is split into 4 gates)
                    # Gate order: input, forget, cell, output
                    n = param.size(0) // 4
                    param.data[n: 2 * n].fill_(1.0)
            elif "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    # ──────────────────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DRQN.

        Parameters
        ----------
        sequence : torch.Tensor
            Normalised pulse window, shape (batch, seq_len, input_dim).
            In training: full sequence (e.g., batch=32, seq_len=10, input_dim=5).
            In inference: single step (batch=1, seq_len=1, input_dim=5).
        context : torch.Tensor
            Engineered context features, shape (batch, context_dim).
        hidden : tuple of (h, c) or None
            Previous LSTM hidden state. If None, initialised to zeros.
            h shape: (n_layers, batch, hidden_dim)
            c shape: (n_layers, batch, hidden_dim)

        Returns
        -------
        q_values : torch.Tensor
            Q-values for each action, shape (batch, n_actions).
        hidden : tuple of (h, c)
            Updated LSTM hidden state for the next step.
        """
        batch_size = sequence.size(0)

        # ── Feature extraction (applied to each time step independently)
        # (batch, seq_len, 5) → (batch, seq_len, 128)
        features = self.feature_extractor(sequence)

        # ── LSTM processing
        # Initialise hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, device=sequence.device)

        # (batch, seq_len, 128) → (batch, seq_len, 256)
        lstm_out, hidden = self.lstm(features, hidden)

        # Take only the LAST time step's output for Q-value computation
        # (batch, seq_len, 256) → (batch, 256)
        lstm_last = lstm_out[:, -1, :]

        # ── Concatenate with context features
        # (batch, 256) + (batch, 5) → (batch, 261)
        combined = torch.cat([lstm_last, context], dim=-1)

        # ── Dueling Q-value computation
        value = self.value_stream(combined)          # (batch, 1)
        advantage = self.advantage_stream(combined)  # (batch, 64)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        # Subtracting the mean advantage ensures identifiability:
        # V(s) truly represents the state value, not an arbitrary offset.
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values, hidden

    # ──────────────────────────────────────────────────────────────────────────
    # HIDDEN STATE MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a fresh zero-initialised hidden state for the LSTM.

        Called at the start of each episode or when the hidden state
        needs to be reset (e.g., after a fallback trigger).

        Returns
        -------
        (h_0, c_0) : tuple of tensors
            h_0 shape: (n_layers, batch_size, hidden_dim)
            c_0 shape: (n_layers, batch_size, hidden_dim)
        """
        h_0 = torch.zeros(
            self.n_layers, batch_size, self.hidden_dim,
            device=device, dtype=torch.float32,
        )
        c_0 = torch.zeros(
            self.n_layers, batch_size, self.hidden_dim,
            device=device, dtype=torch.float32,
        )
        return (h_0, c_0)

    def detach_hidden(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detach hidden state from the computation graph.

        Must be called between episodes or sequence chunks during training
        to prevent backpropagating through the entire episode history
        (which would cause memory explosion and gradient issues).
        """
        return (hidden[0].detach(), hidden[1].detach())

    # ──────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n_params = self.count_parameters()
        return (
            f"DRQNNetwork(\n"
            f"  input_dim={self.input_dim}, context_dim={self.context_dim},\n"
            f"  feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim},\n"
            f"  n_layers={self.n_layers}, n_actions={self.n_actions},\n"
            f"  total_params={n_params:,}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DRQNNetwork — Standalone Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    net = DRQNNetwork(n_actions=64).to(device)
    print(f"\n{net}")

    # ── Test 1: Training mode (full sequence batch)
    print("\n── Test 1: Training forward pass (batch=4, seq_len=10)")
    batch_size = 4
    seq_len = 10

    fake_seq = torch.randn(batch_size, seq_len, 5, device=device)
    fake_ctx = torch.randn(batch_size, 5, device=device)

    q_values, hidden = net(fake_seq, fake_ctx)

    print(f"   Q-values shape: {q_values.shape}")       # (4, 64)
    print(f"   Hidden h shape: {hidden[0].shape}")       # (2, 4, 256)
    print(f"   Hidden c shape: {hidden[1].shape}")       # (2, 4, 256)
    assert q_values.shape == (batch_size, 64), f"Expected (4, 64), got {q_values.shape}"
    print("   ✅ Training forward pass OK")

    # ── Test 2: Inference mode (single step, maintained hidden state)
    print("\n── Test 2: Inference forward pass (batch=1, seq_len=1)")
    hidden = net.init_hidden(batch_size=1, device=device)

    single_seq = torch.randn(1, 1, 5, device=device)
    single_ctx = torch.randn(1, 5, device=device)

    q_values, hidden = net(single_seq, single_ctx, hidden=hidden)

    print(f"   Q-values shape: {q_values.shape}")       # (1, 64)
    action = q_values.argmax(dim=-1).item()
    print(f"   Best action: channel {action}")
    assert 0 <= action < 64, f"Action {action} out of range!"
    print("   ✅ Inference forward pass OK")

    # ── Test 3: Hidden state detach
    print("\n── Test 3: Hidden state detach")
    hidden_detached = net.detach_hidden(hidden)
    assert not hidden_detached[0].requires_grad, "Hidden h should not require grad after detach"
    assert not hidden_detached[1].requires_grad, "Hidden c should not require grad after detach"
    print("   ✅ Hidden state detach OK")

    # ── Test 4: Gradient flow (backward pass)
    print("\n── Test 4: Backward pass (gradient flow)")
    net.zero_grad()
    fake_seq = torch.randn(4, 10, 5, device=device)
    fake_ctx = torch.randn(4, 5, device=device)
    q_values, _ = net(fake_seq, fake_ctx)

    # Simulate a loss: MSE between Q-values and random targets
    target = torch.randn_like(q_values)
    loss = nn.functional.smooth_l1_loss(q_values, target)
    loss.backward()

    # Check gradients exist
    has_grads = all(
        p.grad is not None for p in net.parameters() if p.requires_grad
    )
    print(f"   Loss: {loss.item():.4f}")
    print(f"   All parameters have gradients: {has_grads}")
    assert has_grads, "Some parameters missing gradients!"
    print("   ✅ Backward pass OK")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")
