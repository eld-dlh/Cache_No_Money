"""
tests/test_drqn_pipeline.py — Full Unit Tests for Person 3 (Tier 2 DRQN)
==========================================================================

This test suite validates ALL Person 3 components end-to-end using
synthetic data only — no dataset download required. Run this locally
before pushing to GitHub to catch any issues.

What we check:
  ✅ DRQNNetwork forward pass: (1, 10, 5) → valid action in [0, 63]
  ✅ LSTM hidden state persistence across steps
  ✅ LSTM hidden state reset to zeros
  ✅ Replay buffer: write, sample, verify shapes
  ✅ Training step: loss is finite, gradients flow
  ✅ StateBuilder: all outputs normalised to [0, 1]
  ✅ Pattern-lock trigger fires on cyclic sequences
  ✅ Fallback trigger fires on consecutive misses
  ✅ Checkpoint save/load roundtrip
  ✅ Epsilon decay schedule correctness
  ✅ Full mini-episode integration test

Usage:
    python tests/test_drqn_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# ── Fix Windows terminal encoding (cp1252 can't handle Unicode symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

# ── Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.state_builder import StateBuilder
from models.drqn_network import DRQNNetwork
from models.replay_buffer import RecurrentReplayBuffer
from models.drqn_agent import DRQNAgent
from models.handoff_controller import HandoffController

# ─────────────────────────────────────────────────────────────────────────────
# TEST INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0


def test(name: str):
    """Decorator-like printer for test sections."""
    print(f"\n── {name}")


def check(condition: bool, msg: str):
    """Assert with tracking."""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"   ✅ {msg}")
    else:
        FAIL_COUNT += 1
        print(f"   ❌ FAILED: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: DRQN NETWORK — FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

def test_drqn_forward_pass():
    test("Test 1: DRQNNetwork Forward Pass")

    net = DRQNNetwork(input_dim=5, context_dim=5, n_actions=64)

    # Training mode: batch of 4, sequence of 10 pulses
    seq = torch.randn(4, 10, 5)
    ctx = torch.randn(4, 5)
    q_values, hidden = net(seq, ctx)

    check(
        q_values.shape == (4, 64),
        f"Q-values shape is (4, 64), got {q_values.shape}"
    )
    check(
        hidden[0].shape == (2, 4, 256),
        f"Hidden h shape is (2, 4, 256), got {hidden[0].shape}"
    )
    check(
        hidden[1].shape == (2, 4, 256),
        f"Hidden c shape is (2, 4, 256), got {hidden[1].shape}"
    )
    check(
        torch.isfinite(q_values).all().item(),
        "All Q-values are finite"
    )

    # Inference mode: single step
    seq1 = torch.randn(1, 1, 5)
    ctx1 = torch.randn(1, 5)
    q1, h1 = net(seq1, ctx1)

    action = q1.argmax(dim=-1).item()
    check(0 <= action < 64, f"Action {action} is in [0, 63]")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: LSTM HIDDEN STATE PERSISTENCE & RESET
# ─────────────────────────────────────────────────────────────────────────────

def test_lstm_hidden_state():
    test("Test 2: LSTM Hidden State Persistence & Reset")

    net = DRQNNetwork(n_actions=64)

    # Step 1: initial hidden
    hidden = net.init_hidden(batch_size=1)
    check(
        hidden[0].abs().sum().item() == 0.0,
        "Initial hidden state is all zeros"
    )

    # Step 2: run a forward pass — hidden should change
    seq = torch.randn(1, 5, 5)
    ctx = torch.randn(1, 5)
    _, hidden_after = net(seq, ctx, hidden)

    h_changed = hidden_after[0].abs().sum().item() > 0.0
    check(h_changed, "Hidden state changed after forward pass (LSTM is remembering)")

    # Step 3: reset hidden — should be zeros again
    hidden_reset = net.init_hidden(batch_size=1)
    check(
        hidden_reset[0].abs().sum().item() == 0.0,
        "Hidden state is zeros after reset"
    )

    # Step 4: detach should not change values, only detach from graph
    hidden_detached = net.detach_hidden(hidden_after)
    check(
        torch.allclose(hidden_after[0].data, hidden_detached[0].data),
        "Detached hidden has same values"
    )
    check(
        not hidden_detached[0].requires_grad,
        "Detached hidden does not require grad"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_buffer():
    test("Test 3: Recurrent Replay Buffer")

    buffer = RecurrentReplayBuffer(
        capacity=50, chunk_len=8, window_size=10, context_dim=5,
    )

    # Add 3 episodes of 25 steps each
    for ep in range(3):
        for step in range(25):
            buffer.add_transition(
                obs_seq=np.random.randn(10, 5).astype(np.float32),
                context=np.random.randn(5).astype(np.float32),
                action=np.random.randint(0, 64),
                reward=np.random.randn(),
                done=(step == 24),
            )
        buffer.end_episode()

    check(len(buffer) == 3, f"Buffer has 3 episodes, got {len(buffer)}")
    check(buffer.can_sample(2), "Can sample batch of 2")

    # Sample a batch
    batch = buffer.sample(batch_size=2, device="cpu")

    check(
        batch["obs_seq"].shape == (2, 8, 10, 5),
        f"obs_seq shape is (2, 8, 10, 5), got {batch['obs_seq'].shape}"
    )
    check(
        batch["context"].shape == (2, 8, 5),
        f"context shape is (2, 8, 5), got {batch['context'].shape}"
    )
    check(
        batch["actions"].shape == (2, 8),
        f"actions shape is (2, 8), got {batch['actions'].shape}"
    )
    check(
        batch["rewards"].shape == (2, 8),
        f"rewards shape is (2, 8), got {batch['rewards'].shape}"
    )
    check(
        batch["dones"].shape == (2, 8),
        f"dones shape is (2, 8), got {batch['dones'].shape}"
    )

    # Verify action range
    check(
        batch["actions"].min().item() >= 0 and batch["actions"].max().item() < 64,
        "All sampled actions in [0, 63]"
    )

    # Short episode should be discarded
    n_before = len(buffer)
    for s in range(3):  # Only 3 steps < chunk_len=8
        buffer.add_transition(
            np.zeros((10, 5), dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            0, 0.0, (s == 2),
        )
    buffer.end_episode()
    check(
        len(buffer) == n_before,
        "Short episode (3 steps < chunk_len=8) correctly discarded"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: DRQN AGENT — TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_training():
    test("Test 4: DRQNAgent Training Step")

    agent = DRQNAgent(n_actions=64, device="cpu")

    # Create a fake batch
    B, T = 4, 8
    fake_batch = {
        "obs_seq": torch.randn(B, T, 10, 5),
        "context": torch.randn(B, T, 5),
        "actions": torch.randint(0, 64, (B, T)),
        "rewards": torch.randn(B, T),
        "dones": torch.zeros(B, T, dtype=torch.bool),
    }

    # Run one training step
    loss = agent.train_step(fake_batch)

    check(np.isfinite(loss), f"Loss is finite: {loss:.4f}")
    check(loss >= 0.0, f"Loss is non-negative: {loss:.4f}")

    # Verify gradients exist
    has_grads = all(
        p.grad is not None
        for p in agent.q_network.parameters()
        if p.requires_grad
    )
    check(has_grads, "All Q-network parameters have gradients")

    # Verify target network was NOT updated (only syncs every 500 steps)
    main_p = list(agent.q_network.parameters())[0].data.flatten()[:3]
    target_p = list(agent.target_network.parameters())[0].data.flatten()[:3]
    differ = not torch.allclose(main_p, target_p, atol=1e-6)
    check(differ, "Main and target networks differ after 1 training step")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: STATE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def test_state_builder():
    test("Test 5: StateBuilder Normalisation")

    builder = StateBuilder(n_channels=64, max_steps=500)
    builder.reset()

    # Simulate a realistic observation
    obs = np.zeros((10, 5), dtype=np.float32)
    obs[7] = [100.0, 5000.0, 2.5, 45.0, -30.0]
    obs[8] = [150.0, 5000.0, 2.5, 46.0, -31.0]
    obs[9] = [200.0, 9000.0, 3.0, 90.0, -25.0]

    info = {
        "intercepted": True,
        "chosen_channel": 17,
        "pulse_channel": 17,
        "step": 1,
    }

    state = builder.build_state(obs, info)

    check(
        state["sequence"].shape == (10, 5),
        f"Sequence shape is (10, 5), got {state['sequence'].shape}"
    )
    check(
        state["context"].shape == (5,),
        f"Context shape is (5,), got {state['context'].shape}"
    )
    check(
        state["sequence"].min() >= 0.0,
        f"Sequence min >= 0: {state['sequence'].min():.4f}"
    )
    check(
        state["sequence"].max() <= 1.0,
        f"Sequence max <= 1: {state['sequence'].max():.4f}"
    )
    check(
        state["context"].min() >= 0.0,
        f"Context min >= 0: {state['context'].min():.4f}"
    )
    check(
        state["context"].max() <= 1.0,
        f"Context max <= 1: {state['context'].max():.4f}"
    )

    # Zero-padded rows should remain zero
    check(
        state["sequence"][:7].abs().sum().item() == 0.0,
        "Zero-padded rows remain zero after normalisation"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: HANDOFF CONTROLLER — PATTERN LOCK
# ─────────────────────────────────────────────────────────────────────────────

def test_pattern_lock():
    test("Test 6: HandoffController — Pattern-Lock Trigger")

    ctrl = HandoffController(
        pattern_lock_threshold=6,
        fallback_threshold=5,
        cooldown_steps=0,
    )
    ctrl.reset()

    # Feed a cyclic pattern: 12 → 24 → 36 → repeat
    pattern = [12, 24, 36]
    lock_triggered = False

    for step in range(30):
        ch = pattern[step % 3]
        result = ctrl.update(intercepted=True, channel=ch)

        if result["event"] == "HANDOFF_TO_TIER_2":
            lock_triggered = True
            lock_step = step
            break

    check(lock_triggered, f"Pattern-lock triggered on cyclic sequence")
    check(ctrl.current_tier == 2, f"Current tier is 2 after lock")
    check(
        ctrl.detected_period == 3,
        f"Detected period is 3, got {ctrl.detected_period}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: HANDOFF CONTROLLER — FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback():
    test("Test 7: HandoffController — Fallback Trigger")

    ctrl = HandoffController(
        pattern_lock_threshold=3,
        fallback_threshold=5,
        cooldown_steps=0,
    )
    ctrl.reset()

    # Force into Tier 2 first
    for step in range(15):
        ctrl.update(intercepted=True, channel=[10, 20, 30][step % 3])

    check(ctrl.current_tier == 2, "In Tier 2 after pattern lock")

    # Now simulate 5 consecutive misses
    fallback_triggered = False
    for step in range(10):
        result = ctrl.update(intercepted=False, channel=None)
        if result["event"] == "REVERT_TO_TIER_1":
            fallback_triggered = True
            break

    check(fallback_triggered, "Fallback triggered after consecutive misses")
    check(ctrl.current_tier == 1, "Reverted to Tier 1")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: NO FALSE PATTERN LOCK ON RANDOM DATA
# ─────────────────────────────────────────────────────────────────────────────

def test_no_false_lock():
    test("Test 8: No False Pattern-Lock on Random Channels")

    import random
    random.seed(42)

    ctrl = HandoffController(
        pattern_lock_threshold=6,
        cooldown_steps=0,
    )
    ctrl.reset()

    false_lock = False
    for step in range(500):
        ch = random.randint(0, 63)
        result = ctrl.update(intercepted=True, channel=ch)
        if result["event"] == "HANDOFF_TO_TIER_2":
            false_lock = True
            break

    check(not false_lock, "No false pattern-lock on 500 random steps")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: CHECKPOINT SAVE / LOAD ROUNDTRIP
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_roundtrip():
    test("Test 9: Checkpoint Save/Load Roundtrip")

    agent1 = DRQNAgent(n_actions=64, device="cpu")

    # Do a few training steps to modify weights
    for _ in range(3):
        fake_batch = {
            "obs_seq": torch.randn(2, 8, 10, 5),
            "context": torch.randn(2, 8, 5),
            "actions": torch.randint(0, 64, (2, 8)),
            "rewards": torch.randn(2, 8),
            "dones": torch.zeros(2, 8, dtype=torch.bool),
        }
        agent1.train_step(fake_batch)

    # Save checkpoint
    tmp_path = os.path.join(tempfile.gettempdir(), "test_p3_ckpt.pt")
    agent1.save(tmp_path, eval_reward=0.75)

    check(os.path.exists(tmp_path), f"Checkpoint file created")

    # Load into a fresh agent
    agent2 = DRQNAgent(n_actions=64, device="cpu")
    ckpt = agent2.load(tmp_path)

    # Verify weights match
    weights_match = all(
        torch.allclose(p1, p2, atol=1e-6)
        for p1, p2 in zip(
            agent1.q_network.parameters(),
            agent2.q_network.parameters(),
        )
    )
    check(weights_match, "Loaded weights match saved weights")

    # Verify same action for same input
    test_state = {
        "sequence": torch.randn(10, 5),
        "context": torch.randn(5),
    }
    torch.manual_seed(42)
    a1, _ = agent1.select_action(test_state, evaluate=True)
    torch.manual_seed(42)
    a2, _ = agent2.select_action(test_state, evaluate=True)

    check(a1 == a2, f"Same action from saved and loaded agents: {a1} == {a2}")

    # Verify training state restored
    check(
        ckpt["training_state"]["best_eval_reward"] == 0.75,
        "Best eval reward restored correctly"
    )

    # Clean up
    os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: EPSILON DECAY SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def test_epsilon_decay():
    test("Test 10: Epsilon Decay Schedule")

    agent = DRQNAgent(
        n_actions=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=1000,
    )

    eps_at_0 = agent.epsilon
    check(abs(eps_at_0 - 1.0) < 1e-6, f"ε at step 0 = {eps_at_0:.4f} (expected 1.0)")

    # Advance to midpoint
    for _ in range(500):
        agent.increment_step()
    eps_at_500 = agent.epsilon
    expected_mid = 1.0 + 0.5 * (0.05 - 1.0)  # 0.525
    check(
        abs(eps_at_500 - expected_mid) < 0.01,
        f"ε at step 500 = {eps_at_500:.4f} (expected ~{expected_mid:.3f})"
    )

    # Advance past decay
    for _ in range(600):
        agent.increment_step()
    eps_at_end = agent.epsilon
    check(
        abs(eps_at_end - 0.05) < 1e-6,
        f"ε at step 1100 = {eps_at_end:.4f} (expected 0.05, clamped)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: FULL MINI-EPISODE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def test_full_integration():
    test("Test 11: Full Mini-Episode Integration (synthetic)")

    agent = DRQNAgent(n_actions=64, device="cpu", epsilon_start=0.5)
    state_builder = StateBuilder(n_channels=64, max_steps=50)
    buffer = RecurrentReplayBuffer(capacity=10, chunk_len=8)
    controller = HandoffController(cooldown_steps=0)

    # Simulate a 30-step episode with fake observations
    state_builder.reset()
    controller.reset()
    hidden = agent.q_network.init_hidden(1)

    all_actions = []
    all_rewards = []

    for step in range(30):
        # Create fake observation
        obs = np.random.randn(10, 5).astype(np.float32)
        obs[:, 0] = np.sort(np.abs(obs[:, 0])) * 100  # ToA increasing
        obs[:, 1] = np.abs(obs[:, 1]) * 5000           # Frequency positive
        obs[:, 3] = np.abs(obs[:, 3]) * 360             # AoA positive
        obs[:, 4] = -np.abs(obs[:, 4]) * 50             # Amplitude negative

        fake_info = {
            "intercepted": np.random.random() > 0.5,
            "chosen_channel": np.random.randint(0, 64),
            "pulse_channel": np.random.randint(0, 64),
            "step": step,
        }

        # Build state
        state = state_builder.build_state(obs, fake_info)

        # Select action
        action, hidden = agent.select_action(state, hidden)
        all_actions.append(action)

        # Fake reward
        reward = 1.0 if fake_info["intercepted"] else -0.5
        all_rewards.append(reward)
        done = (step == 29)

        # Store in buffer
        buffer.add_transition(
            state["sequence"].numpy(),
            state["context"].numpy(),
            action, reward, done,
        )

        # Update handoff controller
        controller.update(
            intercepted=fake_info["intercepted"],
            channel=fake_info["pulse_channel"] if fake_info["intercepted"] else None,
        )

        agent.increment_step()

    buffer.end_episode()

    check(len(all_actions) == 30, f"Completed 30 steps, got {len(all_actions)}")
    check(
        all(0 <= a < 64 for a in all_actions),
        "All actions in valid range [0, 63]"
    )
    check(len(buffer) == 1, f"1 episode in buffer, got {len(buffer)}")

    # Try training if we have enough data (need chunk_len=8, ep has 30 steps)
    if buffer.can_sample(1):
        batch = buffer.sample(1, device="cpu")
        loss = agent.train_step(batch)
        check(np.isfinite(loss), f"Training loss is finite: {loss:.4f}")
    else:
        check(False, "Buffer should be sampleable after 30-step episode")

    check(True, "Full integration test completed without crashes")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: PARAMETER COUNT SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def test_parameter_count():
    test("Test 12: Parameter Count Sanity Check")

    net = DRQNNetwork(
        input_dim=5, context_dim=5, feature_dim=128,
        hidden_dim=256, n_layers=2, n_actions=64,
    )

    n_params = net.count_parameters()

    # Rough expected range:
    # Feature: 5*128 + 128 = 768
    # LSTM: ~2 * 4 * (128+256) * 256 ≈ 787,456
    # Value: 261*64 + 64 + 64*1 + 1 ≈ 16,833
    # Advantage: 261*64 + 64 + 64*64 + 64 ≈ 20,928
    # Total: ~826,000 (rough estimate)

    check(
        100_000 < n_params < 2_000_000,
        f"Parameter count {n_params:,} is in reasonable range (100K–2M)"
    )
    print(f"   Exact parameter count: {n_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PERSON 3 — FULL DRQN PIPELINE VALIDATION")
    print("=" * 60)

    # Run all tests
    test_drqn_forward_pass()
    test_lstm_hidden_state()
    test_replay_buffer()
    test_agent_training()
    test_state_builder()
    test_pattern_lock()
    test_fallback()
    test_no_false_lock()
    test_checkpoint_roundtrip()
    test_epsilon_decay()
    test_full_integration()
    test_parameter_count()

    # Final verdict
    print(f"\n{'=' * 60}")
    total = PASS_COUNT + FAIL_COUNT
    print(f"RESULTS: {PASS_COUNT}/{total} checks passed")

    if FAIL_COUNT == 0:
        print("✅ ALL TESTS PASSED — Person 3 pipeline is ready!")
    else:
        print(f"❌ {FAIL_COUNT} CHECK(S) FAILED — review output above")
    print(f"{'=' * 60}\n")

    sys.exit(0 if FAIL_COUNT == 0 else 1)
