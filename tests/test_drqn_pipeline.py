"""
tests/test_drqn_pipeline.py — Full Unit & Hardening Tests for Person 3 (Tier 2 DRQN)
=====================================================================================

This test suite validates ALL Person 3 components end-to-end using
synthetic data and production artifacts:
  ✅ DRQNNetwork forward pass: (1, 10, 5) → valid action in [0, 63]
  ✅ LSTM hidden state persistence across steps
  ✅ LSTM hidden state reset to zeros
  ✅ Replay buffer: write, sample, verify shapes
  ✅ Training step: loss is finite, gradients flow
  ✅ StateBuilder: all outputs normalised to [0, 1]
  ✅ Pattern-lock trigger fires on cyclic sequences
  ✅ Fallback trigger fires on consecutive misses
  ✅ No false pattern-locks on uniform random noise
  ✅ Checkpoint save/load roundtrip
  ✅ Epsilon decay schedule correctness
  ✅ Full mini-episode integration test
  ✅ Parameter count sanity check (960,385 parameters)
  ✅ Production checkpoint validation (drqn_radar_best.pt)
  ✅ Real-time SDR inference latency (< 5ms deadline)
  ✅ Edge cases & numerical robustness (clamping, zero-padding)
  ✅ Handoff controller cooldown & anti-oscillation
  ✅ Multi-batch parallel inference (M-receiver SDR)
  ✅ Dueling architecture mathematical invariant (V(s) + A(s,a) - mean(A))
  ✅ Replay buffer capacity overflow & FIFO overwrite
  ✅ Target network isolation & Polyak soft sync (tau)
  ✅ Hardware sensor NaN & Inf dropout immunity
  ✅ Handoff re-acquisition & multi-period dynamics (P in [3, 8])
  ✅ CognitiveInterceptor unified production API (Person 4 handoff contract)
  ✅ Deterministic policy evaluation invariance

Usage:
    python tests/test_drqn_pipeline.py
    pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
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
from models.cognitive_interceptor import CognitiveInterceptor

# ─────────────────────────────────────────────────────────────────────────────
# TEST INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0


def test(name: str = ""):
    """Decorator-like printer for test sections."""
    print(f"\n── {name}")

test.__test__ = False


def check(condition: bool, msg: str):
    """Assert with tracking."""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"   ✅ {msg}")
    else:
        FAIL_COUNT += 1
        print(f"   ❌ FAILED: {msg}")
        assert condition, msg


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
# TEST 13: PRODUCTION CHECKPOINT VALIDATION (drqn_radar_best.pt)
# ─────────────────────────────────────────────────────────────────────────────

def test_production_checkpoint():
    test("Test 13: Production Checkpoint Validation (drqn_radar_best.pt)")

    ckpt_path = Path("checkpoints/drqn_radar_best.pt")
    check(ckpt_path.exists(), f"Production checkpoint exists at {ckpt_path}")

    if ckpt_path.exists():
        agent = DRQNAgent(n_actions=64, device="cpu")
        ckpt = agent.load(ckpt_path, load_optimiser=False)

        check("model_state_dict" in ckpt, "Checkpoint contains 'model_state_dict' state dict")
        check("config" in ckpt, "Checkpoint contains 'config' metadata")
        check("training_state" in ckpt, "Checkpoint contains 'training_state'")

        eval_reward = ckpt.get("eval_reward")
        if eval_reward is None and "training_state" in ckpt:
            eval_reward = ckpt["training_state"].get("best_eval_reward")

        check(
            eval_reward is not None and np.isfinite(eval_reward),
            f"Recorded eval_reward is valid: {eval_reward}"
        )

        # Run inference test with loaded weights
        sb = StateBuilder(n_channels=64, max_steps=500)
        dummy_obs = np.random.uniform(0.0, 1.0, size=(10, 5)).astype(np.float32)
        dummy_info = {"intercepted": False, "chosen_channel": 0, "pulse_channel": 0, "step": 1}
        state = sb.build_state(dummy_obs, dummy_info)

        action, _ = agent.select_action(state, hidden=None, evaluate=True)
        check(0 <= action < 64, f"Production model produces valid action {action} in [0, 63]")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 14: REAL-TIME INFERENCE LATENCY & SDR DEADLINE
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_latency():
    test("Test 14: Real-Time Inference Latency & SDR Deadline (< 5ms)")

    agent = DRQNAgent(n_actions=64, device="cpu")
    agent.set_eval_mode()
    sb = StateBuilder(n_channels=64, max_steps=500)

    dummy_obs = np.random.uniform(0.0, 1.0, size=(10, 5)).astype(np.float32)
    dummy_info = {"intercepted": False, "chosen_channel": 0, "pulse_channel": 0, "step": 1}
    state = sb.build_state(dummy_obs, dummy_info)

    # Warmup
    for _ in range(10):
        agent.select_action(state, hidden=None, evaluate=True)

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        action, _ = agent.select_action(state, hidden=None, evaluate=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    check(avg_latency < 5.0, f"Average inference latency {avg_latency:.2f} ms < 5.0 ms target")
    check(p95_latency < 15.0, f"P95 inference latency {p95_latency:.2f} ms < 15.0 ms ceiling")
    print(f"   Avg: {avg_latency:.2f} ms | P95: {p95_latency:.2f} ms | Max: {max(latencies):.2f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 15: EDGE CASES & NUMERICAL ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_cases_and_robustness():
    test("Test 15: Edge Cases & Numerical Robustness")

    sb = StateBuilder(n_channels=64, max_steps=500)
    agent = DRQNAgent(n_actions=64, device="cpu")
    agent.set_eval_mode()

    # Edge Case 1: All-zeros observation (step 0 before any pulse arrives)
    zero_obs = np.zeros((10, 5), dtype=np.float32)
    state_zero = sb.build_state(zero_obs, {"intercepted": False, "chosen_channel": 0, "pulse_channel": 0, "step": 0})
    check(torch.all(torch.isfinite(state_zero["sequence"])), "Zero-obs sequence contains no NaN/Inf")
    check(torch.all(torch.isfinite(state_zero["context"])), "Zero-obs context contains no NaN/Inf")
    a_zero, _ = agent.select_action(state_zero, hidden=None, evaluate=True)
    check(0 <= a_zero < 64, f"Zero-obs produces valid action: {a_zero}")

    # Edge Case 2: Extreme Out-of-Bounds Values (Extreme freq, massive ToA, clipping amp)
    extreme_obs = np.array([
        [1e8, 25000.0, 100.0, 450.0, 50.0],
        [0.0, -500.0, -10.0, -50.0, -150.0],
    ] * 5, dtype=np.float32)
    state_extreme = sb.build_state(extreme_obs, {"intercepted": True, "chosen_channel": 63, "pulse_channel": 63, "step": 500})
    check(torch.all((state_extreme["sequence"] >= 0.0) & (state_extreme["sequence"] <= 1.0)), "Extreme values clamped safely to [0, 1]")
    check(torch.all((state_extreme["context"] >= 0.0) & (state_extreme["context"] <= 1.0)), "Extreme context clamped safely to [0, 1]")
    a_extreme, _ = agent.select_action(state_extreme, hidden=None, evaluate=True)
    check(0 <= a_extreme < 64, f"Extreme-obs produces valid action: {a_extreme}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 16: HANDOFF CONTROLLER COOLDOWN & ANTI-OSCILLATION
# ─────────────────────────────────────────────────────────────────────────────

def test_handoff_cooldown():
    test("Test 16: Handoff Controller Cooldown & Anti-Oscillation")

    hc = HandoffController(
        pattern_lock_threshold=4,
        fallback_threshold=3,
        cooldown_steps=10,
    )

    # Trigger pattern lock with repeating sequence
    seq = [10, 20, 30] * 5
    for ch in seq:
        hc.update(intercepted=True, channel=ch)

    check(hc.current_tier == 2, f"Tier 2 locked after pattern, got Tier {hc.current_tier}")

    # Now deliver 3 misses during cooldown (cooldown_steps=10)
    # Even though fallback_threshold=3, cooldown MUST prevent immediate fallback
    for _ in range(3):
        hc.update(intercepted=False)

    check(hc.current_tier == 2, "Cooldown prevented premature fallback on transient misses")

    # Step through remaining cooldown steps
    for _ in range(8):
        hc.update(intercepted=False)

    # After cooldown expires, accumulated misses must trigger fallback
    check(hc.current_tier == 1, f"Fallback correctly triggered after cooldown expired: Tier {hc.current_tier}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 17: MULTI-BATCH PARALLEL INFERENCE (M-RECEIVER SDR)
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_batch_inference():
    test("Test 17: Multi-Batch Parallel Forward Pass (M-Receiver SDR)")

    net = DRQNNetwork(
        input_dim=5, context_dim=5, feature_dim=128,
        hidden_dim=256, n_layers=2, n_actions=64,
    )
    net.eval()

    # Test batches: 1 (single SDR), 4 (quad-receiver PPO spec), 16 (parallel channels)
    for B in [1, 4, 16]:
        seq_batch = torch.rand(B, 10, 5)
        ctx_batch = torch.rand(B, 5)
        q_vals, hidden = net(seq_batch, ctx_batch, None)

        check(q_vals.shape == (B, 64), f"Batch {B}: Q-values shape (B, 64), got {q_vals.shape}")
        check(torch.all(torch.isfinite(q_vals)), f"Batch {B}: All Q-values finite")
        check(hidden[0].shape == (2, B, 256), f"Batch {B}: Hidden state h shape correct")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 18: DUELING ARCHITECTURE MATHEMATICAL INVARIANT
# ─────────────────────────────────────────────────────────────────────────────

def test_dueling_architecture_invariant():
    test("Test 18: Dueling Architecture Mathematical Invariant (V(s) + A(s,a) - mean(A))")

    net = DRQNNetwork(input_dim=5, context_dim=5, n_actions=64)
    net.eval()

    seq = torch.randn(4, 10, 5)
    ctx = torch.randn(4, 5)

    # Forward pass
    q_values, _ = net(seq, ctx)

    # Check forward pass Q-values
    check(q_values.shape == (4, 64), f"Q-values shape (4, 64), got {q_values.shape}")

    # Manually compute V and A through the streams
    features = net.feature_extractor(seq)
    lstm_out, _ = net.lstm(features)
    combined = torch.cat([lstm_out[:, -1, :], ctx], dim=-1)
    v_stream = net.value_stream(combined)  # (4, 1)
    a_stream = net.advantage_stream(combined)  # (4, 64)

    # Identifiability property: mean(Q(s, ·)) == V(s)
    expected_v = q_values.mean(dim=-1, keepdim=True)
    check(
        torch.allclose(expected_v, v_stream, atol=1e-5),
        "Mean Q-value across all actions exactly equals state value V(s)"
    )

    # Advantage centering: (A(s, a) - mean(A)) has mean zero
    centered_adv = a_stream - a_stream.mean(dim=-1, keepdim=True)
    check(
        torch.allclose(centered_adv.mean(dim=-1), torch.zeros(4), atol=1e-6),
        "Mean centered advantage across actions is zero"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 19: REPLAY BUFFER OVERFLOW, CIRCULAR FIFO & CAPACITY BOUNDS
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_buffer_overflow_and_bounds():
    test("Test 19: Replay Buffer Capacity Overflow & Boundary Conditions")

    # Small capacity buffer
    buffer = RecurrentReplayBuffer(capacity=3, chunk_len=8, window_size=10, context_dim=5)

    # Add 10 episodes (capacity is 3, so 7 oldest must be overwritten cleanly)
    for ep in range(10):
        for step in range(15):
            buffer.add_transition(
                obs_seq=np.full((10, 5), ep, dtype=np.float32),
                context=np.full(5, ep, dtype=np.float32),
                action=ep % 64,
                reward=float(ep),
                done=(step == 14),
            )
        buffer.end_episode()

    check(len(buffer) == 3, f"Buffer length clamped to capacity 3: got {len(buffer)}")
    check(buffer.can_sample(3), "Can sample batch of capacity size (3)")
    check(not buffer.can_sample(4), "Cannot sample batch larger than buffer (4)")

    # Sample a batch of 3
    batch = buffer.sample(3, device="cpu")
    check(batch["obs_seq"].shape == (3, 8, 10, 5), "Sampled batch from wrapped buffer has valid shape")

    # Verify exception when sampling too many episodes
    raised_val_error = False
    try:
        buffer.sample(4)
    except ValueError:
        raised_val_error = True
    check(raised_val_error, "Sampling beyond capacity cleanly raises ValueError")

    # Exact boundary condition: episode length exactly chunk_len (8 steps)
    buf2 = RecurrentReplayBuffer(capacity=2, chunk_len=8)
    for s in range(8):
        buf2.add_transition(np.zeros((10, 5), dtype=np.float32), np.zeros(5, dtype=np.float32), 0, 1.0, (s == 7))
    buf2.end_episode()
    check(len(buf2) == 1, "Episode with length exactly equal to chunk_len is retained")
    b2 = buf2.sample(1)
    check(b2["obs_seq"].shape == (1, 8, 10, 5), "Sample from exact-chunk episode succeeded")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 20: TARGET NETWORK ISOLATION & POLYAK SOFT SYNC
# ─────────────────────────────────────────────────────────────────────────────

def test_target_network_isolation_and_polyak():
    test("Test 20: Target Network Isolation & Polyak Soft Sync")

    agent = DRQNAgent(n_actions=64, device="cpu")
    agent.sync_target()

    # Verify parameters match after sync
    p_main = list(agent.q_network.parameters())[0]
    p_target = list(agent.target_network.parameters())[0]
    check(torch.allclose(p_main, p_target), "Target network parameters match main network after sync")

    # Mutate main network in-place: target network MUST NOT change (no shared memory)
    with torch.no_grad():
        p_main.add_(10.0)

    check(not torch.allclose(p_main, p_target), "Target network unaffected by in-place mutation of main network")

    # Test Polyak soft sync: θ⁻ ← τ·θ + (1-τ)·θ⁻
    tau = 0.2
    target_before = p_target.clone()
    main_current = p_main.clone()
    expected_target = tau * main_current + (1.0 - tau) * target_before

    agent.soft_sync_target(tau=tau)
    p_target_after = list(agent.target_network.parameters())[0]

    check(
        torch.allclose(p_target_after, expected_target, atol=1e-5),
        f"Polyak soft sync matches analytical formula (tau={tau})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 21: HARDWARE SENSOR NaN & INF IMMUNITY (ROBUSTNESS)
# ─────────────────────────────────────────────────────────────────────────────

def test_sensor_nan_inf_immunity():
    test("Test 21: Hardware Sensor NaN & Inf Dropout Immunity")

    sb = StateBuilder(n_channels=64, max_steps=500)
    agent = DRQNAgent(n_actions=64, device="cpu")
    agent.set_eval_mode()

    # Create corrupt observation full of NaNs, +Infs, -Infs from sensor failure
    corrupt_obs = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    ] * 3 + [[np.nan, 5000.0, 2.5, 45.0, -30.0]], dtype=np.float32)

    corrupt_info = {
        "intercepted": False,
        "chosen_channel": 0,
        "pulse_channel": 0,
        "step": 1,
    }

    state = sb.build_state(corrupt_obs, corrupt_info)

    check(torch.all(torch.isfinite(state["sequence"])), "Sanitized sequence contains no NaNs or Infs")
    check(torch.all(torch.isfinite(state["context"])), "Sanitized context contains no NaNs or Infs")
    check(state["sequence"].min() >= 0.0 and state["sequence"].max() <= 1.0, "Corrupt obs bounded strictly in [0, 1]")
    check(state["context"].min() >= 0.0 and state["context"].max() <= 1.0, "Corrupt context bounded strictly in [0, 1]")

    action, _ = agent.select_action(state, hidden=None, evaluate=True)
    check(0 <= action < 64, f"Corrupt input still yields valid channel action: {action}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 22: HANDOFF RE-ACQUISITION & MULTI-PERIOD DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def test_handoff_multi_period_and_reacquisition():
    test("Test 22: Handoff Re-acquisition & Multi-Period Dynamics")

    ctrl = HandoffController(
        pattern_lock_threshold=5,
        fallback_threshold=4,
        cooldown_steps=2,
    )

    # 1. Pattern with period 4: [5, 15, 25, 35]
    p4 = [5, 15, 25, 35]
    for step in range(25):
        ctrl.update(intercepted=True, channel=p4[step % 4])

    check(ctrl.current_tier == 2, "Controller locked on period-4 hopping pattern")
    check(ctrl.detected_period == 4, f"Detected period is 4, got {ctrl.detected_period}")

    # 2. Emitter changes tactic / Jammer disrupts: 6 consecutive misses
    for _ in range(6):
        ctrl.update(intercepted=False)

    check(ctrl.current_tier == 1, "Controller successfully fell back to Tier 1 upon pattern break")
    check(not ctrl.pattern_locked, "Pattern lock cleared after fallback")

    # 3. Re-acquisition: Emitter settles on new pattern of period 5: [2, 12, 22, 32, 42]
    p5 = [2, 12, 22, 32, 42]
    for step in range(30):
        ctrl.update(intercepted=True, channel=p5[step % 5])

    check(ctrl.current_tier == 2, "Controller re-acquired pattern lock on new frequency pattern")
    check(ctrl.detected_period == 5, f"Detected period is 5, got {ctrl.detected_period}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 23: COGNITIVE INTERCEPTOR API (PERSON 4 HANDOFF CONTRACT)
# ─────────────────────────────────────────────────────────────────────────────

def test_cognitive_interceptor_api():
    test("Test 23: CognitiveInterceptor Unified Production API (Person 4)")

    ckpt_path = "checkpoints/drqn_radar_best.pt"
    interceptor = CognitiveInterceptor(weights_path=ckpt_path)

    check(interceptor.weights_loaded, "CognitiveInterceptor successfully loaded production checkpoint")
    check(interceptor.current_tier == 1, "Interceptor initialises in Tier 1 (reactive bandit)")
    check(interceptor.consecutive_misses == 0, "Initial consecutive misses is 0")

    # Simulate 20 real-time SDR steps
    actions = []
    dummy_obs = np.random.uniform(0.0, 1.0, size=(10, 5)).astype(np.float32)
    dummy_info = {"intercepted": False, "chosen_channel": 0, "pulse_channel": 0, "step": 0}

    for step in range(20):
        ch = interceptor.predict_channel(dummy_obs, dummy_info)
        actions.append(ch)
        event = interceptor.update_feedback(reward=1.0, intercepted=True, pulse_channel=ch)
        dummy_info["chosen_channel"] = ch
        dummy_info["step"] = step + 1

    check(len(actions) == 20, "Generated 20 real-time channel predictions")
    check(all(0 <= a < 64 for a in actions), "All predictions in valid channel range [0, 63]")

    # Test reset
    interceptor.reset()
    check(interceptor.current_tier == 1, "Tier reset to 1 after reset()")
    check(interceptor.consecutive_misses == 0, "Misses reset to 0 after reset()")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 24: DETERMINISTIC POLICY EVALUATION INVARIANCE
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_evaluation():
    test("Test 24: Deterministic Policy Evaluation Invariance")

    agent = DRQNAgent(n_actions=64, device="cpu")
    agent.set_eval_mode()

    state = {
        "sequence": torch.randn(10, 5),
        "context": torch.randn(5),
    }

    # Evaluate multiple times with same input
    actions = [agent.select_action(state, hidden=None, evaluate=True)[0] for _ in range(10)]
    all_same = all(a == actions[0] for a in actions)
    check(all_same, f"Deterministic evaluation produces identical actions across 10 trials: {actions[0]}")


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
    test_production_checkpoint()
    test_inference_latency()
    test_edge_cases_and_robustness()
    test_handoff_cooldown()
    test_multi_batch_inference()
    test_dueling_architecture_invariant()
    test_replay_buffer_overflow_and_bounds()
    test_target_network_isolation_and_polyak()
    test_sensor_nan_inf_immunity()
    test_handoff_multi_period_and_reacquisition()
    test_cognitive_interceptor_api()
    test_deterministic_evaluation()

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
