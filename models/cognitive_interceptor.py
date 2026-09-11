"""
models/cognitive_interceptor.py — Unified Cognitive Interceptor (Person 4 Handoff API)
========================================================================================

Step 13 Specification:
This module delivers the clean, production-grade interface class `CognitiveInterceptor`
for Person 4 (Systems Engineer) to integrate into real-time SDR receiver software.

It seamlessly combines:
  - DRQNAgent (Tier 2 deep recurrent sequence intelligence)
  - StateBuilder (robust normalisation and context feature engineering)
  - HandoffController (automated pattern-lock detection and fallback triggers)

Usage:
    from models.cognitive_interceptor import CognitiveInterceptor

    # Initialize with trained model weights
    interceptor = CognitiveInterceptor(weights_path="checkpoints/drqn_radar_best.pt")

    # In SDR receiver loop:
    # 1. Get observation window (10, 5) from SDR buffer
    channel = interceptor.predict_channel(obs_window, last_step_info)

    # 2. Tune receiver to predicted channel
    tune_sdr_hardware(channel)

    # 3. Report feedback after dwell slot
    interceptor.update_feedback(reward=+1.0, intercepted=True, pulse_channel=channel)

    # 4. Check telemetry status
    print(f"Tier: {interceptor.current_tier} | Locked: {interceptor.pattern_locked}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.drqn_agent import DRQNAgent
from models.handoff_controller import HandoffController
from models.state_builder import StateBuilder


class CognitiveInterceptor:
    """
    Unified high-level cognitive radar interceptor for production deployment.

    Parameters
    ----------
    weights_path : str or Path or None
        Path to trained PyTorch checkpoint (.pt). If None or file doesn't exist,
        agent runs with randomly initialized weights (for testing).
    n_channels : int
        Number of discrete receiver frequency channels (default: 64).
    max_steps : int
        Maximum steps per episode / operational mission horizon (default: 500).
    fallback_bandit : Any or None
        Optional Tier 1 bandit policy instance with a select_action() method.
        If None, Tier 1 falls back to random channel sampling.
    device : str or torch.device
        Compute device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        n_channels: int = 64,
        max_steps: int = 500,
        fallback_bandit: Any | None = None,
        device: str | torch.device = "cpu",
    ):
        self.n_channels = n_channels
        self.max_steps = max_steps
        self.fallback_bandit = fallback_bandit
        self.device = torch.device(device)

        # 1. State Builder
        self.state_builder = StateBuilder(
            n_channels=n_channels,
            max_steps=max_steps,
        )

        # 2. Handoff Controller
        self.controller = HandoffController(
            pattern_lock_threshold=6,
            fallback_threshold=5,
            cooldown_steps=10,
        )

        # 3. DRQN Agent
        self.agent = DRQNAgent(
            n_actions=n_channels,
            device=self.device,
        )
        self.agent.set_eval_mode()

        self.weights_loaded = False
        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                self.agent.load(p, load_optimiser=False)
                self.agent.set_eval_mode()
                self.weights_loaded = True

        self._last_action: int = 0
        self._step_counter: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # CORE INTERFACE METHODS (STEP 13 SPECIFICATION)
    # ──────────────────────────────────────────────────────────────────────────

    def predict_channel(
        self,
        obs: np.ndarray,
        info: dict | None = None,
    ) -> int:
        """
        Predict the best frequency channel for the next pulse arrival.

        Parameters
        ----------
        obs : np.ndarray, shape (10, 5)
            Raw sliding window of the last 10 PDW pulses from SDR buffer.
            Columns: [ToA, Frequency, PulseWidth, AoA, Amplitude].
        info : dict or None
            Operational metadata from the environment/hardware.

        Returns
        -------
        int : Channel index in [0, n_channels - 1] to tune receiver to.
        """
        if info is None:
            info = {
                "intercepted": False,
                "chosen_channel": self._last_action,
                "pulse_channel": self._last_action,
                "step": self._step_counter,
            }

        # Build normalised sequence + context features
        state = self.state_builder.build_state(obs, info)

        # Handoff routing: Tier 1 vs Tier 2
        if self.controller.current_tier == 2:
            # Tier 2: Deep recurrent Q-network sequence prediction
            action, _ = self.agent.select_action(state, hidden=None, evaluate=True)
        else:
            # Tier 1: Fast reactive fallback (Bandit or uniform exploration)
            if self.fallback_bandit is not None and hasattr(self.fallback_bandit, "select_action"):
                action = int(self.fallback_bandit.select_action())
            else:
                action = int(np.random.randint(0, self.n_channels))

        self._last_action = action
        self._step_counter += 1
        return action

    def update_feedback(
        self,
        reward: float,
        intercepted: bool,
        pulse_channel: int | None = None,
    ) -> dict[str, Any]:
        """
        Deliver feedback from the receiver hardware to update controllers.

        Parameters
        ----------
        reward : float
            Reward received from the environment.
        intercepted : bool
            True if the pulse was successfully intercepted.
        pulse_channel : int or None
            The channel on which the pulse occurred (if intercepted).

        Returns
        -------
        dict : Status event dict from HandoffController.
        """
        if pulse_channel is None and intercepted:
            pulse_channel = self._last_action

        # Inform bandit if available
        if self.fallback_bandit is not None and hasattr(self.fallback_bandit, "update"):
            try:
                self.fallback_bandit.update(self._last_action, reward)
            except Exception:
                pass

        # Update Tier 1 <-> Tier 2 handoff triggers
        event_dict = self.controller.update(
            intercepted=intercepted,
            channel=pulse_channel if intercepted else None,
        )
        return event_dict

    def reset(self) -> None:
        """Reset interceptor state for a new mission or episode."""
        self.state_builder.reset()
        self.controller.reset()
        self._last_action = 0
        self._step_counter = 0

    # ──────────────────────────────────────────────────────────────────────────
    # STATUS & TELEMETRY INDICATORS
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def current_tier(self) -> int:
        """Currently active operational tier (1 = Bandit, 2 = DRQN)."""
        return self.controller.current_tier

    @property
    def pattern_locked(self) -> bool:
        """True if Tier 2 has locked onto a cyclical radar pattern."""
        return self.controller.pattern_locked

    @property
    def consecutive_misses(self) -> int:
        """Number of consecutive misses since last successful intercept."""
        return self.controller.consecutive_misses

    @property
    def detected_period(self) -> int | None:
        """Detected hopping cycle period (P in [3, 8]) if locked, else None."""
        return self.controller.detected_period
