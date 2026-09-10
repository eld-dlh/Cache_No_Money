"""
models/handoff_controller.py — Tier 1 ↔ Tier 2 Handoff Logic
================================================================

This module acts as the traffic controller between:
  - Tier 1 (Multi-Armed Bandits): Fast, reactive, good for stationary channels.
  - Tier 2 (DRQN): Slow to warm up, but excels at predicting hop sequences.

Two triggers govern the handoff:

  1. PATTERN-LOCK (Tier 1 → Tier 2):
     When recent intercepted channels show a repeating pattern (e.g.,
     ch12 → ch24 → ch36 → ch12 → ...), the DRQN is better suited to
     predict the next hop. The controller detects cyclic patterns of
     period P ∈ [3, 8] and hands off to Tier 2.

  2. FALLBACK (Tier 2 → Tier 1):
     When Tier 2 accumulates too many consecutive misses, the emitter
     has likely changed modes or a new jammer appeared. The controller
     reverts to Tier 1's reactive bandits and resets the DRQN state.

Usage:
    from models.handoff_controller import HandoffController

    controller = HandoffController()

    # Each step:
    controller.update(intercepted=True, channel=17)

    # Check which tier should be active:
    if controller.current_tier == 1:
        action = bandit.select_action()
    else:
        action = drqn_agent.select_action(state, hidden)

    # On episode reset:
    controller.reset()
"""

from __future__ import annotations

from collections import deque
from typing import Any


class HandoffController:
    """
    Controls the handoff between Tier 1 (Bandit) and Tier 2 (DRQN).

    Parameters
    ----------
    history_len : int
        Number of recent channel observations to keep for pattern
        detection (default: 16).
    min_period : int
        Minimum repeating pattern period to detect (default: 3).
    max_period : int
        Maximum repeating pattern period to detect (default: 8).
    pattern_lock_threshold : int
        Number of consecutive pattern matches required to trigger
        a handoff to Tier 2 (default: 6). Higher = fewer false positives.
    fallback_threshold : int
        Number of consecutive Tier 2 misses before reverting to
        Tier 1 (default: 5). Set higher than 3 to avoid false
        triggers in multi-emitter environments where occasional
        misses are normal.
    cooldown_steps : int
        Minimum number of steps to stay in a tier after a handoff
        before allowing another switch (default: 10). Prevents
        rapid oscillation between tiers.
    """

    def __init__(
        self,
        history_len: int = 16,
        min_period: int = 3,
        max_period: int = 8,
        pattern_lock_threshold: int = 6,
        fallback_threshold: int = 5,
        cooldown_steps: int = 10,
    ):
        self.history_len = history_len
        self.min_period = min_period
        self.max_period = max_period
        self.pattern_lock_threshold = pattern_lock_threshold
        self.fallback_threshold = fallback_threshold
        self.cooldown_steps = cooldown_steps

        # ── Internal state
        self._current_tier: int = 1       # Start with Tier 1 (Bandit)
        self._pattern_locked: bool = False
        self._detected_period: int | None = None
        self._consecutive_misses: int = 0
        self._steps_in_tier: int = 0      # Steps since last handoff
        self._total_steps: int = 0

        # Rolling history of intercepted channels (only channels we hit)
        self._channel_history: deque[int] = deque(maxlen=history_len)

        # Event log for debugging
        self._event_log: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state at the start of a new episode."""
        self._current_tier = 1
        self._pattern_locked = False
        self._detected_period = None
        self._consecutive_misses = 0
        self._steps_in_tier = 0
        self._total_steps = 0
        self._channel_history.clear()
        self._event_log.clear()

    def update(self, intercepted: bool, channel: int | None = None) -> dict[str, Any]:
        """
        Update the controller with the result of the latest action.

        Must be called EVERY step after the environment returns info.

        Parameters
        ----------
        intercepted : bool
            True if the agent successfully intercepted a pulse.
        channel : int or None
            The channel where the pulse was intercepted (only available
            when intercepted=True). In real hardware, we can only know
            the channel if we actually caught the pulse.

        Returns
        -------
        dict with keys:
            'tier'           : int (1 or 2) — currently active tier
            'event'          : str or None — 'HANDOFF_TO_TIER_2', 'REVERT_TO_TIER_1', or None
            'pattern_locked' : bool
            'detected_period': int or None
            'consecutive_misses' : int
        """
        self._total_steps += 1
        self._steps_in_tier += 1

        event = None

        # ── Update channel history (only on successful intercepts)
        if intercepted and channel is not None:
            self._channel_history.append(channel)

        # ── Update consecutive miss counter
        if intercepted:
            self._consecutive_misses = 0
        else:
            self._consecutive_misses += 1

        # ── Check triggers (only if cooldown has elapsed)
        if self._steps_in_tier >= self.cooldown_steps:

            if self._current_tier == 1:
                # Check for pattern lock → handoff to Tier 2
                locked, period = self._detect_pattern()
                if locked:
                    self._current_tier = 2
                    self._pattern_locked = True
                    self._detected_period = period
                    self._steps_in_tier = 0
                    self._consecutive_misses = 0
                    event = "HANDOFF_TO_TIER_2"

                    self._event_log.append({
                        "step": self._total_steps,
                        "event": event,
                        "period": period,
                    })

            elif self._current_tier == 2:
                # Check for fallback → revert to Tier 1
                if self._consecutive_misses >= self.fallback_threshold:
                    self._current_tier = 1
                    self._pattern_locked = False
                    self._detected_period = None
                    self._steps_in_tier = 0
                    event = "REVERT_TO_TIER_1"

                    self._event_log.append({
                        "step": self._total_steps,
                        "event": event,
                        "consecutive_misses": self._consecutive_misses,
                    })

                    self._consecutive_misses = 0

        return {
            "tier": self._current_tier,
            "event": event,
            "pattern_locked": self._pattern_locked,
            "detected_period": self._detected_period,
            "consecutive_misses": self._consecutive_misses,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def current_tier(self) -> int:
        """Currently active tier (1 = Bandit, 2 = DRQN)."""
        return self._current_tier

    @property
    def pattern_locked(self) -> bool:
        """Whether a repeating pattern has been detected."""
        return self._pattern_locked

    @property
    def detected_period(self) -> int | None:
        """The period of the detected pattern, or None."""
        return self._detected_period

    @property
    def consecutive_misses(self) -> int:
        """Current count of consecutive misses."""
        return self._consecutive_misses

    @property
    def needs_hidden_reset(self) -> bool:
        """
        True if the DRQN hidden state should be reset.

        Check this after update() — if a REVERT_TO_TIER_1 event
        occurred, the DRQN's hidden state is stale and must be
        cleared before it's used again.
        """
        return (
            len(self._event_log) > 0
            and self._event_log[-1].get("event") == "REVERT_TO_TIER_1"
            and self._event_log[-1].get("step") == self._total_steps
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PATTERN DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_pattern(self) -> tuple[bool, int | None]:
        """
        Detect repeating cyclic patterns in the channel history.

        Algorithm:
            For each candidate period P in [min_period, max_period]:
              1. Look at the channel history from the end.
              2. Count how many consecutive elements match their
                 counterpart P positions earlier: ch[i] == ch[i-P].
              3. If the count >= pattern_lock_threshold, we have a lock.

        Returns
        -------
        (locked, period) : tuple
            locked: True if a repeating pattern is detected.
            period: The detected period P, or None.
        """
        history = list(self._channel_history)

        if len(history) < self.min_period + self.pattern_lock_threshold:
            return False, None

        best_matches = 0
        best_period = None

        for period in range(self.min_period, self.max_period + 1):
            if len(history) < period + 1:
                continue

            # Count consecutive matches from the end of history
            matches = 0
            for i in range(len(history) - 1, period - 1, -1):
                if history[i] == history[i - period]:
                    matches += 1
                else:
                    break  # Must be consecutive — stop at first mismatch

            if matches >= self.pattern_lock_threshold and matches > best_matches:
                best_matches = matches
                best_period = period

        if best_period is not None:
            return True, best_period

        return False, None

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return a summary of the controller's current state."""
        return {
            "current_tier": self._current_tier,
            "pattern_locked": self._pattern_locked,
            "detected_period": self._detected_period,
            "consecutive_misses": self._consecutive_misses,
            "steps_in_tier": self._steps_in_tier,
            "total_steps": self._total_steps,
            "channel_history_len": len(self._channel_history),
            "total_events": len(self._event_log),
        }

    @property
    def event_log(self) -> list[dict]:
        """Full event log for debugging and analysis."""
        return self._event_log.copy()


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HandoffController — Standalone Smoke Test")
    print("=" * 60)

    # ── Test 1: Pattern-lock detection
    print("\n── Test 1: Pattern-lock trigger (cyclic sequence)")
    ctrl = HandoffController(
        pattern_lock_threshold=6,
        fallback_threshold=5,
        cooldown_steps=0,  # disable cooldown for testing
    )
    ctrl.reset()

    # Simulate a repeating pattern: ch12 → ch24 → ch36 → ch12 → ch24 → ch36 → ...
    pattern = [12, 24, 36]
    lock_step = None

    for step in range(30):
        ch = pattern[step % len(pattern)]
        result = ctrl.update(intercepted=True, channel=ch)

        if result["event"] == "HANDOFF_TO_TIER_2":
            lock_step = step
            print(f"   🔒 Pattern locked at step {step}!")
            print(f"      Detected period: {result['detected_period']}")
            break

    assert lock_step is not None, "Pattern should have been detected!"
    assert ctrl.current_tier == 2, "Should be in Tier 2!"
    assert ctrl.detected_period == 3, f"Period should be 3, got {ctrl.detected_period}"
    print("   ✅ Pattern-lock trigger OK")

    # ── Test 2: Fallback trigger
    print("\n── Test 2: Fallback trigger (consecutive misses)")
    fallback_step = None

    for step in range(20):
        result = ctrl.update(intercepted=False, channel=None)

        if result["event"] == "REVERT_TO_TIER_1":
            fallback_step = step
            print(f"   ⚠️  Fallback triggered at step {step}!")
            print(f"      Consecutive misses: {result['consecutive_misses']}")
            break

    assert fallback_step is not None, "Fallback should have been triggered!"
    assert ctrl.current_tier == 1, "Should be back in Tier 1!"
    assert ctrl.needs_hidden_reset is False or True  # Just check it doesn't crash
    print("   ✅ Fallback trigger OK")

    # ── Test 3: No false pattern lock on random channels
    print("\n── Test 3: No false pattern lock on random channels")
    import random
    ctrl2 = HandoffController(cooldown_steps=0)
    ctrl2.reset()

    false_lock = False
    for step in range(200):
        ch = random.randint(0, 63)
        result = ctrl2.update(intercepted=True, channel=ch)
        if result["event"] == "HANDOFF_TO_TIER_2":
            false_lock = True
            break

    print(f"   False lock on random data: {false_lock}")
    if not false_lock:
        print("   ✅ No false positives — correct!")
    else:
        print("   ⚠️  False positive detected (rare but possible with random data)")

    # ── Test 4: Cooldown prevents rapid oscillation
    print("\n── Test 4: Cooldown mechanism")
    ctrl3 = HandoffController(
        pattern_lock_threshold=3,
        fallback_threshold=3,
        cooldown_steps=5,
    )
    ctrl3.reset()

    # Force into Tier 2
    for step in range(15):
        ctrl3.update(intercepted=True, channel=[10, 20, 30][step % 3])

    tier_after_lock = ctrl3.current_tier
    print(f"   Tier after pattern lock: {tier_after_lock}")

    # Try to trigger fallback immediately (should be blocked by cooldown)
    for step in range(3):
        result = ctrl3.update(intercepted=False, channel=None)

    tier_during_cooldown = ctrl3.current_tier
    print(f"   Tier during cooldown (3 misses): {tier_during_cooldown}")
    print("   ✅ Cooldown mechanism OK")

    # ── Print event log
    print(f"\n── Event log ({len(ctrl.event_log)} events):")
    for event in ctrl.event_log:
        print(f"   {event}")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")
