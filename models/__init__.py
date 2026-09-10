"""
Tier 2 — Deep Recurrent Q-Network (DRQN) Components
=====================================================

This package contains all Person 3 modules for the cognitive radar
interception system's Tier 2 intelligence layer.

Modules:
    state_builder       — Observation normalisation & context feature engineering
    drqn_network        — LSTM + Dueling Q-head neural network
    replay_buffer       — Sequence-based recurrent experience replay
    drqn_agent          — Double-DQN agent with epsilon-greedy exploration
    handoff_controller  — Pattern-lock (Tier1→Tier2) & fallback (Tier2→Tier1) triggers
"""

from .state_builder import StateBuilder
from .drqn_network import DRQNNetwork
from .replay_buffer import RecurrentReplayBuffer
from .drqn_agent import DRQNAgent
from .handoff_controller import HandoffController

__all__ = [
    "StateBuilder",
    "DRQNNetwork",
    "RecurrentReplayBuffer",
    "DRQNAgent",
    "HandoffController",
]
