"""
Systems & Concurrency Layer — Person 4
=====================================
SIH26055 Smart Scan Strategy for Electronic Warfare.

Core modules:
- replay: Sequential pulse streaming from memmap binary or deterministic synthetic generator.
- buffer: Thread-safe sliding observation buffer (10 x 5).
- dsp_stub: DSP feature extraction stub & frequency-channel mapping.
- receiver_sim: RF receiver retuning latency simulation & passband detection logic.
- metrics: Empirical step latency profiler and interception tracker.
- pipeline: Single-threaded reference pipeline orchestrator.
- concurrent_pipeline: Decoupled producer/consumer concurrent pipeline with bounded queues.
"""

from .replay import PDWReplay
from .buffer import StateBuffer
from .dsp_stub import DSPStub
from .receiver_sim import ReceiverSim, DetectionResult
from .metrics import PipelineMetrics, LatencyBreakdown
from .pipeline import SingleThreadedPipeline, StepResult
from .concurrent_pipeline import ConcurrentPipeline, PulseMessage

__all__ = [
    "PDWReplay",
    "StateBuffer",
    "DSPStub",
    "ReceiverSim",
    "DetectionResult",
    "PipelineMetrics",
    "LatencyBreakdown",
    "SingleThreadedPipeline",
    "StepResult",
    "ConcurrentPipeline",
    "PulseMessage",
]
