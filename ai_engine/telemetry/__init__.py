"""Telemetry: latency instrumentation and source-mode reporting.

Latency is measured, not asserted.  Two clocks are kept apart - monotonic for
durations, source timestamps for event time - and a provider accepting a request
is never reported as a call being answered.
"""

from __future__ import annotations

from ai_engine.telemetry.latency import (
    LATENCY_STAGES,
    LatencyLedger,
    StageTimer,
    percentiles,
    summarise,
    timed,
)

__all__ = [
    "LATENCY_STAGES",
    "LatencyLedger",
    "StageTimer",
    "percentiles",
    "summarise",
    "timed",
]