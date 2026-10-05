"""Stage and end-to-end latency measurement.

Two clocks, deliberately kept apart:

* **monotonic** (``time.perf_counter``) for every *duration*.  It cannot jump
  when NTP corrects the system clock, so an elapsed measurement is trustworthy.
* **source timestamp** for *event time* - when the frame was captured.  This is
  the only time that means anything to a responder.

The central rule: **API acceptance is not answer, and delivery is not
delivery.**  The stage vocabulary separates "we handed it to the provider" from
"the provider took the request" from "a terminal call state" from "the SMS was
delivered to a handset", and a report may never collapse two of them.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

__all__ = [
    "LATENCY_STAGES",
    "LatencyLedger",
    "StageTimer",
    "percentiles",
    "summarise",
]

#: Ordered pipeline stages.  `capture` and `decode` are attributed by the
#: reader; the rest by the pipeline.
LATENCY_STAGES: tuple = (
    "capture_wait",     # blocked waiting for a frame from a live source
    "decode",           # demux + colour convert
    "detect",           # YOLO inference
    "track",            # association
    "analysis",         # motion + trajectory + collision + temporal + scoring
    "verify",           # the verification state machine
    "incident_build",   # assembling the incident record
    "notify_enqueue",   # handing the confirmed incident to the dispatcher
    "provider_request", # provider HTTP round trip
    "call_terminal",    # provider-reported terminal call state
    "sms_status",       # provider-reported SMS state
)


def percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """p50 / p95 / max / mean plus the sample count.

    The sample count is not decoration: a p95 over three samples is noise, and a
    consumer must be able to see that without reading the methodology.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None,
                "mean_ms": None, "min_ms": None}
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(_percentile(ordered, 95), 3),
        "max_ms": round(ordered[-1], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "min_ms": round(ordered[0], 3),
    }


def _percentile(ordered: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile - no interpolation, so the value is a real sample."""
    if not ordered:
        raise ValueError("empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


@dataclass
class StageTimer:
    """Context manager measuring one stage on the monotonic clock.

        with StageTimer(ledger, "detect"):
            result = detector.detect(frame)

    The value is recorded only when the block completes normally; an exception
    propagates and the stage is recorded as ``None`` so a failure cannot be
    silently counted as zero latency.
    """

    ledger: "LatencyLedger"
    stage: str
    _start: float = field(default=0.0, init=False)
    _ok: bool = field(default=False, init=False)

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.ledger.record(self.stage, elapsed_ms, ok=exc_type is None)
        return False  # never swallow


@contextmanager
def timed(ledger: "LatencyLedger", stage: str) -> Iterator[None]:
    """Function-scoped alias for :class:`StageTimer`."""
    timer = StageTimer(ledger, stage)
    with timer:
        yield


@dataclass
class LatencyLedger:
    """Accumulates stage durations and produces a report.

    Bounded by ``max_samples_per_stage`` so a day-long live stream cannot exhaust
    memory; the reservoir keeps the most recent samples, which is what an
    operator debugging "why is it slow now" needs.
    """

    max_samples_per_stage: int = 5000
    mode: str = "UNKNOWN"                 # RECORDED | LIVE | UNKNOWN
    source_kind: str = "UNKNOWN"
    _samples: Dict[str, List[float]] = field(default_factory=dict)
    _errors: Dict[str, int] = field(default_factory=dict)
    #: incident_id -> {stage: monotonic seconds since the incident was created}
    _incident_marks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _last_incident: Optional[str] = None
    _frames_seen: int = 0

    # ------------------------------------------------------------------ #
    def record(self, stage: str, elapsed_ms: float, ok: bool = True) -> None:
        if not ok:
            self._errors[stage] = self._errors.get(stage, 0) + 1
            return
        bucket = self._samples.setdefault(stage, [])
        bucket.append(float(elapsed_ms))
        if len(bucket) > self.max_samples_per_stage:
            del bucket[: len(bucket) - self.max_samples_per_stage]

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        timer = StageTimer(self, name)
        with timer:
            yield

    # -- incident-scoped marks ----------------------------------------- #
    def mark(self, incident_id: Optional[str], stage: str) -> None:
        """Stamp ``stage`` against an incident, on the monotonic clock.

        Used for the notification chain, where the interesting number is "how
        long after confirmation did the provider accept the request", not "how
        long did the HTTP call take".
        """
        if not incident_id:
            return
        marks = self._incident_marks.setdefault(incident_id, {})
        if stage not in marks:
            marks[stage] = time.perf_counter()

    def incident_timeline(self, incident_id: str) -> Dict[str, float]:
        """Milliseconds from the first mark for this incident, per stage."""
        marks = self._incident_marks.get(incident_id) or {}
        if not marks:
            return {}
        origin = min(marks.values())
        return {
            stage: round((value - origin) * 1000.0, 3)
            for stage, value in sorted(marks.items(), key=lambda kv: kv[1])
        }

    # -- frame / queue accounting --------------------------------------- #
    def frame(self, dropped: int = 0) -> None:
        self._frames_seen += 1

    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        stages = {}
        for stage in LATENCY_STAGES:
            samples = self._samples.get(stage)
            if not samples:
                continue
            entry = percentiles(samples)
            if self._errors.get(stage):
                entry["errors"] = self._errors[stage]
            stages[stage] = entry
        return {
            "mode": self.mode,
            "source_kind": self.source_kind,
            "frames_processed": self._frames_seen,
            "stages": stages,
            "incident_timelines": {
                k: self.incident_timeline(k) for k in list(self._incident_marks)
            },
            "methodology": {
                "clock": "time.perf_counter() (monotonic)",
                "start": "first instruction of the stage",
                "end": "return from the stage",
                "percentiles": "nearest-rank; p95 is a real observed sample, not interpolated",
                "event_time": "source timestamp, kept separate from elapsed time",
                "caveat": (
                    "These are pipeline stage durations on THIS machine. They do "
                    "not bound the provider's or the network's contribution to a "
                    "call being answered or an SMS being read."
                ),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.report()

    def text(self) -> str:
        report = self.report()
        lines = [
            "=" * 72,
            f" latency report  mode={report['mode']}  source={report['source_kind']}"
            f"  frames={report['frames_processed']}",
            "=" * 72,
        ]
        if not report["stages"]:
            lines.append("  (no stage samples recorded)")
        for stage, stats in report["stages"].items():
            lines.append(
                f"  {stage:<18} n={stats['count']:<6} "
                f"p50={stats['p50_ms']:>8}ms  p95={stats['p95_ms']:>8}ms  "
                f"max={stats['max_ms']:>8}ms"
            )
        if report["incident_timelines"]:
            lines.append("-" * 72)
            for incident, timeline in report["incident_timelines"].items():
                lines.append(f"  {incident}:")
                for stage, value in timeline.items():
                    lines.append(f"    +{value:>9}ms  {stage}")
        lines.append("-" * 72)
        lines.append(f"  clock: {report['methodology']['clock']}")
        lines.append(f"  caveat: {report['methodology']['caveat']}")
        lines.append("=" * 72)
        return "\n".join(lines)


def summarise(ledgers: Sequence[LatencyLedger]) -> Dict[str, Any]:
    """Merge several ledgers into one distribution set (for a repeated run)."""
    merged: Dict[str, List[float]] = {}
    for ledger in ledgers:
        for stage, samples in ledger._samples.items():  # noqa: SLF001
            merged.setdefault(stage, []).extend(samples)
    return {
        stage: percentiles(values) for stage, values in sorted(merged.items())
    }