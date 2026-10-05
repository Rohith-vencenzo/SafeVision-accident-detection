"""Temporal fusion over a rolling window (Phase 9).

A single frame is never allowed to decide anything.  This module keeps the last
``temporal.window_seconds`` of evidence and produces a *temporal score* that
rewards persistence and punishes isolated spikes.

    frame 1..2   NORMAL        composite ~0.0   temporal ~0.0
    frame 3      possible      composite ~0.4   temporal ~0.15
    frame 4      collision     composite ~0.8   temporal ~0.4
    frame 5      sudden stop   composite ~0.9   temporal ~0.62
    frame 6      abnormal path composite ~0.9   temporal ~0.75
    frame 7      stopped       composite ~0.7   temporal ~0.80  -> VERIFYING

A one-frame false positive (say 1.0 for a single frame) yields a temporal score
around 0.1-0.2 and decays immediately - that is the "confidence decay" of the
spec, implemented as an exponential accumulator plus a recency-weighted mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai_engine.config.classes import TemporalConfig
from ai_engine.utils.geometry import clamp, smoothstep
from ai_engine.utils.logging_utils import get_logger
from ai_engine.utils.ringbuffer import TimeRingBuffer

__all__ = ["EvidenceSnapshot", "TemporalFusion", "TemporalResult"]

_LOGGER = get_logger("temporal")


@dataclass
class EvidenceSnapshot:
    """Everything the analyzers measured for one frame."""

    frame_index: int = 0
    timestamp: float = 0.0
    collision_score: float = 0.0
    motion_score: float = 0.0
    trajectory_score: float = 0.0
    object_evidence_score: float = 0.0
    scene_evidence_score: float = 0.0
    involved_tracks: Tuple[int, ...] = ()
    vehicles: int = 0
    persons: int = 0
    agreement_signals: int = 0
    reasons: List[str] = field(default_factory=list)

    #: how the three physical signals are combined into one "abnormality" value.
    #: Collision dominates because it is the only signal that already requires
    #: *agreement* between independent measurements; motion and trajectory are
    #: corroborating.
    _COMPOSITE_WEIGHTS = (("collision", 0.50), ("motion", 0.30), ("trajectory", 0.20))
    #: how much a single firing signal is trusted when the other two are silent:
    #: 1 signal -> 0.60, 2 -> 0.80, 3 -> 1.00.  Without this, one lone cue (a hard
    #: brake, say) would look exactly as abnormal as a multi-signal impact.
    _AGREEMENT_FACTOR = {0: 0.0, 1: 0.60, 2: 0.80}

    @property
    def composite(self) -> float:
        """Weighted agreement of the physical signals for this frame."""
        total_w = 0.0
        total = 0.0
        firing = 0
        for name, weight in self._COMPOSITE_WEIGHTS:
            value = float(getattr(self, f"{name}_score", 0.0) or 0.0)
            if value <= 0.0:
                continue
            firing += 1
            total += value * weight
            total_w += weight
        if total_w <= 0.0:
            return 0.0
        mean = total / total_w
        return clamp(mean * self._AGREEMENT_FACTOR.get(firing, 1.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 3),
            "composite": round(self.composite, 4),
            "collision_score": round(self.collision_score, 4),
            "motion_score": round(self.motion_score, 4),
            "trajectory_score": round(self.trajectory_score, 4),
            "vehicles": int(self.vehicles),
            "persons": int(self.persons),
            "agreement_signals": int(self.agreement_signals),
        }


@dataclass
class TemporalResult:
    """Fused multi-frame evidence for the current frame."""

    temporal_score: float = 0.0
    consistency: float = 0.0
    persistence: float = 0.0
    accumulator: float = 0.0
    window_frames: int = 0
    window_fill: float = 0.0
    frames_above: int = 0
    consecutive_above: int = 0
    longest_run: int = 0
    peak_composite: float = 0.0
    mean_composite: float = 0.0
    isolated: bool = False
    warmup: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "temporal_score": round(self.temporal_score, 4),
            "consistency": round(self.consistency, 4),
            "persistence": round(self.persistence, 4),
            "accumulator": round(self.accumulator, 4),
            "window_frames": int(self.window_frames),
            "window_fill": round(self.window_fill, 4),
            "frames_above_threshold": int(self.frames_above),
            "consecutive_above_threshold": int(self.consecutive_above),
            "isolated_event": bool(self.isolated),
            "peak_composite": round(self.peak_composite, 4),
            "mean_composite": round(self.mean_composite, 4),
            "reasons": list(self.reasons),
        }


class TemporalFusion:
    """Maintains the rolling evidence window and produces :class:`TemporalResult`."""

    def __init__(self, config: Optional[TemporalConfig] = None) -> None:
        self.config = config or TemporalConfig()
        self._window = TimeRingBuffer(window_seconds=self.config.window_seconds, max_items=600)
        self._accumulator = 0.0
        self._consecutive = 0
        self._longest_run = 0

    def reset(self) -> None:
        self._window.clear()
        self._accumulator = 0.0
        self._consecutive = 0
        self._longest_run = 0

    @property
    def window(self) -> List[EvidenceSnapshot]:
        return self._window.items

    def __len__(self) -> int:
        return len(self._window)

    # ------------------------------------------------------------------ #
    def push(self, snapshot: EvidenceSnapshot) -> TemporalResult:
        """Add one frame of evidence and return the fused result."""
        cfg = self.config
        self._window.push(snapshot.timestamp, snapshot)
        return self.aggregate(now=snapshot.timestamp)

    def aggregate(self, now: Optional[float] = None) -> TemporalResult:
        """Fuse the current window without adding a new snapshot."""
        cfg = self.config
        snapshots = self._window.items
        if now is not None:
            snapshots = self._window.window(now)
        if not snapshots:
            return TemporalResult(warmup=True)

        composites = [s.composite for s in snapshots]
        latest = snapshots[-1]
        # capacity of the configured window, expressed in frames
        expected = max(1, int(round(cfg.window_seconds * _nominal_fps(snapshots))))
        window_fill = clamp(len(snapshots) / expected)

        # --- 1. recency-weighted mean -----------------------------------
        weights: List[float] = []
        for snapshot in snapshots:
            age = max(0.0, float(latest.timestamp) - float(snapshot.timestamp))
            weights.append(0.5 ** (age / max(1e-3, cfg.recency_halflife_seconds)))
        weight_sum = sum(weights) or 1.0
        weighted_mean = sum(c * w for c, w in zip(composites, weights)) / weight_sum

        # --- 2. time-constant accumulator (instant impact, slow release) --
        # The decay is expressed in *seconds*, not frames, so the evidence hold
        # behaves the same at 5 fps and at 30 fps.  This is what keeps a crash
        # confirmed while the aftermath (stopped vehicles) is analysed.
        previous_timestamp = float(snapshots[-2].timestamp) if len(snapshots) > 1 else float(latest.timestamp)
        dt = max(1e-3, float(latest.timestamp) - previous_timestamp)
        release = math.exp(-dt / max(1e-3, cfg.hold_tau_seconds))
        self._accumulator = clamp(self._accumulator * release + latest.composite * (1.0 - release))

        # --- 3. persistence ----------------------------------------------
        above = [c >= cfg.possible_threshold for c in composites]
        frames_above = sum(1 for flag in above if flag)
        if above[-1]:
            self._consecutive += 1
        else:
            self._consecutive = 0
        self._longest_run = max(self._longest_run, self._consecutive)
        persistence = clamp(self._consecutive / max(1, cfg.min_consecutive_frames * 2))
        consistency = 1.0 - clamp(_coefficient_of_variation(composites))
        peak = max(composites)

        # --- 4. isolated spike suppression -------------------------------
        # An event supported by a single frame has no temporal support at all,
        # including on the frame the spike happens.  The first frame of a real
        # burst is also "isolated" for exactly one frame, which is correct: at
        # that instant nothing has confirmed it yet.
        isolated = False
        penalty = 0.0
        if frames_above <= 1 and self._consecutive <= 1:
            isolated = True
            penalty = cfg.isolated_penalty

        score = (
            cfg.weight_recent_mean * weighted_mean
            + cfg.weight_accumulator * self._accumulator
            + cfg.weight_peak * peak
            + cfg.persistence_bonus * persistence
            + cfg.weight_consistency * consistency * min(1.0, frames_above / max(1, cfg.min_consecutive_frames))
        )
        score = clamp(score - penalty)
        # A not-yet-filled window can never reach a confident score.
        score *= smoothstep(0.0, max(1e-3, cfg.min_window_fill), window_fill)
        if len(snapshots) < cfg.warmup_frames:
            score *= 0.5

        result = TemporalResult(
            temporal_score=score,
            consistency=consistency,
            persistence=persistence,
            accumulator=self._accumulator,
            window_frames=len(snapshots),
            window_fill=window_fill,
            frames_above=frames_above,
            consecutive_above=self._consecutive,
            longest_run=self._longest_run,
            peak_composite=max(composites),
            mean_composite=sum(composites) / len(composites),
            isolated=isolated,
            warmup=len(snapshots) < cfg.warmup_frames,
            reasons=self._reasons(
                persistence=persistence,
                consistency=consistency,
                frames_above=frames_above,
                isolated=isolated,
            ),
        )
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reasons(
        persistence: float,
        consistency: float,
        frames_above: int,
        isolated: bool,
    ) -> List[str]:
        reasons: List[str] = []
        if persistence >= 0.5 and frames_above >= 2:
            reasons.append("Persistent abnormal motion across frames")
        if consistency >= 0.7 and frames_above >= 2:
            reasons.append("Consistent evidence over the analysis window")
        if isolated:
            reasons.append("Single-frame event (no temporal support)")
        return reasons

    def snapshot_history(self) -> List[Dict[str, object]]:
        """Serialisable history - handy for the demo report and for tuning."""
        return [s.to_dict() for s in self._window.items]


# --------------------------------------------------------------------------- #
def _nominal_fps(snapshots: Sequence[EvidenceSnapshot]) -> float:
    """Estimate the processing FPS from consecutive timestamps."""
    if len(snapshots) < 2:
        return 30.0
    span = snapshots[-1].timestamp - snapshots[0].timestamp
    if span <= 1e-6:
        return 30.0
    return max(1.0, (len(snapshots) - 1) / span)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    """Normalised standard deviation (0 = perfectly steady evidence)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 1e-9:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean
