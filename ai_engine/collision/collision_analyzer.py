"""Collision analysis (Phase 8).

A collision is *never* inferred from a single cue.  Six independent signals are
measured for every plausible object pair and the score only rises when several
of them **agree**:

1. ``proximity``      - centre distance in object diagonals
2. ``overlap``        - bounding-box IoU / centre inside the other box
3. ``closing``        - radial closing speed (are they actually approaching?)
4. ``deceleration``   - did *both* objects change velocity at the same time?
5. ``post_stop``      - after closing, are they now essentially stopped?
6. ``displacement``   - a position jump that the previous velocity did not
                        predict (the "kick" a collision produces)
7. ``convergence``    - the two paths meet ahead (from the trajectory module)

Gating (``collision.min_signals``) is the important part: proximity alone -
which is all a naive box-overlap rule has - contributes very little and cannot
reach the confirmation threshold on its own.  A single frame of overlap between
two vehicles passing in opposite lanes therefore stays a low score, while
"approaching fast + both decelerate + then both stop" is strong.

Units: distances in object diagonals, speeds in object diagonals per second,
deceleration in object diagonals per second squared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai_engine.config.classes import CollisionConfig
from ai_engine.motion.motion_analyzer import MotionFeatures
from ai_engine.tracking.byte_tracker import Track
from ai_engine.trajectory.trajectory_analyzer import TrajectoryPair
from ai_engine.utils.geometry import clamp, smoothstep, xyxy_iou
from ai_engine.utils.logging_utils import get_logger

__all__ = ["CollisionAnalyzer", "CollisionEvent", "CollisionSignals"]

_LOGGER = get_logger("collision")

#: Signals that carry real physical meaning.  ``proximity``/``overlap`` are
#: listed too but are deliberately weighted low.
_SIGNAL_NAMES: Tuple[str, ...] = (
    "proximity",
    "overlap",
    "closing",
    "deceleration",
    "post_stop",
    "displacement",
    "convergence",
)


@dataclass
class CollisionSignals:
    """Individual, inspectable evidence for one pair."""

    proximity: float = 0.0
    overlap: float = 0.0
    center_inside: bool = False
    closing: float = 0.0
    closing_speed: float = 0.0
    deceleration_a: float = 0.0
    deceleration_b: float = 0.0
    pair_deceleration: float = 0.0
    post_stop: float = 0.0
    displacement: float = 0.0
    convergence: float = 0.0
    time_to_impact: float = 0.0
    relative_speed: float = 0.0
    person_involved: bool = False
    active: Tuple[str, ...] = ()

    @property
    def agreement(self) -> int:
        """How many *independent* signals fired."""
        return len(self.active)

    def to_dict(self) -> Dict[str, object]:
        return {
            "signals_active": list(self.active),
            "agreement": self.agreement,
            "proximity": round(self.proximity, 4),
            "overlap_iou": round(self.overlap, 4),
            "center_inside": bool(self.center_inside),
            "closing_speed": round(self.closing_speed, 5),
            "deceleration_a": round(self.deceleration_a, 5),
            "deceleration_b": round(self.deceleration_b, 5),
            "pair_deceleration": round(self.pair_deceleration, 4),
            "post_stop": round(self.post_stop, 4),
            "displacement_spike": round(self.displacement, 4),
            "convergence": round(self.convergence, 4),
            "time_to_impact_seconds": round(self.time_to_impact, 3) if self.time_to_impact < 1e6 else None,
            "relative_speed": round(self.relative_speed, 5),
            "person_involved": bool(self.person_involved),
        }


@dataclass
class CollisionEvent:
    """A candidate collision between two tracks."""

    track_a: int
    track_b: int
    class_a: str = ""
    class_b: str = ""
    score: float = 0.0
    signals: CollisionSignals = field(default_factory=CollisionSignals)
    reasons: List[str] = field(default_factory=list)
    point: Optional[Tuple[float, float]] = None
    timestamp: float = 0.0
    frame_index: int = 0

    @property
    def track_ids(self) -> Tuple[int, int]:
        return (self.track_a, self.track_b)

    @property
    def agreement(self) -> int:
        return self.signals.agreement

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_a": int(self.track_a),
            "track_b": int(self.track_b),
            "class_a": self.class_a,
            "class_b": self.class_b,
            "score": round(float(self.score), 4),
            "agreement_signals": self.agreement,
            "point": [round(self.point[0], 1), round(self.point[1], 1)] if self.point else None,
            "reasons": list(self.reasons),
            "signals": self.signals.to_dict(),
        }


class CollisionAnalyzer:
    """Scores every plausible object pair for collision evidence."""

    def __init__(self, config: Optional[CollisionConfig] = None) -> None:
        self.config = config or CollisionConfig()
        #: track-pair -> timestamp of the last *physical* event, used to keep
        #: the "post-impact stopping" evidence alive for a few seconds.
        self._impact_memory: Dict[Tuple[int, int], float] = {}

    def reset(self) -> None:
        self._impact_memory.clear()

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        tracks: Sequence[Track],
        motion: Optional[Dict[int, MotionFeatures]] = None,
        trajectory_pairs: Optional[Sequence[TrajectoryPair]] = None,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> Tuple[Optional[CollisionEvent], List[CollisionEvent]]:
        """Return ``(best_event, all_events)`` for this frame.

        ``best_event`` is ``None`` unless at least ``collision.min_signals``
        independent signals agree - that gate is the false-alarm defence at
        the heart of this module.
        """
        cfg = self.config
        motion = motion or {}
        pair_index = {
            (min(p.track_a, p.track_b), max(p.track_a, p.track_b)): p
            for p in (trajectory_pairs or [])
        }

        candidates = [t for t in tracks if t.history_length >= 2]
        if cfg.person_collisions:
            pass  # persons are allowed to be one side of the pair
        else:
            candidates = [t for t in candidates if t.is_vehicle]

        events: List[CollisionEvent] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                event = self._score_pair(
                    candidates[i],
                    candidates[j],
                    motion,
                    pair_index,
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
                if event is not None:
                    events.append(event)

        events.sort(key=lambda e: (e.agreement, e.score), reverse=True)
        capped = events[: max(1, int(cfg.max_pairs))]
        best = None
        for event in capped:
            if event.agreement >= cfg.min_signals and event.score > 0.0:
                best = event
                break
        if best is not None:
            _LOGGER.debug(
                "collision candidate tracks=%s score=%.3f signals=%s",
                best.track_ids,
                best.score,
                list(best.signals.active),
            )
        return best, capped

    # ------------------------------------------------------------------ #
    def _score_pair(
        self,
        a: Track,
        b: Track,
        motion: Dict[int, MotionFeatures],
        pair_index: Dict[Tuple[int, int], TrajectoryPair],
        timestamp: float,
        frame_index: int,
    ) -> Optional[CollisionEvent]:
        cfg = self.config
        signals = CollisionSignals()
        reasons: List[str] = []

        # --- 1. proximity / overlap -------------------------------------
        acx, acy = a.center
        bcx, bcy = b.center
        a_diag = max(1.0, math.hypot(a.width, a.height))
        b_diag = max(1.0, math.hypot(b.width, b.height))
        reference = max(1.0, 0.5 * (a_diag + b_diag))
        distance_norm = math.hypot(bcx - acx, bcy - acy) / reference
        # saturates at ~1.0 when the objects touch, reaches 0 at proximity_radius
        signals.proximity = clamp(1.0 - smoothstep(cfg.proximity_radius * 0.2, cfg.proximity_radius, distance_norm))
        signals.overlap = xyxy_iou(a.bbox, b.bbox)
        signals.center_inside = _point_in(bcx, bcy, a.bbox) or _point_in(acx, acy, b.bbox)
        signals.person_involved = a.is_person or b.is_person

        overlap_score = clamp(
            smoothstep(cfg.iou_threshold, max(cfg.iou_threshold * 3.0, 0.5), signals.overlap)
            + (0.35 if signals.center_inside else 0.0)
        )
        if signals.proximity >= 0.5:
            reasons.append("Collision proximity detected")
        if overlap_score >= 0.4:
            reasons.append("Bounding-box overlap between objects")

        # --- 2. closing speed --------------------------------------------
        rvx, rvy = b.velocity[0] - a.velocity[0], b.velocity[1] - a.velocity[1]
        signals.relative_speed = math.hypot(rvx, rvy) / reference
        closing_speed = 0.0
        distance_px = math.hypot(bcx - acx, bcy - acy)
        if distance_px > 1e-6:
            radial = (rvx * (bcx - acx) + rvy * (bcy - acy)) / distance_px
            closing_speed = max(0.0, -radial) / reference
        signals.closing_speed = closing_speed

        # Two vehicles driving towards each other 400 px apart are not closing
        # on a collision - they are just driving.  The closing signal is scaled
        # by how soon contact would actually happen, so ordinary traffic at a
        # junction never contributes.
        time_to_impact = 1e9
        if distance_norm > 1e-6 and closing_speed > 1e-6:
            time_to_impact = min(distance_norm / closing_speed, 1e6)
        signals.time_to_impact = time_to_impact
        contact_gate = 1.0 - smoothstep(
            cfg.time_to_impact_window * 0.5, cfg.time_to_impact_window, time_to_impact
        )
        signals.closing = clamp(
            smoothstep(cfg.min_closing_speed, cfg.min_closing_speed + cfg.closing_speed_scale, closing_speed)
            * contact_gate
        )
        if signals.closing >= 0.4:
            reasons.append("Rapidly closing distance between objects")

        # --- 3. paired deceleration --------------------------------------
        fa = motion.get(a.track_id)
        fb = motion.get(b.track_id)
        decel_a = _deceleration(a, reference)
        decel_b = _deceleration(b, reference)
        signals.deceleration_a = decel_a
        signals.deceleration_b = decel_b
        if cfg.require_motion:
            if fa is not None and not fa.valid:
                decel_a = 0.0
            if fb is not None and not fb.valid:
                decel_b = 0.0
        pair_ratio = min(decel_a, decel_b)
        signals.pair_deceleration = clamp(
            smoothstep(cfg.decel_ratio, cfg.decel_ratio + cfg.decel_scale, pair_ratio)
        )
        if signals.pair_deceleration >= 0.4:
            reasons.append("Sudden velocity change in both objects")

        # --- 4. post-impact stopping -------------------------------------
        # "Post-impact" means: the object was moving shortly before and is now
        # essentially still.  Two parked cars next to each other must not match
        # this signal, so the *pre-event* speed is checked explicitly.
        #
        # Once a pair has shown a physical event, the "both stopped afterwards"
        # fact stays true for a while even though the pre-event speed is long
        # gone from the history - which is exactly how a human reviewer reads
        # the footage.  ``_impact_memory`` provides that short-term memory.
        pre_a = a.average_speed(cfg.pre_event_offset_seconds, timestamp) / a.scale
        pre_b = b.average_speed(cfg.pre_event_offset_seconds, timestamp) / b.scale
        stopped_now = a.speed_norm <= cfg.post_stop_speed and b.speed_norm <= cfg.post_stop_speed
        both_slow = a.has_been_stopped_for(cfg.post_stop_seconds, cfg.post_stop_speed, timestamp)
        both_slow = both_slow and b.has_been_stopped_for(cfg.post_stop_seconds, cfg.post_stop_speed, timestamp)
        moving_before = max(pre_a, pre_b) > cfg.post_stop_speed

        pair_key = (a.track_id, b.track_id) if a.track_id <= b.track_id else (b.track_id, a.track_id)
        had_impact = (timestamp - self._impact_memory.get(pair_key, -1e9)) <= cfg.post_stop_memory_seconds
        self._impact_memory[pair_key] = self._impact_memory.get(pair_key, -1e9)

        if stopped_now and both_slow and (moving_before or had_impact):
            signals.post_stop = clamp(smoothstep(0.0, cfg.post_stop_speed, max(a.speed_norm, b.speed_norm)))
            reasons.append("Post-collision stopping")
        elif (a.speed_norm <= cfg.post_stop_speed) != (b.speed_norm <= cfg.post_stop_speed):
            # only one vehicle stopped is weaker, but still meaningful
            signals.post_stop = 0.4 * clamp(smoothstep(0.0, cfg.post_stop_speed, min(a.speed_norm, b.speed_norm)))

        # --- 5. displacement spike ---------------------------------------
        spike_a = _displacement_spike(a, reference)
        spike_b = _displacement_spike(b, reference)
        spike = max(spike_a, spike_b)
        signals.displacement = clamp(
            smoothstep(cfg.displacement_spike_ratio, cfg.displacement_spike_ratio + cfg.displacement_spike_scale, spike)
        )
        if signals.displacement >= 0.4:
            reasons.append("Post-impact displacement spike")

        # --- 6. trajectory convergence -----------------------------------
        key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
        traj_pair = pair_index.get(key)
        point: Optional[Tuple[float, float]] = None
        if traj_pair is not None:
            signals.convergence = clamp(traj_pair.score)
            point = traj_pair.crossing_point
            if traj_pair.reasons:
                reasons.extend(traj_pair.reasons)
        if point is None and distance_norm < 1.0:
            point = ((acx + bcx) * 0.5, (acy + bcy) * 0.5)

        # --- agreement + weighted score -----------------------------------
        active: List[str] = []
        if signals.proximity >= 0.5:
            active.append("proximity")
        if overlap_score >= 0.4:
            active.append("overlap")
        if signals.closing >= 0.4:
            active.append("closing")
        if signals.pair_deceleration >= 0.4:
            active.append("deceleration")
        if signals.post_stop >= 0.4:
            active.append("post_stop")
        if signals.displacement >= 0.4:
            active.append("displacement")
        if signals.convergence >= 0.4:
            active.append("convergence")
        signals.active = tuple(name for name in _SIGNAL_NAMES if name in active)

        # Two families of evidence, deliberately weighted very differently:
        #
        # *physical*  - closing / deceleration / post_stop / displacement.
        #   These describe what a collision physically does to a vehicle.
        # *context*   - proximity / overlap / convergence.
        #   These only say two objects are near each other in the image, which
        #   on a road happens constantly (adjacent lanes, following traffic,
        #   perspective overlap).  On their own they must never approach a
        #   confirmable score, which is why they carry a 0.25 cap.
        physical = {
            "closing": signals.closing,
            "deceleration": signals.pair_deceleration,
            "post_stop": signals.post_stop,
            "displacement": signals.displacement,
        }
        active_physical = [name for name in physical if name in active]
        physical_level = max(physical.values()) if active_physical else 0.0
        # one physical signal is 60% convincing, two are 80%, three or more 100%
        physical_confidence = {
            0: 0.0,
            1: 0.60,
            2: 0.80,
        }.get(len(active_physical), 1.0)
        physical_score = physical_level * physical_confidence

        context_score = clamp(
            0.50 * signals.proximity
            + 0.30 * overlap_score
            + 0.20 * signals.convergence
        )

        score = clamp(cfg.physical_weight * physical_score + cfg.context_weight * context_score)
        if signals.person_involved and score > 0.3:
            score = clamp(score + cfg.person_signal_bonus)

        # Below the agreement gate the score is heavily suppressed, which is
        # what stops a single box overlap from ever being an accident.
        if len(active) < cfg.min_signals:
            score = clamp(score * 0.35)

        if not active:
            return None

        # Remember that this pair showed a physical event, so the "both stopped
        # afterwards" evidence survives for ``post_stop_memory_seconds``.
        if physical_score >= cfg.physical_weight * 0.55 and len(active_physical) >= 1:
            self._impact_memory[pair_key] = float(timestamp)

        return CollisionEvent(
            track_a=a.track_id,
            track_b=b.track_id,
            class_a=a.normalized_name,
            class_b=b.normalized_name,
            score=score,
            signals=signals,
            reasons=_dedupe(reasons),
            point=point,
            timestamp=float(timestamp),
            frame_index=int(frame_index),
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _point_in(x: float, y: float, box: Sequence[float]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _dedupe(items: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def _weighted_score(values: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = 0.0
    total = 0.0
    for key, value in values.items():
        weight = weights.get(key, 0.0)
        if weight <= 0.0:
            continue
        total += float(value) * weight
        total_w += weight
    return clamp(total / total_w) if total_w > 0 else 0.0


def _deceleration(track: Track, reference: float) -> float:
    """Speed lost per second, in object diagonals per second (>= 0)."""
    now = track.timestamp
    current = track.speed_norm
    baseline = track.average_speed(0.5, now) / max(1.0, track.scale) if track.history_length >= 2 else current
    if baseline <= 0.0:
        return 0.0
    return clamp((baseline - current) / baseline)


def _displacement_spike(track: Track, reference: float) -> float:
    """Unexplained displacement: |actual step - predicted step| / expected.

    A collision physically shoves a vehicle sideways or rotates it, so its new
    position is not explained by where it was already going.  The ratio between
    the residual and the expected step is what this measures.  When the object
    was (nearly) stationary the "expected" step is floored at a fraction of the
    object's own size, so a real kick still registers instead of dividing by
    zero.
    """
    points = track.recent_points(3)
    if len(points) < 3:
        return 0.0
    p1, p2 = points[-2], points[-1]
    dt = float(p2.timestamp - p1.timestamp)
    if dt <= 1e-6:
        return 0.0
    actual = math.hypot(p2.center[0] - p1.center[0], p2.center[1] - p1.center[1])
    predicted = math.hypot(p1.instant_velocity[0], p1.instant_velocity[1]) * dt
    residual = math.hypot(
        (p2.center[0] - p1.center[0]) - p1.instant_velocity[0] * dt,
        (p2.center[1] - p1.center[1]) - p1.instant_velocity[1] * dt,
    )
    expected = max(predicted, 0.20 * reference)
    if expected < 1e-6:
        return 0.0
    del actual  # documented above; the residual is the meaningful quantity
    return clamp(residual / expected)
