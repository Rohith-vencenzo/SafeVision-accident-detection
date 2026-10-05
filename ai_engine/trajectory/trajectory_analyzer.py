"""Vehicle trajectory analysis (Phase 7).

Two products:

*Per track* - how does this object's path look compared to the straight line it
was already following?  (``→ → → ↘`` = deviation; ``→ → → → →`` = normal.)

*Per pair* - do two trajectories converge or cross?  This is the "possible
collision zone" of the architecture diagram::

        X
   A  →──────╮      A: → → → ↘
             ╳   X  crossing point
      B  ←─────╯      B: ← ← ← ↙

Honesty note
------------
Crossing trajectories are *weak* evidence on their own: on a road viewed from a
fixed camera, two vehicles on different lanes routinely cross in the image
plane without any contact.  The crossing term is therefore capped
(``trajectory.crossing_weight``) and can never carry a decision by itself - the
collision analyzer has to see several *agreeing* signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai_engine.config.classes import TrajectoryConfig
from ai_engine.tracking.byte_tracker import Track
from ai_engine.utils.geometry import (
    angle_between_deg,
    clamp,
    segment_intersection,
    smoothstep,
    unit_vector,
)
from ai_engine.utils.logging_utils import get_logger

__all__ = ["TrajectoryAnalyzer", "TrajectoryFeatures", "TrajectoryPair"]

_LOGGER = get_logger("trajectory")


@dataclass
class TrajectoryFeatures:
    """Path-shape summary for one track."""

    track_id: int
    points: int = 0
    path_length: float = 0.0                 # frame-diagonals
    net_displacement: float = 0.0             # frame-diagonals
    straightness: float = 1.0                 # 1.0 = perfectly straight
    heading_deviation_deg: float = 0.0
    lateral_deviation: float = 0.0            # max distance from the earlier path line
    stopped: bool = False
    stopped_seconds: float = 0.0
    zigzag: float = 0.0
    valid: bool = True
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "points": int(self.points),
            "straightness": round(self.straightness, 4),
            "heading_deviation_deg": round(self.heading_deviation_deg, 2),
            "lateral_deviation": round(self.lateral_deviation, 5),
            "stopped": bool(self.stopped),
            "zigzag": round(self.zigzag, 4),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


@dataclass
class TrajectoryPair:
    """Do two tracks converge / cross?  Supporting evidence, never decisive."""

    track_a: int
    track_b: int
    crossing: bool = False
    crossing_point: Optional[Tuple[float, float]] = None
    converging: bool = False
    distance_norm: float = 0.0                 # current centre distance, object diagonals
    closing_speed: float = 0.0                  # frame-diagonals / s
    time_to_closest: float = 0.0                # seconds (<= max_ttc_seconds)
    speed_a: float = 0.0
    speed_b: float = 0.0
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_a": int(self.track_a),
            "track_b": int(self.track_b),
            "crossing": bool(self.crossing),
            "converging": bool(self.converging),
            "distance": round(self.distance_norm, 4),
            "closing_speed": round(self.closing_speed, 5),
            "time_to_closest_seconds": round(self.time_to_closest, 3),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


class TrajectoryAnalyzer:
    """Derives per-track path features and per-pair convergence."""

    def __init__(self, config: Optional[TrajectoryConfig] = None) -> None:
        self.config = config or TrajectoryConfig()

    def reset(self) -> None:
        return None

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        tracks: Sequence[Track],
        frame_size: Tuple[int, int],
        timestamp: float = 0.0,
    ) -> Tuple[Dict[int, TrajectoryFeatures], List[TrajectoryPair]]:
        """Return ``({track_id: TrajectoryFeatures}, [TrajectoryPair, ...])``."""
        width = float(frame_size[0]) if frame_size and frame_size[0] else 1.0
        height = float(frame_size[1]) if frame_size and frame_size[1] else 1.0
        scale = max(1.0, math.hypot(width, height))
        now = float(timestamp)

        per_track: Dict[int, TrajectoryFeatures] = {}
        for track in tracks:
            per_track[track.track_id] = self._track_features(track, scale, now)

        pairs: List[TrajectoryPair] = []
        candidates = [t for t in tracks if t.history_length >= max(2, self.config.min_points - 1)]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                pair = self._pair_features(candidates[i], candidates[j], scale, now)
                if pair is not None and (pair.score > 0.0 or pair.crossing):
                    pairs.append(pair)
        pairs.sort(key=lambda p: p.score, reverse=True)
        return per_track, pairs

    def aggregate(self, features: Dict[int, TrajectoryFeatures], tracks: Sequence[Track]) -> float:
        """Frame-level trajectory score from the most abnormal vehicle track."""
        vehicle_ids = {t.track_id for t in tracks if t.is_vehicle}
        pool = [f for tid, f in features.items() if (not vehicle_ids or tid in vehicle_ids) and f.valid]
        if not pool:
            return 0.0
        return float(max(f.score for f in pool))

    def aggregate_pair(self, pairs: Sequence[TrajectoryPair], max_pairs: int = 3) -> float:
        """Best convergence score, with a mild bonus for multiple pairs.

        Two or three independent pairs converging at the same spot is stronger
        evidence than one, but the bonus is capped so it cannot dominate.
        """
        if not pairs:
            return 0.0
        best = pairs[0].score
        extra = min(len(pairs) - 1, max_pairs - 1) * 0.05 if len(pairs) > 1 else 0.0
        return clamp(best + extra)

    # ------------------------------------------------------------------ #
    def _track_features(self, track: Track, scale: float, now: float) -> TrajectoryFeatures:
        cfg = self.config
        features = TrajectoryFeatures(track_id=track.track_id)

        points = track.points_between(now - cfg.window_seconds, now)
        features.points = len(points)
        if len(points) < cfg.min_points:
            features.valid = False
            features.reasons = ["insufficient trajectory history"]
            return features

        centers = [p.center for p in points]
        features.path_length = _path_length(centers) / scale
        features.net_displacement = _distance(centers[0], centers[-1]) / scale
        features.straightness = (
            clamp(features.net_displacement / features.path_length) if features.path_length > 1e-6 else 1.0
        )

        # ---- sudden stop over the window ---------------------------------
        features.stopped_seconds = track.stopped_for(cfg.window_seconds, cfg.stop_displacement, now)
        features.stopped = features.net_displacement <= cfg.stop_displacement and features.path_length <= cfg.stop_displacement * 2.5
        recent = [p.speed / scale for p in points[-max(2, min(len(points), 6)):]]
        if len(recent) >= 2 and max(recent[:-1] or recent) > 1e-6:
            drop = clamp((max(recent[:-1]) - recent[-1]) / max(recent[:-1]))
            stop_score = smoothstep(0.35, 1.0, drop)
        else:
            stop_score = 0.0

        # ---- heading deviation -------------------------------------------
        head_len = max(2, len(centers) // 3)
        early_vec = (centers[head_len] [0] - centers[0][0], centers[head_len][1] - centers[0][1])
        late_vec = (centers[-1][0] - centers[-1 - head_len][0], centers[-1][1] - centers[-1 - head_len][1])
        if _norm(early_vec) > 1e-6 and _norm(late_vec) > 1e-6:
            features.heading_deviation_deg = angle_between_deg(early_vec, late_vec)
        heading_score = smoothstep(cfg.heading_threshold_deg * 0.6, cfg.heading_threshold_deg, features.heading_deviation_deg)

        # ---- lateral deviation from the established path ------------------
        features.lateral_deviation = _max_lateral_deviation(centers, scale)
        lateral_score = smoothstep(cfg.lateral_threshold * 0.5, cfg.lateral_threshold * 1.5, features.lateral_deviation)

        # ---- zigzag --------------------------------------------------------
        features.zigzag = _zigzag_score(centers)
        zigzag_score = features.zigzag

        # ---- weighting ------------------------------------------------------
        weights = cfg.weights
        total_w = 0.0
        total = 0.0
        for name, value in (
            ("heading_deviation", heading_score),
            ("lateral_deviation", lateral_score),
            ("sudden_stop", stop_score),
            ("zigzag", zigzag_score),
        ):
            weight = float(weights.get(name, 0.0))
            if weight > 0.0:
                total += value * weight
                total_w += weight
        features.score = clamp(total / total_w) if total_w > 0 else 0.0

        # A path that simply curved and kept moving normally is less suspicious
        # than a path that deviated *and* stopped, so an otherwise clean curve is
        # damped - unless it also ended in a sudden stop.
        if features.score > 0.0 and features.straightness >= cfg.straightness_threshold and stop_score < 0.4:
            features.score *= 0.5

        features.reasons = self._reasons(features, heading_score, lateral_score, stop_score)
        return features

    # ------------------------------------------------------------------ #
    def _pair_features(self, a: Track, b: Track, scale: float, now: float) -> Optional[TrajectoryPair]:
        cfg = self.config
        pair = TrajectoryPair(track_a=a.track_id, track_b=b.track_id)

        ax, ay = a.center
        bx, by = b.center
        reference = max(1.0, 0.5 * (math.hypot(a.width, a.height) + math.hypot(b.width, b.height)))
        pair.distance_norm = _distance((ax, ay), (bx, by)) / reference
        pair.speed_a = a.speed_norm
        pair.speed_b = b.speed_norm

        # radial closing speed: only the component along the line of centres
        vax, vay = a.velocity
        vbx, vby = b.velocity
        pair.closing_speed = _closing_speed((ax, ay), (bx, by), (vax, vay), (vbx, vby)) / scale

        # ---- do the recent path segments cross? -------------------------
        a_pts = a.positions[-max(2, min(len(a.positions), 12)):]
        b_pts = b.positions[-max(2, min(len(b.positions), 12)):]
        crossing_point = _first_crossing(a_pts, b_pts)
        if crossing_point is not None:
            pair.crossing = True
            pair.crossing_point = crossing_point

        # ---- would the *extended* paths cross? (convergence) --------------
        horizon = min(cfg.max_ttc_seconds, 1.0)
        a_future = _extend(a_pts, (vax, vay), horizon)
        b_future = _extend(b_pts, (vbx, vby), horizon)
        if pair.crossing_point is None:
            future_point = _first_crossing(a_future, b_future)
            if future_point is not None:
                pair.converging = True
                pair.crossing_point = future_point
                pair.time_to_closest = _time_to_point(a_pts[-1] if a_pts else (ax, ay), (vax, vay), future_point)

        # ---- score --------------------------------------------------------
        speed_ok = (
            min(pair.speed_a, pair.speed_b) >= cfg.crossing_min_speed
            and max(pair.speed_a, pair.speed_b) <= cfg.crossing_max_speed
        )
        if pair.crossing and speed_ok:
            pair.score = cfg.crossing_weight
            pair.reasons.append("Trajectory crossing ahead of both objects")
        if pair.converging and speed_ok and pair.closing_speed > cfg.crossing_min_speed:
            pair.score = max(pair.score, cfg.crossing_weight)
            pair.reasons.append("Trajectory convergence")
        return pair

    @staticmethod
    def _reasons(
        features: TrajectoryFeatures,
        heading_score: float,
        lateral_score: float,
        stop_score: float,
    ) -> List[str]:
        reasons: List[str] = []
        if heading_score >= 0.4 or lateral_score >= 0.4:
            reasons.append("Path deviates from established heading")
        if stop_score >= 0.4:
            reasons.append("Trajectory ends in a sudden stop")
        if features.zigzag >= 0.5:
            reasons.append("Zig-zag / unstable trajectory")
        return reasons


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _norm(vec: Tuple[float, float]) -> float:
    return math.hypot(vec[0], vec[1])


def _distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _path_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_distance(points[i - 1], points[i]) for i in range(1, len(points)))


def _closing_speed(
    pa: Tuple[float, float],
    pb: Tuple[float, float],
    va: Tuple[float, float],
    vb: Tuple[float, float],
) -> float:
    """Component of the relative velocity that reduces the gap (px/s)."""
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0
    rvx, rvy = vb[0] - va[0], vb[1] - va[1]
    radial = (rvx * dx + rvy * dy) / distance
    return max(0.0, -radial)


def _max_lateral_deviation(points: Sequence[Tuple[float, float]], scale: float) -> float:
    """Largest distance from the line through the first and last points.

    A straight path gives ~0; a path that bulges out (or turns) gives a real
    value.  Normalised by the frame diagonal.
    """
    if len(points) < 3:
        return 0.0
    p0 = points[0]
    p1 = points[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    ux, uy = dx / length, dy / length
    worst = 0.0
    for px, py in points[1:-1]:
        # perpendicular distance to the p0->p1 line
        perpendicular = abs(ux * (py - p0[1]) - uy * (px - p0[0]))
        worst = max(worst, perpendicular)
    return worst / scale


def _zigzag_score(points: Sequence[Tuple[float, float]]) -> float:
    """Normalised oscillation count of the path direction.

    A straight path has zero direction changes; a shaking / rotating object has
    many.  Scaled by the mean segment length so tiny jitter in a stationary box
    does not look like zig-zag.
    """
    if len(points) < 4:
        return 0.0
    directions = []
    for i in range(1, len(points)):
        vec = (points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        if _norm(vec) > 1e-6:
            directions.append(unit_vector(vec))
    if len(directions) < 3:
        return 0.0
    turns = 0
    total_turn = 0.0
    for i in range(1, len(directions)):
        delta = angle_between_deg(directions[i - 1], directions[i])
        total_turn += delta
        if delta > 35.0:
            turns += 1
    mean_turn = total_turn / max(1, len(directions) - 1)
    return clamp((turns / max(1, len(directions) - 1)) * 1.5 + (mean_turn / 180.0) * 0.5)


def _first_crossing(a_pts: Sequence[Tuple[float, float]], b_pts: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    for i in range(1, len(a_pts)):
        for j in range(1, len(b_pts)):
            point = segment_intersection(a_pts[i - 1], a_pts[i], b_pts[j - 1], b_pts[j])
            if point is not None:
                return point
    return None


def _extend(
    points: Sequence[Tuple[float, float]],
    velocity: Tuple[float, float],
    seconds: float,
) -> List[Tuple[float, float]]:
    """Project the last segment forward (constant velocity) for ``seconds``."""
    if len(points) < 2:
        return list(points)
    tail = points[-1]
    direction = velocity if _norm(velocity) > 1e-6 else (tail[0] - points[-2][0], tail[1] - points[-2][1])
    if _norm(direction) < 1e-6:
        return list(points)
    unit = unit_vector(direction)
    step = _norm(direction) * seconds
    extended = list(points)
    steps = 4
    for k in range(1, steps + 1):
        factor = step * (k / steps)
        extended.append((tail[0] + unit[0] * factor, tail[1] + unit[1] * factor))
    return extended


def _time_to_point(
    origin: Tuple[float, float],
    velocity: Tuple[float, float],
    target: Tuple[float, float],
) -> float:
    speed = _norm(velocity)
    if speed < 1e-6:
        return float("inf")
    return _distance(origin, target) / speed
