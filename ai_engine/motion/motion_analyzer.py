"""Per-track motion analysis (Phase 6).

For every tracked object we derive:

* position, displacement, velocity and heading
* acceleration (change of velocity)
* **sudden stop** - was moving, now essentially stopped
* **sudden direction change** - heading flips while still moving
* positional jitter (vibration / rotation blur)

These are combined into one normalised ``score`` in ``[0, 1]`` where ``1.0``
means "as abnormal as this feature can get".

Scope note (honesty): motion alone never proves an accident. Braking for a
red light produces the same signature.  Motion features are *one* evidence
source out of several and are weighted accordingly in
:mod:`ai_engine.scoring.accident_scorer`.

Units
-----
Everything is normalised by the frame diagonal and expressed per second, so the
same thresholds work at 640x360 and 1920x1080.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ai_engine.config.classes import MotionConfig
from ai_engine.tracking.byte_tracker import Track
from ai_engine.utils.geometry import angle_between_deg, clamp, smoothstep
from ai_engine.utils.logging_utils import get_logger

__all__ = ["MotionAnalyzer", "MotionFeatures"]

_LOGGER = get_logger("motion")

#: relative speed drop below which a stop is treated as ordinary noise
_MIN_DROP = 0.25


@dataclass
class MotionFeatures:
    """Motion summary for one track in one frame."""

    track_id: int
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)      # frame-diagonals / s
    speed: float = 0.0                              # frame-diagonals / s
    baseline_speed: float = 0.0                     # speed before the current event
    speed_before: float = 0.0                       # fastest recent instantaneous speed
    instant_speed: float = 0.0                      # latest instantaneous speed
    speed_drop_ratio: float = 0.0
    acceleration: float = 0.0                       # frame-diagonals / s^2
    acceleration_score: float = 0.0                 # normalised [0, 1]
    heading_deg: float = 0.0
    turn_deg: float = 0.0
    displacement: float = 0.0                       # path length in the window
    sudden_stop: float = 0.0
    sudden_turn: float = 0.0
    jitter: float = 0.0
    moving: bool = False
    stationary_seconds: float = 0.0
    history_length: int = 0
    valid: bool = True                              # False -> not enough history
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "speed": round(self.speed, 5),
            "baseline_speed": round(self.baseline_speed, 5),
            "speed_drop_ratio": round(self.speed_drop_ratio, 4),
            "acceleration": round(self.acceleration, 5),
            "turn_deg": round(self.turn_deg, 2),
            "sudden_stop": round(self.sudden_stop, 4),
            "sudden_turn": round(self.sudden_turn, 4),
            "jitter": round(self.jitter, 4),
            "moving": bool(self.moving),
            "stationary_seconds": round(self.stationary_seconds, 3),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


class MotionAnalyzer:
    """Computes :class:`MotionFeatures` for every live track.

    Global (camera) motion compensation
    -----------------------------------
    A panning / PTZ camera makes *every* object appear to move, which would look
    like a crowd of suddenly changing headings.  Before analysing, the median
    velocity of all tracks is estimated and - **only when most tracks agree on
    it** - subtracted from each track.  With a static camera the tracks disagree
    (they drive in different directions), so nothing is subtracted and the
    behaviour is exactly as before.
    """

    def __init__(self, config: Optional[MotionConfig] = None) -> None:
        self.config = config or MotionConfig()
        self._ema: Dict[int, Dict[str, float]] = {}
        self._global_velocity: Tuple[float, float] = (0.0, 0.0)
        self._camera_moving = False
        self._tracks_seen = 0

    def reset(self) -> None:
        self._ema.clear()
        self._global_velocity = (0.0, 0.0)
        self._camera_moving = False
        self._tracks_seen = 0

    @property
    def global_velocity(self) -> Tuple[float, float]:
        """Estimated camera motion in px/s (0, 0 for a static camera)."""
        return self._global_velocity

    @property
    def camera_moving(self) -> bool:
        return self._camera_moving

    # ------------------------------------------------------------------ #
    def _estimate_camera_motion(self, tracks: Sequence[Track]) -> Tuple[float, float]:
        """Median velocity, but only if the tracks agree it is the camera.

        At least ``min_agreeing_tracks`` objects must agree.  With only one or
        two tracks the agreement test is meaningless: two cars driving in the
        same direction on a straight road would look exactly like a panning
        camera, and compensating for that would erase real motion evidence.
        """
        cfg = self.config
        velocities = [(t.velocity[0], t.velocity[1]) for t in tracks]
        if len(velocities) < max(2, int(cfg.camera_motion_min_tracks)):
            self._camera_moving = False
            self._global_velocity = (0.0, 0.0)
            return (0.0, 0.0)

        vx = _median([v[0] for v in velocities])
        vy = _median([v[1] for v in velocities])
        magnitude = math.hypot(vx, vy)
        if magnitude < cfg.camera_motion_min_speed:
            self._camera_moving = False
            self._global_velocity = (0.0, 0.0)
            return (0.0, 0.0)

        agreeing = sum(1 for v in velocities if abs(v[0] - vx) <= cfg.camera_motion_tolerance * magnitude + 1.0
                       and abs(v[1] - vy) <= cfg.camera_motion_tolerance * magnitude + 1.0)
        if agreeing >= max(2, int(cfg.camera_motion_agreement * len(velocities))):
            self._camera_moving = True
            self._global_velocity = (vx, vy)
            return (vx, vy)
        self._camera_moving = False
        self._global_velocity = (0.0, 0.0)
        return (0.0, 0.0)

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        tracks: Sequence[Track],
        frame_size: Tuple[int, int],
        timestamp: float = 0.0,
    ) -> Dict[int, MotionFeatures]:
        """Analyze all tracks.  Returns ``{track_id: MotionFeatures}``."""
        cfg = self.config
        width = float(frame_size[0]) if frame_size and frame_size[0] else 1.0
        height = float(frame_size[1]) if frame_size and frame_size[1] else 1.0
        scale = max(1.0, math.hypot(width, height))

        global_vx, global_vy = self._estimate_camera_motion(tracks)
        self._tracks_seen = len(tracks)

        features: Dict[int, MotionFeatures] = {}
        for track in tracks:
            features[track.track_id] = self._analyze_track(
                track, scale, float(timestamp), (global_vx, global_vy)
            )
        return features

    def aggregate(self, features: Dict[int, MotionFeatures], tracks: Sequence[Track]) -> float:
        """One number for the frame: the most abnormal *vehicle* track.

        People are excluded whenever a vehicle is present - a pedestrian's
        erratic motion says nothing about a vehicle collision.  In a scene with
        no vehicles at all (a footpath, a car park full of bicycles) the
        pedestrian score is used, scaled down by
        ``person_fallback_scale``, because on its own it is weak evidence for a
        *vehicle* accident.
        """
        cfg = self.config
        vehicle_ids = {t.track_id for t in tracks if t.is_vehicle}
        valid = [f for f in features.values() if f.valid]
        if vehicle_ids:
            pool = [f for tid, f in features.items() if tid in vehicle_ids and f.valid]
        else:
            pool = valid
        if not pool:
            return 0.0
        if cfg.aggregate == "mean":
            value = float(sum(f.score for f in pool) / len(pool))
        else:
            value = float(max(f.score for f in pool))
        if not vehicle_ids:
            value *= float(cfg.person_fallback_scale)
        return clamp(value)

    def aggregate_ids(self, features: Dict[int, MotionFeatures], tracks: Sequence[Track]) -> List[int]:
        """Ids of the vehicle tracks that contributed to :meth:`aggregate`."""
        cfg = self.config
        vehicle_ids = {t.track_id for t in tracks if t.is_vehicle}
        candidates = [f for tid, f in features.items() if (not vehicle_ids or tid in vehicle_ids) and f.valid]
        if not candidates:
            return []
        if cfg.aggregate == "mean":
            mean = sum(f.score for f in candidates) / len(candidates)
            return sorted(f.track_id for f in candidates if f.score >= mean * 0.5)
        best = max(candidates, key=lambda f: f.score)
        return [best.track_id] if best.score > 0.0 else []

    # ------------------------------------------------------------------ #
    def _analyze_track(
        self,
        track: Track,
        scale: float,
        timestamp: float,
        global_velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> MotionFeatures:
        cfg = self.config
        now = float(timestamp or track.timestamp)
        features = MotionFeatures(
            track_id=track.track_id,
            position=track.center,
            history_length=track.history_length,
        )
        # all motion features are measured *relative to the camera*
        gvx, gvy = global_velocity
        rel_vx = track.velocity[0] - gvx
        rel_vy = track.velocity[1] - gvy
        rel_speed = math.hypot(rel_vx, rel_vy)
        features.speed = rel_speed / scale
        features.velocity = (rel_vx / scale, rel_vy / scale)
        features.heading_deg = math.degrees(math.atan2(rel_vy, rel_vx)) if rel_speed > 1e-6 else 0.0
        features.moving = features.speed > cfg.stop_exit_speed
        features.stationary_seconds = track.stopped_for(
            seconds=2.0, speed_threshold_norm=cfg.stop_exit_speed, now=now
        )

        if track.history_length < cfg.min_history:
            # Not enough evidence to say anything: an early track must not
            # look "abnormal" simply because it has one velocity sample.
            features.valid = False
            features.reasons = ["insufficient tracking history"]
            return features

        # ---- baseline: how fast the object was moving before this moment ---
        baseline_velocity = track.average_velocity(cfg.baseline_lag_seconds + cfg.baseline_seconds, now)
        baseline_speed = math.hypot(baseline_velocity[0] - gvx, baseline_velocity[1] - gvy) / scale
        if baseline_speed <= 0.0:
            baseline_speed = rel_speed / scale
        features.baseline_speed = baseline_speed

        # ---- sudden stop ----------------------------------------------------
        # Uses the *instantaneous* (unsmoothed) speeds: a crash is an abrupt
        # change, and the tracker's EMA would smear it across several frames.
        window = track.recent_points(cfg.stop_window_frames)
        if len(window) >= 2:
            recent = [(math.hypot(p.instant_velocity[0] - gvx, p.instant_velocity[1] - gvy) / scale) for p in window]
            speed_before = max(recent[:-1])
            speed_now = recent[-1]
            if speed_before > 1e-9:
                drop = clamp((speed_before - speed_now) / speed_before)
            else:
                drop = 0.0
            features.speed_drop_ratio = drop
            features.instant_speed = speed_now
            features.speed_before = speed_before
            moving_gate = smoothstep(cfg.stop_exit_speed, cfg.stop_entry_speed, speed_before)
            features.sudden_stop = moving_gate * smoothstep(_MIN_DROP, _MIN_DROP + 0.6, drop)

        # ---- sudden turn ----------------------------------------------------
        mean_velocity = track.average_velocity(cfg.turn_window_seconds, now)
        mean_vec = (mean_velocity[0] - gvx, mean_velocity[1] - gvy)
        current_vec = (rel_vx, rel_vy)
        if math.hypot(*mean_vec) > 1e-6 and math.hypot(*current_vec) > 1e-6:
            features.turn_deg = angle_between_deg(mean_vec, current_vec)
            turn_gate = smoothstep(cfg.min_turn_speed, cfg.min_turn_speed * 2.0, features.speed)
            features.sudden_turn = turn_gate * smoothstep(cfg.turn_threshold_deg * 0.6, cfg.turn_threshold_deg, features.turn_deg)

        # ---- acceleration ----------------------------------------------------
        if len(window) >= 2:
            span = max(1e-3, float(window[-1].timestamp - window[0].timestamp))
            dvx = (window[-1].instant_velocity[0] - window[0].instant_velocity[0]) / scale
            dvy = (window[-1].instant_velocity[1] - window[0].instant_velocity[1]) / scale
            features.acceleration = math.hypot(dvx, dvy) / span
        features.acceleration_score = _score_accel(features.acceleration, cfg.accel_scale)

        # ---- jitter (high-frequency position noise) ---------------------------
        points = track.recent_points(min(track.history_length, 6))
        if len(points) >= 3:
            total = 0.0
            for i in range(1, len(points) - 1):
                ax, ay = points[i - 1].center
                bx, by = points[i].center
                cx, cy = points[i + 1].center
                total += abs(cx - 2 * bx + ax) + abs(cy - 2 * by + ay)
            mean_jitter = total / (len(points) - 2) / scale
            features.jitter = clamp(smoothstep(cfg.jitter_scale * 0.5, cfg.jitter_scale * 2.0, mean_jitter))

        # ---- displacement ----------------------------------------------------
        features.displacement = track.displacement(cfg.window_seconds, now) / scale

        # ---- smoothing + weighted score --------------------------------------
        features.score = self._smooth_and_score(track.track_id, features)
        features.reasons = self._reasons(features)
        return features

    # ------------------------------------------------------------------ #
    def _smooth_and_score(self, track_id: int, features: MotionFeatures) -> float:
        """EMA-smooth the four terms across frames, then weight them.

        Smoothing matters: a single noisy frame must not spike the score, and
        an already-abnormal object should stay abnormal for a few frames so the
        temporal fusion module can see persistence.
        """
        cfg = self.config
        alpha = cfg.ema_alpha
        previous = self._ema.get(track_id, {})
        accel_score = features.acceleration_score
        terms = {
            "sudden_stop": features.sudden_stop,
            "sudden_turn": features.sudden_turn,
            "acceleration": accel_score,
            "jitter": features.jitter,
        }
        weights = cfg.weights or {}
        total_weight = 0.0
        total = 0.0
        for name, value in terms.items():
            weight = float(weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            smoothed = value if name not in previous else previous[name] * (1.0 - alpha) + value * alpha
            previous[name] = smoothed
            total += smoothed * weight
            total_weight += weight
        previous["score"] = total / total_weight if total_weight > 0 else 0.0
        self._ema[track_id] = previous
        # drop state for tracks that disappeared
        if len(self._ema) > 512:  # pragma: no cover - memory guard
            for key in list(self._ema):
                if key != track_id:
                    del self._ema[key]
        return clamp(total / total_weight if total_weight > 0 else 0.0)

    @staticmethod
    def _reasons(features: MotionFeatures) -> List[str]:
        reasons: List[str] = []
        if features.sudden_stop >= 0.35:
            reasons.append("Sudden stop after sustained motion")
        if features.sudden_turn >= 0.35:
            reasons.append("Sudden direction change while moving")
        if features.acceleration_score >= 0.5:
            reasons.append("Sudden vehicle velocity change")
        if features.jitter >= 0.5:
            reasons.append("Unstable / vibrating motion")
        return reasons


def _score_accel(acceleration: float, scale: float) -> float:
    """Normalise an acceleration magnitude with a soft knee."""
    if scale <= 0:
        return 0.0
    return clamp(smoothstep(scale * 0.4, scale * 2.0, acceleration))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) * 0.5
