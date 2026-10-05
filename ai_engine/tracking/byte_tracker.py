"""Multi-object tracking with stable ids (Phase 5).

Two interchangeable backends:

``tracking.backend = "safevision"`` (default)
    A self-contained ByteTrack implementation: Kalman prediction + **two-stage
    association** (high-confidence detections first, then low-confidence
    detections against the leftovers).  The second stage is what makes ByteTrack
    robust to blurred / partially occluded vehicles, which matters a lot for
    accident footage.

``tracking.backend = "ultralytics"``
    Detection *and* tracking run inside Ultralytics (``model.track``); we adopt
    the supplied ids and keep the history/velocity bookkeeping here.

Both backends produce the same :class:`Track` objects, so nothing downstream
changes.  Velocity for the analyzers is always computed from *measurements*
(EMA of consecutive centre differences); the Kalman filter is used for
*prediction during association only* - it is more stable than a filtered
velocity once a track has been mismatched.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai_engine.config.classes import TrackingConfig
from ai_engine.detection.detection_types import (
    Detection,
    VEHICLE_CLASS_NAMES,
    normalize_class_name,
)
from ai_engine.utils.geometry import ema as ema_step
from ai_engine.utils.geometry import interpolate_bbox, xyxy_iou
from ai_engine.utils.logging_utils import get_logger

__all__ = ["ByteTracker", "Track", "TrackPoint", "TrackerError", "class_group"]

_LOGGER = get_logger("tracking")
_LARGE_COST = 1.0e5
_LARGE_THRESHOLD = 1.0e4


class TrackerError(RuntimeError):
    """Tracking failed in a way the caller must know about."""


def class_group(name: str) -> str:
    """Coarse class group used for association gating.

    ``car``/``truck``/``bus`` all map to ``vehicle`` so a mislabelled box does
    not break a track, while a person can never steal a vehicle's id.
    """
    normalized = normalize_class_name(name)
    return "vehicle" if normalized in VEHICLE_CLASS_NAMES else normalized


# --------------------------------------------------------------------------- #
# Kalman filter (constant-velocity, centre state)
# --------------------------------------------------------------------------- #
class _CenterKalman:
    """4-state constant-velocity filter on the box centre ``[cx, cy, vx, vy]``."""

    __slots__ = ("x", "F", "H", "Q", "R", "P")

    def __init__(self, center: Sequence[float], process_noise: float = 1.0, measurement_noise: float = 10.0) -> None:
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.x[0, 0] = float(center[0])
        self.x[1, 0] = float(center[1])

        self.F = np.eye(4, dtype=np.float64)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0

        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        self.Q = np.eye(4, dtype=np.float64) * float(process_noise)
        self.Q[2, 2] *= 0.01  # velocity is much more stable than position
        self.Q[3, 3] *= 0.01

        self.R = np.eye(2, dtype=np.float64) * float(measurement_noise)
        self.P = np.eye(4, dtype=np.float64) * 100.0

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def correct(self, center: Sequence[float]) -> None:
        z = np.asarray([[float(center[0])], [float(center[1])]], dtype=np.float64)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        try:
            gain = self.P @ self.H.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:  # pragma: no cover - singular R/P is not expected
            return
        self.x = self.x + gain @ y
        identity = np.eye(4, dtype=np.float64)
        self.P = (identity - gain @ self.H) @ self.P

    def velocity(self) -> Tuple[float, float]:
        """Kalman velocity in px/frame (diagnostics only)."""
        return float(self.x[2, 0]), float(self.x[3, 0])


# --------------------------------------------------------------------------- #
# track model
# --------------------------------------------------------------------------- #
@dataclass
class TrackPoint:
    """One historical observation of a track.

    Two velocities are stored on purpose:

    ``velocity`` / ``speed``
        exponentially smoothed - stable, used for display and for trajectory
        shape.
    ``instant_velocity`` / ``instant_speed``
        the raw difference between this measurement and the previous one -
        used by the motion analyzer, because a *sudden stop* is by definition an
        abrupt change that smoothing would hide.
    """

    frame_index: int
    timestamp: float
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)   # px/s (smoothed)
    speed: float = 0.0                            # px/s (smoothed magnitude)
    instant_velocity: Tuple[float, float] = (0.0, 0.0)
    instant_speed: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "bbox": [round(float(v), 2) for v in self.bbox],
            "speed_px_s": round(float(self.speed), 3),
            "instant_speed_px_s": round(float(self.instant_speed), 3),
        }


@dataclass
class Track:
    """A tracked object with its motion history."""

    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    frame_index: int = 0
    timestamp: float = 0.0
    normalized_name: str = ""
    scale: float = 1.0                       # frame diagonal at the last update
    state: str = "new"                       # new | tracked | lost
    hits: int = 1
    age_frames: int = 1
    misses: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)   # px/s, EMA of measurements
    first_frame: int = 0
    first_timestamp: float = 0.0
    #: Last *measured* centre/time.  Velocity must be differenced against a
    #: measurement, never against the Kalman-predicted position, otherwise the
    #: prediction error (not the motion) becomes the "velocity".
    last_measured_center: Tuple[float, float] = (0.0, 0.0)
    last_measured_time: float = 0.0
    history: Deque[TrackPoint] = field(default_factory=deque)
    id_source: str = "safevision"            # or "ultralytics"

    # ------------------------------------------------------------------ #
    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox[0] + (self.bbox[2] - self.bbox[0]) * 0.5, self.bbox[1] + (self.bbox[3] - self.bbox[1]) * 0.5

    @property
    def center_x(self) -> float:
        return self.center[0]

    @property
    def center_y(self) -> float:
        return self.center[1]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_vehicle(self) -> bool:
        return self.normalized_name in VEHICLE_CLASS_NAMES

    @property
    def is_person(self) -> bool:
        return self.normalized_name == "person"

    @property
    def age_seconds(self) -> float:
        return max(0.0, float(self.timestamp) - float(self.first_timestamp))

    @property
    def frames_since_update(self) -> int:
        return int(self.misses)

    @property
    def speed_px_s(self) -> float:
        """Smoothed speed in pixels per second."""
        return float(math.hypot(self.velocity[0], self.velocity[1]))

    @property
    def speed_norm(self) -> float:
        """Smoothed speed in *frame diagonals per second* (resolution free)."""
        if self.scale <= 0:
            return 0.0
        return self.speed_px_s / self.scale

    @property
    def heading_deg(self) -> float:
        """Direction of travel in image coordinates (0 deg = +x, CCW positive)."""
        if self.speed_px_s < 1e-6:
            return float("nan")
        return math.degrees(math.atan2(self.velocity[1], self.velocity[0]))

    # -- history helpers ------------------------------------------------ #
    @property
    def points(self) -> List[TrackPoint]:
        return list(self.history)

    @property
    def positions(self) -> List[Tuple[float, float]]:
        return [p.center for p in self.history]

    @property
    def history_length(self) -> int:
        return len(self.history)

    def points_between(self, start: float, end: float) -> List[TrackPoint]:
        return [p for p in self.history if start <= p.timestamp <= end]

    def recent_points(self, count: int) -> List[TrackPoint]:
        if count <= 0:
            return []
        return list(self.history)[-count:]

    def average_velocity(self, seconds: float, now: Optional[float] = None) -> Tuple[float, float]:
        """Mean velocity over the last ``seconds`` (px/s).

        Used for "what was this vehicle doing *before* the event", which is the
        reference the collision analyzer compares against.
        """
        now = float(now if now is not None else self.timestamp)
        points = self.points_between(now - float(seconds), now)
        if len(points) < 2:
            return (0.0, 0.0)
        vx = [p.velocity[0] for p in points]
        vy = [p.velocity[1] for p in points]
        return (float(sum(vx) / len(vx)), float(sum(vy) / len(vy)))

    def average_speed(self, seconds: float, now: Optional[float] = None) -> float:
        vx, vy = self.average_velocity(seconds, now)
        return float(math.hypot(vx, vy))

    def displacement(self, seconds: float, now: Optional[float] = None) -> float:
        """Distance travelled (path length) over the last ``seconds``, in px."""
        now = float(now if now is not None else self.timestamp)
        points = self.points_between(now - float(seconds), now)
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(points)):
            ax, ay = points[i - 1].center
            bx, by = points[i].center
            total += math.hypot(bx - ax, by - ay)
        return total

    def net_displacement(self, seconds: float, now: Optional[float] = None) -> float:
        """Straight-line distance between the oldest and newest point, in px."""
        now = float(now if now is not None else self.timestamp)
        points = self.points_between(now - float(seconds), now)
        if len(points) < 2:
            return 0.0
        ax, ay = points[0].center
        bx, by = points[-1].center
        return float(math.hypot(bx - ax, by - ay))

    def is_moving(self, threshold_norm: float = 0.01, now: Optional[float] = None) -> bool:
        return self.speed_norm > float(threshold_norm)

    def stopped_for(self, seconds: float, speed_threshold_norm: float = 0.012, now: Optional[float] = None) -> float:
        """How long the track has been slower than ``speed_threshold_norm``.

        The result is capped at ``seconds`` and is always slightly *less* than
        the requested window (the oldest qualifying sample is the first one at
        or after the cut-off), so callers must not compare it with ``>= seconds``.
        Use :meth:`has_been_stopped_for` for that.
        """
        now = float(now if now is not None else self.timestamp)
        cutoff = now - float(seconds)
        stopped_since: Optional[float] = None
        for point in self.history:
            if point.timestamp < cutoff:
                continue
            threshold_px = float(speed_threshold_norm) * self.scale
            if point.speed <= threshold_px:
                stopped_since = point.timestamp if stopped_since is None else stopped_since
            else:
                stopped_since = None
        if stopped_since is None:
            return 0.0
        return max(0.0, now - stopped_since)

    def has_been_stopped_for(
        self,
        seconds: float,
        speed_threshold_norm: float = 0.012,
        now: Optional[float] = None,
        tolerance: float = 0.9,
    ) -> bool:
        """``True`` when the track has been continuously slow for the window.

        ``tolerance`` absorbs the sampling gap: at 15 fps the oldest sample
        inside a 0.35 s window is at most one frame period away from the edge.
        """
        window = max(1e-3, float(seconds))
        return self.stopped_for(window, speed_threshold_norm, now) >= window * float(tolerance)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "class_name": self.class_name,
            "class": self.normalized_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "speed_norm": round(self.speed_norm, 5),
            "state": self.state,
            "hits": int(self.hits),
            "age_frames": int(self.age_frames),
            "history_points": int(self.history_length),
        }


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
class ByteTracker:
    """ByteTrack-style tracker producing stable :class:`Track` ids.

    Usage::

        tracker = ByteTracker(TrackingConfig())
        tracks = tracker.update(detections, frame_size=(w, h), frame_index=i, timestamp=t)
    """

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frame_index = -1
        self._frame_scale = 1.0
        self._kalman: Dict[int, _CenterKalman] = {}
        self._last_seen: Dict[int, int] = {}
        self.total_created = 0
        self.total_lost = 0

    # ------------------------------------------------------------------ #
    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.state != "lost"]

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._kalman.clear()
        self._last_seen.clear()
        self._next_id = 1
        self._frame_index = -1
        self.total_created = 0
        self.total_lost = 0

    def get(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(int(track_id))

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Sequence[Detection],
        frame_size: Tuple[int, int],
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> List[Track]:
        """Associate ``detections`` with existing tracks and return live tracks.

        ``frame_size`` is ``(width, height)``.  Only tracks that have been seen
        at least ``tracking.min_hits`` times are returned, which keeps the
        analyzers free of one-frame noise.
        """
        width = float(frame_size[0]) if frame_size and frame_size[0] else 1.0
        height = float(frame_size[1]) if frame_size and frame_size[1] else 1.0
        self._frame_scale = max(1.0, math.hypot(width, height))
        self._frame_index = int(frame_index)

        usable = [d for d in detections if not d.is_accident_class and d.bbox.area >= self.config.min_box_area]
        if len(usable) != len(detections):
            _LOGGER.debug("Dropped %d detection(s) (accident-class or below min_box_area)", len(detections) - len(usable))

        self._prune_dead()

        if self.config.backend == "ultralytics" and any(d.track_id is not None for d in usable):
            self._update_from_external_ids(usable, frame_index, timestamp)
        else:
            self._update_associate(usable, frame_index, timestamp)

        return [t for t in self._tracks.values() if t.state != "lost" and t.hits >= self.config.min_hits]

    # -- pruning -------------------------------------------------------- #
    def _prune_dead(self) -> None:
        dead: List[int] = []
        for track_id, track in self._tracks.items():
            if self._frame_index - self._last_seen.get(track_id, -1) > self.config.max_age:
                dead.append(track_id)
        for track_id in dead:
            self._tracks.pop(track_id, None)
            self._kalman.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self.total_lost += 1

    # -- backend A: own two-stage association --------------------------- #
    def _update_associate(self, detections: Sequence[Detection], frame_index: int, timestamp: float) -> None:
        cfg = self.config

        for track in self._tracks.values():
            if cfg.use_kalman:
                kalman = self._kalman.get(track.track_id)
                if kalman is not None:
                    px, py = kalman.predict()
                    cx, cy = track.center
                    predicted_box = interpolate_bbox(track.bbox, px - cx, py - cy)
                    track.bbox = predicted_box  # type: ignore[assignment]
            track.misses += 1
            track.state = "lost"
            track.age_frames += 1

        high = [d for d in detections if d.confidence >= cfg.conf_high]
        low = [d for d in detections if cfg.conf_low <= d.confidence < cfg.conf_high]

        # stage 1: high-confidence detections
        matches, unmatched_tracks, unmatched_high = self._associate(high, cfg.match_threshold)
        for track_id, det_idx in matches:
            self._apply(self._tracks[track_id], high[det_idx], frame_index, timestamp)

        # stage 2: low-confidence detections keep the still-unmatched tracks
        # alive (this is what gives ByteTrack its robustness to blur/occlusion)
        matches2, unmatched_tracks2, _ = self._associate(low, cfg.match_threshold_low, restrict=set(unmatched_tracks))
        for track_id, det_idx in matches2:
            self._apply(self._tracks[track_id], low[det_idx], frame_index, timestamp)

        # stage 3 (SafeVision addition): recover tracks that IoU association
        # lost because the object *stopped abruptly* - the Kalman prediction
        # overshoots, IoU collapses, and a plain ByteTrack would emit a brand
        # new id exactly when a crash happens.  Gate on centre distance
        # instead, normalised by the two objects' own size.
        if cfg.use_recovery_association and unmatched_tracks2 and unmatched_high:
            remaining = [high[i] for i in unmatched_high]
            matches3, _, _ = self._associate_by_distance(unmatched_tracks2, remaining, cfg.recovery_distance)
            recovered = {pos for _, pos in matches3}
            for track_id, det_pos in sorted(matches3, key=lambda pair: pair[1]):
                self._apply(self._tracks[track_id], remaining[det_pos], frame_index, timestamp)
            # detections consumed by the recovery must not also start a new track
            unmatched_high = [i for pos, i in enumerate(unmatched_high) if pos not in recovered]

        # new tracks are created from unmatched *high* detections only: a lone
        # low-confidence detection is usually noise, not a new object.
        for det_idx in unmatched_high:
            self._spawn(high[det_idx], frame_index, timestamp)

    # -- backend B: adopt ids from an external tracker ------------------ #
    def _update_from_external_ids(self, detections: Sequence[Detection], frame_index: int, timestamp: float) -> None:
        """Consume ``Detection.track_id`` values (Ultralytics ByteTrack).

        We still own history, velocity and cleanup, but ids come from outside,
        which is slightly better at recovering long occlusions.
        """
        seen: set[int] = set()
        for detection in detections:
            tid = detection.track_id
            if tid is None:
                continue
            tid = int(tid)
            seen.add(tid)
            track = self._tracks.get(tid)
            if track is None:
                track = self._spawn(detection, frame_index, timestamp, track_id=tid, source="ultralytics")
            else:
                self._apply(track, detection, frame_index, timestamp)

        for tid, track in self._tracks.items():
            if tid not in seen:
                track.misses += 1
                track.state = "lost"
                track.age_frames += 1

    # -- association ---------------------------------------------------- #
    def _associate(
        self,
        detections: Sequence[Detection],
        iou_threshold: float,
        restrict: Optional[set] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to (predicted) tracks.

        Returns ``(matches, unmatched_track_ids, unmatched_det_indices)`` where
        each match is ``(track_id, detection_index)``.  Pairs that fail the IoU
        gate or the class gate stay unmatched, so a low-quality detection can
        never steal a track id.
        """
        cfg = self.config
        candidates = [tid for tid in self._tracks if restrict is None or tid in restrict]
        if not candidates or not detections:
            return [], list(candidates), list(range(len(detections)))

        cost = np.full((len(candidates), len(detections)), _LARGE_COST, dtype=np.float64)
        for i, tid in enumerate(candidates):
            track = self._tracks[tid]
            track_group = class_group(track.class_name)
            for j, det in enumerate(detections):
                if cfg.class_gate and class_group(det.class_name) != track_group:
                    continue
                iou = xyxy_iou(track.bbox, det.bbox)
                if iou < iou_threshold:
                    continue
                cost[i, j] = 1.0 - (iou * float(det.confidence) if cfg.score_weighted_iou else iou)

        matches: List[Tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for i, j in _solve_assignment(cost):
            if cost[i, j] >= _LARGE_THRESHOLD:
                continue
            matches.append((candidates[i], j))
            used_tracks.add(i)
            used_dets.add(j)

        unmatched_tracks = [candidates[i] for i in range(len(candidates)) if i not in used_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _associate_by_distance(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        max_distance: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Distance-based association used to recover abruptly stopped tracks.

        Cost is the centre distance normalised by the two objects' own
        diagonal, so a car and a motorcycle are judged on the same relative
        scale.  Pairs further apart than ``max_distance`` object-diagonals are
        gated out.
        """
        if not track_ids or not detections or max_distance <= 0:
            return [], list(track_ids), list(range(len(detections)))

        cost = np.full((len(track_ids), len(detections)), _LARGE_COST, dtype=np.float64)
        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            track_group = class_group(track.class_name)
            tcx, tcy = track.center
            tdiag = math.hypot(track.width, track.height)
            for j, det in enumerate(detections):
                if self.config.class_gate and class_group(det.class_name) != track_group:
                    continue
                dcx, dcy = det.bbox.center
                ddiag = math.hypot(det.bbox.width, det.bbox.height)
                reference = max(1.0, 0.5 * (tdiag + ddiag))
                distance = math.hypot(dcx - tcx, dcy - tcy) / reference
                if distance > max_distance:
                    continue
                cost[i, j] = distance / max_distance

        matches: List[Tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for i, j in _solve_assignment(cost):
            if cost[i, j] >= _LARGE_THRESHOLD:
                continue
            matches.append((track_ids[i], j))
            used_tracks.add(i)
            used_dets.add(j)

        unmatched_tracks = [tid for i, tid in enumerate(track_ids) if i not in used_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    # -- create / update ------------------------------------------------ #
    def _spawn(
        self,
        detection: Detection,
        frame_index: int,
        timestamp: float,
        track_id: Optional[int] = None,
        source: str = "safevision",
    ) -> Track:
        tid = int(track_id) if track_id is not None else self._next_id
        if track_id is None:
            self._next_id += 1
        elif tid >= self._next_id:
            self._next_id = tid + 1

        track = Track(
            track_id=tid,
            class_name=detection.class_name,
            normalized_name=detection.normalized_name or normalize_class_name(detection.class_name),
            bbox=(float(detection.bbox.x1), float(detection.bbox.y1), float(detection.bbox.x2), float(detection.bbox.y2)),
            confidence=float(detection.confidence),
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            scale=self._frame_scale,
            state="new",
            hits=1,
            age_frames=1,
            first_frame=int(frame_index),
            first_timestamp=float(timestamp),
            id_source=source,
        )
        track.history.append(TrackPoint(frame_index, float(timestamp), track.center, track.bbox))
        track.last_measured_center = track.center
        track.last_measured_time = float(timestamp)
        self._tracks[tid] = track
        self._last_seen[tid] = int(frame_index)
        if self.config.use_kalman:
            self._kalman[tid] = _CenterKalman(track.center)
        self.total_created += 1
        return track

    def _apply(self, track: Track, detection: Detection, frame_index: int, timestamp: float) -> None:
        cfg = self.config
        previous_center = track.last_measured_center
        previous_time = track.last_measured_time
        previous_speed = track.speed_px_s

        cx, cy = detection.bbox.center
        if cfg.use_kalman:
            kalman = self._kalman.get(track.track_id)
            if kalman is not None:
                kalman.correct((cx, cy))

        track.bbox = (float(detection.bbox.x1), float(detection.bbox.y1), float(detection.bbox.x2), float(detection.bbox.y2))
        track.confidence = float(detection.confidence)
        track.class_name = detection.class_name
        track.normalized_name = detection.normalized_name or normalize_class_name(detection.class_name)
        track.frame_index = int(frame_index)
        track.timestamp = float(timestamp)
        track.scale = self._frame_scale
        track.hits += 1
        track.misses = 0
        track.state = "tracked"
        track.age_frames = max(1, int(frame_index) - track.first_frame + 1)

        dt = float(timestamp) - float(previous_time)
        if dt > 1e-6:
            instant_vx = (cx - previous_center[0]) / dt
            instant_vy = (cy - previous_center[1]) / dt
        else:
            instant_vx, instant_vy = track.velocity
        vx = ema_step(track.velocity[0], instant_vx, cfg.ema_alpha)
        vy = ema_step(track.velocity[1], instant_vy, cfg.ema_alpha)
        # magnitude is smoothed separately: a direction flip must not instantly
        # zero the speed, otherwise "sudden stop" triggers on every turn.
        speed = ema_step(previous_speed, math.hypot(vx, vy), cfg.ema_alpha)
        track.velocity = (vx, vy)
        track.last_measured_center = (cx, cy)
        track.last_measured_time = float(timestamp)

        track.history.append(
            TrackPoint(
                frame_index=int(frame_index),
                timestamp=float(timestamp),
                center=track.center,
                bbox=track.bbox,
                velocity=track.velocity,
                speed=speed,
                instant_velocity=(instant_vx, instant_vy),
                instant_speed=math.hypot(instant_vx, instant_vy),
            )
        )
        self._trim_history(track, float(timestamp))
        self._last_seen[track.track_id] = int(frame_index)

    def _trim_history(self, track: Track, now: float) -> None:
        window = float(self.config.history_seconds)
        max_items = int(self.config.max_history)
        while len(track.history) > max_items:
            track.history.popleft()
        if window > 0:
            cutoff = now - window
            while len(track.history) > 1 and track.history[0].timestamp < cutoff:
                track.history.popleft()

    def __repr__(self) -> str:  # pragma: no cover
        return f"ByteTracker(backend={self.config.backend!r}, live={len(self._tracks)}, created={self.total_created})"


def _solve_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian assignment, with a greedy fallback if SciPy is unavailable."""
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except ImportError:  # pragma: no cover - scipy is in requirements.txt
        pairs: List[Tuple[int, int]] = []
        work = cost.copy()
        while work.size:
            idx = int(np.argmin(work))
            i, j = divmod(idx, work.shape[1])
            if work[i, j] >= _LARGE_THRESHOLD:
                break
            pairs.append((i, j))
            work[i, :] = _LARGE_COST
            work[:, j] = _LARGE_COST
        return pairs
