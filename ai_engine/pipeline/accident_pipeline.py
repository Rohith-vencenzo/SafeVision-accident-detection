"""The accident detection pipeline (orchestration layer).

Wires the phases together and defines the **integration contract** for the rest
of the project (Phase 27)::

    from ai_engine.pipeline.accident_pipeline import AccidentPipeline

    pipeline = AccidentPipeline(load_config("config.yaml"))
    result   = pipeline.process_frame(frame)          # -> FrameResult
    incidents = pipeline.process_video("clip.mp4")     # -> RunResult (incidents)

Every :class:`FrameResult` carries the full evidence breakdown, so a UI or a
backend can subscribe per frame if it wants to; every confirmed incident is
returned as a plain JSON dict (see
:mod:`ai_engine.schemas.incident_result`).

This module never talks to a backend, a database, a socket or a notification
service.  That is the teammate's 40%.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ai_engine.collision.collision_analyzer import CollisionAnalyzer, CollisionEvent
from ai_engine.config.classes import Config
from ai_engine.config.settings import load_config, project_path
from ai_engine.detection.detection_types import DetectionResult
from ai_engine.detection.yolo_detector import BaseDetector, DetectionError, YoloDetector, resolve_device
from ai_engine.emergency.dispatcher import DemoCallDispatcher
from ai_engine.emergency.recipient import mask_phone
from ai_engine.location.location_provider import LocationProviderRegistry
from ai_engine.evidence.evidence_capture import EvidenceCapture, EvidenceSession
from ai_engine.location.location_manager import LocationManager
from ai_engine.motion.motion_analyzer import MotionAnalyzer, MotionFeatures
from ai_engine.reporting.incident_report import IncidentReportGenerator
from ai_engine.schemas.enums import DetectionState, IncidentStatus
from ai_engine.telemetry.latency import LatencyLedger
from ai_engine.schemas.incident_result import (
    AccidentInfo,
    CameraRef,
    DemoCallInfo,
    DemoSmsInfo,
    EvidenceInfo,
    IncidentResult,
    LocationInfo,
    MediaInfo,
    ObjectsInfo,
    PerformanceInfo,
)
from ai_engine.scoring.accident_scorer import AccidentScore, AccidentScorer
from ai_engine.severity.severity_analyzer import SeverityAnalysis, SeverityAnalyzer, SeverityContext
from ai_engine.temporal.temporal_fusion import EvidenceSnapshot, TemporalFusion, TemporalResult
from ai_engine.tracking.byte_tracker import ByteTracker, Track, TrackerError
from ai_engine.trajectory.trajectory_analyzer import TrajectoryAnalyzer, TrajectoryFeatures, TrajectoryPair
from ai_engine.utils.geometry import clamp
from ai_engine.utils.jsonio import atomic_write_json, to_jsonable
from ai_engine.utils.logging_utils import get_logger
from ai_engine.utils.timeutil import now_iso
from ai_engine.verification.false_alarm_prevention import (
    FalseAlarmPrevention,
    VerificationResult,
    assess_negative_evidence,
)
from ai_engine.video.source import VideoReader, VideoSourceError
from ai_engine.visualization.overlay import OverlayRenderer

__all__ = ["AccidentPipeline", "FrameResult", "PerfStats", "RunResult", "IncidentCallback"]

_LOGGER = get_logger("pipeline")

IncidentCallback = Callable[[Dict[str, Any]], None]


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #
@dataclass
class PerfStats:
    """Per-frame timings (Phase 21)."""

    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    analysis_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    source_fps: float = 0.0
    frames_processed: int = 0
    frame_size: Tuple[int, int] = (0, 0)
    detector: Optional[str] = None
    device: Optional[str] = None
    detection_failures: int = 0
    dropped_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": round(self.fps, 2),
            "source_fps": round(self.source_fps, 2),
            "inference_ms": round(self.inference_ms, 3),
            "tracking_ms": round(self.tracking_ms, 3),
            "analysis_ms": round(self.analysis_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "frames_processed": int(self.frames_processed),
            "frame_size": list(self.frame_size),
            "detector": self.detector,
            "device": self.device,
            "detection_failures": int(self.detection_failures),
            "dropped_frames": int(self.dropped_frames),
        }


@dataclass
class FrameResult:
    """Everything known about one processed frame."""

    frame_index: int = 0
    timestamp: float = 0.0
    frame_size: Tuple[int, int] = (0, 0)
    camera_id: Optional[str] = None

    detections: DetectionResult = field(default_factory=DetectionResult)
    tracks: List[Track] = field(default_factory=list)
    motion: Dict[int, MotionFeatures] = field(default_factory=dict)
    trajectories: Dict[int, TrajectoryFeatures] = field(default_factory=dict)
    trajectory_pairs: List[TrajectoryPair] = field(default_factory=list)
    collision: Optional[CollisionEvent] = None
    collision_events: List[CollisionEvent] = field(default_factory=list)
    temporal: TemporalResult = field(default_factory=TemporalResult)
    verification: VerificationResult = field(default_factory=VerificationResult)

    score_detail: AccidentScore = field(default_factory=AccidentScore)
    accident_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    negative_evidence: Dict[str, float] = field(default_factory=dict)
    severity: Optional[SeverityAnalysis] = None

    vehicles_involved: int = 0
    persons_in_scene: int = 0
    object_counts: Dict[str, int] = field(default_factory=dict)

    #: state of the demonstration emergency call (demo mode only); ``{}`` when
    #: the feature is disabled.  Present on every frame while it is wired up,
    #: so a live overlay can follow the call as it progresses.
    demo_call: Dict[str, Any] = field(default_factory=dict)

    performance: PerfStats = field(default_factory=PerfStats)
    warnings: List[str] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    raw_frame: Optional[np.ndarray] = None
    incident: Optional[Dict[str, Any]] = None

    @property
    def state(self) -> str:
        return str(self.verification.state)

    @property
    def status(self) -> str:
        return str(self.verification.status)

    def to_dict(self, include_frame: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
            "frame_size": list(self.frame_size),
            "camera_id": self.camera_id,
            "state": self.state,
            "status": self.status,
            "accident_score": round(float(self.accident_score), 4),
            "component_scores": {k: round(float(v), 4) for k, v in self.component_scores.items()},
            "negative_evidence": {k: round(float(v), 4) for k, v in self.negative_evidence.items()},
            "vehicles_involved": int(self.vehicles_involved),
            "persons_in_scene": int(self.persons_in_scene),
            "object_counts": dict(self.object_counts),
            "detections": self.detections.to_dict(),
            "tracks": [t.to_dict() for t in self.tracks],
            "collision": self.collision.to_dict() if self.collision else None,
            "temporal": self.temporal.to_dict(),
            "verification": self.verification.to_dict(),
            "score": self.score_detail.to_dict(),
            "demo_emergency_call": dict(self.demo_call),
            "performance": self.performance.to_dict(),
            "warnings": list(self.warnings),
        }
        if self.severity is not None:
            payload["severity"] = self.severity.to_dict()
        if self.incident is not None:
            payload["incident"] = self.incident
        if include_frame and self.annotated_frame is not None:
            payload["annotated_frame_shape"] = list(self.annotated_frame.shape)
        return to_jsonable(payload)


@dataclass
class RunResult:
    """Summary of a whole video / stream run."""

    source: str = ""
    frames_processed: int = 0
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    confirmed: int = 0
    false_alarms: int = 0
    elapsed_seconds: float = 0.0
    performance: PerfStats = field(default_factory=PerfStats)
    video_stats: Dict[str, Any] = field(default_factory=dict)
    incidents_file: Optional[str] = None
    output_video: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "source": self.source,
                "frames_processed": int(self.frames_processed),
                "confirmed_incidents": int(self.confirmed),
                "false_alarms": int(self.false_alarms),
                "elapsed_seconds": round(float(self.elapsed_seconds), 3),
                "incidents": self.incidents,
                "transitions": self.transitions,
                "performance": self.performance.to_dict(),
                "video": self.video_stats,
                "incidents_file": self.incidents_file,
                "output_video": self.output_video,
            }
        )


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
class AccidentPipeline:
    """End-to-end accident detection and verification.

    Parameters
    ----------
    config:
        A :class:`~ai_engine.config.classes.Config`.  Loaded from
        ``config.yaml`` when omitted.
    detector:
        Any object with ``detect(frame, frame_index, timestamp) ->
        DetectionResult`` (and optionally ``track(...)``).  Injecting one is how
        the tests run the whole pipeline without downloading YOLO weights.
    on_incident:
        Callback invoked with the incident JSON every time a new incident is
        confirmed.  This is the intended hook for the backend (write to a
        queue, POST to an API, ... - the AI module does not do it for you).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        detector: Optional[BaseDetector] = None,
        tracker: Optional[ByteTracker] = None,
        location_manager: Optional[LocationManager] = None,
        evidence_capture: Optional[EvidenceCapture] = None,
        severity_analyzer: Optional[SeverityAnalyzer] = None,
        report_generator: Optional[IncidentReportGenerator] = None,
        camera_id: Optional[str] = None,
        on_incident: Optional[IncidentCallback] = None,
        source_label: Optional[str] = None,
        emergency: Optional[DemoCallDispatcher] = None,
        on_call_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        ledger: Optional[LatencyLedger] = None,
        source_kind: str = "UNKNOWN",
        on_transition: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = (config or load_config()).validate()
        self.camera_id = camera_id or self.config.location.default_camera_id
        self.source_label = source_label

        self.detector: BaseDetector = detector if detector is not None else YoloDetector(self.config.model)
        self.tracker = tracker if tracker is not None else ByteTracker(self.config.tracking)
        self.motion = MotionAnalyzer(self.config.motion)
        self.trajectory = TrajectoryAnalyzer(self.config.trajectory)
        self.collision = CollisionAnalyzer(self.config.collision)
        self.temporal = TemporalFusion(self.config.temporal)
        self.scorer = AccidentScorer(self.config.scoring)
        self.verification = FalseAlarmPrevention(self.config.verification)
        self.verification.min_consecutive_frames = max(1, int(self.config.temporal.min_consecutive_frames))
        self.severity = severity_analyzer or SeverityAnalyzer(self.config.severity)
        self.reports = report_generator or IncidentReportGenerator()
        self.renderer = OverlayRenderer(self.config.visualization)

        self.locations = location_manager or LocationManager.from_config(self.config.location)
        self.evidence = evidence_capture or EvidenceCapture(self.config.evidence)
        self.pre_event = self.evidence.pre_event_buffer()
        self._callbacks: List[IncidentCallback] = [on_incident] if on_incident else []

        # Demonstration emergency call.  Built only when the feature is enabled
        # (emergency.enabled) and DEMO_MODE is true - otherwise the recipient is
        # never read from the environment at all.
        self._owns_emergency = emergency is None
        self.emergency: Optional[DemoCallDispatcher] = None
        if self.config.emergency.enabled:
            self.emergency = emergency or DemoCallDispatcher(
                self.config.emergency, on_status=on_call_status
            )
            if not self.emergency.demo_mode:
                self.emergency = None
                self._owns_emergency = False
                _LOGGER.info(
                    "demonstration emergency call is off "
                    "(set DEMO_MODE=true in .env to enable it)"
                )
            else:
                usable, why = self.emergency.provider.available()
                _LOGGER.warning(
                    "DEMO MODE is on - an automatic DEMONSTRATION call will be placed "
                    "to %s for every CONFIRMED incident (%s)",
                    self.emergency.raw_recipient() and mask_phone(self.emergency.raw_recipient())
                    or "an unconfigured recipient",
                    self.emergency.provider.name if usable else f"blocked: {why}",
                )

        # runtime state
        self._frame_index = 0
        self._detection_failures = 0
        self._dropped = 0
        self._fps_ema: Optional[float] = None
        self._started = time.monotonic()
        self._closed = False
        self._incidents: List[IncidentResult] = []
        self._transitions: List[Dict[str, Any]] = []
        #: live state-change callback, used by the CLI to render the workflow
        #: as it happens instead of only in the final summary.
        self._on_transition = on_transition
        self._open_evidence: Optional[EvidenceSession] = None
        self._open_incident_id: Optional[str] = None
        self._peak_collision_score = 0.0
        self._abnormal_since: Optional[float] = None
        self._last_pairs: List[TrajectoryPair] = []
        self._last_frame_size: Tuple[int, int] = (0, 0)
        self._event_peak_signals: Dict[str, float] = {
            "closing_speed": 0.0,
            "deceleration": 0.0,
            "displacement": 0.0,
        }
        self._last_result: Optional[FrameResult] = None
        self._device = resolve_device(self.config.model.device)
        self._warmed = False
        # Phase 3: stage timings on a monotonic clock, and an explicit
        # RECORDED / LIVE distinction so offline throughput is never quoted as a
        # live response time.
        self.ledger = ledger if ledger is not None else LatencyLedger()
        # Phase 3: location provenance.  A registered camera position and a
        # live GPS fix are different facts and must never be merged.
        self._location_registry = LocationProviderRegistry.from_config(self.config.location)
        kind = str(source_kind).upper()
        self.ledger.source_kind = source_kind
        self.ledger.mode = "LIVE" if kind in ("LIVE", "WEBCAM", "RTSP") else (
            "RECORDED" if kind == "RECORDED" else "UNKNOWN"
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def add_callback(self, callback: IncidentCallback) -> None:
        """Register an extra incident callback (e.g. a queue writer)."""
        if callback is not None:
            self._callbacks.append(callback)

    def warmup(self) -> None:
        """Load the model and run one throwaway inference (Phase 21)."""
        if self._warmed:
            return
        loader = getattr(self.detector, "warmup", None)
        if callable(loader):
            width = int(self.config.video.target_width or 640)
            loader((width, int(width * 9 / 16)))
        self._warmed = True

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> FrameResult:
        """Run the whole pipeline on a single BGR frame."""
        if self._closed:
            raise RuntimeError("this pipeline has been closed")
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("process_frame() needs a non-empty frame")

        total_start = time.perf_counter()
        self._frame_index = self._frame_index if frame_index is None else int(frame_index)
        # ``stamp`` is the *analysis* timeline: the media position for a file
        # (so the motion maths line up with the footage) and wall clock for a
        # live source.  It is 0-based for a file, so it must never be reported
        # as the incident time - hence ``wall`` below.
        stamp = float(timestamp) if timestamp is not None else time.time()
        wall = time.time()
        height, width = int(frame.shape[0]), int(frame.shape[1])
        frame_size = (width, height)
        self._last_frame_size = frame_size
        warnings: List[str] = []
        perf = PerfStats(frame_size=frame_size, device=self._device, detector=getattr(self.detector, "model_name", None))

        # ---- 1. detection ------------------------------------------------
        t0 = time.perf_counter()
        detections = self._detect(frame, stamp, warnings)
        perf.inference_ms = (time.perf_counter() - t0) * 1000.0
        self.ledger.record("detect", perf.inference_ms)

        # ---- 2. tracking -------------------------------------------------
        t0 = time.perf_counter()
        tracks = self._track(detections, frame_size, stamp, warnings)
        perf.tracking_ms = (time.perf_counter() - t0) * 1000.0
        self.ledger.record("track", perf.tracking_ms)

        # ---- 3. motion / trajectory / collision --------------------------
        t0 = time.perf_counter()
        motion = self.motion.analyze(tracks, frame_size, stamp)
        trajectories, traj_pairs = self.trajectory.analyze(tracks, frame_size, stamp)
        self._last_pairs = traj_pairs
        collision, collision_events = self.collision.analyze(
            tracks, motion, traj_pairs, timestamp=stamp, frame_index=self._frame_index
        )
        perf.analysis_ms = (time.perf_counter() - t0) * 1000.0
        self.ledger.record("analysis", perf.analysis_ms)

        # ---- 4. object / scene evidence ----------------------------------
        vehicles = [t for t in tracks if t.is_vehicle]
        persons = [t for t in tracks if t.is_person]
        involved_ids = self._involved_track_ids(collision, motion, vehicles)
        vehicles_involved = len([tid for tid in involved_ids if self._is_vehicle(tid, vehicles)])
        object_evidence = self._object_evidence(vehicles_involved, persons, collision)
        scene_evidence, scene_detail = self._scene_evidence(tracks, vehicles, motion, stamp)

        # ---- 4b. peak physical signals over the whole event -----------------
        # Severity must describe the *impact*, not the quiet aftermath.  The
        # closing speed and the deceleration are transient, so their peaks are
        # remembered for as long as the event lasts.
        if self.verification.state is DetectionState.NORMAL:
            self._event_peak_signals = {"closing_speed": 0.0, "deceleration": 0.0, "displacement": 0.0}
        else:
            peaks = self._event_peak_signals
            if collision is not None:
                peaks["closing_speed"] = max(peaks["closing_speed"], collision.signals.closing_speed)
            for track in tracks:
                if involved_ids and track.track_id not in set(involved_ids):
                    continue
                features = motion.get(track.track_id)
                if features is not None and features.valid:
                    peaks["deceleration"] = max(peaks["deceleration"], features.acceleration)
                if track.scale > 0:
                    peaks["displacement"] = max(
                        peaks["displacement"], track.net_displacement(0.3, stamp) / track.scale
                    )

        # ---- 5. temporal fusion ------------------------------------------
        snapshot = EvidenceSnapshot(
            frame_index=self._frame_index,
            timestamp=stamp,
            collision_score=collision.score if collision else 0.0,
            motion_score=self.motion.aggregate(motion, tracks),
            trajectory_score=self.trajectory.aggregate(trajectories, tracks),
            object_evidence_score=object_evidence,
            scene_evidence_score=scene_evidence,
            involved_tracks=tuple(sorted(involved_ids)),
            vehicles=len(vehicles),
            persons=len(persons),
            agreement_signals=collision.agreement if collision else 0,
            reasons=list(collision.reasons) if collision else [],
        )
        temporal = self.temporal.push(snapshot)

        # ---- 6. negative evidence + score --------------------------------
        negative = assess_negative_evidence(
            tracks=tracks,
            motion=motion,
            detection_confidence=detections.mean_confidence,
            temporal=temporal,
            involved_track_ids=involved_ids,
            min_history=int(self.config.motion.min_history),
            physical_signals=collision.agreement if collision else 0,
        )
        components = {
            "collision": collision.score if collision else None,
            "motion": snapshot.motion_score,
            "trajectory": snapshot.trajectory_score,
            "temporal": temporal.temporal_score,
            "object_evidence": object_evidence,
            "scene_evidence": scene_evidence,
        }
        score = self.scorer.score(components, negative.strengths(), snapshot.reasons)

        # ---- 7. verification state machine -------------------------------
        t_verify = time.perf_counter()
        verification = self.verification.update(
            score=score,
            temporal=temporal,
            negative=negative,
            vehicles_involved=vehicles_involved,
            persons_involved=len(persons),
            agreement_signals=collision.agreement if collision else 0,
            timestamp=stamp,
        )
        self.ledger.record("verify", (time.perf_counter() - t_verify) * 1000.0)
        if verification.changed:
            event = {
                **verification.transitions[-1].to_dict(),
                "timestamp_iso": now_iso(),
                # The ANALYSIS stage, captured at the moment of the transition.
                # Without this a caller can only print the state name; with it
                # the console and the dashboard can show *why* the decision was
                # made - which signals agreed, what was negative, how long the
                # evidence held.
                "analysis": {
                    "detections": len(detections),
                    "vehicles_involved": int(verification.vehicles_involved),
                    "persons_involved": int(verification.persons_involved),
                    "score": round(float(score.total), 4),
                    "physical_score": round(float(verification.physical_score), 4),
                    "event_score": round(float(verification.event_score), 4),
                    "temporal_score": round(float(verification.temporal_score), 4),
                    "components": {k: round(float(v), 4)
                                   for k, v in (score.components or {}).items()},
                    "agreement_signals": int(verification.agreement_signals),
                    "evidence_sufficient": bool(verification.evidence_sufficient),
                    "negative_evidence": list(verification.negative_evidence),
                    "blocking_reasons": list(verification.blocking_reasons),
                    "frames_in_state": int(verification.frames_in_state),
                    "seconds_in_state": round(float(verification.seconds_in_state), 2),
                    "collision": (
                        {
                            "tracks": list(collision.track_ids),
                            "classes": [collision.class_a, collision.class_b],
                            "score": round(float(collision.score), 4),
                            "reasons": list(collision.reasons),
                        }
                        if collision is not None else None
                    ),
                },
            }
            self._transitions.append(event)
            if self._on_transition is not None:
                try:
                    self._on_transition(event)
                except Exception as exc:  # noqa: BLE001 - a UI bug must not stop a run
                    _LOGGER.warning("on_transition callback failed: %s", exc)
        if verification.state is not DetectionState.NORMAL:
            if self._abnormal_since is None:
                self._abnormal_since = stamp
        elif verification.state is DetectionState.NORMAL:
            self._abnormal_since = None

        # ---- 8. evidence + incident construction -------------------------
        # The frame's timings are completed *before* the incident is built, so
        # the incident's own performance block is meaningful.
        perf.total_ms = (time.perf_counter() - total_start) * 1000.0
        perf.frames_processed = self._frame_index + 1
        perf.fps = self._update_fps(perf.total_ms)
        perf.detection_failures = self._detection_failures
        perf.dropped_frames = self._dropped

        incident_payload: Optional[Dict[str, Any]] = None
        severity_result: Optional[SeverityAnalysis] = None
        t_build = time.perf_counter()
        if verification.is_new_incident:
            severity_result = self._estimate_severity(
                collision=collision,
                involved_ids=involved_ids,
                tracks=tracks,
                persons=len(persons),
                motion=motion,
                stamp=stamp,
            )
            incident_payload = self._build_incident(
                score=score,
                temporal=temporal,
                verification=verification,
                severity=severity_result,
                collision=collision,
                involved_ids=involved_ids,
                tracks=tracks,
                persons=persons,
                detections=detections,
                stamp=stamp,
                wall_stamp=wall,
                frame=frame,
                perf=perf,
                warnings=warnings,
                scene_detail=scene_detail,
            )
        else:
            self._maintain_evidence(verification, collision, frame, stamp)
        self.ledger.record("incident_build", (time.perf_counter() - t_build) * 1000.0)
        self.ledger.frame()

        # ---- 9. annotate + performance ------------------------------------
        result = FrameResult(
            frame_index=self._frame_index,
            timestamp=stamp,
            frame_size=frame_size,
            camera_id=self.camera_id,
            detections=detections,
            tracks=tracks,
            motion=motion,
            trajectories=trajectories,
            trajectory_pairs=traj_pairs,
            collision=collision,
            collision_events=collision_events,
            temporal=temporal,
            verification=verification,
            score_detail=score,
            accident_score=score.total,
            component_scores=score.components,
            negative_evidence=negative.strengths(),
            severity=severity_result,
            vehicles_involved=vehicles_involved,
            persons_in_scene=len(persons),
            object_counts=detections.count_by_class(),
            performance=perf,
            warnings=warnings,
            raw_frame=frame,
        )
        if self.config.visualization.enabled:
            result.annotated_frame = self.renderer.render(frame, result)
        if self.emergency is not None:
            # refreshed every frame so the overlay follows the live call
            result.demo_call = self.emergency.status()
        if incident_payload is not None:
            result.incident = incident_payload

        perf.total_ms = (time.perf_counter() - total_start) * 1000.0
        perf.frames_processed = self._frame_index + 1
        perf.fps = self._update_fps(perf.total_ms)
        perf.detection_failures = self._detection_failures
        perf.dropped_frames = self._dropped
        result.performance = perf
        self._last_result = result

        # keep a rolling copy of raw frames for the pre-event evidence buffer
        if self.evidence.enabled:
            self.pre_event.push(frame, stamp)

        self._frame_index += 1
        return result

    # ------------------------------------------------------------------ #
    def wait_for_notifications(
        self,
        timeout: Optional[float] = None,
    ) -> int:
        """Block until the confirmed incident's call and SMS have finished.

        Returns the number of notifications still unresolved. A caller that
        wants to *report* what happened must call this first: the incident
        payload is a snapshot taken when the incident was built, so before this
        returns it still shows ``INITIATED`` / ``PENDING``.

        Uses ``emergency.notification_wait_seconds`` (not
        ``request_timeout_seconds``) - a real call can ring for ~30 s before the
        provider reports a terminal state.
        """
        if self.emergency is None:
            return 0
        if timeout is None:
            timeout = self.config.emergency.notification_wait_seconds
        return self.emergency.wait(timeout=timeout)

    def _reader_for(self, source: Any) -> Tuple[VideoReader, bool]:
        """Build the reader for ``source``.  Returns ``(reader, is_owned)``."""
        cfg_source = self.config.video.source if source is None else source
        if isinstance(cfg_source, VideoReader):
            return cfg_source, False
        return (
            VideoReader(
                cfg_source,
                self.config.video,
                credentials={
                    "env_username": self.config.location.env_username,
                    "env_password": self.config.location.env_password,
                },
            ),
            True,
        )

    def stream(self, source: Any = None, max_frames: int = 0) -> Iterator[FrameResult]:
        """Yield one :class:`FrameResult` per processed frame.

        The streaming API - use it for a live dashboard or a preview window::

            for result in pipeline.stream("rtsp://camera/stream"):
                print(result.state, result.accident_score)

        A frame that raises is logged and skipped: one bad frame must never end
        a live stream.
        """
        reader, owned = self._reader_for(source)
        limit = int(max_frames or self.config.runtime.max_frames or 0)
        # a file is RECORDED; a webcam/RTSP is LIVE.  Never quote one as the other.
        self.ledger.source_kind = getattr(reader, "kind", "UNKNOWN")
        self.ledger.mode = "RECORDED" if not getattr(reader, "is_live", False) else "LIVE"
        try:
            reader.open()
        except VideoSourceError as exc:
            _LOGGER.error("video source error: %s", exc)
            if owned:
                reader.close()
            return
        try:
            for packet in reader.frames(max_frames=limit):
                self._dropped += int(max(0, packet.dropped))
                try:
                    yield self.process_frame(packet.frame, packet.index, packet.timestamp)
                except Exception as exc:  # noqa: BLE001 - never die on one frame
                    _LOGGER.exception("frame %s failed, skipping: %s", packet.index, exc)
                    continue
        except VideoSourceError as exc:
            _LOGGER.error("video source error: %s", exc)
        except KeyboardInterrupt:  # pragma: no cover
            _LOGGER.info("interrupted by user")
        finally:
            if owned:
                reader.close()
            if self.evidence.enabled:
                self.evidence.prune()

    def process_video(
        self,
        source: Any = None,
        max_frames: int = 0,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
        output_video: Optional[str] = None,
    ) -> RunResult:
        """Process a whole video / stream and return the run summary.

        The batch counterpart of :meth:`stream`.  ``source`` may be a path, a
        webcam index, an RTSP URL or a
        :class:`~ai_engine.video.source.VideoReader`.
        """
        reader, owned = self._reader_for(source)
        started = time.perf_counter()
        run = RunResult(source=str(source if source is not None else self.config.video.source))
        writer: Optional[Dict[str, Any]] = None
        reader_used = reader

        try:
            reader_used.open()
        except VideoSourceError as exc:
            _LOGGER.error("video source error: %s", exc)
            run.video_stats = {"error": str(exc)}
            if owned:
                reader_used.close()
            return run

        try:
            for result in self._stream_with_reader(reader_used, max_frames):
                if writer is None:
                    writer = self._open_writer(result.frame_size, output_video)
                if writer is not None and result.annotated_frame is not None:
                    writer["write"](result.annotated_frame)
                if on_frame is not None:
                    on_frame(result)
                self._maybe_report(result)
            run.output_video = str(writer["path"]) if writer is not None else None
        except KeyboardInterrupt:  # pragma: no cover
            _LOGGER.info("interrupted by user")
        finally:
            if writer is not None:
                writer["release"]()
            if owned:
                reader_used.close()
            if self.evidence.enabled:
                self.evidence.prune()

        run.frames_processed = self._frame_index
        run.incidents = [incident.to_dict() for incident in self._incidents]
        run.transitions = list(self._transitions)
        run.confirmed = self.verification.confirmed_count
        run.false_alarms = self.verification.false_alarm_count
        run.elapsed_seconds = time.perf_counter() - started
        run.performance = PerfStats(
            frames_processed=self._frame_index,
            detector=getattr(self.detector, "model_name", None),
            device=self._device,
            detection_failures=self._detection_failures,
            dropped_frames=self._dropped,
        )
        run.performance.fps = self._fps_ema or 0.0
        if run.video_stats == {}:
            run.video_stats = reader_used.stats()

        if self.config.runtime.write_incidents_file and run.incidents:
            try:
                path = project_path(self.config.runtime.incidents_file)
                run.incidents_file = atomic_write_json(path, run.to_dict())
                _LOGGER.info("wrote %d incident(s) to %s", len(run.incidents), run.incidents_file)
            except OSError as exc:
                _LOGGER.error("could not write %s: %s", self.config.runtime.incidents_file, exc)

        return run

    def _stream_with_reader(self, reader: VideoReader, max_frames: int = 0) -> Iterator[FrameResult]:
        """Iterate an already-opened reader (shared by stream/process_video)."""
        limit = int(max_frames or self.config.runtime.max_frames or 0)
        for packet in reader.frames(max_frames=limit):
            self._dropped += int(max(0, packet.dropped))
            try:
                yield self.process_frame(packet.frame, packet.index, packet.timestamp)
            except Exception as exc:  # noqa: BLE001 - never die on one frame
                _LOGGER.exception("frame %s failed, skipping: %s", packet.index, exc)
                continue

    #: Convenience alias - ``pipeline.run(source)``
    run = process_video

    # ------------------------------------------------------------------ #
    def get_incident_result(self) -> Dict[str, Any]:
        """The most recent incident as plain JSON (``{}`` when there is none)."""
        if not self._incidents:
            return {}
        return self._incidents[-1].to_dict()

    def generate_emergency_incident(self, minimum_score: float = 0.0) -> Dict[str, Any]:
        """The **only** interface the downstream 40% needs (Phase 26).

        Returns the compact alert payload for the most recent confirmed
        incident, or ``{}`` when there is none (or it is below
        ``minimum_score``).
        """
        from ai_engine.schemas.incident_result import generate_emergency_incident

        if not self._incidents:
            return {}
        return generate_emergency_incident(self._incidents[-1], minimum_score=minimum_score)

    @property
    def incidents(self) -> List[IncidentResult]:
        return list(self._incidents)

    @property
    def state(self) -> DetectionState:
        return self.verification.state

    @property
    def dropped_frames(self) -> int:
        """Frames the source could not keep up with.

        On a live source a slow detector shows up here first. The queue is
        bounded and drops rather than queues, because queuing would convert a
        slow pipeline into unbounded latency and an ever-growing memory.
        """
        return int(self._dropped)

    def dashboard_fields(self) -> Dict[str, Any]:
        """Live/recorded mode, stage latency and dropped-frame counters.

        Phase 3 dashboard contract.  Returns the processing mode so a viewer can
        never mistake offline throughput for a live response.
        """
        report = self.ledger.report()
        return {
            "processing_mode": report["mode"],
            "source_kind": report["source_kind"],
            "frames_processed": report["frames_processed"],
            "dropped_frames": self.dropped_frames,
            "stage_latency_p95_ms": {
                stage: stats.get("p95_ms") for stage, stats in report["stages"].items()
            },
            "stage_latency_samples": {
                stage: stats.get("count") for stage, stats in report["stages"].items()
            },
            "incident_timelines_ms": report.get("incident_timelines", {}),
            "latency_methodology": report["methodology"],
        }

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all per-stream state (keeps the loaded model)."""
        self._close_open_evidence("RESET")
        self.tracker.reset()
        self.motion.reset()
        self.trajectory.reset()
        self.collision.reset()
        self.temporal.reset()
        self.verification.reset()
        self.pre_event.clear()
        self._frame_index = 0
        self._detection_failures = 0
        self._dropped = 0
        self._fps_ema = None
        self._started = time.monotonic()
        self._incidents.clear()
        self._transitions.clear()
        self._open_evidence = None
        self._open_incident_id = None
        self._peak_collision_score = 0.0
        self._abnormal_since = None
        self._event_peak_signals = {"closing_speed": 0.0, "deceleration": 0.0, "displacement": 0.0}

    def close(self) -> None:
        if self._closed:
            return
        self._close_open_evidence("STREAM_ENDED")
        self._closed = True
        if self.emergency is not None:
            try:
                self.emergency.close(
                    timeout=self.config.emergency.notification_wait_seconds
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("could not close the demonstration-call dispatcher: %s", exc)
            finally:
                if self._owns_emergency:
                    self.emergency = None
        try:
            self.detector.close()
        except Exception as exc:  # pragma: no cover
            _LOGGER.debug("detector close failed: %s", exc)

    def _close_open_evidence(self, reason: str) -> None:
        """Finalise an evidence session that was still open when the run ended.

        Without this, a stream that stops during VERIFYING would leave a
        directory with frames but no ``meta.json`` - evidence nobody can
        interpret.  The outcome is recorded explicitly instead.
        """
        if self._open_evidence is None:
            return
        try:
            self._open_evidence.finalize(
                {
                    "outcome": "INCOMPLETE",
                    "reason": reason,
                    "note": "the stream ended while the event was still being verified",
                    "captured_at": now_iso(),
                }
            )
            _LOGGER.info(
                "evidence for %s finalised as INCOMPLETE (%s)", self._open_incident_id, reason
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("could not finalise the open evidence session: %s", exc)
        self._open_evidence = None
        self._open_incident_id = None
        self._peak_collision_score = 0.0

    def __enter__(self) -> "AccidentPipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # internals: detection / tracking
    # ------------------------------------------------------------------ #
    def _detect(self, frame: np.ndarray, stamp: float, warnings: List[str]) -> DetectionResult:
        self.warmup()
        track_backend = str(self.config.tracking.backend)
        detector = self.detector
        use_external = track_backend == "ultralytics" and hasattr(detector, "track")
        try:
            if use_external:
                result = detector.track(  # type: ignore[attr-defined]
                    frame,
                    frame_index=self._frame_index,
                    timestamp=stamp,
                    tracker=str(self.config.tracking.ultralytics_tracker),
                )
            else:
                result = detector.detect(frame, frame_index=self._frame_index, timestamp=stamp)
            return result
        except DetectionError as exc:
            self._detection_failures += 1
            if len(warnings) < 5:
                warnings.append(f"detection failed on frame {self._frame_index}: {exc}")
            _LOGGER.warning("detection failed on frame %s: %s", self._frame_index, exc)
            return DetectionResult(frame_index=self._frame_index, timestamp=stamp, frame_size=frame.shape[1::-1])
        except Exception as exc:  # noqa: BLE001 - any detector bug
            self._detection_failures += 1
            if len(warnings) < 5:
                warnings.append(f"unexpected detector error on frame {self._frame_index}: {exc}")
            _LOGGER.exception("unexpected detector error")
            return DetectionResult(frame_index=self._frame_index, timestamp=stamp, frame_size=frame.shape[1::-1])

    def _track(
        self,
        detections: DetectionResult,
        frame_size: Tuple[int, int],
        stamp: float,
        warnings: List[str],
    ) -> List[Track]:
        try:
            return self.tracker.update(
                detections.detections,
                frame_size=frame_size,
                frame_index=self._frame_index,
                timestamp=stamp,
            )
        except TrackerError as exc:
            if len(warnings) < 5:
                warnings.append(f"tracking failed on frame {self._frame_index}: {exc}")
            _LOGGER.warning("tracking failed: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            if len(warnings) < 5:
                warnings.append(f"unexpected tracking error on frame {self._frame_index}: {exc}")
            _LOGGER.exception("unexpected tracking error")
            return []

    # ------------------------------------------------------------------ #
    # internals: evidence scoring helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_vehicle(track_id: int, vehicles: Sequence[Track]) -> bool:
        return any(t.track_id == track_id and t.is_vehicle for t in vehicles)

    @staticmethod
    def _involved_track_ids(
        collision: Optional[CollisionEvent],
        motion: Dict[int, MotionFeatures],
        vehicles: Sequence[Track],
    ) -> List[int]:
        """Which tracks are "involved" in the current evidence."""
        if collision is not None:
            return [collision.track_a, collision.track_b]
        candidates = [
            tid
            for tid, features in motion.items()
            if features.valid and features.score >= 0.35
        ]
        if candidates:
            return sorted(candidates)[:3]
        return []

    def _object_evidence(
        self,
        vehicles_involved: int,
        persons: Sequence[Track],
        collision: Optional[CollisionEvent],
    ) -> float:
        """How much the *object census* supports an accident."""
        cfg = self.config.scoring
        if vehicles_involved <= 0:
            return 0.0
        score = cfg.object_evidence_per_vehicle * float(vehicles_involved - 1)
        if persons:
            score += cfg.object_evidence_person_bonus
        if collision is not None and collision.signals.person_involved:
            score += cfg.object_evidence_person_bonus * 0.5
        return clamp(score)

    def _scene_evidence(
        self,
        tracks: Sequence[Track],
        vehicles: Sequence[Track],
        motion: Dict[int, MotionFeatures],
        stamp: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Scene-level signals: disruption, blockage, instability, duration."""
        cfg = self.config.scoring
        disrupted = [tid for tid, f in motion.items() if f.valid and f.score >= 0.3]
        disruption = clamp(len(disrupted) / 3.0) if disrupted else 0.0

        # "traffic blockage": vehicles that have stopped for at least a second
        stopped_low: List[Track] = []
        for track in vehicles:
            if track.speed_norm > 0.02:
                continue
            stopped_for = track.stopped_for(1.0, 0.02, stamp)
            if stopped_for <= 0.0:
                continue
            stopped_low.append(track)
        blocked_ratio = clamp(len(stopped_low) / max(1, len(vehicles)))
        blocked_seconds = max((t.stopped_for(2.0, 0.02, stamp) for t in stopped_low), default=0.0)

        lost = [t for t in tracks if t.state == "lost"]
        instability = clamp(len(lost) / max(1, len(tracks))) if tracks else 0.0

        duration = 0.0
        if self._abnormal_since is not None:
            duration = clamp((stamp - self._abnormal_since) / max(1e-3, cfg.scene_duration_cap_seconds))

        score = clamp(
            cfg.scene_disruption_weight * disruption
            + cfg.scene_stopped_weight * blocked_ratio
            + cfg.scene_occlusion_weight * instability
            + cfg.scene_duration_weight * duration
        )
        detail = {
            "disruption": round(disruption, 4),
            "blocked_ratio": round(blocked_ratio, 4),
            "blocked_seconds": round(blocked_seconds, 3),
            "tracking_instability": round(instability, 4),
            "abnormal_duration": round(duration, 3),
            "disrupted_track_ids": sorted(disrupted),
        }
        return score, detail

    # ------------------------------------------------------------------ #
    # internals: evidence capture
    # ------------------------------------------------------------------ #
    def _maintain_evidence(
        self,
        verification: VerificationResult,
        collision: Optional[CollisionEvent],
        frame: np.ndarray,
        stamp: float,
    ) -> None:
        """Open / update the evidence directory while an event is being judged."""
        if not self.evidence.enabled:
            return

        if verification.state in (DetectionState.VERIFYING, DetectionState.CONFIRMED_ACCIDENT):
            if self._open_evidence is None:
                self._open_incident_id = self.evidence.next_incident_id()
                self._open_evidence = self.evidence.begin(self._open_incident_id)
                self._peak_collision_score = 0.0
                if self._open_evidence is not None:
                    self._save_pre_event(self._open_evidence)
            if self._open_evidence is not None and collision is not None:
                if collision.score > self._peak_collision_score:
                    self._peak_collision_score = collision.score
                    self._open_evidence.save_frame("collision", frame)
        elif verification.state in (DetectionState.FALSE_ALARM, DetectionState.NORMAL):
            if self._open_evidence is not None:
                self._open_evidence.finalize({"outcome": str(verification.status), "note": "rejected by verification"})
                _LOGGER.info(
                    "evidence for %s marked %s (false-alarm review copy kept)",
                    self._open_incident_id,
                    verification.status,
                )
                self._open_evidence = None
                self._open_incident_id = None
                self._peak_collision_score = 0.0

    def _save_pre_event(self, session: EvidenceSession) -> None:
        frames = self.pre_event.frames()
        if not frames:
            return
        # the oldest buffered frame is the most useful "before" evidence
        stamp, frame = frames[0]
        session.save_frame("before", frame)
        if len(frames) > 1:
            middle = frames[len(frames) // 2]
            session.save_frame("before_mid", middle[1])

    # ------------------------------------------------------------------ #
    # internals: severity
    # ------------------------------------------------------------------ #
    def _estimate_severity(
        self,
        collision: Optional[CollisionEvent],
        involved_ids: Sequence[int],
        tracks: Sequence[Track],
        persons: int,
        motion: Dict[int, MotionFeatures],
        stamp: float,
    ) -> SeverityAnalysis:
        cfg = self.config.severity

        impact_speed = self._event_peak_signals["closing_speed"]
        deceleration = self._event_peak_signals["deceleration"]
        displacement = self._event_peak_signals["displacement"]

        vehicles = [t for t in tracks if t.is_vehicle and (not involved_ids or t.track_id in set(involved_ids))]
        stopped = [t for t in vehicles if t.speed_norm <= 0.02]
        blocked_ratio = clamp(len(stopped) / max(1, len(vehicles))) if vehicles else 0.0
        blocked_seconds = max((t.stopped_for(2.0, 0.02, stamp) for t in stopped), default=0.0)
        duration = (stamp - self._abnormal_since) if self._abnormal_since is not None else 0.0

        context = SeverityContext(
            vehicles_involved=len(vehicles) if involved_ids else 0,
            persons_in_scene=persons,
            impact_speed=impact_speed,
            deceleration=deceleration,
            displacement=displacement,
            blocked_ratio=blocked_ratio,
            blocked_seconds=blocked_seconds,
            abnormal_duration=max(0.0, duration),
            optional_provider=cfg.optional_signal_provider,
            optional_signals={},
        )
        return self.severity.analyze(context)

    # ------------------------------------------------------------------ #
    # internals: incident assembly
    # ------------------------------------------------------------------ #
    def _build_incident(
        self,
        score: AccidentScore,
        temporal: TemporalResult,
        verification: VerificationResult,
        severity: SeverityAnalysis,
        collision: Optional[CollisionEvent],
        involved_ids: Sequence[int],
        tracks: Sequence[Track],
        persons: Sequence[Track],
        detections: DetectionResult,
        stamp: float,
        wall_stamp: float,
        frame: np.ndarray,
        perf: PerfStats,
        warnings: List[str],
        scene_detail: Dict[str, Any],
    ) -> Dict[str, Any]:
        # The state machine can confirm on the very first VERIFYING frame when
        # min_verification_seconds is 0, so make sure a session exists.
        if self.evidence.enabled and self._open_evidence is None:
            self._open_incident_id = self.evidence.next_incident_id()
            self._open_evidence = self.evidence.begin(self._open_incident_id)
            if self._open_evidence is not None:
                self._save_pre_event(self._open_evidence)
                self._open_evidence.save_frame("collision", frame)

        incident_id = self._open_incident_id or self.evidence.next_incident_id()

        # ---- evidence finalisation ------------------------------------
        manifest = None
        if self._open_evidence is not None:
            self._open_evidence.save_frame("after", frame)
            if self.config.visualization.enabled and perf.frame_size:
                annotated = self.renderer.render(
                    frame,
                    self._frame_result_stub(verification, score, temporal, collision, tracks, perf, severity),
                )
                self._open_evidence.save_annotated(annotated)
            manifest = self._open_evidence.finalize(
                {
                    "outcome": IncidentStatus.CONFIRMED_ACCIDENT.value,
                    "accident_score": round(score.total, 4),
                    "components": {k: round(v, 4) for k, v in score.components.items()},
                    "temporal": temporal.to_dict(),
                    "state_transitions": [t.to_dict() for t in verification.transitions],
                    "collision": collision.to_dict() if collision else None,
                    "scene": scene_detail,
                    "severity": severity.to_dict(),
                    "captured_at": now_iso(),
                }
            )
            self._open_evidence = None
            self._open_incident_id = None
            self._peak_collision_score = 0.0

        # ---- camera + location ----------------------------------------
        camera = self.locations.get(self.camera_id)
        warnings_out = list(warnings)
        if camera is None and not self.config.location.allow_unregistered:
            warnings_out.append("camera metadata missing and location.allow_unregistered is false")
        if camera is None:
            warnings_out.append("camera_metadata_missing: incident reported without coordinates")

        camera_ref = CameraRef(
            camera_id=self.camera_id or (camera.camera_id if camera else None),
            name=camera.name if camera else self.camera_id,
            source=self.source_label or _describe_source_hint(self.camera_id),
            camera_type="cctv" if self.camera_id else None,
            location_source="camera_registered_metadata" if camera else "unregistered",
        )
        location = (
            LocationInfo(
                name=camera.location_name or camera.name,
                latitude=camera.latitude,
                longitude=camera.longitude,
                road=camera.road,
                address=camera.address,
                zone=camera.zone,
                is_camera_registered=True,
                source=camera.source,
            )
            if camera
            else LocationInfo(is_camera_registered=False, source="unregistered")
        )

        # ---- evidence payload ------------------------------------------
        involved_tracks = [
            {
                "track_id": int(t.track_id),
                "class": t.normalized_name,
                "speed_norm": round(t.speed_norm, 5),
                "bbox": [round(v, 1) for v in t.bbox],
                "history_points": int(t.history_length),
            }
            for t in tracks
            if not involved_ids or t.track_id in set(involved_ids)
        ]
        evidence_info = EvidenceInfo(
            collision_score=score.component("collision"),
            motion_score=score.component("motion"),
            trajectory_score=score.component("trajectory"),
            temporal_score=score.component("temporal"),
            object_evidence_score=score.component("object_evidence"),
            scene_evidence_score=score.component("scene_evidence"),
            weights=dict(score.weights),
            signals={
                "collision": collision.to_dict() if collision else None,
                "trajectory_pairs": [p.to_dict() for p in self._last_pairs[:5]],
                "scene": scene_detail,
                "temporal": temporal.to_dict(),
                "penalties": score.penalties,
            },
            reasons=score.reasons,
            negative_evidence=score.negative_evidence,
            involved_tracks=involved_tracks,
            agreement_signals=collision.agreement if collision else 0,
        )

        vehicles_involved = len([t for t in tracks if t.is_vehicle and (not involved_ids or t.track_id in set(involved_ids))])
        incident = IncidentResult(
            incident_id=incident_id,
            status=IncidentStatus.CONFIRMED_ACCIDENT.value,
            # the wall-clock time this frame was processed, NOT the media
            # position: an incident record that says 1970 is useless
            timestamp=_iso(wall_stamp),
            detected_at_unix=wall_stamp,
            camera=camera_ref,
            location=location,
            accident=AccidentInfo(
                score=score.total,
                severity=str(severity.level),
                severity_score=severity.score,
                severity_reasons=list(severity.reasons),
                confirmed=True,
                confirmation="multi-frame evidence (temporal fusion + independent signals)",
                frames_of_evidence=temporal.window_frames,
                verification_seconds=verification.verification_seconds,
            ),
            evidence=evidence_info,
            objects=ObjectsInfo(
                vehicles_involved=vehicles_involved,
                persons_detected=len(persons),
                total_objects=len(tracks),
                object_counts=detections.count_by_class(),
                involved_track_ids=sorted(int(t) for t in involved_ids),
                classes=sorted({t.normalized_name for t in tracks}),
            ),
            media=MediaInfo(
                evidence_dir=manifest.directory if manifest else None,
                annotated_frame=manifest.annotated if manifest else None,
                evidence_frames=manifest.frame_paths if manifest else [],
                clip=manifest.clip if manifest else None,
                files=dict(manifest.hashes) if manifest else {},
            ),
            performance=PerformanceInfo(
                fps=round(perf.fps or 0.0, 2),
                source_fps=round(perf.source_fps or 0.0, 2),
                inference_ms=round(perf.inference_ms, 3),
                tracking_ms=round(perf.tracking_ms, 3),
                analysis_ms=round(perf.analysis_ms, 3),
                total_ms=round(perf.total_ms, 3),
                frames_processed=int(perf.frames_processed),
                detector=perf.detector,
                device=perf.device,
            ),
            warnings=warnings_out,
        )
        # keep the analysis timeline available for alignment/debugging
        incident.extra["media_time_seconds"] = round(float(stamp), 3)
        # Phase 3: label the position.  A camera coordinate is approximate and
        # is never presented as a live GPS fix.
        incident.extra["location_provenance"] = self._location_provenance(wall_stamp)

        # attach the AI incident report
        try:
            report = self.reports.generate(incident.to_dict(include_report=False))
            report["markdown"] = self.reports.to_markdown(incident.to_dict(include_report=False))
            incident.report = report
        except Exception as exc:  # noqa: BLE001 - a report must never break an incident
            _LOGGER.warning("report generation failed: %s", exc)
            incident.warnings.append(f"report_generation_failed: {exc}")

        problems = incident.validate()
        if problems:
            _LOGGER.warning("incident %s has schema problems: %s", incident_id, problems)
            incident.warnings.extend(f"schema:{p}" for p in problems)

        incident.demo_emergency_call = self._demo_call_info()
        incident.demo_emergency_sms = self._demo_sms_info()
        payload = incident.to_dict()
        self._incidents.append(incident)
        _LOGGER.info(
            "CONFIRMED incident %s | score %.2f | severity %s | vehicles %d | evidence %s",
            incident_id,
            score.total,
            severity.level,
            vehicles_involved,
            manifest.directory if manifest else "(disabled)",
        )

        # ---- demonstration emergency call ---------------------------------
        # Attached *before* the callbacks run, so the incident a consumer
        # receives already carries its call status.  The dispatcher enforces
        # "confirmed only, exactly once per incident".
        self.ledger.mark(incident_id, "incident_created")
        t_notify = time.perf_counter()
        record = self._dispatch_demo_call(payload)
        # "queued for notification" - NOT "notified".  The provider round trip
        # and the terminal call state are recorded separately by the dispatcher.
        self.ledger.record("notify_enqueue", (time.perf_counter() - t_notify) * 1000.0)
        self.ledger.mark(incident_id, "notify_enqueue")
        if record is not None:
            demo_mode = self.emergency is not None and self.emergency.demo_mode
            incident.demo_emergency_call = DemoCallInfo.from_record(record, demo_mode=demo_mode)
            # The SMS follows the call on the worker thread, so this snapshot
            # shows PENDING; the live state arrives via on_call_status /
            # pipeline.demo_calls, exactly like the call's later transitions.
            incident.demo_emergency_sms = DemoSmsInfo.from_record(record.sms, demo_mode=demo_mode)
            payload = incident.to_dict()
        for callback in self._callbacks:
            try:
                callback(payload)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("incident callback failed: %s", exc)
        return payload

    # ------------------------------------------------------------------ #
    # internals: demonstration emergency call
    # ------------------------------------------------------------------ #
    def _demo_call_info(self) -> DemoCallInfo:
        """The ``demo_emergency_call`` block for an incident just confirmed."""
        if self.emergency is None:
            reason = (
                "emergency.enabled is false"
                if not self.config.emergency.enabled
                else "DEMO_MODE is not true - automatic calling is disabled"
            )
            return DemoCallInfo.inactive(reason=reason)
        return DemoCallInfo.inactive(
            reason="no demonstration call has been attempted yet",
            demo_mode=self.emergency.demo_mode,
        )

    def _location_provenance(self, at_unix: float) -> Dict[str, Any]:
        """Where the position came from, and how much to trust it.

        Delegates to the location registry so the camera position and any
        (future) live fix stay distinguishable all the way to the dashboard.
        """
        try:
            return self._location_registry.resolve(self.camera_id, at_unix).to_dict()
        except Exception as exc:  # noqa: BLE001 - provenance must never break an incident
            _LOGGER.warning("location provenance failed: %s", exc)
            return {
                "source": "LOCATION_UNAVAILABLE",
                "camera_id": self.camera_id,
                "confidence": "NONE",
                "reason": f"location provider failed: {exc}",
            }

    def _demo_sms_info(self) -> DemoSmsInfo:
        """The ``demo_emergency_sms`` block for an incident just confirmed."""
        if self.emergency is None:
            reason = (
                "emergency.enabled is false"
                if not self.config.emergency.enabled
                else "DEMO_MODE is not true - automatic calling is disabled"
            )
            return DemoSmsInfo.inactive(reason=reason)
        return DemoSmsInfo.inactive(
            reason="no demonstration SMS has been attempted yet",
            demo_mode=self.emergency.demo_mode,
        )

    def _dispatch_demo_call(self, payload: Dict[str, Any]) -> Any:
        """Hand the confirmed incident to the dispatcher.

        Returns the call record, or ``None``.  Never raises: a failed call must
        not cost us the incident.
        """
        if self.emergency is None:
            return None
        try:
            return self.emergency.handle_incident(payload)
        except Exception as exc:  # noqa: BLE001 - the incident is already built
            _LOGGER.exception("demonstration call dispatch failed")
            self.emergency = None
            return None

    @property
    def demo_calls(self) -> List[Dict[str, Any]]:
        """Call records as plain JSON (audit trail for the dashboard)."""
        if self.emergency is None:
            return []
        return [
            call.to_dict(include_recipient=self.config.emergency.expose_recipient)
            for call in self.emergency.calls
        ]

    def _frame_result_stub(
        self,
        verification: VerificationResult,
        score: AccidentScore,
        temporal: TemporalResult,
        collision: Optional[CollisionEvent],
        tracks: Sequence[Track],
        perf: PerfStats,
        severity: Optional[SeverityAnalysis] = None,
    ) -> FrameResult:
        """Minimal FrameResult used only to draw the annotated evidence frame."""
        return FrameResult(
            frame_index=self._frame_index,
            timestamp=time.time(),
            frame_size=perf.frame_size,
            camera_id=self.camera_id,
            tracks=list(tracks),
            collision=collision,
            temporal=temporal,
            verification=verification,
            score_detail=score,
            accident_score=score.total,
            component_scores=dict(score.components),
            severity=severity,
            performance=perf,
        )

    # ------------------------------------------------------------------ #
    def _update_fps(self, total_ms: float) -> float:
        instant = 1000.0 / total_ms if total_ms > 0 else 0.0
        self._fps_ema = instant if self._fps_ema is None else 0.85 * self._fps_ema + 0.15 * instant
        return self._fps_ema

    def _open_writer(self, frame_size: Tuple[int, int], output_video: Optional[str]):
        """Open the debug-video writer (Phase 19/21) when requested."""
        target = output_video or (self.config.runtime.output_video if self.config.runtime.save_output_video else None)
        if not target:
            return None
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return None
        width, height = frame_size
        if not width or not height:
            _LOGGER.warning("cannot open the output writer: unknown frame size")
            return None
        target_path = project_path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(self.config.runtime.output_video_fps or max(1.0, min(30.0, self._fps_ema or 15.0)))
        writer = cv2.VideoWriter(str(target_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            _LOGGER.error("could not open output video writer for %s", target_path)
            return None
        _LOGGER.info("writing debug video to %s (%dx%d @ %.1f fps)", target_path, width, height, fps)
        return {
            "path": str(target_path),
            "write": writer.write,
            "release": writer.release,
            "fps": fps,
            "size": (width, height),
        }

    def _maybe_report(self, result: FrameResult) -> None:
        """Periodic one-line status (video.report_every > 0)."""
        every = int(self.config.video.report_every or 0)
        if every <= 0 or result.frame_index % every:
            return
        _LOGGER.info(
            "frame %d | state %-16s | score %.2f | temporal %.2f | %d tracks | %.1f fps",
            result.frame_index,
            result.state,
            result.accident_score,
            result.temporal.temporal_score,
            len(result.tracks),
            result.performance.fps,
        )


# --------------------------------------------------------------------------- #
def _iso(stamp: float) -> str:
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(float(stamp), tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return now_iso()


def _describe_source_hint(camera_id: Optional[str]) -> Optional[str]:
    return camera_id
