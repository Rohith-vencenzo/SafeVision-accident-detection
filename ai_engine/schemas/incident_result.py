"""The structured incident contract (Phase 17 + Phase 26).

This module is the boundary between my 60% (AI/CV) and my teammate's 40%
(backend / database / dashboard / notifications).  Nothing else in the engine
talks to a backend, and nothing here knows about HTTP, SQL or SMS.

Two payloads are produced:

``IncidentResult.to_dict()``
    the full, versioned record (everything that was measured).
``IncidentResult.to_alert_dict()``
    the compact payload a dashboard / dispatcher view needs.

Both are plain JSON: no numpy types, no NaN, no custom objects.

Schema versioning
-----------------
``schema_version`` follows ``MAJOR.MINOR.PATCH``:

* MAJOR - a field was removed or its meaning changed
* MINOR - a field was added (consumers should ignore unknown fields)
* PATCH - clarification only

Consumers must reject an unknown MAJOR version and may safely ignore unknown
MINOR fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ai_engine.schemas.enums import CallStatus, IncidentStatus, SeverityLevel, SmsStatus
from ai_engine.utils.jsonio import to_jsonable
from ai_engine.utils.timeutil import now_iso

__all__ = [
    "DEMO_CALL_DISCLAIMER",
    "DEMO_CALL_LABEL",
    "DEMO_SMS_DISCLAIMER",
    "DEMO_SMS_LABEL",
    "DISCLAIMER",
    "INCIDENT_JSON_SCHEMA",
    "SCHEMA_VERSION",
    "AccidentInfo",
    "CameraRef",
    "DemoCallInfo",
    "DemoSmsInfo",
    "EvidenceInfo",
    "IncidentResult",
    "LocationInfo",
    "MediaInfo",
    "ObjectsInfo",
    "PerformanceInfo",
    "generate_emergency_incident",
    "get_incident_result",
    "validate_incident",
]

SCHEMA_VERSION = "1.3.0"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float.  ``to_jsonable`` sanitises the rest, but the
    percentage fields are computed here, so NaN has to be caught before it."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number

DISCLAIMER = (
    "SafeVision output is an AI-estimated accident probability computed from "
    "multi-frame video evidence. It is not a verified human-injury assessment and "
    "the location is the camera's registered location, not a GPS fix. A human "
    "operator should confirm before any emergency action is taken."
)

#: Re-exported so the contract and the dispatcher can never disagree about the
#: wording shown next to a call status on the dashboard.
DEMO_CALL_DISCLAIMER = (
    "Demonstration call to the authorised SafeVision demo recipient. It is not "
    "connected to any ambulance, police or 112 line and no emergency service is "
    "notified."
)

#: The literal wording the debug overlay and the dashboard display.  Defined
#: here (not in the dispatcher) so the contract cannot drift from the UI.
DEMO_CALL_LABEL = "DEMO EMERGENCY CALL"

#: Same for the SMS that follows the call.
DEMO_SMS_LABEL = "DEMO SMS"

DEMO_SMS_DISCLAIMER = (
    "Demonstration SMS to the authorised SafeVision demo recipient. It is not "
    "connected to any ambulance, police or 112 line and no emergency service is "
    "notified."
)

#: Reason strings are short, machine-stable phrases.  This list documents the
#: vocabulary downstream systems can key on.
KNOWN_REASONS: tuple[str, ...] = (
    "Sudden vehicle velocity change",
    "Sudden stop after sustained motion",
    "Trajectory convergence",
    "Trajectory crossing ahead of both objects",
    "Collision proximity detected",
    "Bounding-box overlap between objects",
    "Post-collision stopping",
    "Post-impact displacement spike",
    "Path deviates from established heading",
    "Zig-zag / unstable trajectory",
    "Persistent abnormal motion across frames",
    "Multiple vehicles involved",
    "Person detected near involved vehicles",
    "Traffic blockage (stopped vehicles in lane)",
    "Scene remained abnormal for an extended period",
    "Parked vehicles only (no meaningful movement change)",
    "Objects stable, no motion change",
    "Single-frame event (no temporal support)",
    "Unstable tracking",
    "Low detection confidence",
    "Insufficient tracking history",
    "Occlusion / conflicting assignment",
)


# --------------------------------------------------------------------------- #
# nested payloads
# --------------------------------------------------------------------------- #
@dataclass
class CameraRef:
    """Which camera produced the incident."""

    camera_id: Optional[str] = None
    name: Optional[str] = None
    source: Optional[str] = None          # redacted source (file / rtsp host / webcam index)
    camera_type: Optional[str] = None     # cctv | webcam | video_file
    location_source: str = "camera_registered_metadata"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": self.source,
            "type": self.camera_type,
            "location_source": self.location_source,
        }


@dataclass
class LocationInfo:
    """Camera-registered location (Phase 14).

    ``is_camera_registered`` is ``False`` when the camera has no metadata; the
    incident is still emitted (with a warning) because dropping it would be
    worse than reporting it without coordinates.
    """

    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road: Optional[str] = None
    address: Optional[str] = None
    zone: Optional[str] = None
    is_camera_registered: bool = False
    source: str = "camera_metadata"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road": self.road,
            "address": self.address,
            "zone": self.zone,
            "is_camera_registered": bool(self.is_camera_registered),
            "source": self.source,
        }


@dataclass
class AccidentInfo:
    """Final score + AI-estimated scene severity."""

    score: float = 0.0
    severity: str = SeverityLevel.LOW.value
    severity_score: float = 0.0
    severity_reasons: List[str] = field(default_factory=list)
    confirmed: bool = False
    confirmation: str = "multi-frame evidence"  # how the decision was reached
    frames_of_evidence: int = 0
    verification_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        score = _safe_float(self.score)
        return {
            "score": round(score, 4),
            "confidence_percent": int(round(score * 100)),
            "severity": str(self.severity),
            "severity_score": round(_safe_float(self.severity_score), 4),
            "severity_reasons": list(self.severity_reasons),
            "confirmed": bool(self.confirmed),
            "confirmation": self.confirmation,
            "frames_of_evidence": int(self.frames_of_evidence),
            "verification_seconds": round(_safe_float(self.verification_seconds), 3),
        }


@dataclass
class EvidenceInfo:
    """Per-component scores, the weights used, and why the system decided."""

    collision_score: float = 0.0
    motion_score: float = 0.0
    trajectory_score: float = 0.0
    temporal_score: float = 0.0
    object_evidence_score: float = 0.0
    scene_evidence_score: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)
    involved_tracks: List[Dict[str, Any]] = field(default_factory=list)
    agreement_signals: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collision_score": round(float(self.collision_score), 4),
            "motion_score": round(float(self.motion_score), 4),
            "trajectory_score": round(float(self.trajectory_score), 4),
            "temporal_score": round(float(self.temporal_score), 4),
            "object_evidence_score": round(float(self.object_evidence_score), 4),
            "scene_evidence_score": round(float(self.scene_evidence_score), 4),
            "agreement_signals": int(self.agreement_signals),
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "signals": to_jsonable(self.signals),
            "reasons": list(self.reasons),
            "negative_evidence": list(self.negative_evidence),
            "involved_tracks": to_jsonable(self.involved_tracks),
        }


@dataclass
class ObjectsInfo:
    """Object census at the moment of the incident."""

    vehicles_involved: int = 0
    persons_detected: int = 0
    total_objects: int = 0
    object_counts: Dict[str, int] = field(default_factory=dict)
    involved_track_ids: List[int] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicles_involved": int(self.vehicles_involved),
            "persons_detected": int(self.persons_detected),
            "total_objects": int(self.total_objects),
            "object_counts": dict(self.object_counts),
            "involved_track_ids": [int(t) for t in self.involved_track_ids],
            "classes": list(self.classes),
        }


@dataclass
class MediaInfo:
    """Paths to the captured evidence (Phase 15)."""

    evidence_dir: Optional[str] = None
    annotated_frame: Optional[str] = None
    evidence_frames: List[str] = field(default_factory=list)
    clip: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_dir": self.evidence_dir,
            "annotated_frame": self.annotated_frame,
            "evidence_frames": list(self.evidence_frames),
            "clip": self.clip,
            "files": dict(self.files),
        }


@dataclass
class PerformanceInfo:
    """Pipeline timings (Phase 21).  Useful for the dashboard's health view."""

    fps: float = 0.0
    source_fps: float = 0.0
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    analysis_ms: float = 0.0
    total_ms: float = 0.0
    frames_processed: int = 0
    detector: Optional[str] = None
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": round(float(self.fps), 2),
            "source_fps": round(float(self.source_fps), 2),
            "inference_ms": round(float(self.inference_ms), 3),
            "tracking_ms": round(float(self.tracking_ms), 3),
            "analysis_ms": round(float(self.analysis_ms), 3),
            "total_ms": round(float(self.total_ms), 3),
            "frames_processed": int(self.frames_processed),
            "detector": self.detector,
            "device": self.device,
        }


@dataclass
class DemoCallInfo:
    """Status of the **demonstration** emergency call for this incident.

    Present on every incident so a dashboard can render the same card whether
    demo calling is on or off.  ``attempted`` tells the two apart: when it is
    ``False`` nothing was dialled and ``status`` explains why.

    ``status`` moves on after the incident is emitted (``INITIATED`` at first),
    so a live view must subscribe to the status feed as well - see
    :class:`~ai_engine.emergency.DemoCallDispatcher`.
    """

    attempted: bool = False
    status: str = CallStatus.SKIPPED.value
    call_id: Optional[str] = None
    incident_id: Optional[str] = None
    label: str = DEMO_CALL_LABEL
    provider: str = "none"
    trigger: str = "automatic_on_confirmation"
    recipient: Optional[str] = None          # masked, e.g. "+91******3949"
    recipient_full: Optional[str] = None     # only when emergency.expose_recipient
    reason: Optional[str] = None              # why it was skipped / blocked
    provider_status: Optional[str] = None
    provider_call_sid: Optional[str] = None
    error: Optional[str] = None
    call_placed: bool = False
    started_at: Optional[str] = None
    duration_seconds: float = 0.0
    demo_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": bool(self.attempted),
            "status": str(self.status),
            "label": DEMO_CALL_LABEL,
            "call_id": self.call_id,
            "incident_id": self.incident_id,
            "provider": self.provider,
            "trigger": self.trigger,
            "recipient": self.recipient,
            "recipient_full": self.recipient_full,
            "reason": self.reason,
            "provider_status": self.provider_status,
            "provider_call_sid": self.provider_call_sid,
            "error": self.error,
            "call_placed": bool(self.call_placed),
            "started_at": self.started_at,
            "duration_seconds": round(_safe_float(self.duration_seconds), 2),
            "demo_mode": bool(self.demo_mode),
            "disclaimer": DEMO_CALL_DISCLAIMER,
        }

    @classmethod
    def from_record(cls, record: Any, demo_mode: bool = False) -> "DemoCallInfo":
        """Build from a :class:`~ai_engine.emergency.DemoCallRecord`."""
        payload = record.to_dict(include_recipient=True)
        return cls(
            attempted=True,
            status=str(payload.get("status") or CallStatus.INITIATED.value),
            label=DEMO_CALL_LABEL,
            call_id=payload.get("call_id"),
            incident_id=payload.get("incident_id"),
            provider=str(payload.get("provider") or "none"),
            trigger=str(payload.get("trigger") or "automatic_on_confirmation"),
            recipient=payload.get("recipient"),
            recipient_full=payload.get("recipient_full"),
            reason=payload.get("reason"),
            provider_status=payload.get("provider_status"),
            provider_call_sid=payload.get("provider_call_sid"),
            error=payload.get("error"),
            call_placed=bool(payload.get("call_placed")),
            started_at=payload.get("started_at"),
            duration_seconds=float(payload.get("duration_seconds") or 0.0),
            demo_mode=bool(demo_mode),
        )

    @classmethod
    def inactive(cls, reason: str = "automatic calling is disabled", demo_mode: bool = False) -> "DemoCallInfo":
        """The payload used when demo calling is off - explicit, never null."""
        return cls(attempted=False, status=CallStatus.SKIPPED.value, reason=reason, demo_mode=demo_mode)


# --------------------------------------------------------------------------- #
# the incident
# --------------------------------------------------------------------------- #
@dataclass
class DemoSmsInfo:
    """Status of the **demonstration** SMS sent after the call.

    Same rules as :class:`DemoCallInfo`: present on every incident so the
    dashboard renders one card, and ``attempted`` distinguishes "SMS off" from
    "SMS attempted".
    """

    attempted: bool = False
    status: str = "PENDING"
    label: str = DEMO_SMS_LABEL
    call_id: Optional[str] = None
    incident_id: Optional[str] = None
    provider: str = "none"
    trigger: str = "automatic_after_call"
    recipient: Optional[str] = None
    recipient_full: Optional[str] = None
    body: Optional[str] = None
    reason: Optional[str] = None
    provider_status: Optional[str] = None
    provider_message_sid: Optional[str] = None
    error: Optional[str] = None
    sent_at: Optional[str] = None
    sent: bool = False
    demo_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": bool(self.attempted),
            "status": str(self.status),
            "label": DEMO_SMS_LABEL,
            "call_id": self.call_id,
            "incident_id": self.incident_id,
            "provider": self.provider,
            "trigger": self.trigger,
            "recipient": self.recipient,
            "recipient_full": self.recipient_full,
            "body": self.body,
            "reason": self.reason,
            "provider_status": self.provider_status,
            "provider_message_sid": self.provider_message_sid,
            "error": self.error,
            "sent_at": self.sent_at,
            # derived as well as stored, so a caller that sets only the status
            # can never report sent=False for a message that went out
            "sent": bool(self.sent or str(self.status) == SmsStatus.SENT.value),
            "demo_mode": bool(self.demo_mode),
            "disclaimer": DEMO_SMS_DISCLAIMER,
        }

    @classmethod
    def from_record(cls, record: Any, demo_mode: bool = False) -> "DemoSmsInfo":
        payload = record.to_dict(include_recipient=True)
        return cls(
            attempted=payload.get("status") not in (None, "PENDING"),
            status=str(payload.get("status") or "PENDING"),
            call_id=payload.get("call_id"),
            incident_id=payload.get("incident_id"),
            provider=str(payload.get("provider") or "none"),
            trigger=str(payload.get("trigger") or "automatic_after_call"),
            recipient=payload.get("recipient"),
            recipient_full=payload.get("recipient_full"),
            body=payload.get("body"),
            reason=payload.get("reason"),
            provider_status=payload.get("provider_status"),
            provider_message_sid=payload.get("provider_message_sid"),
            error=payload.get("error"),
            sent_at=payload.get("sent_at"),
            sent=bool(payload.get("sent")),
            demo_mode=bool(demo_mode),
        )

    @classmethod
    def inactive(cls, reason: str = "SMS is disabled", demo_mode: bool = False) -> "DemoSmsInfo":
        return cls(attempted=False, status=SmsStatus.SKIPPED, reason=reason, demo_mode=demo_mode)


# --------------------------------------------------------------------------- #
# the incident
# --------------------------------------------------------------------------- #
@dataclass
class IncidentResult:
    """A single AI incident record - the object my teammate consumes."""

    incident_id: str
    status: str = IncidentStatus.CONFIRMED_ACCIDENT.value
    timestamp: str = field(default_factory=now_iso)
    detected_at_unix: float = 0.0
    camera: CameraRef = field(default_factory=CameraRef)
    location: LocationInfo = field(default_factory=LocationInfo)
    accident: AccidentInfo = field(default_factory=AccidentInfo)
    evidence: EvidenceInfo = field(default_factory=EvidenceInfo)
    objects: ObjectsInfo = field(default_factory=ObjectsInfo)
    media: MediaInfo = field(default_factory=MediaInfo)
    performance: PerformanceInfo = field(default_factory=PerformanceInfo)
    demo_emergency_call: DemoCallInfo = field(default_factory=DemoCallInfo)
    demo_emergency_sms: DemoSmsInfo = field(default_factory=DemoSmsInfo)
    source: str = "safevision.ai_engine"
    warnings: List[str] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self, include_report: bool = True) -> Dict[str, Any]:
        """Full JSON-serialisable record."""
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_by": self.source,
            "incident_id": self.incident_id,
            "status": str(self.status),
            "timestamp": self.timestamp,
            "detected_at_unix": round(float(self.detected_at_unix), 3),
            "camera": self.camera.to_dict(),
            "location": self.location.to_dict(),
            "accident": self.accident.to_dict(),
            "evidence": self.evidence.to_dict(),
            "objects": self.objects.to_dict(),
            "media": self.media.to_dict(),
            "performance": self.performance.to_dict(),
            "demo_emergency_call": self.demo_emergency_call.to_dict(),
            "demo_emergency_sms": self.demo_emergency_sms.to_dict(),
            "warnings": list(self.warnings),
            "disclaimer": DISCLAIMER,
        }
        if include_report and self.report is not None:
            payload["report"] = to_jsonable(self.report)
        if self.extra:
            payload["extra"] = to_jsonable(self.extra)
        return to_jsonable(payload)

    def to_alert_dict(self, minimum_score: float = 0.0) -> Dict[str, Any]:
        """Compact payload for an alert / dashboard card.

        ``minimum_score`` filters out low-confidence records so a consumer can
        subscribe to everything and only be woken up for real events.
        """
        if float(self.accident.score) < float(minimum_score):
            return {}
        return to_jsonable(
            {
                "schema_version": SCHEMA_VERSION,
                "incident_id": self.incident_id,
                "status": str(self.status),
                "timestamp": self.timestamp,
                "camera_id": self.camera.camera_id,
                "camera_name": self.camera.name,
                "location_name": self.location.name,
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "accident_score": round(float(self.accident.score), 4),
                "confidence_percent": int(round(float(self.accident.score) * 100)),
                "severity": str(self.accident.severity),
                "severity_score": round(float(self.accident.severity_score), 4),
                "vehicles_involved": int(self.objects.vehicles_involved),
                "persons_detected": int(self.objects.persons_detected),
                "demo_emergency_call": self.demo_emergency_call.to_dict(),
                "demo_emergency_sms": self.demo_emergency_sms.to_dict(),
                "reasons": list(self.evidence.reasons)[:6],
                "annotated_frame": self.media.annotated_frame,
                "clip": self.media.clip,
                "evidence_dir": self.media.evidence_dir,
                "disclaimer": DISCLAIMER,
            }
        )

    @property
    def is_confirmed(self) -> bool:
        return str(self.status) == IncidentStatus.CONFIRMED_ACCIDENT.value

    def validate(self) -> List[str]:
        """Return a list of problems (empty == valid).  Never raises."""
        problems: List[str] = []
        if not self.incident_id:
            problems.append("incident_id is empty")
        try:
            IncidentStatus.coerce(self.status)
        except ValueError:
            problems.append(f"status {self.status!r} is not a valid IncidentStatus")
        try:
            SeverityLevel.coerce(self.accident.severity)
        except ValueError:
            problems.append(f"severity {self.accident.severity!r} is not a valid SeverityLevel")
        for label, value in (
            ("accident.score", self.accident.score),
            ("accident.severity_score", self.accident.severity_score),
        ):
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                problems.append(f"{label} is not numeric: {value!r}")
                continue
            if not 0.0 <= as_float <= 1.0:
                problems.append(f"{label}={as_float} is outside [0, 1]")
        if self.accident.confirmed and not self.accident.frames_of_evidence:
            problems.append("confirmed accident without any supporting frame")
        lat, lon = self.location.latitude, self.location.longitude
        if lat is not None and not -90.0 <= float(lat) <= 90.0:
            problems.append(f"latitude {lat} is outside [-90, 90]")
        if lon is not None and not -180.0 <= float(lon) <= 180.0:
            problems.append(f"longitude {lon} is outside [-180, 180]")
        if not self.evidence.reasons and self.accident.confirmed:
            problems.append("confirmed accident without any evidence reason")
        return problems

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IncidentResult":
        """Rebuild an incident from its own JSON (used by tests and backends)."""
        def _sub(key: str, klass):
            raw = data.get(key) or {}
            known = {f for f in klass.__dataclass_fields__}  # type: ignore[attr-defined]
            return klass(**{k: v for k, v in raw.items() if k in known})

        return cls(
            incident_id=str(data.get("incident_id", "")),
            status=str(data.get("status", IncidentStatus.NORMAL.value)),
            timestamp=str(data.get("timestamp", now_iso())),
            detected_at_unix=float(data.get("detected_at_unix", 0.0) or 0.0),
            camera=_sub("camera", CameraRef),
            location=_sub("location", LocationInfo),
            accident=_sub("accident", AccidentInfo),
            evidence=_sub("evidence", EvidenceInfo),
            objects=_sub("objects", ObjectsInfo),
            media=_sub("media", MediaInfo),
            performance=_sub("performance", PerformanceInfo),
            demo_emergency_call=_sub("demo_emergency_call", DemoCallInfo),
            demo_emergency_sms=_sub("demo_emergency_sms", DemoSmsInfo),
            source=str(data.get("generated_by", "safevision.ai_engine")),
            warnings=list(data.get("warnings", []) or []),
            report=data.get("report"),
            extra=dict(data.get("extra", {}) or {}),
        )


# --------------------------------------------------------------------------- #
# JSON Schema (optional, machine readable)
# --------------------------------------------------------------------------- #
INCIDENT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://safevision.local/schemas/incident_result.json",
    "title": "SafeVision incident result",
    "description": DISCLAIMER,
    "type": "object",
    "required": [
        "schema_version",
        "incident_id",
        "status",
        "timestamp",
        "camera",
        "location",
        "accident",
        "evidence",
        "objects",
        "media",
    ],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "generated_by": {"type": "string"},
        "incident_id": {"type": "string", "minLength": 1},
        "status": {"enum": [s.value for s in IncidentStatus]},
        "timestamp": {"type": "string"},
        "detected_at_unix": {"type": "number"},
        "camera": {
            "type": "object",
            "required": ["camera_id"],
            "properties": {
                "camera_id": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
                "source": {"type": ["string", "null"]},
                "type": {"type": ["string", "null"]},
                "location_source": {"type": "string"},
            },
        },
        "location": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "latitude": {"type": ["number", "null"], "minimum": -90, "maximum": 90},
                "longitude": {"type": ["number", "null"], "minimum": -180, "maximum": 180},
                "road": {"type": ["string", "null"]},
                "address": {"type": ["string", "null"]},
                "zone": {"type": ["string", "null"]},
                "is_camera_registered": {"type": "boolean"},
                "source": {"type": "string"},
            },
        },
        "accident": {
            "type": "object",
            "required": ["score", "severity", "severity_score"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence_percent": {"type": "integer", "minimum": 0, "maximum": 100},
                "severity": {"enum": [s.value for s in SeverityLevel]},
                "severity_score": {"type": "number", "minimum": 0, "maximum": 1},
                "severity_reasons": {"type": "array", "items": {"type": "string"}},
                "confirmed": {"type": "boolean"},
                "confirmation": {"type": "string"},
                "frames_of_evidence": {"type": "integer", "minimum": 0},
                "verification_seconds": {"type": "number", "minimum": 0},
            },
        },
        "evidence": {
            "type": "object",
            "required": [
                "collision_score",
                "motion_score",
                "trajectory_score",
                "temporal_score",
                "reasons",
            ],
            "properties": {
                "collision_score": {"type": "number", "minimum": 0, "maximum": 1},
                "motion_score": {"type": "number", "minimum": 0, "maximum": 1},
                "trajectory_score": {"type": "number", "minimum": 0, "maximum": 1},
                "temporal_score": {"type": "number", "minimum": 0, "maximum": 1},
                "object_evidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                "scene_evidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                "agreement_signals": {"type": "integer", "minimum": 0},
                "weights": {"type": "object"},
                "signals": {"type": "object"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "negative_evidence": {"type": "array", "items": {"type": "string"}},
                "involved_tracks": {"type": "array"},
            },
        },
        "objects": {
            "type": "object",
            "properties": {
                "vehicles_involved": {"type": "integer", "minimum": 0},
                "persons_detected": {"type": "integer", "minimum": 0},
                "total_objects": {"type": "integer", "minimum": 0},
                "object_counts": {"type": "object"},
                "involved_track_ids": {"type": "array", "items": {"type": "integer"}},
                "classes": {"type": "array", "items": {"type": "string"}},
            },
        },
        "media": {
            "type": "object",
            "properties": {
                "evidence_dir": {"type": ["string", "null"]},
                "annotated_frame": {"type": ["string", "null"]},
                "evidence_frames": {"type": "array", "items": {"type": "string"}},
                "clip": {"type": ["string", "null"]},
                "files": {"type": "object"},
            },
        },
        "performance": {"type": "object"},
        "demo_emergency_call": {
            "type": "object",
            "description": (
                "Status of the optional DEMONSTRATION call. Never an emergency-service "
                "notification. 'attempted' is false when demo calling is disabled."
            ),
            "required": ["attempted", "status"],
            "properties": {
                "attempted": {"type": "boolean"},
                "status": {"enum": [s.value for s in CallStatus]},
                "label": {"const": DEMO_CALL_LABEL},
                "call_id": {"type": ["string", "null"]},
                "incident_id": {"type": ["string", "null"]},
                "provider": {"type": "string"},
                "trigger": {"type": "string"},
                "recipient": {"type": ["string", "null"]},
                "recipient_full": {"type": ["string", "null"]},
                "reason": {"type": ["string", "null"]},
                "provider_status": {"type": ["string", "null"]},
                "provider_call_sid": {"type": ["string", "null"]},
                "error": {"type": ["string", "null"]},
                "call_placed": {"type": "boolean"},
                "started_at": {"type": ["string", "null"]},
                "duration_seconds": {"type": "number", "minimum": 0},
                "demo_mode": {"type": "boolean"},
                "disclaimer": {"type": "string"},
            },
        },
        "demo_emergency_sms": {
            "type": "object",
            "description": (
                "Status of the DEMONSTRATION SMS sent after the call for the same "
                "incident. Never an emergency-service notification."
            ),
            "required": ["attempted", "status"],
            "properties": {
                "attempted": {"type": "boolean"},
                "status": {"enum": [s.value for s in SmsStatus]},
                "label": {"const": DEMO_SMS_LABEL},
                "call_id": {"type": ["string", "null"]},
                "incident_id": {"type": ["string", "null"]},
                "provider": {"type": "string"},
                "trigger": {"type": "string"},
                "recipient": {"type": ["string", "null"]},
                "recipient_full": {"type": ["string", "null"]},
                "body": {"type": ["string", "null"]},
                "reason": {"type": ["string", "null"]},
                "provider_status": {"type": ["string", "null"]},
                "provider_message_sid": {"type": ["string", "null"]},
                "error": {"type": ["string", "null"]},
                "sent_at": {"type": ["string", "null"]},
                "sent": {"type": "boolean"},
                "demo_mode": {"type": "boolean"},
                "disclaimer": {"type": "string"},
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "disclaimer": {"type": "string"},
        "report": {"type": ["object", "null"]},
        "extra": {"type": "object"},
    },
}


def validate_incident(payload: Mapping[str, Any], strict_schema: bool = False) -> List[str]:
    """Validate a JSON payload.

    Uses the built-in dataclass validation, and additionally the JSON Schema
    when ``jsonschema`` happens to be installed.  Returns a list of problems;
    an empty list means valid.
    """
    problems = IncidentResult.from_dict(payload).validate()
    if strict_schema:
        try:
            import jsonschema  # type: ignore
        except ImportError:
            problems.append("jsonschema is not installed (pip install jsonschema) - strict validation skipped")
        else:
            validator = jsonschema.Draft7Validator(INCIDENT_JSON_SCHEMA)
            for error in validator.iter_errors(payload):
                problems.append(f"schema: {'/'.join(str(p) for p in error.path) or '<root>'}: {error.message}")
    return problems


# --------------------------------------------------------------------------- #
# emergency-response interface (Phase 26)
# --------------------------------------------------------------------------- #
def generate_emergency_incident(
    incident: Any,
    include_evidence_paths: bool = True,
    minimum_score: float = 0.0,
) -> Dict[str, Any]:
    """Return the structured incident JSON for the downstream 40% of the project.

    This is the *entire* interface the AI module exposes to the backend: it
    sends no notification, writes to no database and opens no socket.  (The one
    outbound action in the whole engine is the opt-in demonstration call in
    :mod:`ai_engine.emergency`, whose status this payload carries.)

    Parameters
    ----------
    incident:
        An :class:`IncidentResult` or an already-serialised incident dict.
    include_evidence_paths:
        Keep the captured-frame paths in the payload.
    minimum_score:
        Return ``{}`` below this accident score, so a subscriber can receive
        every incident and filter here.

    Returns
    -------
    dict
        JSON-serialisable alert payload (empty dict when filtered out).
    """
    if isinstance(incident, IncidentResult):
        payload = incident.to_alert_dict(minimum_score=minimum_score)
        if payload and not include_evidence_paths:
            payload.pop("evidence_dir", None)
            payload.pop("annotated_frame", None)
        return payload
    if isinstance(incident, Mapping):
        result = IncidentResult.from_dict(incident)
        payload = result.to_alert_dict(minimum_score=minimum_score)
        if payload and not include_evidence_paths:
            payload.pop("evidence_dir", None)
            payload.pop("annotated_frame", None)
            payload.pop("clip", None)
        return payload
    raise TypeError(f"expected IncidentResult or Mapping, got {type(incident).__name__}")


def get_incident_result(source: Any) -> Dict[str, Any]:
    """Fetch the most recent incident as plain JSON.

    ``source`` may be an :class:`IncidentResult`, a mapping, a
    :class:`~ai_engine.pipeline.accident_pipeline.AccidentPipeline`, or a
    sequence of any of those (the most recent usable record wins).
    """
    from ai_engine.pipeline.accident_pipeline import AccidentPipeline  # local import: avoid a cycle

    if isinstance(source, AccidentPipeline):
        return source.get_incident_result()
    if isinstance(source, IncidentResult):
        return source.to_dict()
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, Sequence):
        for item in reversed(list(source)):
            if isinstance(item, IncidentResult):
                return item.to_dict()
            if isinstance(item, Mapping) and item.get("incident_id"):
                return dict(item)
        return {}
    raise TypeError(f"cannot read an incident from {type(source).__name__}")
