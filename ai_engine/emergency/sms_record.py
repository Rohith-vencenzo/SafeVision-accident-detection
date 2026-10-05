"""The record of the demonstration **SMS**.

Sent *after* the call for the same confirmed incident, and stored on the same
:class:`~ai_engine.emergency.call_record.DemoCallRecord` as the call, so
"exactly one call and exactly one SMS per confirmed incident" is structural
rather than a convention.

The body is built from the confirmed incident only - incident id, timestamp,
camera-registered location, camera id, severity and score. Nothing is
hard-coded and nothing is invented: a missing value is written as an explicit
"unavailable" rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ai_engine.emergency.recipient import mask_phone
from ai_engine.schemas.enums import SmsStatus
from ai_engine.schemas.incident_result import DEMO_SMS_LABEL
from ai_engine.utils.timeutil import now_iso

__all__ = [
    "DEMO_SMS_LABEL",
    "NO_GPS",
    "DemoSmsRecord",
    "SmsStatus",
    "build_sms_body",
]

#: Shown when the camera has no registered coordinates.  Never substitute a
#: guess: the engine has no GPS receiver.
NO_GPS = "no GPS fix available - this is the camera's registered location"


def _location_line(location: Mapping[str, Any]) -> Dict[str, str]:
    """Name the location *and* say where it came from.

    The distinction matters operationally: a registered camera position is
    "somewhere near this junction", not the vehicle's position.
    """
    registered = bool(location.get("is_camera_registered"))
    parts = [
        str(location.get("name") or "").strip(),
        str(location.get("road") or "").strip(),
        str(location.get("address") or "").strip(),
    ]
    name = next((p for p in parts if p), "unregistered camera")
    latitude, longitude = location.get("latitude"), location.get("longitude")
    if registered and latitude is not None and longitude is not None:
        coords = f"({float(latitude):.6f}, {float(longitude):.6f})"
        source = "CAMERA-REGISTERED (not a GPS fix)"
    elif registered:
        coords = "(no coordinates registered)"
        source = "CAMERA-REGISTERED (no coordinates registered)"
    else:
        coords = NO_GPS
        source = "UNREGISTERED CAMERA - " + NO_GPS
    return {"name": name, "coordinates": coords, "source": source}


def build_sms_body(incident: Mapping[str, Any], instruction: str = "") -> str:
    """Compose the alert text from a confirmed incident.

    Every value comes from the payload.  ``incident_date`` / ``incident_time``
    are the confirmation timestamp, formatted in the camera's local sense via
    the recorded ISO string.
    """
    accident = incident.get("accident") or {}
    camera = incident.get("camera") or {}
    detected = str(incident.get("timestamp") or "")
    where = _location_line(incident.get("location") or {})

    try:
        score = float(accident.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    date, clock = _split_timestamp(detected)
    camera_id = str(camera.get("camera_id") or "unregistered")

    lines = [
        "\U0001F6A8 ACCIDENT ALERT",
        f"Incident: {incident.get('incident_id') or 'unknown'}",
        f"Date: {date}",
        f"Time: {clock}",
        f"Location: {where['name']} {where['coordinates']}",
        f"Camera: {camera_id}",
        f"Severity: {accident.get('severity') or 'unknown'}"
        f" (AI-estimated scene severity, confidence {int(round(score * 100))}%)",
        "",
        "A traffic accident has been confirmed by SafeVision from multi-frame "
        "video evidence. Emergency assistance may be required."
        if not instruction
        else str(instruction),
        "",
        f"Location source: {where['source']}",
        "This is an AI estimate, not a verified injury assessment. "
        "Please confirm before acting.",
    ]
    return "\n".join(lines)


def _split_timestamp(iso: str) -> tuple:
    """``2026-10-02T18:31:35.123+00:00`` -> ``('02-10-2026', '18:31:35')``."""
    text = str(iso or "").strip()
    if not text:
        return ("unknown", "unknown")
    date_part, _, time_part = text.partition("T")
    clock = time_part.split("+")[0].split(".")[0]
    try:
        year, month, day = date_part.split("-")
        return (f"{day}-{month}-{year}", clock)
    except ValueError:
        return (date_part or "unknown", clock or "unknown")


@dataclass
class DemoSmsRecord:
    """Audit trail for the SMS half of one incident's notification."""

    status: str = SmsStatus.PENDING.value
    provider: str = "simulation"
    trigger: str = "automatic_after_call"
    recipient: str = ""
    recipient_exposed: Optional[str] = None
    body: str = ""
    provider_status: str = ""
    provider_message_sid: str = ""
    error: str = ""
    reason: str = ""
    sent_at: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return SmsStatus(self.status).is_terminal

    def set_status(
        self,
        status: str,
        provider_status: str = "",
        provider_message_sid: str = "",
        error: str = "",
        reason: str = "",
    ) -> None:
        self.status = str(status)
        if provider_status:
            self.provider_status = provider_status
        if provider_message_sid:
            self.provider_message_sid = provider_message_sid
        if error:
            self.error = error
        if reason:
            self.reason = reason
        if self.status == SmsStatus.SENT.value and not self.sent_at:
            self.sent_at = now_iso()
        self.history.append(
            {"status": self.status, "at": now_iso(), "provider_status": self.provider_status or None}
        )

    def to_dict(self, include_recipient: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "label": DEMO_SMS_LABEL,
            "status": self.status,
            "provider": self.provider,
            "trigger": self.trigger,
            "recipient": self.recipient,
            "body": self.body,
            "provider_status": self.provider_status or None,
            "provider_message_sid": self.provider_message_sid or None,
            "error": self.error or None,
            "reason": self.reason or None,
            "sent_at": self.sent_at,
            "sent": self.status == SmsStatus.SENT.value,
            "history": list(self.history),
        }
        if include_recipient and self.recipient_exposed:
            payload["recipient_full"] = self.recipient_exposed
        return payload

    @classmethod
    def masked(cls, number: Optional[str]) -> str:
        return mask_phone(number)