"""The record of one demonstration call (the dashboard's data source).

A :class:`DemoCallRecord` is the audit trail for "did we call, whom, when and
how did it end".  It is deliberately JSON-shaped: it goes straight into the
incident payload, into ``demo_calls.jsonl`` and into the debug overlay.

The recipient appears **masked** by default.  The unmasked number is only ever
read from the environment at dial time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_engine.emergency.recipient import mask_phone
from ai_engine.emergency.sms_record import DEMO_SMS_LABEL, DemoSmsRecord
from ai_engine.schemas.enums import CallStatus
from ai_engine.utils.timeutil import iso_timestamp, now_iso, utc_now

__all__ = ["CALL_LOG_SCHEMA_VERSION", "DemoCallRecord"]

#: bumped if the call-log line shape changes
CALL_LOG_SCHEMA_VERSION = "1.0.0"

#: How the call came to be made.  There is deliberately no other source: the
#: engine never waits for a human to press a button, and it never calls on a
#: per-frame basis.
TRIGGER_AUTOMATIC = "automatic_on_confirmation"


@dataclass
class DemoCallRecord:
    """State of the notification sent for one confirmed incident.

    The record holds **both** channels: the call and, for the same incident,
    the SMS that follows it. Keeping them on one record is what makes
    "exactly one call and exactly one SMS per confirmed incident" structural -
    there is a single row per ``incident_id`` and therefore a single place where
    that can be enforced.
    """

    #: the exact string shown on the overlay and expected on the dashboard
    LABEL = "DEMO EMERGENCY CALL"

    call_id: str = ""
    incident_id: str = ""
    status: CallStatus = CallStatus.INITIATED
    provider: str = "simulation"
    trigger: str = TRIGGER_AUTOMATIC
    recipient: str = ""                      # masked, e.g. "+91******3949"
    recipient_exposed: Optional[str] = None  # unmasked, only when opted in
    camera_id: Optional[str] = None
    accident_score: Optional[float] = None
    severity: Optional[str] = None
    message: str = ""
    provider_status: str = ""
    provider_call_sid: str = ""
    #: **Phase 4.** Ties every log line, outbox row and provider request for one
    #: notification attempt together, so an operator can reconstruct the whole
    #: exchange from the logs alone.  Never contains a phone number or a secret.
    correlation_id: str = ""
    error: str = ""
    reason: str = ""                         # why it was skipped / blocked
    deliveries: int = 1                      # times the incident was handed over
    started_unix: float = 0.0
    ended_unix: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    #: the SMS for this same incident, sent after the call
    sms: DemoSmsRecord = field(default_factory=DemoSmsRecord)

    # ------------------------------------------------------------------ #
    @property
    def sms_sent(self) -> bool:
        return self.sms.status == "SENT"

    # ------------------------------------------------------------------ #
    @property
    def is_active(self) -> bool:
        """True while the call is still in progress (drives the overlay banner)."""
        return not self.status.is_terminal

    @property
    def duration_seconds(self) -> float:
        if self.started_unix <= 0.0:
            return 0.0
        end = self.ended_unix if self.ended_unix > 0.0 else utc_now().timestamp()
        return max(0.0, end - self.started_unix)

    def set_status(
        self,
        status: CallStatus,
        provider_status: str = "",
        provider_call_sid: str = "",
        error: str = "",
        reason: str = "",
    ) -> None:
        """Transition the record, keeping a history for the dashboard."""
        self.status = status
        if provider_status:
            self.provider_status = provider_status
        if provider_call_sid:
            self.provider_call_sid = provider_call_sid
        if error:
            self.error = error
        if reason:
            self.reason = reason
        if status.is_terminal and self.ended_unix <= 0.0:
            self.ended_unix = utc_now().timestamp()
        self.history.append(
            {
                "status": str(status),
                "at": now_iso(),
                "provider_status": self.provider_status or None,
            }
        )

    # ------------------------------------------------------------------ #
    def to_dict(self, include_recipient: bool = False) -> Dict[str, Any]:
        """JSON payload for the incident contract / call log / dashboard.

        ``label`` is the literal string the debug overlay and the dashboard show,
        so a consumer never has to invent its own wording.
        """
        payload: Dict[str, Any] = {
            "schema_version": CALL_LOG_SCHEMA_VERSION,
            "label": self.LABEL,
            "call_id": self.call_id,
            "incident_id": self.incident_id,
            "status": str(self.status),
            "provider": self.provider,
            "trigger": self.trigger,
            "recipient": self.recipient,
            "camera_id": self.camera_id,
            "accident_score": (
                None if self.accident_score is None else round(float(self.accident_score), 4)
            ),
            "severity": self.severity,
            "message": self.message,
            "provider_status": self.provider_status or None,
            "provider_call_sid": self.provider_call_sid or None,
            "correlation_id": self.correlation_id or None,
            "error": self.error or None,
            "reason": self.reason or None,
            "deliveries": int(self.deliveries),
            "call_placed": bool(self.status.was_placed),
            "started_at": iso_timestamp(self.started_unix) if self.started_unix else None,
            "ended_at": iso_timestamp(self.ended_unix) if self.ended_unix else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "history": list(self.history),
            "sms": self.sms.to_dict(include_recipient=include_recipient),
        }
        if include_recipient and self.recipient_exposed:
            payload["recipient_full"] = self.recipient_exposed
        return payload

    def to_banner_text(self) -> str:
        """One line for the console / overlay: ``DEMO EMERGENCY CALL - RINGING``."""
        return f"{self.LABEL} - {self.status}"

    def to_sms_banner_text(self) -> str:
        """``DEMO SMS - SENT``"""
        return f"{DEMO_SMS_LABEL} - {self.sms.status}"

    @classmethod
    def masked(cls, number: Optional[str]) -> str:
        """Convenience wrapper so callers never format a number themselves."""
        return mask_phone(number)
