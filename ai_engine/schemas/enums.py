"""Shared enumerations.

They live in ``schemas`` (not in the analysis modules) so that the verification
state machine, the severity analyzer and the JSON output can all reference the
same symbols without circular imports.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["CallStatus", "DetectionState", "IncidentStatus", "SeverityLevel", "SmsStatus"]


class _StrEnum(str, Enum):
    """String enum whose ``str()`` is the value - keeps JSON clean."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    @classmethod
    def coerce(cls, value: object, default: "_StrEnum | None" = None):
        """Best-effort conversion from a raw value (config, CLI, JSON)."""
        if isinstance(value, cls):
            return value
        if value is None:
            return default
        text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value == text or member.name == text:
                return member
        if default is not None:
            return default
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")


class DetectionState(_StrEnum):
    """Internal state machine states (Phase 12)."""

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    VERIFYING = "VERIFYING"
    CONFIRMED_ACCIDENT = "CONFIRMED_ACCIDENT"
    FALSE_ALARM = "FALSE_ALARM"


class IncidentStatus(_StrEnum):
    """Status reported to downstream systems (Phase 10 / 17)."""

    NORMAL = "NORMAL"
    POSSIBLE_ACCIDENT = "POSSIBLE_ACCIDENT"
    CONFIRMED_ACCIDENT = "CONFIRMED_ACCIDENT"
    FALSE_ALARM = "FALSE_ALARM"


class SeverityLevel(_StrEnum):
    """AI-estimated *scene* severity (Phase 13).

    This is not a medical or injury classification.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CallStatus(_StrEnum):
    """Status of the **demonstration** emergency call (demo mode only).

    These are call-control states, not medical states.  ``SKIPPED`` and
    ``BLOCKED`` mean *no call was placed* - a safety interlock refused, which
    is always a success of the false-alarm prevention, never a failure.
    """

    #: accepted for processing, no call placed yet
    INITIATED = "INITIATED"
    RINGING = "RINGING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    #: terminal, unsuccessful
    BUSY = "BUSY"
    NO_ANSWER = "NO_ANSWER"
    FAILED = "FAILED"
    #: deliberately not dialled (demo mode off, no recipient, duplicate, ...)
    SKIPPED = "SKIPPED"
    #: a safety interlock refused to dial (misconfiguration, not a failure)
    BLOCKED = "BLOCKED"
    #: **Phase 4.** No provider is configured, so nothing was dialled.  This is
    #: deliberately distinct from both ``COMPLETED`` and ``FAILED``: it is not a
    #: delivery outcome at all, it is the absence of the ability to attempt one.
    #: A run in which this appears has notified nobody.
    NOT_CONFIGURED = "NOT_CONFIGURED"

    @property
    def is_terminal(self) -> bool:
        """True once no further transition is expected."""
        return self in (
            CallStatus.COMPLETED,
            CallStatus.BUSY,
            CallStatus.NO_ANSWER,
            CallStatus.FAILED,
            CallStatus.SKIPPED,
            CallStatus.BLOCKED,
            CallStatus.NOT_CONFIGURED,
        )

    @property
    def was_placed(self) -> bool:
        """True when a provider took the call and it did not fail outright."""
        return self in (
            CallStatus.RINGING,
            CallStatus.IN_PROGRESS,
            CallStatus.COMPLETED,
            CallStatus.BUSY,
            CallStatus.NO_ANSWER,
        )

    @property
    def reached_recipient(self) -> bool:
        """True when the call reached the recipient's network.

        ``BUSY`` / ``NO_ANSWER`` count: the incident is real whether or not
        somebody picked up.  ``INITIATED`` (queued) and ``FAILED`` (the provider
        refused or was unreachable) do not - which is why the SMS waits for this.
        """
        return self.was_placed


class SmsStatus(_StrEnum):
    """States of the demonstration SMS.

    An SMS has no ringing phase, so the call states do not apply.  Keeping a
    separate vocabulary stops the contract from reporting a call state for a
    channel that never had one.
    """

    #: accepted, nothing sent yet
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    #: **Phase 4.** No SMS provider is configured.  Not a delivery outcome -
    #: see :attr:`CallStatus.NOT_CONFIGURED`.
    NOT_CONFIGURED = "NOT_CONFIGURED"

    @property
    def is_terminal(self) -> bool:
        """True once the outcome is known (sent, failed or deliberately not sent)."""
        return self in (
            SmsStatus.SENT,
            SmsStatus.FAILED,
            SmsStatus.SKIPPED,
            SmsStatus.BLOCKED,
            SmsStatus.NOT_CONFIGURED,
        )
