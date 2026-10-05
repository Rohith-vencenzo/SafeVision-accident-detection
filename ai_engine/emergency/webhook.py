"""Provider callback verification.  Phase 4.

The Phase 1 audit found the code **polls** and registers no ``StatusCallback``:
it reads the provider on a timer and never accepts an inbound request.  That is
not unsafe, but it is the wrong shape - it burns the notification thread for up
to 60 s and cannot learn about a state change the poll interval misses.

Twilio supports ``StatusCallback`` on call and message creation.  Accepting
those callbacks requires answering a question the old polling design never had
to: *is this request really from the provider?*  Anyone on the internet can POST
to a webhook URL, and a forged ``completed`` callback would make the engine
report that a call connected.

So a callback is only applied when its ``X-Twilio-Signature`` verifies, per
Twilio's documented algorithm:

    HMAC-SHA1(auth_token, url + sorted_concatenated_post_params), base64

Two consequences this module enforces:

* **No signature, no state change.**  An unverifiable request is rejected and
  logged; it never touches the delivery record.
* **Provider acceptance is not delivery.**  A verified callback is still only
  evidence that *the provider* reported a state.  ``completed`` means a
  connection was established - possibly with voicemail - so nothing here ever
  reports that a human answered.

The receiver is opt-in and off by default; no HTTP server is started unless an
operator enables it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

from ai_engine.schemas.enums import CallStatus
from ai_engine.utils.logging_utils import get_logger

__all__ = [
    "CallbackDecision",
    "VerifiedCallback",
    "verify_twilio_signature",
    "verify_request",
    "call_status_from_twilio",
    "sms_status_from_twilio",
]

_LOGGER = get_logger("emergency.webhook")

#: Twilio's documented call status vocabulary, mapped to ours.  Unchanged from
#: the polling implementation so a state means the same thing either way.
_CALL_STATUS_MAP = {
    "queued": CallStatus.INITIATED,
    "initiated": CallStatus.INITIATED,
    "ringing": CallStatus.RINGING,
    "in-progress": CallStatus.IN_PROGRESS,
    "completed": CallStatus.COMPLETED,
    "busy": CallStatus.BUSY,
    "no-answer": CallStatus.NO_ANSWER,
    "failed": CallStatus.FAILED,
    "canceled": CallStatus.FAILED,
}

_SMS_STATUS_MAP = {
    "queued": "PENDING",
    "sending": "PENDING",
    "sent": "SENT",
    "delivered": "SENT",
    "failed": "FAILED",
    "undelivered": "FAILED",
}


# --------------------------------------------------------------------------- #
# signature
# --------------------------------------------------------------------------- #
def compute_signature(url: str, params: Mapping[str, str], auth_token: str) -> str:
    """Twilio's documented algorithm: HMAC-SHA1 over url + sorted params."""
    payload = url
    for key in sorted(params):
        payload += f"{key}{params[key]}"
    digest = hmac.new(
        auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def verify_twilio_signature(
    url: str,
    params: Mapping[str, str],
    auth_token: str,
    signature: str,
) -> bool:
    """Constant-time signature check.

    ``hmac.compare_digest`` is used so the comparison cannot leak the expected
    value through timing.
    """
    if not auth_token or not signature:
        return False
    expected = compute_signature(url, params, auth_token)
    return hmac.compare_digest(expected, signature.strip())


def verify_request(
    url: str,
    params: Mapping[str, str],
    auth_token: str,
    signature: Optional[str],
    *,
    tolerance_seconds: float = 300.0,
    now: Optional[float] = None,
    require_https: bool = True,
) -> Tuple[bool, str]:
    """``(accepted, reason)`` for one inbound callback.

    Checks, in order: HTTPS, signature presence, signature correctness, then
    replay freshness.  Rejection reasons are specific on purpose - an operator
    debugging a webhook needs to know *which* check failed.
    """
    if require_https and urlparse(url).scheme != "https":
        return False, f"callback URL must be https, got {urlparse(url).scheme!r}"
    if not signature:
        return False, "missing X-Twilio-Signature header"
    if not verify_twilio_signature(url, params, auth_token, signature):
        return False, "signature does not match (forged, or the auth token changed)"
    # Twilio does not sign its own timestamp, so freshness is checked on the
    # value Twilio sends.  Without it a captured callback could be replayed
    # forever and keep re-applying states.
    stamp = params.get("Timestamp") or params.get("timestamp")
    if stamp:
        try:
            age = (time.time() if now is None else now) - float(stamp)
        except (TypeError, ValueError):
            return False, "Timestamp is not numeric"
        if age > tolerance_seconds:
            return False, f"callback is {age:.0f}s old, tolerance is {tolerance_seconds:.0f}s"
        if age < -tolerance_seconds:
            return False, "callback timestamp is in the future"
    return True, "verified"


# --------------------------------------------------------------------------- #
# status mapping
# --------------------------------------------------------------------------- #
def call_status_from_twilio(provider_status: str) -> Optional[CallStatus]:
    return _CALL_STATUS_MAP.get((provider_status or "").strip().lower())


def sms_status_from_twilio(provider_status: str) -> Optional[str]:
    return _SMS_STATUS_MAP.get((provider_status or "").strip().lower())


# --------------------------------------------------------------------------- #
# idempotent application
# --------------------------------------------------------------------------- #
@dataclass
class VerifiedCallback:
    """A callback that passed verification and has been interpreted."""

    channel: str                    # "call" | "sms"
    incident_id: str
    provider_ref: str               # CallSid / MessageSid
    provider_status: str
    status: Any                     # CallStatus or SmsStatus
    applied: bool
    reason: str = ""
    correlation_id: str = ""


@dataclass
class CallbackDecision:
    accepted: bool
    reason: str
    callbacks: list = field(default_factory=list)


class CallbackApplier:
    """Applies verified callbacks to the outbox, exactly once per state.

    Replay protection is durable: an already-applied ``(provider_ref,
    provider_status)`` pair is a no-op that is logged, not an error.  Providers
    legitimately resend callbacks, and a webhook that fails on a duplicate is a
    webhook that gets disabled.
    """

    def __init__(self, outbox: Any, correlation_for: Optional[Callable[[str], str]] = None) -> None:
        self.outbox = outbox
        self.correlation_for = correlation_for or (lambda ref: ref)
        #: (provider_ref, provider_status) pairs already applied, in memory.
        #: The outbox row state is the durable record; this is a fast path.
        self._applied: set = set()
        self._lock_free = True

    def apply(
        self,
        channel: str,
        incident_id: str,
        provider_ref: str,
        provider_status: str,
    ) -> VerifiedCallback:
        from ai_engine.schemas.enums import SmsStatus

        key = (provider_ref, (provider_status or "").lower())
        if key in self._applied:
            return VerifiedCallback(
                channel, incident_id, provider_ref, provider_status, None,
                applied=False, reason="duplicate callback for an already-applied state",
            )

        if channel == "call":
            status = call_status_from_twilio(provider_status)
            terminal_map = {
                CallStatus.COMPLETED: "DONE",
                CallStatus.BUSY: "DONE",
                CallStatus.NO_ANSWER: "DONE",
                CallStatus.FAILED: "FAILED",
            }
        else:
            mapped = sms_status_from_twilio(provider_status)
            status = SmsStatus(mapped) if mapped else None
            terminal_map = {
                SmsStatus.SENT: "DONE",
                SmsStatus.FAILED: "FAILED",
            }

        if status is None:
            return VerifiedCallback(
                channel, incident_id, provider_ref, provider_status, None,
                applied=False,
                reason=f"unrecognised provider status {provider_status!r}; ignored",
            )

        row = self.outbox.get(incident_id, channel)
        if row is None:
            return VerifiedCallback(
                channel, incident_id, provider_ref, provider_status, status,
                applied=False,
                reason="no outbox row for this incident/channel; ignoring",
            )
        if row.is_terminal:
            self._applied.add(key)
            return VerifiedCallback(
                channel, incident_id, provider_ref, provider_status, status,
                applied=False,
                reason=f"already terminal ({row.state}); not overwritten",
            )

        target = terminal_map.get(status)
        if target:
            self.outbox.complete(
                incident_id, channel, state=target,
                provider=row.provider or channel,
                provider_ref=provider_ref,
                provider_status=provider_status,
                detail=f"callback {provider_status}",
            )
        else:
            self.outbox._update(   # noqa: SLF001 - non-terminal progress note
                incident_id, channel, provider_ref=provider_ref,
                provider_status=provider_status, detail=f"callback {provider_status}",
            )
        self._applied.add(key)
        return VerifiedCallback(
            channel, incident_id, provider_ref, provider_status, status,
            applied=True, reason="applied", correlation_id=self.correlation_for(provider_ref),
        )