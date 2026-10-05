"""Outbound-call providers for the demonstration call.

The engine never talks to a telephony SDK directly; it asks a
:class:`CallProvider`.  Two are shipped:

``simulation``
    Walks the same status lifecycle a real call goes through
    (``INITIATED -> RINGING -> IN_PROGRESS -> COMPLETED``) without touching a
    network.  This is the default whenever no telephony credentials are
    configured, which is what makes the feature safe to ship and to demo: **no
    credentials, no real call**.

``twilio``
    Places a genuine call through Twilio's REST API using only the standard
    library (``urllib``), so no new dependency is added.  The credentials come
    from the environment / ``.env`` and are never logged.

Neither provider knows anything about accidents - they are handed a recipient
and a message.  All accident-specific policy lives in the dispatcher.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ai_engine.schemas.enums import CallStatus
from ai_engine.utils.logging_utils import get_logger

__all__ = [
    "CallProvider",
    "NullProvider",
    "NullSmsProvider",
    "ProviderResult",
    "SimulationProvider",
    "SimulationSmsProvider",
    "SmsProvider",
    "TwilioProvider",
    "TwilioSmsProvider",
    "build_provider",
    "build_sms_provider",
]

_LOGGER = get_logger("emergency")

# --------------------------------------------------------------------------- #
# Twilio Messaging Templates (trial-account requirement)
# --------------------------------------------------------------------------- #
# Declared up here because they are used as default argument values further
# down, which are evaluated at import time.
#
#: Env var naming the Twilio Messaging Template to send. A **trial** account
#: cannot send free-form SMS: Twilio rejects it with ``572006 Invalid template
#: name. Trial accounts can only use predefined SMS templates.`` A template is
#: addressed by its SID and sent as ``ContentSid``.
TEMPLATE_ENV_VAR = "TWILIO_CONTENT_SID"
#: Optional JSON object of template variables, sent as ``ContentVariables``.
CONTENT_VARIABLES_ENV_VAR = "TWILIO_CONTENT_VARIABLES"

#: The exact Twilio error code for "trial account sent free-form SMS".
TWILIO_TRIAL_TEMPLATE_CODE = "572006"

#: ``on_status(status, provider_status, provider_call_sid, error)``
StatusCallback = Callable[[CallStatus, str, str, str], None]


@dataclass
class ProviderResult:
    """Outcome of handing a call to a provider."""

    ok: bool = False
    status: CallStatus = CallStatus.FAILED
    provider_status: str = ""
    provider_call_sid: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "status": str(self.status),
            "provider_status": self.provider_status or None,
            "provider_call_sid": self.provider_call_sid or None,
            "error": self.error or None,
        }


class CallProvider:
    """Interface for an outbound voice-call backend."""

    name = "base"

    def available(self) -> Tuple[bool, str]:
        """``(usable, reason)`` - ``reason`` explains a ``False``."""
        raise NotImplementedError

    def place(
        self,
        recipient: str,
        message: str,
        on_status: StatusCallback,
    ) -> ProviderResult:
        """Place one call.  Must never raise; report problems in the result."""
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        usable, reason = self.available()
        return {"provider": self.name, "usable": bool(usable), "reason": reason or None}


class NullProvider(CallProvider):
    """Does nothing at all - used when calling is switched off."""

    name = "none"

    def available(self) -> Tuple[bool, str]:
        return False, "call provider is 'none'"

    def place(self, recipient: str, message: str, on_status: StatusCallback) -> ProviderResult:
        return ProviderResult(
            ok=False,
            status=CallStatus.SKIPPED,
            provider_status="disabled",
            error="call provider is 'none'",
        )


class SimulationProvider(CallProvider):
    """Reproduces a call's lifecycle without placing a call.

    The timing mirrors a real call closely enough for a demonstration
    (``ring_after`` then ``duration``), and every transition is reported
    through the same callback the live provider uses - so the dashboard, the
    overlay and the call log are exercised identically either way.
    """

    name = "simulation"

    def __init__(
        self,
        ring_after_seconds: float = 2.0,
        duration_seconds: float = 4.0,
        step_seconds: float = 0.5,
    ) -> None:
        self.ring_after_seconds = max(0.0, float(ring_after_seconds))
        self.duration_seconds = max(0.0, float(duration_seconds))
        self.step_seconds = max(0.05, float(step_seconds))

    def available(self) -> Tuple[bool, str]:
        return True, "simulation provider needs no credentials"

    def place(self, recipient: str, message: str, on_status: StatusCallback) -> ProviderResult:
        call_sid = f"SIM{time.strftime('%H%M%S')}{id(self) % 10000:04d}"
        on_status(CallStatus.RINGING, "ringing", call_sid, "")
        elapsed = 0.0
        while elapsed < self.ring_after_seconds:
            time.sleep(min(self.step_seconds, self.ring_after_seconds - elapsed))
            elapsed += self.step_seconds
        on_status(CallStatus.IN_PROGRESS, "in-progress", call_sid, "")
        remaining = self.duration_seconds
        while remaining > 0.0:
            time.sleep(min(self.step_seconds, remaining))
            remaining -= self.step_seconds
        on_status(CallStatus.COMPLETED, "completed", call_sid, "")
        return ProviderResult(
            ok=True,
            status=CallStatus.COMPLETED,
            provider_status="completed",
            provider_call_sid=call_sid,
        )


class TwilioProvider(CallProvider):
    """Places a real outbound call through Twilio's REST API.

    Uses :mod:`urllib` from the standard library, so SafeVision gains no new
    dependency.  Credentials come from the environment (``.env``) and are never
    written to a log, a file or the incident payload.

    ``TwiML`` is supplied inline, so no web server or public URL is needed.
    """

    name = "twilio"

    API_BASE = "https://api.twilio.com/2010-04-01"

    #: Twilio call states -> our CallStatus
    _STATE_MAP = {
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

    def __init__(
        self,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
        timeout_seconds: float = 10.0,
        poll_attempts: int = 15,
        poll_interval_seconds: float = 4.0,
        voice_message: str = "",
    ) -> None:
        self.account_sid = (account_sid or "").strip()
        self.auth_token = (auth_token or "").strip()
        self.from_number = (from_number or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.poll_attempts = max(0, int(poll_attempts))
        self.poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self.voice_message = voice_message

    # ------------------------------------------------------------------ #
    def available(self) -> Tuple[bool, str]:
        missing = [
            name
            for name, value in (
                ("TWILIO_ACCOUNT_SID", self.account_sid),
                ("TWILIO_AUTH_TOKEN", self.auth_token),
                ("TWILIO_FROM_NUMBER", self.from_number),
            )
            if not value
        ]
        if missing:
            return False, f"missing Twilio credential(s) in .env: {', '.join(missing)}"
        if not self.account_sid.startswith("AC"):
            return False, "TWILIO_ACCOUNT_SID does not look like an Account SID (AC...)"
        return True, "Twilio credentials found in the environment"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({"from_number": self.from_number or None})
        return info

    # ------------------------------------------------------------------ #
    def _auth_header(self) -> str:
        raw = f"{self.account_sid}:{self.auth_token}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _request(self, url: str, data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        body = urllib.parse.urlencode(data).encode("utf-8") if data else None
        request = urllib.request.Request(url, data=body)
        request.add_header("Authorization", self._auth_header())
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    # ------------------------------------------------------------------ #
    def place(self, recipient: str, message: str, on_status: StatusCallback) -> ProviderResult:
        usable, reason = self.available()
        if not usable:
            return ProviderResult(ok=False, status=CallStatus.BLOCKED, error=reason)

        twiml = (
            "<Response><Say>"
            + _xml_escape(message or "SafeVision demonstration alert.")
            + "</Say></Response>"
        )
        try:
            payload = self._request(
                f"{self.API_BASE}/Accounts/{self.account_sid}/Calls.json",
                {"To": recipient, "From": self.from_number, "TwiML": twiml},
            )
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            _LOGGER.error("Twilio rejected the call (%s): %s", exc.code, detail)
            return ProviderResult(
                ok=False,
                status=CallStatus.FAILED,
                provider_status=f"http_{exc.code}",
                error=detail,
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            _LOGGER.error("could not reach the Twilio API: %s", exc)
            return ProviderResult(
                ok=False, status=CallStatus.FAILED, provider_status="network_error", error=str(exc)
            )

        sid = str(payload.get("sid", ""))
        state = str(payload.get("status", "")).lower()
        status = self._STATE_MAP.get(state, CallStatus.INITIATED)
        on_status(status, state, sid, "")

        if status.is_terminal:
            return ProviderResult(
                ok=status is CallStatus.COMPLETED,
                status=status,
                provider_status=state,
                provider_call_sid=sid,
                error="" if status is CallStatus.COMPLETED else f"Twilio reported '{state}'",
            )

        self._poll(sid, on_status)
        return ProviderResult(
            ok=True, status=status, provider_status=state, provider_call_sid=sid
        )

    def _poll(self, call_sid: str, on_status: StatusCallback) -> None:
        """Follow the call until it reaches a terminal state (best effort)."""
        url = f"{self.API_BASE}/Accounts/{self.account_sid}/Calls/{call_sid}.json"
        for _ in range(self.poll_attempts):
            time.sleep(self.poll_interval_seconds)
            try:
                payload = self._request(url)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                _LOGGER.debug("Twilio status poll failed: %s", exc)
                return
            state = str(payload.get("status", "")).lower()
            status = self._STATE_MAP.get(state, CallStatus.IN_PROGRESS)
            on_status(status, state, call_sid, "")
            if status.is_terminal:
                return


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
class NotConfiguredCallProvider(CallProvider):
    """No telephony provider is configured.  **Phase 4.**

    This provider exists to make one specific failure mode impossible: a run
    with no credentials must never *report* a successful notification.  It
    refuses to dial and reports :attr:`CallStatus.NOT_CONFIGURED`, which is
    distinct from ``FAILED`` (the provider refused) and from ``COMPLETED``
    (a connection was established).

    Before this class existed, ``provider: auto`` with no credentials silently
    resolved to the simulation provider and reported ``COMPLETED`` / ``SENT``.
    A dashboard could not tell that from a real notification. That was the
    highest-severity finding of the Phase 1 audit.
    """

    name = "not_configured"

    def __init__(self, reason: str = "no telephony credentials are configured") -> None:
        self.reason = reason

    def available(self) -> Tuple[bool, str]:
        return False, self.reason

    def place(
        self,
        recipient: str,
        message: str,
        on_status: StatusCallback,
    ) -> ProviderResult:
        # Deliberately does NOT call on_status: no call was attempted, so there is
        # no transition to report.
        return ProviderResult(
            ok=False,
            status=CallStatus.NOT_CONFIGURED,
            error=f"NOT_CONFIGURED: {self.reason}",
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "usable": False,
            "reason": self.reason,
            "notifies": False,
        }


def build_provider(
    provider: str = "auto",
    environ: Optional[Mapping[str, str]] = None,
    *,
    ring_after_seconds: float = 2.0,
    duration_seconds: float = 4.0,
    timeout_seconds: float = 10.0,
    poll_attempts: int = 15,
    poll_interval_seconds: float = 4.0,
    voice_message: str = "",
    sid_env: str = "TWILIO_ACCOUNT_SID",
    token_env: str = "TWILIO_AUTH_TOKEN",
    from_env: str = "TWILIO_FROM_NUMBER",
    demo_mode: bool = False,
) -> CallProvider:
    """Pick a call provider.

    ``auto`` uses Twilio when its credentials are usable.  When they are not, the
    answer depends on ``demo_mode``:

    * ``demo_mode=True`` - the simulation provider.  This is an **explicit**
      request to simulate, and simulation is the correct answer to it.
    * ``demo_mode=False`` - :class:`NotConfiguredCallProvider`, which reports
      ``NOT_CONFIGURED`` and notifies nobody.

    The ``demo_mode`` flag must be passed through, not inferred.  Inferring it
    is exactly how the pre-Phase-4 silent substitution happened: the engine had
    no credentials, guessed "simulation", and reported a completed call.

    An explicit ``provider: twilio`` never falls back at all - it returns the
    real provider, whose ``available()`` is False with an explicit reason.
    """
    env = os.environ if environ is None else environ
    choice = (provider or "auto").strip().lower()

    def twilio() -> TwilioProvider:
        return TwilioProvider(
            account_sid=env.get(sid_env, ""),
            auth_token=env.get(token_env, ""),
            from_number=env.get(from_env, ""),
            timeout_seconds=timeout_seconds,
            poll_attempts=poll_attempts,
            poll_interval_seconds=poll_interval_seconds,
            voice_message=voice_message,
        )

    if choice == "none":
        return NullProvider()
    if choice == "simulation":
        return SimulationProvider(ring_after_seconds, duration_seconds)
    if choice == "twilio":
        # never silently degrade an explicit choice
        return twilio()

    usable, reason = twilio().available()
    if usable:
        return twilio()
    if demo_mode:
        return SimulationProvider(ring_after_seconds, duration_seconds)
    return NotConfiguredCallProvider(
        reason=f"provider=auto and Twilio is unusable ({reason}); "
        "set DEMO_MODE to simulate deliberately, or configure credentials"
    )


# --------------------------------------------------------------------------- #
# SMS providers
# --------------------------------------------------------------------------- #
class SmsProvider:
    """Interface for an outbound SMS backend.

    Separate from :class:`CallProvider` on purpose: a deployment may be able to
    place voice calls but not send SMS, or the reverse.
    """

    name = "base"

    def available(self) -> Tuple[bool, str]:
        raise NotImplementedError

    def send(self, recipient: str, body: str) -> ProviderResult:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        usable, reason = self.available()
        return {"provider": self.name, "usable": bool(usable), "reason": reason or None}


class NullSmsProvider(SmsProvider):
    """SMS switched off."""

    name = "none"

    def available(self) -> Tuple[bool, str]:
        return False, "SMS provider is 'none'"

    def send(self, recipient: str, body: str) -> ProviderResult:
        return ProviderResult(
            ok=False, status=CallStatus.SKIPPED, error="SMS provider is 'none'"
        )


class SimulationSmsProvider(SmsProvider):
    """Records the SMS without contacting any network.

    Used by default whenever no telephony credentials are present, which is
    what makes DEMO_MODE safe to show on a machine with no Twilio account.
    """

    name = "simulation"

    def available(self) -> Tuple[bool, str]:
        return True, "simulation SMS provider needs no credentials"

    def send(self, recipient: str, body: str) -> ProviderResult:
        sid = f"SMSIM{time.strftime('%H%M%S')}{len(body) % 10000:04d}"
        return ProviderResult(
            ok=True,
            status=CallStatus.COMPLETED,
            provider_status="queued",
            provider_call_sid=sid,
        )


class NotConfiguredSmsProvider(SmsProvider):
    """No SMS provider is configured.  **Phase 4.**

    Mirrors :class:`NotConfiguredCallProvider`: it reports ``NOT_CONFIGURED``
    instead of returning a fake ``SENT``, so a run without credentials can
    never be mistaken for one that messaged somebody.
    """

    name = "not_configured"

    def __init__(self, reason: str = "no SMS credentials are configured") -> None:
        self.reason = reason

    def available(self) -> Tuple[bool, str]:
        return False, self.reason

    def send(self, recipient: str, body: str) -> ProviderResult:
        return ProviderResult(
            ok=False,
            status=CallStatus.NOT_CONFIGURED,
            error=f"NOT_CONFIGURED: {self.reason}",
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "usable": False,
            "reason": self.reason,
            "notifies": False,
        }


class TwilioSmsProvider(SmsProvider):
    """Sends a real SMS through Twilio's Messages REST API (stdlib only)."""

    name = "twilio"

    API_BASE = "https://api.twilio.com/2010-04-01"

    def __init__(
        self,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
        timeout_seconds: float = 10.0,
        max_body_length: int = 1500,
        content_sid: str = "",
        content_variables: str = "",
    ) -> None:
        self.account_sid = (account_sid or "").strip()
        self.auth_token = (auth_token or "").strip()
        self.from_number = (from_number or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_body_length = max(160, int(max_body_length))
        #: Twilio Messaging Template SID. Required for TRIAL accounts, which
        #: reject free-form bodies with error 572006. Empty = free-form.
        self.content_sid = (content_sid or "").strip()
        self.content_variables = (content_variables or "").strip()

    def available(self) -> Tuple[bool, str]:
        missing = [
            name
            for name, value in (
                ("TWILIO_ACCOUNT_SID", self.account_sid),
                ("TWILIO_AUTH_TOKEN", self.auth_token),
                ("TWILIO_PHONE_NUMBER", self.from_number),
            )
            if not value
        ]
        if missing:
            return False, f"missing Twilio credential(s) in .env: {', '.join(missing)}"
        if not self.account_sid.startswith("AC"):
            return False, "TWILIO_ACCOUNT_SID does not look like an Account SID (AC...)"
        return True, "Twilio credentials found in the environment"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update({"from_number": self.from_number or None})
        return info

    def _auth_header(self) -> str:
        raw = f"{self.account_sid}:{self.auth_token}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def send(self, recipient: str, body: str) -> ProviderResult:
        usable, reason = self.available()
        if not usable:
            return ProviderResult(ok=False, status=CallStatus.BLOCKED, error=reason)
        url = f"{self.API_BASE}/Accounts/{self.account_sid}/Messages.json"
        data = build_message_payload(
            recipient,
            self.from_number,
            body,
            content_sid=self.content_sid,
            content_variables=self.content_variables,
            max_body_length=self.max_body_length,
        )
        try:
            request = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode("utf-8"))
            request.add_header("Authorization", self._auth_header())
            request.add_header("Accept", "application/json")
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = explain_sms_error(exc.code, _http_error_detail(exc))
            _LOGGER.error("Twilio rejected the SMS (%s): %s", exc.code, detail)
            return ProviderResult(
                ok=False,
                status=CallStatus.FAILED,
                provider_status=f"http_{exc.code}",
                error=detail,
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            _LOGGER.error("could not reach the Twilio API for the SMS: %s", exc)
            return ProviderResult(
                ok=False, status=CallStatus.FAILED, provider_status="network_error", error=str(exc)
            )
        return ProviderResult(
            ok=True,
            status=CallStatus.COMPLETED,
            provider_status=str(payload.get("status") or "queued"),
            provider_call_sid=str(payload.get("sid") or ""),
        )


def build_sms_provider(
    provider: str = "auto",
    environ: Optional[Mapping[str, str]] = None,
    *,
    timeout_seconds: float = 10.0,
    max_body_length: int = 1500,
    sid_env: str = "TWILIO_ACCOUNT_SID",
    token_env: str = "TWILIO_AUTH_TOKEN",
    phone_env: str = "TWILIO_PHONE_NUMBER",
    content_sid_env: str = TEMPLATE_ENV_VAR,
    content_variables_env: str = CONTENT_VARIABLES_ENV_VAR,
    demo_mode: bool = False,
) -> SmsProvider:
    """Pick an SMS provider; same safety rule as :func:`build_provider`.

    With ``demo_mode=False`` and no credentials, this returns
    :class:`NotConfiguredSmsProvider` (reports ``NOT_CONFIGURED``) instead of
    silently simulating a sent message.
    """
    env = os.environ if environ is None else environ
    choice = (provider or "auto").strip().lower()

    def twilio() -> TwilioSmsProvider:
        return TwilioSmsProvider(
            account_sid=env.get(sid_env, ""),
            auth_token=env.get(token_env, ""),
            # the brief's canonical name is TWILIO_PHONE_NUMBER; fall back to the
            # call's TWILIO_FROM_NUMBER so one credential set drives both
            from_number=env.get(phone_env, "") or env.get("TWILIO_FROM_NUMBER", ""),
            timeout_seconds=timeout_seconds,
            max_body_length=max_body_length,
            # Twilio trial accounts require a Messaging Template SID; without it
            # Twilio rejects the request with 572006.
            content_sid=env.get(content_sid_env, ""),
            content_variables=env.get(content_variables_env, ""),
        )

    if choice == "none":
        return NullSmsProvider()
    if choice == "simulation":
        return SimulationSmsProvider()
    if choice == "twilio":
        return twilio()
    usable, reason = twilio().available()
    if usable:
        return twilio()
    if demo_mode:
        return SimulationSmsProvider()
    return NotConfiguredSmsProvider(
        reason=f"provider=auto and Twilio is unusable ({reason}); "
        "set DEMO_MODE to simulate deliberately, or configure credentials"
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _xml_escape(text: str) -> str:
    """Escape a message for inline TwiML (``<Say>``)."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------- #
# Twilio Messaging Templates (trial-account requirement)
# --------------------------------------------------------------------------- #
# Trial-account errors we can explain better than Twilio does.
SMS_ERROR_HINTS = {
    TWILIO_TRIAL_TEMPLATE_CODE: (
        "This Twilio account is a TRIAL account, so Twilio will not accept a "
        "free-form SMS body. Set {var}=HX... (a Twilio Messaging Template SID, "
        "Console > Messaging > Templates > your approved template) to send via a "
        "predefined template instead. Upgrade the account to send free-form SMS."
    ),
    "21612": (
        "Twilio could not route the SMS to that number. A trial account can only "
        "send to VERIFIED recipient numbers - add the number under Console > "
        "Phone Numbers > Manage > Verified Caller IDs, or upgrade the account."
    ),
    "21614": (
        "The recipient number is not verified for this trial account. Add it under "
        "Console > Phone Numbers > Manage > Verified Caller IDs, or upgrade."
    ),
    "21211": (
        "'From' is not a valid Twilio number for SMS. Trial accounts cannot send "
        "from an alphanumeric sender; buy an SMS-capable number or upgrade."
    ),
}


def build_message_payload(
    recipient: str,
    sender: str,
    body: str = "",
    *,
    content_sid: str = "",
    content_variables: Optional[str] = None,
    max_body_length: int = 1500,
) -> Dict[str, str]:
    """Build the ``Messages.json`` form fields.

    Two mutually exclusive shapes, because Twilio rejects ``Body`` together with
    ``ContentSid``:

    * **template** (``content_sid`` given): ``To``, ``From``, ``ContentSid`` and
      optional ``ContentVariables``. This is the only shape a trial account
      accepts.
    * **free-form** (no ``content_sid``): ``To``, ``From``, ``Body``. Works on a
      paid account, and is rejected by trial accounts with 572006.

    Shared by the production provider and the gated live test so the two can
    never drift apart.
    """
    data: Dict[str, str] = {"To": recipient, "From": sender}
    template = (content_sid or "").strip()
    if template:
        data["ContentSid"] = template
        variables = (content_variables or "").strip()
        if variables:
            data["ContentVariables"] = variables
    else:
        data["Body"] = (body or "")[: max(160, int(max_body_length))]
    return data


def explain_sms_error(code: int, detail: str) -> str:
    """Turn a Twilio rejection into something the operator can act on.

    The provider's own wording is always preserved; the hint is appended. The
    auth token is never part of ``detail`` and never added here.
    """
    text = detail or f"HTTP {code}"
    for marker, hint in SMS_ERROR_HINTS.items():
        if marker in text:
            return f"{text} | {hint.format(var=TEMPLATE_ENV_VAR)}"
    return text


def _http_error_detail(exc: "urllib.error.HTTPError") -> str:
    """Twilio reports the reason in the JSON body - surface it verbatim."""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        return str(payload.get("message") or payload.get("code") or raw[:200])
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return f"HTTP {exc.code}"
