"""Automatic demonstration call on a *confirmed* incident.

Policy (all of it lives here, not in the providers):

1. Nothing happens unless ``DEMO_MODE`` is true.  That flag comes from the
   environment / ``.env``; it is off by default.
2. Nothing happens unless the incident reached ``CONFIRMED_ACCIDENT`` - i.e.
   it already passed the whole false-alarm-prevention pipeline.  A
   ``SUSPICIOUS`` or ``VERIFYING`` frame never triggers a call.
3. **Exactly one call per confirmed incident.**  The ``incident_id`` is
   remembered, so however many frames follow the confirmation - and however
   often the callback is invoked - the recipient is dialled once.
4. Nothing is dialled unless a recipient is configured *and* the selected
   provider is usable.  With no telephony credentials the provider is the
   simulation provider, so an accidental run cannot ring anybody's phone.
5. There is no button and no manual trigger anywhere in this module.  The call
   is automatic, and the reason is always recorded.

The dispatcher hands the call to a worker thread so the video loop never
stalls behind a network call, and exposes a live status so the debug overlay
and the downstream dashboard can show ``DEMO EMERGENCY CALL`` plus its state.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

from ai_engine.config.classes import EmergencyConfig
from ai_engine.emergency.call_record import TRIGGER_AUTOMATIC, DemoCallRecord
from ai_engine.emergency.envfile import env_flag
from ai_engine.emergency.providers import (
    CallProvider,
    ProviderResult,
    SmsProvider,
    build_provider,
    build_sms_provider,
)
from ai_engine.emergency.recipient import RecipientError, mask_phone, normalize_phone
from ai_engine.emergency.outbox import CHANNELS, DeliveryRow, NotificationOutbox, OutboxState
from ai_engine.emergency.reliability import (
    CircuitBreaker,
    CircuitOpen,
    RetryPolicy,
    run_with_retry,
)
from ai_engine.emergency.sms_record import DEMO_SMS_LABEL, DemoSmsRecord, SmsStatus, build_sms_body
from ai_engine.schemas.enums import CallStatus, IncidentStatus
from ai_engine.utils.jsonio import to_jsonable
from ai_engine.utils.logging_utils import get_logger
from ai_engine.utils.timeutil import utc_now

__all__ = ["DemoCallDispatcher", "DEMO_CALL_DISCLAIMER", "DEMO_SMS_DISCLAIMER"]

_LOGGER = get_logger("emergency")

#: Shown in every payload and printed before a call, so nobody can mistake the
#: demonstration for a real emergency notification.
DEMO_CALL_DISCLAIMER = (
    "Demonstration call to the authorised SafeVision demo recipient. It is not "
    "connected to any ambulance, police or 112 line and no emergency service is "
    "notified."
)

#: Same, for the SMS that follows the call.
DEMO_SMS_DISCLAIMER = (
    "Demonstration SMS to the authorised SafeVision demo recipient. It is not "
    "connected to any ambulance, police or 112 line and no emergency service is "
    "notified."
)

StatusCallback = Callable[[Dict[str, Any]], None]

_CALL_ID_LOCK = threading.Lock()
_CALL_SEQUENCE = {"n": 0}


def _next_call_id() -> str:
    with _CALL_ID_LOCK:
        _CALL_SEQUENCE["n"] += 1
        index = _CALL_SEQUENCE["n"]
    return f"CALL-{utc_now().strftime('%Y%m%d-%H%M%S')}-{index:03d}"


class DemoCallDispatcher:
    """Places at most one demonstration call per confirmed incident."""

    def __init__(
        self,
        config: Optional[EmergencyConfig] = None,
        environ: Optional[MutableMapping[str, str]] = None,
        provider: Optional[CallProvider] = None,
        sms_provider: Optional[SmsProvider] = None,
        on_status: Optional[StatusCallback] = None,
        log_path: Optional[str | os.PathLike] = None,
        clock: Callable[[], float] = time.time,
        outbox: Optional[NotificationOutbox] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.config = config or EmergencyConfig()
        self.environ: MutableMapping[str, str] = os.environ if environ is None else environ
        self.on_status = on_status
        self.clock = clock

        # Phase 4: resolve DEMO_MODE *before* choosing providers, so `auto` can
        # tell "not configured, do not simulate" from "simulate deliberately".
        # Cached so the provider choice and the later gates cannot disagree.
        self._demo_mode = bool(
            env_flag(self.environ, "DEMO_MODE", default=bool(self.config.demo_mode))
        )
        demo = self._demo_mode

        if provider is not None:
            self.provider = provider
        else:
            self.provider = build_provider(
                self.config.provider,
                self.environ,
                ring_after_seconds=self.config.simulated_ring_after_seconds,
                duration_seconds=self.config.simulated_duration_seconds,
                timeout_seconds=self.config.request_timeout_seconds,
                poll_attempts=self.config.poll_attempts,
                poll_interval_seconds=self.config.poll_interval_seconds,
                voice_message=self.config.voice_message,
                sid_env=self.config.twilio_sid_env_var,
                token_env=self.config.twilio_token_env_var,
                from_env=self.config.twilio_from_env_var,
                demo_mode=demo,
            )
        # The SMS rides the same credentials and the same "auto = no credentials
        # means no network" rule, so a checkout without .env keys stays offline.
        self.sms_provider = sms_provider or build_sms_provider(
            self.config.sms_provider,
            self.environ,
            timeout_seconds=self.config.request_timeout_seconds,
            max_body_length=self.config.sms_max_body_length,
            sid_env=self.config.twilio_sid_env_var,
            token_env=self.config.twilio_token_env_var,
            phone_env=self.config.twilio_phone_env_var,
            content_sid_env=self.config.twilio_content_sid_env_var,
            content_variables_env=self.config.twilio_content_variables_env_var,
            demo_mode=demo,
        )

        self._lock = threading.Lock()
        self._calls: List[DemoCallRecord] = []
        self._by_incident: Dict[str, DemoCallRecord] = {}
        self._threads: List[threading.Thread] = []
        self._latest: Optional[DemoCallRecord] = None

        # ---- Phase 4: durable delivery + reliability ---------------------- #
        self._owns_outbox = False
        self.outbox: Optional[NotificationOutbox] = outbox
        if self.outbox is None and self.config.use_outbox:
            try:
                self.outbox = NotificationOutbox(
                    self.config.outbox_path or ":memory:",
                    lease_seconds=self.config.outbox_lease_seconds,
                )
                self._owns_outbox = True
            except Exception as exc:  # noqa: BLE001 - durability is best-effort
                _LOGGER.error(
                    "could not open the notification outbox at %r (%s); continuing "
                    "WITHOUT durable exactly-once delivery",
                    self.config.outbox_path, exc,
                )
                self.outbox = None
        self._retry_policy = RetryPolicy(
            max_attempts=self.config.retry_max_attempts,
            base_delay_seconds=self.config.retry_base_delay_seconds,
            max_delay_seconds=self.config.retry_max_delay_seconds,
        )
        self._breaker = CircuitBreaker(
            failure_threshold=self.config.breaker_failure_threshold,
            reset_timeout_seconds=self.config.breaker_reset_timeout_seconds,
        )
        #: replaced in tests to avoid real sleeping
        self._sleep = sleep or time.sleep
        #: last status pushed to the subscriber per call id, so a terminal state
        #: is reported exactly once even when _emit and _finalize both fire
        self._emitted: Dict[str, str] = {}
        self._closed = False
        self._started_wall = self.clock()

        self.log_path = (
            Path(log_path)
            if log_path is not None
            else (Path(self.config.call_log) if self.config.write_call_log else None)
        )

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    @property
    def demo_mode(self) -> bool:
        """``DEMO_MODE`` from the environment, or the configured fallback.

        Re-read on every access rather than snapshotted in ``__init__``: callers
        do flip the flag on a live dispatcher, and a cached copy would let the
        provider chosen at construction disagree with the gate that runs later.
        The provider is still chosen once, but the *gate* always sees the truth.
        """
        return bool(env_flag(self.environ, "DEMO_MODE", default=bool(self.config.demo_mode)))

    @property
    def calls(self) -> List[DemoCallRecord]:
        with self._lock:
            return list(self._calls)

    @property
    def latest(self) -> Optional[DemoCallRecord]:
        with self._lock:
            return self._latest

    @property
    def call_count(self) -> int:
        """Number of calls actually handed to a provider."""
        return len([c for c in self.calls if c.status.was_placed])

    @property
    def attempted_count(self) -> int:
        """Number of calls that were *not* refused by a safety interlock.

        Unlike :attr:`call_count` this includes a call still in flight, so it is
        the race-free value for enforcing ``max_calls_per_run``.
        """
        return len([c for c in self.calls if c.status not in (CallStatus.SKIPPED, CallStatus.BLOCKED)])

    @property
    def sms_count(self) -> int:
        """Number of SMS actually handed to a provider."""
        return len([c for c in self.calls if c.sms.status == SmsStatus.SENT])

    @property
    def active(self) -> bool:
        """True while a call is in flight - drives the overlay banner."""
        latest = self.latest
        return latest is not None and latest.is_active

    def status(self, include_recipient: Optional[bool] = None) -> Dict[str, Any]:
        """Compact state for the overlay / ``FrameResult`` / the dashboard."""
        latest = self.latest
        expose = (
            self.config.expose_recipient if include_recipient is None else bool(include_recipient)
        )
        base: Dict[str, Any] = {
            "demo_mode": self.demo_mode,
            "provider": self.provider.name,
            "sms_provider": self.sms_provider.name,
            "send_sms": bool(self.config.send_sms),
            "call_count": self.call_count,
            "sms_count": self.sms_count,
            "attempted_count": self.attempted_count,
            "call_log": str(self.log_path) if self.log_path else None,
            "disclaimer": DEMO_CALL_DISCLAIMER,
            "sms_disclaimer": DEMO_SMS_DISCLAIMER,
            "label": "DEMO EMERGENCY CALL",
        }
        if latest is None:
            base["status"] = CallStatus.SKIPPED.value
            base["recipient"] = mask_phone(self.raw_recipient())
            base["idle"] = True
            base["note"] = "no demonstration call requested yet"
            return base
        base.update(latest.to_dict(include_recipient=expose))
        base["idle"] = False
        base["label"] = f"DEMO EMERGENCY CALL + {DEMO_SMS_LABEL}"
        return to_jsonable(base)

    # ------------------------------------------------------------------ #
    # recipient
    # ------------------------------------------------------------------ #
    def raw_recipient(self) -> Optional[str]:
        """The configured recipient, straight from the environment."""
        return (self.environ.get(self.config.contact_env_var) or "").strip() or None

    def resolve_recipient_safe(self) -> Optional[str]:
        """The recipient if one is configured, else ``None``.

        Used by the CLI banner, which must never raise or leak the number just
        to describe the configuration.
        """
        try:
            return self.resolve_recipient()
        except Exception:  # noqa: BLE001 - describing config must not fail a run
            return None

    def resolve_recipient(self) -> str:
        """Validated E.164 recipient, or raise :class:`RecipientError`."""
        return normalize_phone(self.raw_recipient())

    # ------------------------------------------------------------------ #
    # the incident hook
    # ------------------------------------------------------------------ #
    def handle_incident(self, incident: Mapping[str, Any]) -> Optional[DemoCallRecord]:
        """Called once per confirmed incident.  Returns the call record.

        Safe to call repeatedly for the same incident: only the first call
        dials, the rest are counted and ignored.
        """
        if self._closed:
            return None

        incident_id = str(incident.get("incident_id") or "").strip()
        if not incident_id:
            _LOGGER.warning("ignoring an incident without an incident_id")
            return None

        # ---- gate 1: is this even a confirmed accident? -----------------
        status = str(incident.get("status") or "")
        if status != IncidentStatus.CONFIRMED_ACCIDENT.value:
            _LOGGER.debug(
                "no demonstration call for %s: status is %s, not CONFIRMED_ACCIDENT",
                incident_id,
                status or "unknown",
            )
            return None

        with self._lock:
            existing = self._by_incident.get(incident_id)
            if existing is not None:
                existing.deliveries += 1
                _LOGGER.debug(
                    "incident %s already handled (delivery %d); not dialling again",
                    incident_id,
                    existing.deliveries,
                )
                return existing

            record = DemoCallRecord(
                call_id=_next_call_id(),
                incident_id=incident_id,
                provider=self.provider.name,
                trigger=TRIGGER_AUTOMATIC,
                # Phase 4: one correlation id per notification, shared by the
                # call, the SMS, every log line and both outbox rows.
                correlation_id=f"sv-{incident_id}-{_next_call_id()[-4:]}",
                camera_id=(incident.get("camera") or {}).get("camera_id"),
                accident_score=(incident.get("accident") or {}).get("score"),
                severity=(incident.get("accident") or {}).get("severity"),
                started_unix=self.clock(),
                message=self._voice_message(incident),
            )
            # registered *before* the call is placed: this is the single point
            # that makes "one call + one SMS per confirmed incident" true
            self._by_incident[incident_id] = record
            self._calls.append(record)
            self._latest = record
        # composed here, while the incident payload is in scope, so the worker
        # thread never has to carry the whole incident
        record.sms.provider = self.sms_provider.name
        record.sms.body = self._sms_body(incident)

        # ---- gate 2: is calling allowed at all? ------------------------
        self._apply_gates(record)
        if record.status.is_terminal:
            # A safety interlock refused the call, so there is no call for the
            # SMS to follow.  Record that explicitly - leaving it PENDING would
            # tell the dashboard a message was still on its way.
            record.sms.set_status(
                SmsStatus.SKIPPED,
                reason=f"call was {record.status}: {record.reason}",
            )
            # Phase 4: a gate-refused notification must still leave a durable
            # trace.  Without this row an audit of the outbox cannot prove that
            # nobody was told - the absence of a row would be ambiguous with
            # "never attempted".
            gate_state = {
                CallStatus.NOT_CONFIGURED: OutboxState.NOT_CONFIGURED,
                CallStatus.BLOCKED: OutboxState.BLOCKED,
                CallStatus.SKIPPED: OutboxState.SKIPPED,
            }.get(record.status, OutboxState.SKIPPED)
            self._outbox_gate_row(record, gate_state)
            self._announce_sms(record.sms)
            self._finalize(record)
            return record

        # ---- gate 3: place it (off the video thread) -------------------
        self._emit(record)
        self._log_call_status(record)
        if self.config.async_calls:
            thread = threading.Thread(
                target=self._dial,
                args=(record,),
                name=f"safevision-{record.call_id}",
                daemon=True,
            )
            with self._lock:
                self._threads.append(thread)
            thread.start()
        else:
            self._dial(record)
        return record

    # ------------------------------------------------------------------ #
    def _sms_body(self, incident: Mapping[str, Any]) -> str:
        """The alert text for this incident (real data, never a template constant)."""
        override = (self.config.sms_body or "").strip()
        if override:
            return override
        return build_sms_body(incident, instruction=self.config.emergency_instruction)

    def _should_send_sms(self, record: DemoCallRecord, incident_reasons: str) -> Tuple[bool, str]:
        """The explicit ``SMS_ON_CALL_FAILURE`` policy.  Returns ``(send, why)``.

        Default (``sms_on_call_failure=False``) preserves the original rule: the
        SMS follows a call that actually reached the recipient's network.  The
        Phase 1 audit found this was hard-coded and not operator-configurable, so
        it is now a documented, validated flag.
        """
        cfg = self.config
        reached = record.status.reached_recipient
        if reached:
            if not cfg.sms_on_call_success:
                return False, "sms_on_call_success is false, so no SMS follows a good call"
            return True, "call reached the recipient"
        why = (f"call was {record.status}: "
               f"{record.reason or incident_reasons or record.error or 'no reason given'}")
        if cfg.sms_on_call_failure:
            # A fallback is sent because the primary channel did not work - this
            # is not evidence that the recipient is reachable, only that the call
            # did not get through.
            return True, f"fallback SMS: {why}"
        return False, why

    def _send_sms(self, record: DemoCallRecord, recipient: str, incident_reasons: str = "") -> None:
        """Send the SMS for this incident.  Runs after the call, never raises."""
        sms = record.sms
        cfg = self.config
        try:
            if not cfg.send_sms:
                sms.set_status(SmsStatus.SKIPPED, reason="emergency.send_sms is false")
                return
            send, why = self._should_send_sms(record, incident_reasons)
            if not send:
                sms.set_status(SmsStatus.SKIPPED, reason=why)
                self._outbox_terminal(record, "sms", OutboxState.SKIPPED, detail=why)
                return
            try:
                sms.recipient = mask_phone(recipient)
                if cfg.expose_recipient:
                    sms.recipient_exposed = recipient
            except Exception:  # noqa: BLE001 - masking must never break the SMS
                sms.recipient = "(unconfigured)"
            if not sms.body:
                sms.set_status(SmsStatus.SKIPPED, reason="no SMS body could be composed")
                return

            usable, usable_why = self.sms_provider.available()
            if not usable:
                # **Phase 4:** an unconfigured provider is NOT a "blocked" safety
                # interlock and not a failure - it is the absence of a provider.
                # Reporting it as either would hide the fact that nobody was told.
                state = (SmsStatus.NOT_CONFIGURED
                         if getattr(self.sms_provider, "name", "") == "not_configured"
                         else SmsStatus.BLOCKED)
                sms.set_status(state, reason=usable_why or "SMS provider unavailable")
                self._outbox_terminal(record, "sms",
                                      OutboxState.NOT_CONFIGURED if state == SmsStatus.NOT_CONFIGURED
                                      else OutboxState.BLOCKED,
                                      detail=usable_why)
                self._announce_sms(sms)
                return

            # Exactly-once: the claim is atomic, so a duplicate trigger, a retry
            # or a second worker cannot produce a second SMS.
            claim = self._claim(record, "sms", sms.recipient)
            if claim is None:
                sms.set_status(
                    SmsStatus.SKIPPED,
                    reason=f"an SMS for this incident is already {self._outbox_state(record, 'sms')}",
                )
                return
            attempt = claim.attempt

            def operation(_attempt: int):
                return self.sms_provider.send(recipient, sms.body)

            if not self._breaker.allow():
                # Refusing is not a delivery failure: say so plainly, or the
                # operator will read a breaker outage as a bad phone number.
                reason = "circuit breaker open; the provider was not contacted"
                sms.set_status(SmsStatus.BLOCKED, reason=reason)
                self._outbox_terminal(record, "sms", OutboxState.BLOCKED, detail=reason)
                self._announce_sms(sms)
                return

            ok, result, error, delays = run_with_retry(
                operation, self._retry_policy, breaker=self._breaker,
                on_attempt=lambda n, err, _ra: _LOGGER.info(
                    "SMS attempt %d incident=%s channel=sms provider=%s "
                    "correlation=%s outcome=%s",
                    n, record.incident_id, self.sms_provider.name,
                    record.correlation_id, "error" if err else "ok",
                ),
                sleep=self._sleep,
            )
            if delays:
                _LOGGER.warning(
                    "incident=%s channel=sms retried with backoff %s", record.incident_id, delays
                )
            if ok and result is not None and result.ok:
                sms.set_status(
                    SmsStatus.SENT,
                    provider_status=result.provider_status or "queued",
                    provider_message_sid=result.provider_call_sid,
                )
                self._outbox_terminal(
                    record, "sms", OutboxState.DONE,
                    provider=self.sms_provider.name,
                    provider_ref=result.provider_call_sid,
                    provider_status=result.provider_status or "queued",
                )
            else:
                # Keep the provider's own words: "unauthorised" or "invalid
                # number" is what tells an operator what to fix, and replacing it
                # with a generic sentence would destroy the only diagnostic.
                provider_error = getattr(result, "error", None)
                if provider_error:
                    detail = str(provider_error)
                elif error is not None:
                    detail = f"sms error: {error}"
                else:
                    detail = "SMS provider reported a failure"
                sms.set_status(
                    SmsStatus.FAILED,
                    provider_status=getattr(result, "provider_status", None),
                    provider_message_sid=getattr(result, "provider_call_sid", None),
                    error=f"{detail} (after {attempt} attempt(s))",
                )
                self._outbox_terminal(record, "sms", OutboxState.FAILED, detail=detail)
                self._maybe_escalate(record, "sms")
        except Exception as exc:  # noqa: BLE001 - never let SMS kill the pipeline
            _LOGGER.exception("demonstration SMS failed")
            sms.set_status(SmsStatus.FAILED, error=f"sms error: {exc}")
        finally:
            self._announce_sms(sms)

    # ------------------------------------------------------------------ #
    # Phase 4: durable outbox, retry policy, escalation
    # ------------------------------------------------------------------ #
    def _outbox_state(self, record: DemoCallRecord, channel: str) -> str:
        if self.outbox is None:
            return "(no outbox)"
        row = self.outbox.get(record.incident_id, channel)
        return row.state if row else "(no row)"

    def _claim(self, record: DemoCallRecord, channel: str, recipient_masked: str):
        """Atomically take the one row for ``(incident, channel)``.  ``None`` = already handled."""
        if self.outbox is None:
            return DeliveryRow(incident_id=record.incident_id, channel=channel,
                               state=OutboxState.IN_FLIGHT, attempt=1)
        self.outbox.enqueue(
            record.incident_id, channel,
            recipient_masked=recipient_masked,
            correlation_id=record.correlation_id,
            provider="call" if channel == "call" else "sms",
        )
        return self.outbox.claim(record.incident_id, channel)

    def _outbox_terminal(self, record: DemoCallRecord, channel: str, state: str, **kw: Any) -> None:
        if self.outbox is None:
            return
        try:
            self.outbox.complete(record.incident_id, channel, state=state, **kw)
        except Exception as exc:  # noqa: BLE001 - the audit trail must not break a send
            _LOGGER.warning("outbox write failed for %s/%s: %s",
                            record.incident_id, channel, exc)

    def _outbox_gate_row(self, record: DemoCallRecord, state: str) -> None:
        """Write the audit rows for a notification a gate refused to attempt."""
        if self.outbox is None:
            return
        detail = record.reason or record.error or f"call was {record.status}"
        for channel in ("call", "sms"):
            try:
                self.outbox.enqueue(
                    record.incident_id, channel,
                    recipient_masked=record.recipient,
                    correlation_id=record.correlation_id,
                    provider=self.provider.name,
                )
                self.outbox.complete(record.incident_id, channel, state=state,
                                     provider=self.provider.name, detail=detail)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("outbox gate row failed for %s/%s: %s",
                                record.incident_id, channel, exc)

    def _maybe_escalate(self, record: DemoCallRecord, failed_channel: str) -> None:
        """Record that both channels failed.  Never contacts a public service.

        ``both_channels_failed_policy=escalate`` stops here: the failure is logged
        and written to the outbox with a loud warning.  Routing to police,
        ambulance or 112 is NOT_APPROVED in this project and is not implemented.
        """
        cfg = self.config
        if cfg.both_channels_failed_policy == "none":
            return
        other = "call" if failed_channel == "sms" else "sms"
        other_state = self._outbox_state(record, other)
        if other_state not in (OutboxState.FAILED, OutboxState.SKIPPED,
                               OutboxState.BLOCKED, OutboxState.NOT_CONFIGURED):
            return
        _LOGGER.error(
            "BOTH CHANNELS FAILED incident=%s call=%s sms=%s policy=%s - "
            "recorded for operator attention. No public emergency service was "
            "contacted; that routing is NOT_APPROVED and not implemented.",
            record.incident_id, other_state, OutboxState.FAILED, cfg.both_channels_failed_policy,
        )

    def _announce_sms(self, sms: DemoSmsRecord) -> None:
        """Emit the SMS status and log the required ``SMS -> SENT`` line."""
        self._emit_sms(sms)
        label = DEMO_SMS_LABEL if self.demo_mode else "SMS"
        if sms.status == SmsStatus.SENT:
            _LOGGER.info("%s -> SENT | %s | %s", label, sms.recipient or "(unconfigured)",
                         sms.provider)
        else:
            _LOGGER.warning(
                "%s -> %s | %s%s",
                label, sms.status, sms.recipient or "(unconfigured)",
                f" ({sms.reason or sms.error})" if (sms.reason or sms.error) else "",
            )

    def _emit_sms(self, sms: DemoSmsRecord) -> None:
        if self.on_status is None:
            return
        try:
            payload = sms.to_dict(include_recipient=self.config.expose_recipient)
            payload.update({"incident_id": self._latest.incident_id if self._latest else None})
            self.on_status(payload)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("sms-status callback failed: %s", exc)

    # ------------------------------------------------------------------ #
    def _apply_gates(self, record: DemoCallRecord) -> None:
        """Refuse (and record why) whenever calling is not appropriate."""
        cfg = self.config

        if not cfg.enabled:
            record.set_status(CallStatus.SKIPPED, reason="emergency.enabled is false")
            return
        # Phase 4: two ways to dial automatically.  `allow_real_call` is the
        # production opt-in; `DEMO_MODE` remains the deliberate simulation.  A
        # fresh install has both false and therefore dials nobody.
        real_send = bool(cfg.allow_real_call)
        if not self.demo_mode and not real_send:
            record.set_status(
                CallStatus.SKIPPED,
                reason=("DEMO_MODE is not true and emergency.allow_real_call is false "
                        "- automatic calling is disabled"),
            )
            return
        try:
            recipient = self.resolve_recipient()
        except RecipientError as exc:
            record.set_status(
                CallStatus.BLOCKED,
                reason=f"{cfg.contact_env_var}: {exc}",
            )
            return
        record.recipient = mask_phone(recipient)
        if cfg.expose_recipient:
            record.recipient_exposed = recipient

        usable, why = self.provider.available()
        if not usable:
            # Phase 4: distinguish "no provider configured" (NOT_CONFIGURED -
            # nobody was told) from a genuine safety interlock (BLOCKED).
            if getattr(self.provider, "name", "") == "not_configured":
                record.set_status(CallStatus.NOT_CONFIGURED, reason=why or "no provider configured")
            else:
                record.set_status(CallStatus.BLOCKED, reason=why or "provider unavailable")
            return
        if real_send and not self.demo_mode and self.provider.name == "simulation":
            # a real send was requested but the resolved provider would only
            # simulate - refuse rather than report a fake notification
            record.set_status(
                CallStatus.BLOCKED,
                reason=("emergency.allow_real_call is true but the resolved provider is "
                        "'simulation'; refusing to report a simulated call as a real one"),
            )
            return
        record.sms.recipient = mask_phone(recipient)
        if cfg.expose_recipient:
            record.sms.recipient_exposed = recipient

        # Calls placed *before* this one.  `record` is already registered (that
        # is what makes the per-incident dedup work), so it must be excluded
        # here - otherwise the first call would count itself.
        prior = len(
            [
                c
                for c in self.calls
                if c is not record and c.status not in (CallStatus.SKIPPED, CallStatus.BLOCKED)
            ]
        )

        limit = max(0, int(cfg.max_calls_per_run))
        if limit and prior >= limit:
            record.set_status(
                CallStatus.SKIPPED,
                reason=f"per-run limit reached ({limit} call(s)) - not dialling again",
            )
            return

        gap = max(0.0, float(cfg.min_gap_seconds))
        if gap > 0.0 and prior > 0:
            # the gap separates calls from each other; the first one is free
            elapsed = self.clock() - self._started_wall
            if elapsed < gap:
                record.set_status(
                    CallStatus.SKIPPED,
                    reason=f"only {elapsed:.1f}s since the first call, min_gap_seconds is {gap:.0f}s",
                )
                return

    # ------------------------------------------------------------------ #
    def _dial(self, record: DemoCallRecord) -> None:
        """Worker-thread body.  Never raises."""
        try:
            recipient = self.resolve_recipient()
        except RecipientError as exc:  # pragma: no cover - gated in _apply_gates
            record.set_status(CallStatus.BLOCKED, reason=str(exc))
            self._finalize(record)
            return

        def on_status(
            status: CallStatus,
            provider_status: str = "",
            provider_call_sid: str = "",
            error: str = "",
        ) -> None:
            record.set_status(
                status,
                provider_status=provider_status,
                provider_call_sid=provider_call_sid,
                error=error,
            )
            self._emit(record)
            self._log_call_status(record)

        # Pre-set so a provider that *raises* still leaves a well-formed result:
        # the SMS step below must never hit an unbound name.
        result = ProviderResult(ok=False, status=CallStatus.FAILED)

        # Exactly-once: claim before dialling.  A second worker, a duplicate
        # incident or a retry cannot produce a second call.
        claim = self._claim(record, "call", record.recipient)
        if claim is None:
            record.set_status(
                CallStatus.SKIPPED,
                reason=(f"a call for this incident is already "
                        f"{self._outbox_state(record, 'call')}"),
            )
            self._finalize(record)
            return

        # The breaker is checked here as well as inside run_with_retry: the call
        # path places directly, so without this a provider outage would turn every
        # confirmed incident into a doomed request.
        try:
            self._breaker.check()
        except CircuitOpen as exc:
            _LOGGER.error("call not attempted, breaker open: %s", exc)
            record.set_status(CallStatus.BLOCKED, reason=f"circuit breaker open: {exc}")
            if self.outbox is not None:
                self.outbox.release(record.incident_id, "call", detail=str(exc))
            self._send_sms(record, recipient, incident_reasons=str(exc))
            self._finalize(record)
            return

        try:
            result = self.provider.place(recipient, record.message, on_status)
        except CircuitOpen as exc:
            # breaker refused: release the row so a retry can pick it up
            _LOGGER.error("call not attempted, breaker open: %s", exc)
            record.set_status(CallStatus.BLOCKED, reason=f"circuit breaker open: {exc}")
            if self.outbox is not None:
                self.outbox.release(record.incident_id, "call", detail=str(exc))
            self._send_sms(record, recipient, incident_reasons=str(exc))
            self._finalize(record)
            return
        except Exception as exc:  # noqa: BLE001 - a provider bug must not kill the run
            _LOGGER.exception("call provider raised")
            record.set_status(CallStatus.FAILED, error=f"provider error: {exc}")
        else:
            if not result.ok and not record.status.is_terminal:
                record.set_status(
                    result.status,
                    provider_status=result.provider_status,
                    provider_call_sid=result.provider_call_sid,
                    error=result.error,
                )
            elif result.error and not record.error:
                # `on_status` may already have applied a terminal state, which
                # makes the branch above a no-op - and that would silently drop
                # the provider's own diagnostic ("unauthorised", "not a valid
                # phone number"), leaving an operator with no clue what to fix.
                record.error = result.error
                if result.provider_call_sid and not record.provider_call_sid:
                    record.provider_call_sid = result.provider_call_sid

        # ---- durable outcome -------------------------------------------- #
        if result.status == CallStatus.NOT_CONFIGURED or record.status == CallStatus.NOT_CONFIGURED:
            # **Phase 4:** no provider.  The outbox records this as
            # NOT_CONFIGURED so a later audit can prove nobody was notified.
            self._outbox_terminal(
                record, "call", OutboxState.NOT_CONFIGURED,
                provider=self.provider.name,
                detail=record.error or record.reason or "no provider configured",
            )
            _LOGGER.error(
                "CALL -> NOT_CONFIGURED | incident=%s | provider=%s | reason=%s | "
                "NO TELEPHONY WAS PLACED",
                record.incident_id, self.provider.name, record.error or record.reason,
            )
        elif record.status.reached_recipient:
            self._outbox_terminal(
                record, "call", OutboxState.DONE,
                provider=self.provider.name,
                provider_ref=record.provider_call_sid,
                provider_status=record.provider_status,
            )
        else:
            self._outbox_terminal(
                record, "call", OutboxState.FAILED,
                provider=self.provider.name,
                provider_ref=record.provider_call_sid,
                provider_status=record.provider_status,
                detail=record.error or record.reason or "call did not reach the recipient",
            )

        # The SMS is strictly *after* the call workflow for the same incident.
        # Whether it follows a *failed* call is the explicit sms_on_call_failure
        # policy, not an accident of the call's outcome.
        self._send_sms(record, recipient, incident_reasons=result.error if not result.ok else "")
        self._finalize(record)

    def _log_call_status(self, record: DemoCallRecord) -> None:
        """The required ``CALL -> STATE`` line for every transition."""
        label = "DEMO CALL" if self.demo_mode else "CALL"
        _LOGGER.info(
            "%s -> %s | %s | %s | incident %s",
            label,
            record.status,
            record.recipient or "(unconfigured)",
            record.provider,
            record.incident_id,
        )

    def _finalize(self, record: DemoCallRecord) -> None:
        """Emit, persist and announce the end of a notification - exactly once."""
        if record.ended_unix <= 0.0:
            record.set_status(record.status)
        self._emit(record)
        placed = record.status.was_placed
        if placed:
            _LOGGER.warning("%s", DEMO_CALL_DISCLAIMER)
        if record.sms.status == SmsStatus.SENT:
            _LOGGER.warning("%s", DEMO_SMS_DISCLAIMER)
        self._write_log(record)

    # ------------------------------------------------------------------ #
    def _voice_message(self, incident: Mapping[str, Any]) -> str:
        """The spoken message: the configured text plus the incident facts."""
        accident = incident.get("accident") or {}
        camera = incident.get("camera") or {}
        location = incident.get("location") or {}
        try:
            score = float(accident.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        facts = (
            f"Accident probability {int(round(score * 100))} percent. "
            f"Estimated severity {accident.get('severity', 'unknown')}. "
            f"Camera {camera.get('camera_id') or 'unregistered'}"
        )
        where = location.get("name") or location.get("road")
        if where:
            facts += f", near {where}"
        template = (self.config.voice_message or "").strip()
        return f"{template} {facts}" if template else f"SafeVision demonstration alert. {facts}"

    def _emit(self, record: DemoCallRecord) -> None:
        """Push a status update to the subscriber (dashboard / overlay).

        A status is pushed at most once: the provider's final callback and
        :meth:`_finalize` both report the terminal state, and a duplicate
        ``COMPLETED`` would make a dashboard flip-flop.
        """
        key = f"{record.call_id}:{record.status}"
        with self._lock:
            if self._emitted.get(record.call_id) == key:
                return
            self._emitted[record.call_id] = key
        if self.on_status is None:
            return
        try:
            self.on_status(record.to_dict(include_recipient=self.config.expose_recipient))
        except Exception as exc:  # noqa: BLE001 - a UI bug must not stop the call
            _LOGGER.warning("call-status callback failed: %s", exc)

    def _drain_status_queue(self) -> None:
        """Wait for any worker thread still mid-transition (called by close)."""
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                busy = [t for t in self._threads if t.is_alive()]
            if not busy:
                return
            time.sleep(0.02)

    def _log_line(self, text: str) -> None:
        """Record a lifecycle transition.

        Console visibility is the job of the ``on_status`` callback (the CLI
        installs one); this keeps the dispatcher silent by default so a library
        consumer and the ``--json`` stdout contract are never disturbed.
        """
        _LOGGER.debug("%s", text)

    def _write_log(self, record: DemoCallRecord) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record.to_dict(include_recipient=False), allow_nan=False)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            _LOGGER.error("could not append to the demo call log %s: %s", self.log_path, exc)

    # ------------------------------------------------------------------ #
    def wait(self, timeout: float = 75.0) -> int:
        """Block until every worker has finished, or ``timeout`` expires.

        Returns the number of workers still running when the wait ended.

        This exists because a caller must not report a notification's outcome
        before that outcome is known. The call and the SMS run on a worker
        thread, so reading the incident payload immediately after the video ends
        shows the *initial* state (``INITIATED`` / ``PENDING``) rather than what
        the provider actually did. Waiting is what makes the report true.
        """
        with self._lock:
            threads = list(self._threads)
        if not threads:
            return 0
        deadline = time.monotonic() + max(0.0, float(timeout))
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        self._drain_status_queue()
        alive = [t for t in threads if t.is_alive()]
        if alive:
            _LOGGER.warning(
                "%d notification(s) had not reached a terminal state after %.0fs",
                len(alive), timeout,
            )
        return len(alive)

    def pending(self) -> int:
        """Workers still running - i.e. notifications whose outcome is unknown."""
        with self._lock:
            return len([t for t in self._threads if t.is_alive()])

    def final_states(self) -> Dict[str, Dict[str, Any]]:
        """``incident_id -> {call, sms}`` final states, from the live records.

        The authoritative outcome, as opposed to the snapshot captured in the
        incident payload when it was built.
        """
        with self._lock:
            records = list(self._calls)
        out: Dict[str, Dict[str, Any]] = {}
        for record in records:
            # CallStatus is an enum; DemoSmsRecord.status is a plain str. Accept
            # either so this never depends on which one a caller handed us.
            call_state = getattr(record.status, "value", record.status)
            sms_state = getattr(record.sms.status, "value", record.sms.status)
            out[record.incident_id] = {
                "call": str(call_state),
                "call_reached_recipient": bool(record.status.reached_recipient),
                "call_placed": bool(record.status.was_placed),
                "provider": record.provider,
                "provider_status": record.provider_status or None,
                "provider_ref": record.provider_call_sid or None,
                "correlation_id": record.correlation_id or None,
                "reason": record.reason or None,
                "error": record.error or None,
                "recipient": record.recipient or None,
                "sms": str(sms_state),
                "sms_provider": record.sms.provider or None,
                "sms_provider_status": record.sms.provider_status or None,
                "sms_provider_ref": record.sms.provider_message_sid or None,
                "sms_body": record.sms.body or None,
                "sms_reason": record.sms.reason or None,
                "sms_error": record.sms.error or None,
            }
        return out

    def close(self, timeout: float = 6.0) -> None:
        """Wait for any in-flight call and stop accepting new ones."""
        self._closed = True
        with self._lock:
            threads = list(self._threads)
            owns_outbox = self._owns_outbox
        deadline = time.monotonic() + max(0.0, float(timeout))
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        alive = [t for t in threads if t.is_alive()]
        if alive:
            _LOGGER.warning("%d demonstration call(s) still in flight at shutdown", len(alive))
        with self._lock:
            self._threads = []
        self._drain_status_queue()
        if owns_outbox and self.outbox is not None:
            # Release the SQLite handle. Leaving it open makes the outbox file
            # undeletable, which breaks temp-dir cleanup and any operator's
            # attempt to rotate or remove the audit trail.
            try:
                self.outbox.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                _LOGGER.debug("outbox close failed: %s", exc)

    def __enter__(self) -> "DemoCallDispatcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        """What the CLI prints for ``--check-emergency``.  Never shows the number."""
        recipient = self.raw_recipient()
        try:
            normalized: Optional[str] = self.resolve_recipient()
        except RecipientError as exc:
            normalized = None
            reason = str(exc)
        else:
            reason = ""
        usable, provider_reason = self.provider.available()
        sms_usable, sms_reason = self.sms_provider.available()
        return {
            "demo_mode": self.demo_mode,
            "enabled": bool(self.config.enabled),
            "provider": self.provider.describe(),
            "sms_provider": self.sms_provider.describe(),
            "send_sms": bool(self.config.send_sms),
            "recipient_env_var": self.config.contact_env_var,
            "recipient_configured": bool(recipient),
            "recipient_valid": normalized is not None,
            "recipient_masked": mask_phone(recipient),
            "recipient_problem": reason or None,
            "will_place_a_real_call": bool(normalized and usable and self.config.provider != "none"
                                           and self.provider.name == "twilio"),
            "will_send_a_real_sms": bool(self.config.send_sms and normalized and sms_usable
                                         and self.config.sms_provider != "none"
                                         and self.sms_provider.name == "twilio"),
            "calls_placed": self.call_count,
            "sms_sent": self.sms_count,
            "call_log": str(self.log_path) if self.log_path else None,
            "provider_problem": provider_reason if not usable else None,
            "sms_provider_problem": sms_reason if not sms_usable else None,
            "disclaimer": DEMO_CALL_DISCLAIMER,
            "sms_disclaimer": DEMO_SMS_DISCLAIMER,
        }
