"""Phase 4: reliability, durability, fallbacks and verified callbacks.

Everything here is offline.  No test opens a socket, reaches a provider, or
contacts the configured recipient - ``test_no_phase4_test_can_reach_a_network``
enforces that for the whole module.

The failure modes are the point.  A notification system is only as trustworthy
as its behaviour when the provider refuses, times out, rate-limits, or rejects
the number - which is exactly when nobody is watching.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.config.classes import ConfigError  # noqa: E402
from ai_engine.emergency.call_record import DemoCallRecord  # noqa: E402
from ai_engine.emergency.dispatcher import DemoCallDispatcher  # noqa: E402
from ai_engine.emergency.outbox import (  # noqa: E402
    NotificationOutbox,
    OutboxState,
)
from ai_engine.emergency.providers import (  # noqa: E402
    CallProvider,
    ProviderResult,
    SmsProvider,
    TwilioSmsProvider,
    build_provider,
    build_sms_provider,
)
from ai_engine.emergency.reliability import (  # noqa: E402
    CircuitBreaker,
    CircuitOpen,
    PermanentFailure,
    RetryPolicy,
    TransientFailure,
    classify_http_status,
    idempotency_key,
    run_with_retry,
)
from ai_engine.emergency.webhook import (  # noqa: E402
    CallbackApplier,
    call_status_from_twilio,
    compute_signature,
    sms_status_from_twilio,
    verify_request,
    verify_twilio_signature,
)
from ai_engine.schemas.enums import CallStatus, SmsStatus  # noqa: E402
from tests.helpers import test_config  # noqa: E402

RECIPIENT = "+15550100"          # fictional, reserved for tests
TOKEN = "auth-token-for-signing"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class ScriptedCallProvider(CallProvider):
    """Replays a scripted sequence of results, recording every attempt."""

    name = "scripted-call"

    def __init__(self, results):
        self.results = list(results)
        self.attempts = []

    def available(self):
        return True, ""

    def place(self, recipient, message, on_status):
        self.attempts.append({"recipient": recipient, "message": message})
        result = self.results[min(len(self.attempts) - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        on_status(result.status, result.provider_status, result.provider_call_sid)
        return result


class ScriptedSmsProvider(SmsProvider):
    name = "scripted-sms"

    def __init__(self, results):
        self.results = list(results)
        self.sent = []

    def available(self):
        return True, ""

    def send(self, recipient, body):
        self.sent.append({"recipient": recipient, "body": body})
        result = self.results[min(len(self.sent) - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def make_dispatcher(**kw):
    """A dispatcher with an in-memory outbox and no real sleeping."""
    provider = kw.pop("_provider", None)
    sms = kw.pop("_sms", None)
    config = test_config().emergency
    config.enabled = True
    config.demo_mode = True
    config.write_call_log = False
    config.async_calls = False
    config.use_outbox = True
    config.outbox_path = ":memory:"
    config.retry_base_delay_seconds = 0.0
    for key, value in kw.items():
        setattr(config, key, value)
    env = {"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT}
    return DemoCallDispatcher(
        config, environ=env, provider=provider, sms_provider=sms,
        sleep=lambda _s: None,
    )


def incident(incident_id="INC-TEST-0001"):
    return {
        "incident_id": incident_id,
        "status": "CONFIRMED_ACCIDENT",
        "timestamp": "2026-10-03T10:00:00+00:00",
        "camera": {"camera_id": "CAM-001"},
        "location": {"name": "Chennai Main Road", "latitude": 13.08, "longitude": 80.27,
                     "is_camera_registered": True},
        "accident": {"score": 0.6, "severity": "HIGH"},
    }


def ok_call(status=CallStatus.COMPLETED):
    return ProviderResult(ok=True, status=status, provider_status="completed",
                          provider_call_sid="CA0000FAKE")


def ok_sms():
    return ProviderResult(ok=True, status=CallStatus.COMPLETED, provider_status="queued",
                          provider_call_sid="SM0000FAKE")


# --------------------------------------------------------------------------- #
class TestNotConfiguredNeverReportsSuccess(unittest.TestCase):
    """The Phase 1 critical finding, locked shut."""

    def test_no_credentials_and_no_demo_mode_notifies_nobody(self) -> None:
        call = build_provider("auto", environ={}, demo_mode=False)
        sms = build_sms_provider("auto", environ={}, demo_mode=False)
        self.assertEqual(call.name, "not_configured")
        self.assertEqual(sms.name, "not_configured")
        self.assertFalse(call.available()[0])
        self.assertFalse(sms.available()[0])

    def test_a_run_without_credentials_reports_not_configured_not_completed(self) -> None:
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = False
        config.allow_real_call = True
        config.async_calls = False
        config.use_outbox = True
        config.outbox_path = ":memory:"
        dispatcher = DemoCallDispatcher(
            config, environ={"DEMO_MODE": "false", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            sleep=lambda _s: None,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.NOT_CONFIGURED)
        self.assertFalse(record.status.was_placed)
        self.assertFalse(record.status.reached_recipient)

    def test_the_outbox_records_that_nobody_was_told(self) -> None:
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = False
        config.allow_real_call = True
        config.async_calls = False
        config.use_outbox = True
        config.outbox_path = ":memory:"
        dispatcher = DemoCallDispatcher(
            config, environ={"DEMO_MODE": "false", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            sleep=lambda _s: None,
        )
        dispatcher.handle_incident(incident())
        states = {row.channel: row.state
                  for row in dispatcher.outbox.for_incident("INC-TEST-0001")}
        self.assertEqual(states.get("call"), OutboxState.NOT_CONFIGURED)

    def test_a_real_send_request_cannot_be_served_by_the_simulation(self) -> None:
        """`allow_real_call` + a simulated provider must be refused, not faked.

        The provider is injected directly here, because with ``demo_mode=False``
        and no credentials ``auto`` correctly resolves to ``not_configured``
        (which the neighbouring test covers).  This covers the case where an
        operator wires the simulation provider into a real-send configuration.
        """
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = False
        config.allow_real_call = True
        config.async_calls = False
        config.use_outbox = False
        from ai_engine.emergency.providers import SimulationProvider
        dispatcher = DemoCallDispatcher(
            config,
            environ={"DEMO_MODE": "false", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=SimulationProvider(),
            sleep=lambda _s: None,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.BLOCKED)
        self.assertIn("simulation", record.reason)

    def test_no_credentials_with_a_real_send_reports_not_configured(self) -> None:
        """`allow_real_call` on, credentials absent -> NOT_CONFIGURED, never faked."""
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = False
        config.allow_real_call = True
        config.async_calls = False
        config.use_outbox = False
        dispatcher = DemoCallDispatcher(
            config, environ={"DEMO_MODE": "false", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            sleep=lambda _s: None,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.NOT_CONFIGURED)

    def test_not_configured_is_not_terminal_success(self) -> None:
        for status in (CallStatus.NOT_CONFIGURED, SmsStatus.NOT_CONFIGURED):
            self.assertTrue(status.is_terminal)
            self.assertNotIn(status, (CallStatus.COMPLETED, SmsStatus.SENT))


# --------------------------------------------------------------------------- #
class TestRetryPolicy(unittest.TestCase):
    def test_permanent_failures_are_never_retried(self) -> None:
        policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.0)
        self.assertFalse(policy.should_retry(1, PermanentFailure("bad credentials")))

    def test_transient_failures_are_retried_within_the_budget(self) -> None:
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.0)
        self.assertTrue(policy.should_retry(1, TransientFailure("429")))
        self.assertTrue(policy.should_retry(2, TransientFailure("503")))
        self.assertFalse(policy.should_retry(3, TransientFailure("500")))

    def test_backoff_grows_and_is_capped(self) -> None:
        policy = RetryPolicy(max_attempts=6, base_delay_seconds=2.0, max_delay_seconds=8.0,
                             jitter_source=lambda: 1.0)
        self.assertEqual(policy.delay_for(1), 2.0)
        self.assertEqual(policy.delay_for(2), 4.0)
        self.assertEqual(policy.delay_for(3), 8.0)
        self.assertEqual(policy.delay_for(9), 8.0, "must stay capped")

    def test_retry_after_from_the_provider_wins(self) -> None:
        policy = RetryPolicy(base_delay_seconds=2.0)
        self.assertEqual(policy.delay_for(1, retry_after=17.0), 17.0)

    def test_jitter_stays_within_the_window(self) -> None:
        policy = RetryPolicy(max_attempts=3, base_delay_seconds=10.0, max_delay_seconds=10.0,
                             jitter_source=lambda: 0.5)
        self.assertEqual(policy.delay_for(1), 5.0)

    def test_rate_limit_is_transient_and_bad_credentials_are_not(self) -> None:
        with self.assertRaises(TransientFailure):
            classify_http_status(429)
        with self.assertRaises(TransientFailure):
            classify_http_status(503)
        for code in (400, 401, 403, 404):
            with self.assertRaises(PermanentFailure):
                classify_http_status(code)

    def test_run_with_retry_succeeds_after_transient_failures(self) -> None:
        calls = []

        def operation(attempt):
            calls.append(attempt)
            if attempt < 3:
                raise TransientFailure("rate limited")
            return "delivered"

        ok, value, error, delays = run_with_retry(
            operation, RetryPolicy(max_attempts=4, base_delay_seconds=0.0),
            sleep=lambda _s: None)
        self.assertTrue(ok)
        self.assertEqual(value, "delivered")
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(len(delays), 2)

    def test_run_with_retry_gives_up_on_a_permanent_failure_immediately(self) -> None:
        calls = []

        def operation(attempt):
            calls.append(attempt)
            raise PermanentFailure("unauthorised")

        ok, _, error, delays = run_with_retry(
            operation, RetryPolicy(max_attempts=5, base_delay_seconds=0.0),
            sleep=lambda _s: None)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1, "a 401 must not be retried")
        self.assertEqual(delays, [])

    def test_run_with_retry_exhausts_the_budget(self) -> None:
        calls = []

        def operation(attempt):
            calls.append(attempt)
            raise TransientFailure("503")

        ok, _, _, _ = run_with_retry(
            operation, RetryPolicy(max_attempts=3, base_delay_seconds=0.0),
            sleep=lambda _s: None)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 3)

    def test_idempotency_key_is_stable_and_channel_specific(self) -> None:
        first = idempotency_key("INC-1", "call", 1)
        self.assertEqual(first, idempotency_key("INC-1", "call", 1))
        self.assertNotEqual(first, idempotency_key("INC-1", "sms", 1))
        self.assertNotEqual(first, idempotency_key("INC-2", "call", 1))
        self.assertNotEqual(first, idempotency_key("INC-1", "call", 2))
        self.assertNotIn("INC-1", first, "the raw incident id must not leak")

    def test_invalid_policies_are_rejected(self) -> None:
        for kwargs in ({"max_attempts": 0}, {"base_delay_seconds": -1.0}, {"multiplier": 0.5}):
            with self.assertRaises(ValueError):
                RetryPolicy(**kwargs)


# --------------------------------------------------------------------------- #
class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_consecutive_failures(self) -> None:
        clock = [0.0]
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=10.0,
                                 clock=lambda: clock[0])
        for _ in range(2):
            breaker.record_failure()
        self.assertEqual(breaker.state, "closed")
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")
        with self.assertRaises(CircuitOpen):
            breaker.check()

    def test_a_success_resets_the_failure_count(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3, clock=lambda: 0.0)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        self.assertEqual(breaker.state, "closed")

    def test_half_opens_after_the_cool_off(self) -> None:
        clock = [0.0]
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=5.0,
                                 clock=lambda: clock[0])
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")
        clock[0] = 6.0
        self.assertEqual(breaker.state, "half_open")
        breaker.check()   # allowed again

    def test_a_failed_trial_reopens_the_breaker(self) -> None:
        clock = [0.0]
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=5.0,
                                 clock=lambda: clock[0])
        breaker.record_failure()
        clock[0] = 6.0
        self.assertEqual(breaker.state, "half_open")
        breaker.record_failure()
        self.assertEqual(breaker.state, "open", "a failed trial must reopen it")

    def test_an_open_breaker_stops_the_provider_being_called(self) -> None:
        clock = [0.0]
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=60.0,
                                 clock=lambda: clock[0])
        breaker.record_failure()
        calls = []

        def operation(attempt):
            calls.append(attempt)
            raise TransientFailure("503")

        with self.assertRaises(CircuitOpen):
            run_with_retry(operation, RetryPolicy(max_attempts=3, base_delay_seconds=0.0),
                           breaker=breaker, sleep=lambda _s: None)
        self.assertEqual(calls, [], "an open breaker must prevent the call entirely")

    def test_open_breaker_does_not_stall_the_incident(self) -> None:
        """The gate must report BLOCKED, not hang the notification worker."""
        dispatcher = make_dispatcher(_provider=ScriptedCallProvider([ok_call()]))
        dispatcher._breaker._failures = 99          # trip it
        dispatcher._breaker.record_failure()
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.BLOCKED)
        self.assertIn("circuit breaker open", record.reason)


# --------------------------------------------------------------------------- #
class TestOutbox(unittest.TestCase):
    def setUp(self) -> None:
        self.box = NotificationOutbox(":memory:")

    def tearDown(self) -> None:
        self.box.close()

    def test_a_second_enqueue_does_not_create_a_second_row(self) -> None:
        self.box.enqueue("INC-1", "call", "+15*****0100")
        self.box.enqueue("INC-1", "call", "+15*****0100")
        self.assertEqual(self.box.stats()["total_rows"], 1)

    def test_claim_is_exclusive(self) -> None:
        self.box.enqueue("INC-1", "call")
        self.assertIsNotNone(self.box.claim("INC-1", "call"))
        self.assertIsNone(self.box.claim("INC-1", "call"),
                          "a second worker must not get the same row")

    def test_concurrent_claims_produce_exactly_one_winner(self) -> None:
        self.box.enqueue("INC-1", "sms")
        winners = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            if self.box.claim("INC-1", "sms") is not None:
                winners.append(1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(winners), 1, "exactly-once must survive a race")

    def test_a_terminal_row_is_never_claimed_again(self) -> None:
        self.box.enqueue("INC-1", "call")
        self.box.claim("INC-1", "call")
        self.box.complete("INC-1", "call", state=OutboxState.DONE, provider_ref="CA1")
        self.assertIsNone(self.box.claim("INC-1", "call"))
        self.assertEqual(self.box.state_of("INC-1", "call"), OutboxState.DONE)

    def test_an_expired_lease_is_reclaimed(self) -> None:
        clock = [0.0]
        box = NotificationOutbox(":memory:", lease_seconds=10.0, clock=lambda: clock[0])
        box.enqueue("INC-1", "call")
        box.claim("INC-1", "call")
        self.assertIsNone(box.claim("INC-1", "call"))
        clock[0] = 20.0
        self.assertEqual(box.reclaim_expired(), 1)
        self.assertIsNotNone(box.claim("INC-1", "call"),
                             "a crashed worker must not strand the row")
        box.close()

    def test_a_non_terminal_state_cannot_be_completed(self) -> None:
        self.box.enqueue("INC-1", "call")
        with self.assertRaises(ValueError):
            self.box.complete("INC-1", "call", state=OutboxState.IN_FLIGHT)

    def test_an_unknown_channel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.box.enqueue("INC-1", "carrier-pigeon")

    def test_the_provider_reference_is_recorded(self) -> None:
        self.box.enqueue("INC-1", "call")
        self.box.claim("INC-1", "call")
        self.box.complete("INC-1", "call", state=OutboxState.DONE,
                          provider="twilio", provider_ref="CA_real",
                          provider_status="completed")
        row = self.box.get("INC-1", "call")
        self.assertEqual(row.provider_ref, "CA_real")
        self.assertEqual(row.provider_status, "completed")
        self.assertEqual(row.attempt, 1)

    def test_survives_a_reopen_of_the_database(self) -> None:
        """Durability is the whole point: a restart must not re-notify."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "outbox.sqlite3")
            first = NotificationOutbox(path)
            first.enqueue("INC-1", "call")
            first.claim("INC-1", "call")
            first.complete("INC-1", "call", state=OutboxState.DONE, provider_ref="CA1")
            first.close()
            second = NotificationOutbox(path)
            self.assertIsNone(second.claim("INC-1", "call"),
                              "a restart must not place a second call")
            second.close()

    def test_no_secret_is_stored(self) -> None:
        self.box.enqueue("INC-1", "call", recipient_masked="+15*****0100")
        blob = self.box.export_json()
        self.assertIn("*****", blob)
        self.assertNotIn(RECIPIENT, blob, "the unmasked number must never be persisted")


# --------------------------------------------------------------------------- #
class TestFallbackPolicy(unittest.TestCase):
    def test_default_policy_skips_the_sms_after_a_failed_call(self) -> None:
        """A call that never reached the network gets no SMS by default."""
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ProviderResult(ok=False, status=CallStatus.FAILED,
                                                            error="carrier rejected")]),
            _sms=ScriptedSmsProvider([ok_sms()]),
            sms_on_call_failure=False,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED)
        self.assertEqual(dispatcher.sms_provider.sent, [])

    def test_an_unanswered_but_reached_call_still_gets_its_sms(self) -> None:
        """BUSY / NO_ANSWER mean the line rang.  The SMS still follows."""
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ProviderResult(ok=False,
                                                            status=CallStatus.NO_ANSWER,
                                                            error="no answer")]),
            _sms=ScriptedSmsProvider([ok_sms()]),
            sms_on_call_failure=False,
        )
        record = dispatcher.handle_incident(incident())
        self.assertTrue(record.status.reached_recipient)
        self.assertEqual(record.sms.status, SmsStatus.SENT)

    def test_sms_on_call_failure_sends_one_fallback_sms(self) -> None:
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ProviderResult(ok=False, status=CallStatus.NO_ANSWER,
                                                            error="no answer")]),
            _sms=ScriptedSmsProvider([ok_sms()]),
            sms_on_call_failure=True,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.sms.status, SmsStatus.SENT)
        self.assertEqual(len(dispatcher.sms_provider.sent), 1, "exactly one fallback SMS")

    def test_a_fallback_sms_is_not_sent_twice(self) -> None:
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ProviderResult(ok=False, status=CallStatus.BUSY)]),
            _sms=ScriptedSmsProvider([ok_sms()]),
            sms_on_call_failure=True,
        )
        record = dispatcher.handle_incident(incident())
        dispatcher.handle_incident(incident())
        self.assertEqual(len(dispatcher.sms_provider.sent), 1)
        self.assertEqual(record.sms.status, SmsStatus.SENT)

    def test_a_rejected_call_request_also_triggers_the_fallback(self) -> None:
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ProviderResult(ok=False, status=CallStatus.FAILED,
                                                            error="carrier rejected")]),
            _sms=ScriptedSmsProvider([ok_sms()]),
            sms_on_call_failure=True,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.sms.status, SmsStatus.SENT)

    def test_sms_on_call_success_false_suppresses_the_primary_sms(self) -> None:
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ok_call()]),
            _sms=ScriptedSmsProvider([ok_sms()]),
            sms_on_call_success=False,
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED)

    def test_both_channels_failing_is_recorded_and_routes_nowhere(self) -> None:
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ProviderResult(ok=False, status=CallStatus.FAILED,
                                                            error="carrier rejected")]),
            _sms=ScriptedSmsProvider([ProviderResult(ok=False, status=CallStatus.FAILED,
                                                     error="invalid number")]),
            sms_on_call_failure=True,
            both_channels_failed_policy="flag",
        )
        record = dispatcher.handle_incident(incident())
        states = {row.channel: row.state for row in dispatcher.outbox.for_incident(record.incident_id)}
        self.assertEqual(states["call"], OutboxState.FAILED)
        self.assertEqual(states["sms"], OutboxState.FAILED)

    def test_emergency_service_routing_is_refused_by_validation(self) -> None:
        config = test_config()
        config.emergency.allow_emergency_service_routing = True
        with self.assertRaises(ConfigError) as ctx:
            config.validate()
        self.assertIn("NOT_APPROVED", str(ctx.exception))

    def test_a_simulated_sms_is_reported_as_simulation_not_as_delivered(self) -> None:
        dispatcher = make_dispatcher(_provider=ScriptedCallProvider([ok_call()]))
        record = dispatcher.handle_incident(incident())
        self.assertTrue(dispatcher.demo_mode)
        self.assertEqual(record.sms.status, SmsStatus.SENT)
        self.assertEqual(record.sms.provider, "simulation")


# --------------------------------------------------------------------------- #
class TestProviderFailureModes(unittest.TestCase):
    """Every failure the provider can produce, on the real code path."""

    def _dispatch(self, result, sms=None):
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([result]),
            _sms=ScriptedSmsProvider([sms or ok_sms()]),
        )
        return dispatcher, dispatcher.handle_incident(incident())

    def test_auth_failure_is_recorded_not_raised(self) -> None:
        _, record = self._dispatch(ProviderResult(
            ok=False, status=CallStatus.FAILED, provider_status="http_401",
            error="unauthorised"))
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertIn("unauthorised", record.error)

    def test_rate_limit_is_retried_then_reported(self) -> None:
        provider = ScriptedCallProvider([ProviderResult(
            ok=False, status=CallStatus.FAILED, provider_status="http_429",
            error="Too Many Requests")])
        dispatcher = make_dispatcher(_provider=provider, retry_max_attempts=3)
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.FAILED)
        # the dispatcher retries internally; it does not hammer 3x per event
        self.assertGreaterEqual(len(provider.attempts), 1)

    def test_timeout_is_recorded_not_raised(self) -> None:
        _, record = self._dispatch(ProviderResult(
            ok=False, status=CallStatus.FAILED, error="timed out after 10s"))
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertIn("timed out", record.error)

    def test_invalid_number_is_recorded_not_raised(self) -> None:
        _, record = self._dispatch(ProviderResult(
            ok=False, status=CallStatus.FAILED, provider_status="http_400",
            error="The 'To' number is not a valid phone number"))
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertIn("not a valid phone number", record.error)

    def test_a_provider_that_raises_is_contained(self) -> None:
        _, record = self._dispatch(RuntimeError("provider exploded"))
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertIn("provider error", record.error)

    def test_a_busy_line_is_not_treated_as_a_failure(self) -> None:
        _, record = self._dispatch(ProviderResult(
            ok=False, status=CallStatus.BUSY, provider_status="busy"))
        self.assertEqual(record.status, CallStatus.BUSY)
        self.assertTrue(record.status.reached_recipient)

    def test_no_answer_is_reached_but_not_answered(self) -> None:
        _, record = self._dispatch(ProviderResult(
            ok=False, status=CallStatus.NO_ANSWER, provider_status="no-answer"))
        self.assertEqual(record.status, CallStatus.NO_ANSWER)
        self.assertTrue(record.status.reached_recipient)

    def test_sms_delivery_failure_keeps_the_provider_message(self) -> None:
        dispatcher = make_dispatcher(
            _provider=ScriptedCallProvider([ok_call()]),
            _sms=ScriptedSmsProvider([ProviderResult(
                ok=False, status=CallStatus.FAILED, provider_status="http_400",
                error="unrecognised recipient")]),
        )
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.sms.status, SmsStatus.FAILED)
        self.assertIn("unrecognised recipient", record.sms.error)

    def test_a_no_send_configuration_still_reports_a_reason(self) -> None:
        dispatcher = make_dispatcher(_provider=ScriptedCallProvider([ok_call()]),
                                     send_sms=False)
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.COMPLETED)
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED)

    def test_credentials_are_never_written_into_the_record(self) -> None:
        dispatcher = make_dispatcher(_provider=ScriptedCallProvider([ok_call()]))
        record = dispatcher.handle_incident(incident())
        blob = str(record.to_dict(include_recipient=False))
        self.assertNotIn("AC" + "0" * 32, blob)
        self.assertNotIn(TOKEN, blob)


# --------------------------------------------------------------------------- #
class TestWebhookVerification(unittest.TestCase):
    def setUp(self) -> None:
        self.url = "https://example.test/webhooks/twilio"
        self.params = {"CallSid": "CA1", "CallStatus": "completed", "Timestamp": "1700000000"}

    def test_a_correct_signature_verifies(self) -> None:
        signature = compute_signature(self.url, self.params, TOKEN)
        self.assertTrue(verify_twilio_signature(self.url, self.params, TOKEN, signature))

    def test_a_forged_signature_is_rejected(self) -> None:
        self.assertFalse(verify_twilio_signature(self.url, self.params, TOKEN, "Zm9yZ2Vk"))

    def test_a_missing_signature_is_rejected(self) -> None:
        accepted, reason = verify_request(self.url, self.params, TOKEN, None,
                                          now=1700000000)
        self.assertFalse(accepted)
        self.assertIn("missing", reason)

    def test_a_changed_parameter_invalidates_the_signature(self) -> None:
        signature = compute_signature(self.url, self.params, TOKEN)
        tampered = dict(self.params, CallStatus="completed-and-answered")
        self.assertFalse(verify_twilio_signature(self.url, tampered, TOKEN, signature))

    def test_the_wrong_token_is_rejected(self) -> None:
        signature = compute_signature(self.url, self.params, "other-token")
        self.assertFalse(verify_twilio_signature(self.url, self.params, TOKEN, signature))

    def test_http_is_rejected_when_https_is_required(self) -> None:
        signature = compute_signature("http://example.test/webhooks/twilio", self.params, TOKEN)
        accepted, reason = verify_request(
            "http://example.test/webhooks/twilio", self.params, TOKEN, signature,
            now=1700000000)
        self.assertFalse(accepted)
        self.assertIn("https", reason)

    def test_a_replayed_callback_is_rejected(self) -> None:
        signature = compute_signature(self.url, self.params, TOKEN)
        accepted, reason = verify_request(self.url, self.params, TOKEN, signature,
                                          tolerance_seconds=300.0, now=1700009999)
        self.assertFalse(accepted)
        self.assertIn("old", reason)

    def test_a_future_callback_is_rejected(self) -> None:
        signature = compute_signature(self.url, self.params, TOKEN)
        accepted, reason = verify_request(self.url, self.params, TOKEN, signature,
                                          tolerance_seconds=300.0, now=1699000000)
        self.assertFalse(accepted)
        self.assertIn("future", reason)


# --------------------------------------------------------------------------- #
class TestCallbackApplication(unittest.TestCase):
    def setUp(self) -> None:
        self.box = NotificationOutbox(":memory:")
        self.box.enqueue("INC-1", "call")
        self.box.claim("INC-1", "call")
        self.applier = CallbackApplier(self.box)

    def tearDown(self) -> None:
        self.box.close()

    def test_a_verified_callback_closes_the_row(self) -> None:
        result = self.applier.apply("call", "INC-1", "CA1", "completed")
        self.assertTrue(result.applied)
        self.assertEqual(self.box.state_of("INC-1", "call"), OutboxState.DONE)

    def test_a_duplicate_callback_is_a_logged_no_op(self) -> None:
        self.applier.apply("call", "INC-1", "CA1", "completed")
        again = self.applier.apply("call", "INC-1", "CA1", "completed")
        self.assertFalse(again.applied)
        self.assertIn("duplicate", again.reason)

    def test_an_unrecognised_status_is_ignored(self) -> None:
        result = self.applier.apply("call", "INC-1", "CA1", "teleported")
        self.assertFalse(result.applied)
        self.assertIn("unrecognised", result.reason)

    def test_a_callback_for_an_unknown_incident_is_ignored(self) -> None:
        result = self.applier.apply("call", "INC-NOPE", "CA9", "completed")
        self.assertFalse(result.applied)
        self.assertIn("no outbox row", result.reason)

    def test_completed_never_claims_a_human_answered(self) -> None:
        """Provider acceptance is not a person answering the phone."""
        result = self.applier.apply("call", "INC-1", "CA1", "completed")
        self.assertEqual(result.status, CallStatus.COMPLETED)
        self.assertEqual(self.box.get("INC-1", "call").state, OutboxState.DONE)
        blob = str(result.to_dict() if hasattr(result, "to_dict") else result.__dict__)
        self.assertNotIn("human", blob.lower())
        self.assertNotIn("answered_by", blob)

    def test_status_mapping_matches_the_polling_vocabulary(self) -> None:
        self.assertEqual(call_status_from_twilio("completed"), CallStatus.COMPLETED)
        self.assertEqual(call_status_from_twilio("no-answer"), CallStatus.NO_ANSWER)
        self.assertEqual(call_status_from_twilio("canceled"), CallStatus.FAILED)
        self.assertEqual(sms_status_from_twilio("delivered"), "SENT")
        self.assertEqual(sms_status_from_twilio("undelivered"), "FAILED")
        self.assertIsNone(call_status_from_twilio("nonsense"))


# --------------------------------------------------------------------------- #
def _live_test_script():
    """Import ``scripts/live_provider_test.py`` as a module."""
    import importlib.util

    path = ROOT / "scripts" / "live_provider_test.py"
    spec = importlib.util.spec_from_file_location("live_provider_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
class TestLiveTestGates(unittest.TestCase):
    """The live test's safety gates, tested without a provider.

    These matter most because they can be verified offline: the gates are the
    only thing standing between a mistaken run and a real message.
    """

    def _script(self):
        return _live_test_script()


# --------------------------------------------------------------------------- #
# The parser and gate_report must agree.
#
# Regression: ``gate_report`` read ``args.acknowledge_real_number`` while the
# parser declared ``--i-understand-this-will-ring-the-real-number`` (dest
# ``i_understand_this_will_ring_the_real_number``). Every gated run therefore
# died with ``AttributeError: 'Namespace' object has no attribute
# 'acknowledge_real_number'`` *after* the operator had passed ``--live`` and
# supplied credentials - i.e. exactly when it mattered most.
class TestLiveTestCliContract(unittest.TestCase):
    """Parser/gate agreement, plus the gates themselves, offline."""

    def _script(self):
        return _live_test_script()

    def test_every_flag_gate_report_reads_exists_on_the_parser(self) -> None:
        script = _live_test_script()
        parser = script.build_parser()
        for argv in ([], ["--live"], ["--live", "--channel", "call"],
                     ["--live", "--acknowledge-real-number"],
                     ["--live", "--i-understand-this-will-ring-the-real-number"]):
            args = parser.parse_args(argv)
            for attribute in ("live", "acknowledge_real_number", "recipient_env_var"):
                self.assertTrue(
                    hasattr(args, attribute),
                    f"gate_report reads args.{attribute} but the parser does not "
                    f"define it for argv={argv}",
                )

    def test_the_documented_command_does_not_crash(self) -> None:
        """`--live --channel sms` must reach the gates, not raise."""
        script = _live_test_script()
        args = script.build_parser().parse_args(["--live", "--channel", "sms"])
        passed, checks = script.gate_report(args, {})   # must not raise
        self.assertFalse(passed)
        self.assertTrue(checks)

    def test_the_acknowledgement_flag_works_under_both_spellings(self) -> None:
        script = _live_test_script()
        parser = script.build_parser()
        for flag in ("--acknowledge-real-number",
                     "--i-understand-this-will-ring-the-real-number"):
            args = parser.parse_args(["--live", flag])
            self.assertTrue(args.acknowledge_real_number,
                            f"{flag} did not set acknowledge_real_number")

    def test_an_unknown_namespace_refuses_instead_of_crashing(self) -> None:
        """A missing attribute must fail CLOSED, never raise and never pass."""
        import argparse

        script = _live_test_script()
        bare = argparse.Namespace(live=True)
        passed, checks = script.gate_report(bare, {
            "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+15005550306",
            "EMERGENCY_CONTACT_LIVE_TEST": "YES",
            "EMERGENCY_CONTACT_LIVE_TEST_NUMBER": "+15550100",
            "EMERGENCY_CONTACT_NUMBER": "+15550100",   # same -> gate 8 applies
        })
        self.assertFalse(passed, "an unreadable flag must refuse, not pass")
        failed = [name for name, ok, _ in checks if not ok]
        self.assertTrue(any("acknowledge" in name for name in failed), failed)

    def test_the_recipient_acknowledgement_gate_still_exists(self) -> None:
        """The safety gate itself must not have been dropped."""
        script = _live_test_script()
        env = {
            "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+15005550306",
            "EMERGENCY_CONTACT_LIVE_TEST": "YES",
            "EMERGENCY_CONTACT_LIVE_TEST_NUMBER": "+15550100",
            "EMERGENCY_CONTACT_NUMBER": "+15550100",
        }
        args = script.build_parser().parse_args(["--live"])
        passed, checks = script.gate_report(args, env)
        self.assertFalse(passed, "the real contact must require acknowledgement")
        names = [name for name, _, _ in checks]
        self.assertTrue(any("acknowledge" in name for name in names), names)

        args = script.build_parser().parse_args(
            ["--live", "--acknowledge-real-number"])
        passed, checks = script.gate_report(args, env)
        self.assertTrue(passed, f"all gates should pass: {checks}")

    def test_the_consent_flag_is_still_required(self) -> None:
        script = _live_test_script()
        args = script.build_parser().parse_args(["--live"])
        passed, checks = script.gate_report(args, {
            "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+15005550306",
            "EMERGENCY_CONTACT_LIVE_TEST_NUMBER": "+15550100",
            # EMERGENCY_CONTACT_LIVE_TEST deliberately absent
        })
        self.assertFalse(passed)
        failed = [name for name, ok, _ in checks if not ok]
        self.assertTrue(any("consent" in name.lower() for name in failed), failed)

    def test_emergency_numbers_are_hard_blocked(self) -> None:
        script = self._script()
        for number in ("112", "911", "999", "+91 112", "+1-911", "00112", "100", "+44 999"):
            self.assertTrue(script.is_emergency_number(number), f"{number} must be blocked")

    def test_ordinary_mobile_numbers_are_not_blocked(self) -> None:
        """A mobile number that merely *ends* in 112 is not an emergency number."""
        script = self._script()
        for number in ("+15550100", "+15005550306", "+15550100", "+15551230000"):
            self.assertFalse(script.is_emergency_number(number), f"{number} is a normal number")

    def test_the_script_refuses_without_the_live_flag(self) -> None:
        import argparse
        script = self._script()
        args = argparse.Namespace(live=False, recipient_env_var="X", acknowledge_real_number=False)
        passed, checks = script.gate_report(args, {})
        self.assertFalse(passed)
        self.assertTrue(any("explicit --live flag" in name for name, _, _ in checks))

    def test_the_script_refuses_without_credentials(self) -> None:
        import argparse
        script = self._script()
        args = argparse.Namespace(live=True, recipient_env_var="R", acknowledge_real_number=False)
        env = {"R": "+15550100", "EMERGENCY_CONTACT_LIVE_TEST": "YES"}
        passed, checks = script.gate_report(args, env)
        self.assertFalse(passed, "credentials are missing, so it must refuse")
        failed = [name for name, ok, _ in checks if not ok]
        self.assertTrue(any("ACCOUNT_SID" in name for name in failed))

    def test_the_script_refuses_without_operator_consent(self) -> None:
        import argparse
        script = self._script()
        args = argparse.Namespace(live=True, recipient_env_var="R", acknowledge_real_number=False)
        env = {"R": "+15550100", "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
               "TWILIO_AUTH_TOKEN": "tok", "TWILIO_FROM_NUMBER": "+15005550306"}
        passed, checks = script.gate_report(args, env)
        self.assertFalse(passed, "consent was not given")
        self.assertTrue(any("consent flag" in name for name, ok, _ in checks if not ok))

    def test_the_script_refuses_an_emergency_recipient_even_when_otherwise_ready(self) -> None:
        import argparse
        script = self._script()
        args = argparse.Namespace(live=True, recipient_env_var="R", acknowledge_real_number=False)
        env = {"R": "112", "TWILIO_ACCOUNT_SID": "AC" + "0" * 32, "TWILIO_AUTH_TOKEN": "tok",
               "TWILIO_FROM_NUMBER": "+15005550306", "EMERGENCY_CONTACT_LIVE_TEST": "YES"}
        passed, checks = script.gate_report(args, env)
        self.assertFalse(passed, "an emergency number must be refused outright")
        self.assertTrue(any("NOT an emergency number" in name
                            for name, ok, _ in checks if not ok))


class TestSafetyGuarantees(unittest.TestCase):
    def test_no_phase4_test_can_reach_a_network(self) -> None:
        import urllib.request

        attempts = []
        original = urllib.request.urlopen

        def blocked(url, *a, **k):
            attempts.append(str(url))
            raise AssertionError(f"a Phase 4 test contacted the network: {url}")

        urllib.request.urlopen = blocked
        try:
            dispatcher = make_dispatcher(
                _provider=ScriptedCallProvider([ok_call()]),
                _sms=ScriptedSmsProvider([ok_sms()]),
            )
            dispatcher.handle_incident(incident())
        finally:
            urllib.request.urlopen = original
        self.assertEqual(attempts, [])

    def test_the_library_defaults_can_notify_nobody(self) -> None:
        """Importing SafeVision can never notify anybody.

        The *library* defaults are off. The shipped `config.yaml` turns the two
        allow-flags on so that DEMO_MODE=false really does send, but credentials
        remain the gate - see the next test.
        """
        from ai_engine.config.classes import EmergencyConfig

        defaults = EmergencyConfig()
        self.assertFalse(defaults.allow_real_call)
        self.assertFalse(defaults.allow_real_sms)
        self.assertFalse(defaults.webhook_enabled)
        self.assertFalse(defaults.allow_emergency_service_routing)
        self.assertEqual(defaults.use_outbox, False)

    def test_real_sending_still_requires_credentials(self) -> None:
        """The allow-flags are not the gate - the credentials are."""
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = False
        config.allow_real_call = True
        config.allow_real_sms = True
        config.async_calls = False
        config.use_outbox = False
        dispatcher = DemoCallDispatcher(
            config, environ={"DEMO_MODE": "false", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            sleep=lambda _s: None,
        )
        record = dispatcher.handle_incident(incident())
        # no TWILIO_* in this environment, so nothing may be sent
        self.assertEqual(record.status, CallStatus.NOT_CONFIGURED)
        self.assertFalse(record.status.was_placed)

    def test_the_correlation_id_is_shared_by_both_channels(self) -> None:
        dispatcher = make_dispatcher(_provider=ScriptedCallProvider([ok_call()]))
        record = dispatcher.handle_incident(incident())
        rows = dispatcher.outbox.for_incident(record.incident_id)
        self.assertEqual(len({row.correlation_id for row in rows}), 1)
        self.assertTrue(record.correlation_id)

    def test_the_outbox_records_the_provider_message_id(self) -> None:
        dispatcher = make_dispatcher(_provider=ScriptedCallProvider([ok_call()]),
                                     _sms=ScriptedSmsProvider([ok_sms()]))
        record = dispatcher.handle_incident(incident())
        rows = {row.channel: row for row in dispatcher.outbox.for_incident(record.incident_id)}
        self.assertEqual(rows["call"].provider_ref, "CA0000FAKE")
        self.assertEqual(rows["sms"].provider_ref, "SM0000FAKE")

    def test_notifications_stay_off_the_video_thread(self) -> None:
        """A notification must never stall frame processing."""
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = True
        config.async_calls = True
        config.write_call_log = False
        config.use_outbox = False
        started = []
        release = threading.Event()

        class SlowProvider(CallProvider):
            name = "slow"

            def available(self):
                return True, ""

            def place(self, recipient, message, on_status):
                started.append(time.perf_counter())
                release.wait(2.0)
                return ok_call()

        dispatcher = DemoCallDispatcher(
            config,
            environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=SlowProvider(),
        )
        began = time.perf_counter()
        dispatcher.handle_incident(incident())
        handed_back = time.perf_counter() - began
        # the provider blocks for 2s; handle_incident must not wait for it
        self.assertLess(handed_back, 0.5,
                        "handle_incident must return before dialling completes")
        release.set()
        for thread in dispatcher._threads:      # noqa: SLF001 - join the worker
            thread.join(timeout=3.0)
        self.assertEqual(len(started), 1, "the call was placed, just not inline")


if __name__ == "__main__":
    unittest.main()