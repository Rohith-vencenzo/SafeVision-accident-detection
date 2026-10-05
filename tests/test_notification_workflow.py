"""The notification contract the CLI depends on.

    CONFIRMED_ACCIDENT -> call -> CALL COMPLETED -> SMS -> SENT

These tests exist because the engine *did* place the call and SMS correctly while
the console reported ``INITIATED`` / ``PENDING``: the incident payload is a
snapshot taken before the provider was contacted, and the process exited without
waiting for the worker. That is a correctness bug for anything that acts on the
reported state, so it is pinned here.

Order is asserted on the recorded timeline, not on log text.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.emergency.call_record import DemoCallRecord  # noqa: E402
from ai_engine.emergency.dispatcher import DemoCallDispatcher  # noqa: E402
from ai_engine.emergency.outbox import NotificationOutbox, OutboxState  # noqa: E402
from ai_engine.emergency.providers import (  # noqa: E402
    CallProvider,
    ProviderResult,
    SmsProvider,
)
from ai_engine.schemas.enums import CallStatus, SmsStatus  # noqa: E402
from tests.helpers import test_config  # noqa: E402

RECIPIENT = "+15550100"


class LifecycleProvider(CallProvider):
    """Emits the full call lifecycle and records the order it was invoked in."""

    name = "lifecycle"

    def __init__(self, order: list) -> None:
        self.order = order

    def available(self):
        return True, ""

    def place(self, recipient, message, on_status):
        self.order.append("call_start")
        on_status(CallStatus.INITIATED, "queued", "CA-LIFE")
        on_status(CallStatus.RINGING, "ringing", "CA-LIFE")
        on_status(CallStatus.IN_PROGRESS, "in-progress", "CA-LIFE")
        on_status(CallStatus.COMPLETED, "completed", "CA-LIFE")
        self.order.append("call_completed")
        return ProviderResult(ok=True, status=CallStatus.COMPLETED,
                              provider_status="completed", provider_call_sid="CA-LIFE")


class LifecycleSms(SmsProvider):
    name = "lifecycle-sms"

    def __init__(self, order: list) -> None:
        self.order = order
        self.bodies = []

    def available(self):
        return True, ""

    def send(self, recipient, body):
        self.order.append("sms_send")
        self.bodies.append(body)
        return ProviderResult(ok=True, status=CallStatus.COMPLETED,
                              provider_status="queued", provider_call_sid="SM-LIFE")


def incident(incident_id="INC-ORDER-0001"):
    return {
        "incident_id": incident_id,
        "status": "CONFIRMED_ACCIDENT",
        "timestamp": "2026-10-03T10:00:00+00:00",
        "camera": {"camera_id": "CAM-001"},
        "location": {"name": "Chennai Main Road", "latitude": 13.08, "longitude": 80.27,
                     "is_camera_registered": True},
        "accident": {"score": 0.6, "severity": "HIGH"},
    }


def dispatcher(order, sms_provider, **kw):
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
    return DemoCallDispatcher(
        config,
        environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
        provider=LifecycleProvider(order), sms_provider=sms_provider,
        sleep=lambda _s: None,
    )


# --------------------------------------------------------------------------- #
class TestCallBeforeSms(unittest.TestCase):
    def test_the_sms_is_sent_only_after_the_call_completes(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        box.handle_incident(incident())
        self.assertEqual(
            order,
            ["call_start", "call_completed", "sms_send"],
            "the SMS must never be attempted before the call has completed",
        )

    def test_no_sms_is_sent_when_the_call_never_reaches_the_network(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order), sms_on_call_failure=False)
        # replace the provider with one that fails before any resource id
        class DeadProvider(CallProvider):
            name = "dead"

            def available(self):
                return True, ""

            def place(self, recipient, message, on_status):
                order.append("call_start")
                on_status(CallStatus.FAILED, "failed", "")
                return ProviderResult(ok=False, status=CallStatus.FAILED,
                                      error="carrier rejected")

        box.provider = DeadProvider()
        record = box.handle_incident(incident())
        self.assertNotIn("sms_send", order)
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED)

    def test_a_fallback_sms_still_follows_a_failed_call_when_configured(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order), sms_on_call_failure=True)

        class DeadProvider(CallProvider):
            name = "dead"

            def available(self):
                return True, ""

            def place(self, recipient, message, on_status):
                order.append("call_start")
                on_status(CallStatus.FAILED, "failed", "")
                return ProviderResult(ok=False, status=CallStatus.FAILED,
                                      error="carrier rejected")

        box.provider = DeadProvider()
        record = box.handle_incident(incident())
        self.assertEqual(order, ["call_start", "sms_send"])
        self.assertEqual(record.sms.status, SmsStatus.SENT)


# --------------------------------------------------------------------------- #
class TestExactlyOnce(unittest.TestCase):
    def test_one_confirmed_incident_produces_one_call_and_one_sms(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        for _ in range(5):                     # duplicate triggers
            box.handle_incident(incident())
        self.assertEqual(order.count("call_start"), 1)
        self.assertEqual(order.count("sms_send"), 1)

    def test_two_distinct_incidents_each_get_their_own_call_and_sms(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        box.handle_incident(incident("INC-A"))
        box.handle_incident(incident("INC-B"))
        self.assertEqual(order.count("call_start"), 2)
        self.assertEqual(order.count("sms_send"), 2)

    def test_a_replayed_incident_after_a_restart_sends_nothing(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "outbox.sqlite3")
            order: list = []
            config = test_config().emergency
            config.enabled = True
            config.demo_mode = True
            config.write_call_log = False
            config.async_calls = False
            config.use_outbox = True
            config.outbox_path = path
            config.retry_base_delay_seconds = 0.0

            first = DemoCallDispatcher(
                config,
                environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
                provider=LifecycleProvider(order), sms_provider=LifecycleSms(order),
                sleep=lambda _s: None,
            )
            first.handle_incident(incident())
            first.close()
            self.assertEqual(order.count("call_start"), 1)

            # a fresh process, the same incident id
            second = DemoCallDispatcher(
                config,
                environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
                provider=LifecycleProvider(order), sms_provider=LifecycleSms(order),
                sleep=lambda _s: None,
            )
            second.handle_incident(incident())
            second.close()
            self.assertEqual(order.count("call_start"), 1,
                             "a restart must not re-notify a done incident")
            self.assertEqual(order.count("sms_send"), 1)

    def test_only_a_confirmed_incident_notifies(self) -> None:
        for status in ("SUSPICIOUS", "VERIFYING", "NORMAL", "FALSE_ALARM", "POSSIBLE_ACCIDENT"):
            order: list = []
            box = dispatcher(order, LifecycleSms(order))
            record = box.handle_incident({**incident(f"INC-{status}"), "status": status})
            self.assertEqual(order, [], f"{status} must not notify")
            # an unconfirmed incident never even creates a notification record
            self.assertIsNone(record)
            self.assertEqual(box.attempted_count, 0)
            self.assertEqual(box.sms_count, 0)

    def test_the_outbox_holds_one_row_per_channel_for_the_incident(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        box.handle_incident(incident())
        rows = box.outbox.for_incident("INC-ORDER-0001")
        self.assertEqual(sorted(row.channel for row in rows), ["call", "sms"])
        for row in rows:
            self.assertEqual(row.state, OutboxState.DONE)
            self.assertEqual(row.attempt, 1)


# --------------------------------------------------------------------------- #
class TestReportedStateIsFinalState(unittest.TestCase):
    """The bug this file exists for: the console showed the wrong state."""

    def test_the_record_snapshot_is_stale_but_final_states_are_not(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        record = box.handle_incident(incident())

        # the snapshot taken when the incident was built: still in progress
        snapshot_call = record.to_dict(include_recipient=False)["status"]
        snapshot_sms = record.sms.status

        # the authoritative outcome, after the worker finished
        live = box.final_states()["INC-ORDER-0001"]
        self.assertEqual(live["call"], "COMPLETED")
        self.assertEqual(live["sms"], "SENT")
        self.assertTrue(live["call_placed"])
        self.assertEqual(live["provider_ref"], "CA-LIFE")
        self.assertEqual(live["sms_provider_ref"], "SM-LIFE")
        # the snapshot is allowed to lag; that is exactly why the CLI must not
        # report it as the outcome
        self.assertIn(snapshot_call, ("INITIATED", "COMPLETED"))
        self.assertIn(snapshot_sms, ("PENDING", "SENT"))

    def test_wait_returns_zero_when_everything_has_finished(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        box.handle_incident(incident())
        self.assertEqual(box.wait(timeout=5.0), 0)
        self.assertEqual(box.pending(), 0)

    def test_the_full_lifecycle_is_recorded_in_order(self) -> None:
        order: list = []
        box = dispatcher(order, LifecycleSms(order))
        record = box.handle_incident(incident())
        states = [entry["status"] for entry in record.history]
        self.assertEqual(
            states,
            ["INITIATED", "RINGING", "IN_PROGRESS", "COMPLETED"],
            "the whole call lifecycle must be recorded, in order",
        )


# --------------------------------------------------------------------------- #
class TestSmsContent(unittest.TestCase):
    def test_the_sms_carries_every_required_field(self) -> None:
        order: list = []
        sms = LifecycleSms(order)
        box = dispatcher(order, sms)
        box.handle_incident(incident())
        body = sms.bodies[0]
        for required in ("Incident: INC-ORDER-0001",
                         "Date:", "Time:", "Camera: CAM-001",
                         "Chennai Main Road", "Severity: HIGH",
                         "Emergency assistance may be required"):
            self.assertIn(required, body, f"SMS is missing {required!r}")

    def test_the_sms_names_its_location_source(self) -> None:
        order: list = []
        sms = LifecycleSms(order)
        box = dispatcher(order, sms)
        box.handle_incident(incident())
        self.assertIn("CAMERA-REGISTERED", sms.bodies[0])
        self.assertIn("not a GPS fix", sms.bodies[0])

    def test_the_sms_is_not_empty_and_respects_the_length_cap(self) -> None:
        order: list = []
        sms = LifecycleSms(order)
        box = dispatcher(order, sms)
        box.handle_incident(incident())
        self.assertTrue(sms.bodies[0].strip())
        self.assertLessEqual(len(sms.bodies[0]), box.config.sms_max_body_length)


# --------------------------------------------------------------------------- #
class TestProviderFailuresAreNeverSuccess(unittest.TestCase):
    def _failing(self, call_result, sms_result, **kw):
        order: list = []

        class C(CallProvider):
            name = "c"

            def available(self):
                return True, ""

            def place(self, recipient, message, on_status):
                order.append("call")
                if isinstance(call_result, Exception):
                    raise call_result
                on_status(call_result.status, "", call_result.provider_call_sid)
                return call_result

        class S(SmsProvider):
            name = "s"

            def available(self):
                return True, ""

            def send(self, recipient, body):
                order.append("sms")
                if isinstance(sms_result, Exception):
                    raise sms_result
                return sms_result

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
        box = DemoCallDispatcher(
            config,
            environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=C(), sms_provider=S(), sleep=lambda _s: None,
        )
        return box

    def test_a_failing_call_is_not_reported_as_completed(self) -> None:
        box = self._failing(
            ProviderResult(ok=False, status=CallStatus.FAILED, error="carrier rejected"),
            ProviderResult(ok=True, status=CallStatus.COMPLETED))
        record = box.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertNotEqual(record.status, CallStatus.COMPLETED)
        self.assertIn("carrier rejected", record.error)

    def test_a_raising_call_provider_is_contained(self) -> None:
        box = self._failing(
            RuntimeError("twilio exploded"),
            ProviderResult(ok=True, status=CallStatus.COMPLETED))
        record = box.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertIn("provider error", record.error)

    def test_a_failing_sms_keeps_the_call_result_and_reports_the_sms_error(self) -> None:
        box = self._failing(
            ProviderResult(ok=True, status=CallStatus.COMPLETED,
                           provider_call_sid="CA-OK"),
            ProviderResult(ok=False, status=CallStatus.FAILED,
                           error="unrecognised recipient"))
        record = box.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.COMPLETED)
        self.assertEqual(record.sms.status, SmsStatus.FAILED)
        self.assertIn("unrecognised recipient", record.sms.error)

    def test_a_raising_sms_provider_is_contained(self) -> None:
        box = self._failing(
            ProviderResult(ok=True, status=CallStatus.COMPLETED,
                           provider_call_sid="CA-OK"),
            RuntimeError("sms backend exploded"))
        record = box.handle_incident(incident())
        self.assertEqual(record.status, CallStatus.COMPLETED)
        self.assertEqual(record.sms.status, SmsStatus.FAILED)

    def test_a_real_provider_id_is_recorded_for_the_audit_trail(self) -> None:
        box = self._failing(
            ProviderResult(ok=True, status=CallStatus.COMPLETED,
                           provider_call_sid="CA-real-123"),
            ProviderResult(ok=True, status=CallStatus.COMPLETED,
                           provider_call_sid="SM-real-456"))
        box.handle_incident(incident())
        live = box.final_states()["INC-ORDER-0001"]
        self.assertEqual(live["provider_ref"], "CA-real-123")
        self.assertEqual(live["sms_provider_ref"], "SM-real-456")
        self.assertTrue(live["correlation_id"])


if __name__ == "__main__":
    unittest.main()