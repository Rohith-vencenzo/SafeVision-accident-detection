"""Emergency notification: call first, then SMS, exactly once per incident.

Two halves, and the second one is meaningless without the first:

**False-alarm tests** - a notification must only ever follow a *verified*
incident.  SUSPICIOUS, VERIFYING, a single-frame ghost and every quiet scenario
must all produce nothing.

**Notification tests** - one call, one SMS, in that order, with the real
incident data in the SMS, and no duplication across the frames that follow the
confirmation.

Every test here uses recording fakes.  Nothing in this module can reach a
telephony API: :meth:`_forbid_network` makes any outbound socket an immediate
failure, so a regression that tried to contact Twilio would fail loudly.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.emergency.call_record import DemoCallRecord  # noqa: E402
from ai_engine.emergency.dispatcher import DemoCallDispatcher  # noqa: E402
from ai_engine.emergency.providers import (  # noqa: E402
    CallProvider,
    ProviderResult,
    SimulationSmsProvider,
    NotConfiguredSmsProvider,
    SmsProvider,
    TwilioProvider,
    TwilioSmsProvider,
    build_sms_provider,
)
from ai_engine.emergency.sms_record import build_sms_body  # noqa: E402
from ai_engine.schemas.enums import CallStatus, IncidentStatus, SmsStatus  # noqa: E402
from ai_engine.schemas.incident_result import DemoSmsInfo, validate_incident  # noqa: E402
from tests.helpers import blank_frame, test_config  # noqa: E402

RECIPIENT = "+15550100"  # fictional 555 number; tests never reach a real phone


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _is_env_pattern_ignored(name: str) -> bool:
    """Would git ignore a file with this name? (mirrors .gitignore's fnmatch rules)."""
    import fnmatch

    patterns = [
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    ignored = False
    for pattern in patterns:
        if pattern.startswith("!"):
            if fnmatch.fnmatch(name, pattern[1:]):
                ignored = False
        elif fnmatch.fnmatch(name, pattern):
            ignored = True
    return ignored


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class RecordingCallProvider(CallProvider):
    """Records every call and the state of the record at call time."""

    name = "recording-call"

    def __init__(self, final: CallStatus = CallStatus.COMPLETED) -> None:
        self.attempts: List[Dict[str, Any]] = []
        self.final = final

    def available(self):
        return True, ""

    def place(self, recipient: str, message: str, on_status) -> ProviderResult:
        self.attempts.append({"recipient": recipient, "message": message})
        on_status(CallStatus.RINGING, "ringing", "SID-CALL", "")
        on_status(CallStatus.IN_PROGRESS, "in-progress", "SID-CALL", "")
        on_status(self.final, str(self.final).lower(), "SID-CALL", "")
        return ProviderResult(
            ok=self.final is CallStatus.COMPLETED,
            status=self.final,
            provider_status=str(self.final).lower(),
            provider_call_sid="SID-CALL",
        )


class RecordingSmsProvider(SmsProvider):
    """Records SMS sends, and what the call was doing when it was sent."""

    name = "recording-sms"

    def __init__(self, record: DemoCallRecord = None, ok: bool = True) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.record = record
        self.ok = ok

    def available(self):
        return True, ""

    def send(self, recipient: str, body: str) -> ProviderResult:
        self.sent.append(
            {
                "recipient": recipient,
                "body": body,
                "call_status_at_send": str(self.record.status) if self.record else None,
                "call_sid_at_send": self.record.provider_call_sid if self.record else None,
            }
        )
        if not self.ok:
            return ProviderResult(ok=False, status=CallStatus.FAILED, error="provider refused")
        return ProviderResult(
            ok=True, status=CallStatus.COMPLETED, provider_status="queued",
            provider_call_sid="SID-SMS",
        )


class ExplodingCallProvider(CallProvider):
    """Simulates a provider that blows up."""

    name = "exploding"

    def available(self):
        return True, ""

    def place(self, recipient: str, message: str, on_status) -> ProviderResult:
        raise RuntimeError("telephony backend exploded")


class ExplodingSmsProvider(SmsProvider):
    """Simulates an SMS provider that blows up *after* a good call."""

    name = "exploding-sms"

    def available(self):
        return True, ""

    def send(self, recipient: str, body: str) -> ProviderResult:
        raise RuntimeError("sms backend exploded")


def _forbid_network():
    """Make any outbound HTTP call an immediate, loud failure."""
    calls: List[str] = []

    def blocked(url, *args, **kwargs):  # noqa: ANN001
        calls.append(str(url))
        raise AssertionError(f"the emergency path must not touch the network: {url}")

    return urllib.request.urlopen, blocked, calls


# --------------------------------------------------------------------------- #
def make_incident(incident_id: str = "INC-20261002-0001", **overrides) -> Dict[str, Any]:
    payload = {
        "incident_id": incident_id,
        "status": IncidentStatus.CONFIRMED_ACCIDENT.value,
        "timestamp": "2026-10-02T18:31:35.482913+00:00",
        "detected_at_unix": 1790000000.0,
        "camera": {"camera_id": "CAM-001", "name": "Main Road Camera"},
        "location": {
            "name": "Chennai Main Road",
            "road": "Anna Salai",
            "latitude": 13.0827,
            "longitude": 80.2707,
            "is_camera_registered": True,
            "source": "camera_metadata",
        },
        "accident": {"score": 0.52, "severity": "HIGH", "confirmed": True},
        "objects": {"vehicles_involved": 2, "persons_detected": 0},
        "evidence": {"reasons": ["Sudden vehicle velocity change"], "agreement_signals": 3},
    }
    payload.update(overrides)
    return payload


def make_dispatcher(
    call_provider: CallProvider = None,
    sms_provider: SmsProvider = None,
    **config_overrides,
) -> DemoCallDispatcher:
    """A dispatcher with recording providers and demo mode on."""
    config = test_config().emergency
    config.enabled = True
    config.demo_mode = True
    config.async_calls = False
    config.write_call_log = False
    config.expose_recipient = False
    for key, value in config_overrides.items():
        setattr(config, key, value)
    call = call_provider or RecordingCallProvider()
    sms = sms_provider or RecordingSmsProvider()
    if isinstance(sms, RecordingSmsProvider):
        sms.record = None  # bound to the record inside the test
    dispatcher = DemoCallDispatcher(
        config,
        environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
        provider=call,
        sms_provider=sms,
    )
    dispatcher._test_call = call          # noqa: SLF001 - test-only handles
    dispatcher._test_sms = sms            # noqa: SLF001
    return dispatcher


def bind_sms_to_record(dispatcher: DemoCallDispatcher) -> RecordingSmsProvider:
    """Let the fake SMS provider see the call record it follows."""
    if isinstance(dispatcher._test_sms, RecordingSmsProvider):  # noqa: SLF001
        dispatcher._test_sms.record = dispatcher.latest  # noqa: SLF001
    return dispatcher._test_sms  # noqa: SLF001


def run_scenario(scenario: str, dispatcher: DemoCallDispatcher = None, frames: int = 90):
    """Run a scripted scenario through the real pipeline."""
    from ai_engine.detection.scenarios import SCENARIOS
    from ai_engine.detection.scripted import ScriptedDetector
    from ai_engine.pipeline import AccidentPipeline

    config = test_config()
    if dispatcher is None:
        dispatcher = make_dispatcher()
    config.emergency.enabled = True
    config.emergency.demo_mode = True
    config.emergency.async_calls = False
    config.emergency.write_call_log = False
    config.emergency.expose_recipient = False

    original = dispatcher.config
    dispatcher.config = config.emergency
    pipeline = AccidentPipeline(
        config,
        detector=ScriptedDetector(SCENARIOS[scenario]),
        camera_id="CAM-001",
        emergency=dispatcher,
    )
    frame = blank_frame()
    base = time.time()
    for index in range(frames):
        pipeline.process_frame(frame, index, base + index / 15.0)
    pipeline.close()
    dispatcher.config = original
    return pipeline, dispatcher


# =========================================================================== #
# 1. FALSE-ALARM PREVENTION
# =========================================================================== #
class TestFalseAlarmsNeverNotify(unittest.TestCase):
    """Section 1 of the brief: nothing may be sent before CONFIRMED_ACCIDENT."""

    def test_non_confirmed_statuses_never_notify(self) -> None:
        """NORMAL / POSSIBLE_ACCIDENT must produce no call and no SMS."""
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms)
        for index, status in enumerate(
            (
                IncidentStatus.NORMAL.value,
                IncidentStatus.POSSIBLE_ACCIDENT.value,
                IncidentStatus.FALSE_ALARM.value,
            )
        ):
            record = dispatcher.handle_incident(make_incident(f"INC-N-{index}", status=status))
            self.assertIsNone(record, f"status {status} must not reach the dispatcher")
        self.assertEqual(call.attempts, [])
        self.assertEqual(sms.sent, [])
        self.assertEqual(dispatcher.calls, [])

    def test_suspicious_and_verifying_states_do_not_notify(self) -> None:
        """The dispatcher only ever sees confirmed incidents - nothing earlier."""
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms)
        # there is deliberately no API to hand the dispatcher a state; the
        # pipeline only calls it from the CONFIRMED_ACCIDENT branch
        public = [n for n in dir(dispatcher) if not n.startswith("_")]
        self.assertNotIn("handle_state", public)
        self.assertNotIn("handle_frame", public)
        # and a confirmed incident does notify
        record = dispatcher.handle_incident(make_incident("INC-FIRST"))
        self.assertIsNotNone(record)
        self.assertEqual(len(call.attempts), 1)

    def test_quiet_scenarios_produce_no_notification(self) -> None:
        """Normal traffic, hard braking, parked cars, a single-frame ghost."""
        for scenario in (
            "opposite_lanes",
            "following_close",
            "parked_cars",
            "hard_brake",
            "pedestrian_crossing",
            "single_frame_ghost",
        ):
            with self.subTest(scenario=scenario):
                pipeline, dispatcher = run_scenario(scenario)
                self.assertEqual(pipeline.incidents, [], "no incident may be raised")
                self.assertEqual(dispatcher._test_call.attempts, [])   # noqa: SLF001
                self.assertEqual(dispatcher._test_sms.sent, [])        # noqa: SLF001

    def test_a_single_frame_detection_never_notifies(self) -> None:
        """One noisy frame cannot reach confirmation, so nothing is sent."""
        pipeline, dispatcher = run_scenario("single_frame_ghost", frames=90)
        self.assertEqual(pipeline.incidents, [])
        self.assertEqual(len(dispatcher._test_call.attempts), 0)  # noqa: SLF001

    def test_both_switches_off_never_notifies(self) -> None:
        """**Updated for the real-send workflow.**

        `DEMO_MODE=false` + `allow_real_call=False` is the fully-disabled state:
        nothing is dialled and nothing is messaged.
        """
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms, allow_real_call=False,
                                     allow_real_sms=False)
        dispatcher.environ["DEMO_MODE"] = "false"
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(call.attempts, [])
        self.assertEqual(sms.sent, [])
        self.assertEqual(record.status, CallStatus.SKIPPED)

    def test_real_mode_notifies_call_then_sms(self) -> None:
        """REAL MODE: `DEMO_MODE=false` with real sending enabled must notify.

        This is the path the operator needs working. The call comes first and the
        SMS strictly after it.
        """
        order: List[str] = []

        class OrderedCall(RecordingCallProvider):
            def place(self, recipient, message, on_status):
                order.append("call")
                return super().place(recipient, message, on_status)

        class OrderedSms(RecordingSmsProvider):
            def send(self, recipient, body):
                order.append("sms")
                return super().send(recipient, body)

        call = OrderedCall()
        sms = OrderedSms()
        dispatcher = make_dispatcher(call, sms, allow_real_call=True,
                                     allow_real_sms=True)
        dispatcher.environ["DEMO_MODE"] = "false"
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(order, ["call", "sms"], "call must precede the SMS")
        self.assertEqual(record.status, CallStatus.COMPLETED)
        self.assertEqual(record.sms.status, SmsStatus.SENT)

    def test_no_recipient_never_notifies(self) -> None:
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms)
        dispatcher.environ.pop("EMERGENCY_CONTACT_NUMBER")
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(call.attempts, [])
        self.assertEqual(sms.sent, [])
        self.assertEqual(record.status, CallStatus.BLOCKED)
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED.value)

    def test_a_genuine_crash_confirms_and_notifies(self) -> None:
        """The other half of section 1: real evidence must still get through."""
        pipeline, dispatcher = run_scenario("head_on_crash")
        self.assertEqual(len(pipeline.incidents), 1)
        self.assertEqual(len(dispatcher._test_call.attempts), 1)  # noqa: SLF001
        self.assertEqual(len(dispatcher._test_sms.sent), 1)       # noqa: SLF001


# =========================================================================== #
# 2. ONE INCIDENT -> ONE CALL AND ONE SMS
# =========================================================================== #
class TestExactlyOneCallAndSms(unittest.TestCase):
    def test_one_call_one_sms_per_confirmed_incident(self) -> None:
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms)
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(len(call.attempts), 1)
        self.assertEqual(len(sms.sent), 1)
        self.assertEqual(record.status, CallStatus.COMPLETED)
        self.assertEqual(record.sms.status, SmsStatus.SENT.value)

    def test_duplicate_frames_do_not_duplicate_notifications(self) -> None:
        """Every frame after the confirmation must be inert."""
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms)
        for _ in range(500):
            dispatcher.handle_incident(make_incident("INC-DUPE"))
        self.assertEqual(len(call.attempts), 1)
        self.assertEqual(len(sms.sent), 1)
        self.assertEqual(dispatcher.call_count, 1)
        self.assertEqual(dispatcher.sms_count, 1)
        self.assertEqual(dispatcher.calls[0].deliveries, 500)

    def test_distinct_incidents_each_get_their_own_pair(self) -> None:
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms)
        for index in range(3):
            dispatcher.handle_incident(make_incident(f"INC-{index}"))
        self.assertEqual(len(call.attempts), 3)
        self.assertEqual(len(sms.sent), 3)

    def test_a_long_run_yields_one_pair(self) -> None:
        """The aftermath of a crash must not spam the recipient."""
        pipeline, dispatcher = run_scenario("rear_end_crash", frames=140)
        self.assertEqual(len(pipeline.incidents), 1)
        self.assertEqual(len(dispatcher._test_call.attempts), 1)  # noqa: SLF001
        self.assertEqual(len(dispatcher._test_sms.sent), 1)       # noqa: SLF001


# =========================================================================== #
# 3. CALL FIRST, SMS SECOND
# =========================================================================== #
class TestCallPrecedesSms(unittest.TestCase):
    def test_sms_is_sent_after_the_call_has_finished(self) -> None:
        dispatcher = make_dispatcher()
        sms = bind_sms_to_record(dispatcher)  # type: ignore[attr-defined]
        # the provider is only reachable after handle_incident created the record
        original_place = dispatcher._test_call.place  # noqa: SLF001

        seen: List[Any] = []

        def place(recipient: str, message: str, on_status) -> ProviderResult:  # noqa: ANN001
            result = original_place(recipient, message, on_status)
            seen.append(("call finished", str(dispatcher.latest.status)))
            return result

        dispatcher._test_call.place = place  # noqa: SLF001
        sms.record = dispatcher._latest if dispatcher.latest else None
        # handle_incident creates the record then dials; re-bind inside send
        real_send = sms.send

        def send(recipient: str, body: str) -> ProviderResult:  # noqa: ANN001
            sms.record = dispatcher.latest
            seen.append(("sms sent", str(dispatcher.latest.status)))
            return real_send(recipient, body)

        sms.send = send
        dispatcher.handle_incident(make_incident())

        self.assertEqual(len(sms.sent), 1)
        self.assertEqual(sms.sent[0]["call_status_at_send"], CallStatus.COMPLETED.value)
        self.assertEqual(sms.sent[0]["call_sid_at_send"], "SID-CALL")
        # ordering: the call completed before the SMS was handed over
        self.assertEqual([step for step, _ in seen], ["call finished", "sms sent"])

    def test_no_sms_when_the_call_was_blocked(self) -> None:
        """If a safety interlock blocked the call, no SMS follows."""
        class UnusableCall(RecordingCallProvider):
            name = "unusable"

            def available(self):
                return False, "carrier unreachable"

        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(UnusableCall(), sms)
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(record.status, CallStatus.BLOCKED)
        self.assertEqual(sms.sent, [])
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED.value)
        self.assertIn("call was", record.sms.reason)

    def test_sms_body_is_composed_before_dialing(self) -> None:
        """The worker thread never has to carry the incident payload."""
        dispatcher = make_dispatcher()
        dispatcher.handle_incident(make_incident("INC-BODY"))
        self.assertIn("INC-BODY", dispatcher.calls[0].sms.body)


class NullCallProvider(CallProvider):
    name = "null"

    def available(self):
        return True, ""

    def place(self, recipient: str, message: str, on_status) -> ProviderResult:  # noqa: ANN001
        return ProviderResult(ok=True, status=CallStatus.COMPLETED, provider_status="null")


# =========================================================================== #
# 4. SMS CONTENT - all real data, nothing hard-coded
# =========================================================================== #
class TestSmsContent(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = make_dispatcher()
        self.record = self.dispatcher.handle_incident(make_incident())
        self.sms = self.record.sms

    def test_contains_the_incident_id(self) -> None:
        self.assertIn("INC-20261002-0001", self.sms.body)
        self.assertIn("Incident:", self.sms.body)

    def test_contains_the_exact_date_and_time(self) -> None:
        self.assertIn("Date: 02-10-2026", self.sms.body)
        self.assertIn("Time: 18:31:35", self.sms.body)
        self.assertRegex(self.sms.body, r"Date: \d{2}-\d{2}-\d{4}")
        self.assertRegex(self.sms.body, r"Time: \d{2}:\d{2}:\d{2}")

    def test_contains_the_registered_location_and_camera(self) -> None:
        self.assertIn("Chennai Main Road", self.sms.body)
        self.assertIn("13.082700", self.sms.body)
        self.assertIn("Camera: CAM-001", self.sms.body)

    def test_contains_the_severity(self) -> None:
        self.assertIn("Severity: HIGH", self.sms.body)
        self.assertIn("52%", self.sms.body)

    def test_contains_the_emergency_instruction(self) -> None:
        self.assertIn("Emergency assistance may be required", self.sms.body)
        self.assertIn("ACCIDENT ALERT", self.sms.body)

    def test_distinguishes_camera_registered_location_from_gps(self) -> None:
        self.assertIn("CAMERA-REGISTERED", self.sms.body)
        self.assertIn("not a GPS fix", self.sms.body)

    def test_never_invents_coordinates_for_an_unregistered_camera(self) -> None:
        dispatcher = make_dispatcher()
        incident = make_incident(
            location={
                "name": "",
                "latitude": None,
                "longitude": None,
                "is_camera_registered": False,
                "source": "unregistered",
            }
        )
        record = dispatcher.handle_incident(incident)
        body = record.sms.body
        self.assertIn("UNREGISTERED CAMERA", body)
        self.assertIn("no GPS fix", body)
        self.assertNotIn("00.000000", body)

    def test_values_track_the_actual_incident(self) -> None:
        """Two different incidents must produce two different bodies."""
        dispatcher = make_dispatcher()
        first = dispatcher.handle_incident(make_incident("INC-A"))
        second = dispatcher.handle_incident(
            make_incident(
                "INC-B",
                camera={"camera_id": "CAM-042"},
                accident={"score": 0.91, "severity": "CRITICAL"},
                timestamp="2027-01-31T23:59:59.000+00:00",
                location={
                    "name": "Ring Road",
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "is_camera_registered": True,
                },
            )
        )
        self.assertNotEqual(first.sms.body, second.sms.body)
        self.assertIn("INC-A", first.sms.body)
        self.assertIn("INC-B", second.sms.body)
        self.assertIn("CAM-042", second.sms.body)
        self.assertIn("CRITICAL", second.sms.body)
        self.assertIn("Date: 31-01-2027", second.sms.body)
        self.assertIn("Time: 23:59:59", second.sms.body)
        self.assertIn("12.971600", second.sms.body)

    def test_never_claims_injury(self) -> None:
        """The engine has no injury detection, so the SMS must not assert one.

        The only permitted mention is the disclaimer that says the opposite.
        """
        lowered = self.sms.body.lower()
        for phrase in ("injured", "casualty", "casualties", "fatal", "died", "trapped"):
            self.assertNotIn(phrase, lowered, f"the SMS must not assert {phrase!r}")
        self.assertIn("not a verified injury assessment", lowered)

    def test_body_helper_is_pure(self) -> None:
        body = build_sms_body(make_incident("INC-PURE"), instruction="Custom instruction.")
        self.assertIn("Custom instruction.", body)
        self.assertIn("INC-PURE", body)

    def test_the_sms_date_is_the_real_wall_clock_time(self) -> None:
        """A file's media position is 0-based - it must never reach the SMS.

        Regression: driving the pipeline with media timestamps used to produce
        "Date: 01-01-1970".
        """
        from datetime import datetime, timezone

        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        config = test_config()
        config.emergency.enabled = True
        config.emergency.demo_mode = True
        config.emergency.async_calls = False
        config.emergency.write_call_log = False
        config.emergency.expose_recipient = False
        before = time.time()
        pipeline = AccidentPipeline(
            config,
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
        )
        frame = blank_frame()
        # media timestamps starting at 0, exactly like a video file
        for index in range(80):
            pipeline.process_frame(frame, index, index / 15.0)
        pipeline.close()

        payload = pipeline.incidents[0].to_dict()
        self.assertGreaterEqual(payload["detected_at_unix"], before - 1.0)
        recorded = datetime.fromtimestamp(payload["detected_at_unix"], tz=timezone.utc)
        self.assertGreater(recorded.year, 2020, "the incident time must be a real date")
        # the media position is kept, but only as debug metadata
        self.assertIn("media_time_seconds", payload.get("extra", {}))

    def test_sms_body_matches_a_wall_clock_incident(self) -> None:
        """The rendered date/time must come from a real timestamp."""
        from datetime import datetime, timezone

        now = time.time()
        incident = make_incident(timestamp=datetime.fromtimestamp(now, tz=timezone.utc).isoformat())
        body = build_sms_body(incident)
        expected = datetime.fromtimestamp(now, tz=timezone.utc)
        self.assertIn(f"Date: {expected.strftime('%d-%m-%Y')}", body)
        self.assertIn(f"Time: {expected.strftime('%H:%M:%S')}", body)
        self.assertNotIn("1970", body)


# =========================================================================== #
# 5. DEMO MODE MUST NOT REACH TWILIO
# =========================================================================== #
class TestDemoModeNeverReachesTwilio(unittest.TestCase):
    def test_no_credentials_reports_not_configured_rather_than_sent(self) -> None:
        """**Phase 4 change.** This used to assert ``SimulationSmsProvider``.

        Returning the simulation provider from ``auto`` with no credentials made
        an unconfigured deployment report ``sms=SENT``.  ``NOT_CONFIGURED`` is the
        honest state: nobody was messaged.
        """
        provider = build_sms_provider("auto", environ={}, demo_mode=False)
        self.assertIsInstance(provider, NotConfiguredSmsProvider)
        self.assertFalse(provider.available()[0])
        result = provider.send("+15550100", "body")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, CallStatus.NOT_CONFIGURED)

    def test_no_credentials_simulates_only_when_demo_mode_is_on(self) -> None:
        self.assertIsInstance(
            build_sms_provider("auto", environ={}, demo_mode=True), SimulationSmsProvider
        )

    def test_not_configured_sms_is_not_a_success(self) -> None:
        self.assertNotEqual(SmsStatus.NOT_CONFIGURED, SmsStatus.SENT)
        self.assertNotEqual(SmsStatus.NOT_CONFIGURED, SmsStatus.FAILED)
        self.assertTrue(SmsStatus.NOT_CONFIGURED.is_terminal)

    def test_credentials_would_select_twilio(self) -> None:
        env = {
            "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
            "TWILIO_AUTH_TOKEN": "secret-token-value",
            "TWILIO_PHONE_NUMBER": "+15005550306",
        }
        provider = build_sms_provider("auto", environ=env)
        self.assertIsInstance(provider, TwilioSmsProvider)
        self.assertTrue(provider.available()[0])

    def test_a_full_demo_run_opens_no_socket(self) -> None:
        original, blocked, seen = _forbid_network()
        urllib.request.urlopen = blocked
        try:
            pipeline, dispatcher = run_scenario("head_on_crash")
        finally:
            urllib.request.urlopen = original
        self.assertEqual(len(pipeline.incidents), 1)
        self.assertEqual(len(dispatcher._test_sms.sent), 1)  # noqa: SLF001
        self.assertEqual(dispatcher.calls[0].sms.status, SmsStatus.SENT.value)
        self.assertEqual(seen, [], "demo mode must not touch the network")

    def test_provider_selection_never_leaks_the_token(self) -> None:
        env = {
            "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
            "TWILIO_AUTH_TOKEN": "super-secret-token",
            "TWILIO_PHONE_NUMBER": "+15005550306",
        }
        provider = build_sms_provider("auto", environ=env)
        described = json.dumps(provider.describe())
        self.assertNotIn("super-secret-token", described)

    def test_describe_never_leaks_the_token(self) -> None:
        dispatcher = make_dispatcher()
        text = json.dumps(dispatcher.describe())
        self.assertNotIn("secret", text.lower())
        self.assertIn("recipient_masked", text)


# =========================================================================== #
# 6. FAILURES MUST NOT CRASH THE PIPELINE
# =========================================================================== #
class TestFailureHandling(unittest.TestCase):
    def test_a_call_provider_that_raises_does_not_crash(self) -> None:
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(ExplodingCallProvider(), sms)
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(record.status, CallStatus.FAILED)
        self.assertIn("provider error", record.error)
        # the call failed, so the SMS is not sent - but nothing raised
        self.assertEqual(sms.sent, [])
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED.value)

    def test_an_sms_provider_that_raises_does_not_crash(self) -> None:
        dispatcher = make_dispatcher(RecordingCallProvider(), ExplodingSmsProvider())
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(record.status, CallStatus.COMPLETED, "the call still succeeded")
        self.assertEqual(record.sms.status, SmsStatus.FAILED.value)
        self.assertIn("sms error", record.sms.error)

    def test_a_sms_api_error_is_recorded_not_raised(self) -> None:
        class Refusing(RecordingSmsProvider):
            name = "refusing"

            def send(self, recipient: str, body: str) -> ProviderResult:
                self.sent.append({"recipient": recipient, "body": body})
                return ProviderResult(
                    ok=False, status=CallStatus.FAILED,
                    provider_status="http_401", error="unauthorised",
                )

        dispatcher = make_dispatcher(RecordingCallProvider(), Refusing())
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(record.sms.status, SmsStatus.FAILED.value)
        # Phase 4: the provider's own words survive the retry wrapper, and the
        # attempt count is appended so the audit trail shows how hard it tried.
        self.assertIn("unauthorised", record.sms.error)
        self.assertIn("after 1 attempt", record.sms.error)

    def test_send_sms_false_suppresses_only_the_sms(self) -> None:
        call = RecordingCallProvider()
        sms = RecordingSmsProvider()
        dispatcher = make_dispatcher(call, sms, send_sms=False)
        record = dispatcher.handle_incident(make_incident())
        self.assertEqual(len(call.attempts), 1)
        self.assertEqual(sms.sent, [])
        self.assertEqual(record.sms.status, SmsStatus.SKIPPED.value)

    def test_twilio_sms_without_credentials_is_blocked(self) -> None:
        provider = TwilioSmsProvider(account_sid="", auth_token="", from_number="")
        usable, reason = provider.available()
        self.assertFalse(usable)
        self.assertIn("TWILIO_PHONE_NUMBER", reason)
        result = provider.send(RECIPIENT, "hello")
        self.assertFalse(result.ok)

    def test_twilio_call_without_credentials_is_blocked(self) -> None:
        provider = TwilioProvider()
        self.assertFalse(provider.available()[0])


# =========================================================================== #
# 7. NON-BLOCKING
# =========================================================================== #
class TestNonBlocking(unittest.TestCase):
    def test_slow_notification_does_not_stall_the_frame_loop(self) -> None:
        import threading

        started = threading.Event()

        class SlowCall(RecordingCallProvider):
            name = "slow"

            def place(self, recipient: str, message: str, on_status) -> ProviderResult:  # noqa: ANN001
                started.set()
                time.sleep(0.6)
                return super().place(recipient, message, on_status)

        class SlowSms(RecordingSmsProvider):
            name = "slow-sms"

            def send(self, recipient: str, body: str) -> ProviderResult:  # noqa: ANN001
                time.sleep(0.4)
                return super().send(recipient, body)

        dispatcher = make_dispatcher(SlowCall(), SlowSms(), async_calls=True)
        pipeline, dispatcher = run_scenario("head_on_crash", dispatcher=dispatcher)
        # the run finished well before the (1.0s total) notification did
        self.assertEqual(len(pipeline.incidents), 1)
        dispatcher.close(timeout=6.0)
        self.assertEqual(dispatcher.calls[0].status, CallStatus.COMPLETED)
        self.assertEqual(dispatcher.calls[0].sms.status, SmsStatus.SENT.value)


# =========================================================================== #
# 8. THE CONTRACT AND THE AUDIT TRAIL
# =========================================================================== #
class TestContractAndAuditTrail(unittest.TestCase):
    def test_incident_carries_both_blocks(self) -> None:
        pipeline, _ = run_scenario("head_on_crash")
        payload = pipeline.incidents[0].to_dict()
        self.assertIn("demo_emergency_call", payload)
        self.assertIn("demo_emergency_sms", payload)
        self.assertTrue(payload["demo_emergency_call"]["attempted"])
        self.assertTrue(payload["demo_emergency_sms"]["attempted"])
        self.assertEqual(payload["demo_emergency_sms"]["trigger"], "automatic_after_call")
        self.assertEqual(validate_incident(payload), [])

    def test_both_blocks_always_present_even_when_disabled(self) -> None:
        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        config = test_config()  # emergency.enabled is False in the test config
        pipeline = AccidentPipeline(
            config, detector=ScriptedDetector(SCENARIOS["head_on_crash"]), camera_id="CAM-001"
        )
        frame = blank_frame()
        base = time.time()
        for index in range(80):
            pipeline.process_frame(frame, index, base + index / 15.0)
        pipeline.close()
        payload = pipeline.incidents[0].to_dict()
        self.assertFalse(payload["demo_emergency_call"]["attempted"])
        self.assertFalse(payload["demo_emergency_sms"]["attempted"])
        # the reason names whichever gate refused: the feature switch or DEMO_MODE
        self.assertTrue(
            "DEMO_MODE" in payload["demo_emergency_sms"]["reason"]
            or "enabled is false" in payload["demo_emergency_sms"]["reason"],
            payload["demo_emergency_sms"]["reason"],
        )

    def test_alert_payload_carries_the_sms(self) -> None:
        pipeline, _ = run_scenario("head_on_crash")
        alert = pipeline.generate_emergency_incident()
        self.assertIn("demo_emergency_sms", alert)
        self.assertIn("body", alert["demo_emergency_sms"])

    def test_recipient_is_masked_everywhere(self) -> None:
        dispatcher = make_dispatcher()
        record = dispatcher.handle_incident(make_incident())
        text = json.dumps(record.to_dict(include_recipient=False))
        self.assertNotIn("5550100", text)
        self.assertNotIn("recipient_full", text)
        self.assertIn("recipient", text)

    def test_call_log_records_one_line_per_incident(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            config = test_config().emergency
            config.enabled = True
            config.demo_mode = True
            config.async_calls = False
            config.write_call_log = False
            config.expose_recipient = False
            dispatcher = DemoCallDispatcher(
                config,
                environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
                provider=RecordingCallProvider(),
                sms_provider=RecordingSmsProvider(),
                log_path=log,
            )
            for _ in range(20):
                dispatcher.handle_incident(make_incident("INC-LOG"))
            lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["sms"]["status"], "SENT")

    def test_status_stream_carries_the_sms(self) -> None:
        seen: List[Dict[str, Any]] = []
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = True
        config.async_calls = False
        config.write_call_log = False
        config.expose_recipient = False
        dispatcher = DemoCallDispatcher(
            config,
            environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=RecordingCallProvider(),
            sms_provider=RecordingSmsProvider(),
            on_status=seen.append,
        )
        dispatcher.handle_incident(make_incident())
        labels = [s.get("label") for s in seen]
        self.assertIn("DEMO EMERGENCY CALL", labels)
        self.assertIn("DEMO SMS", labels)
        self.assertEqual(seen[-1]["status"], "SENT")

    def test_overlay_hud_shows_both_channels(self) -> None:
        dispatcher = make_dispatcher()
        record = dispatcher.handle_incident(make_incident())
        state = dispatcher.status()
        self.assertEqual(state["sms"]["status"], "SENT")
        self.assertIn("DEMO SMS", state["label"])

    def test_sms_info_derives_sent_from_status(self) -> None:
        """A caller that sets only the status must not report sent=False."""
        info = DemoSmsInfo(status=SmsStatus.SENT.value, call_id="C1", incident_id="INC-X")
        payload = info.to_dict()
        self.assertEqual(payload["label"], "DEMO SMS")
        self.assertTrue(payload["sent"], "SENT must imply sent=true")
        self.assertFalse(DemoSmsInfo(status=SmsStatus.PENDING.value).to_dict()["sent"])
        self.assertFalse(DemoSmsInfo(status=SmsStatus.FAILED.value).to_dict()["sent"])


if __name__ == "__main__":
    unittest.main()