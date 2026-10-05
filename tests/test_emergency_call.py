"""The demonstration emergency call.

These tests use a **recording fake provider** and the simulation provider.
No test contacts a telephony API and no test can reach the real recipient, so
the suite is hermetic and safe to run anywhere.

What is locked in here
----------------------
* ``.env`` is the only place the recipient is stored, and it is git-ignored.
* No call happens unless ``DEMO_MODE=true``.
* No call happens unless the incident was CONFIRMED.
* **Exactly one call per confirmed incident** - not one per frame.
* There is no button: nothing in the package accepts a manual trigger.
* The number is masked everywhere except the incident JSON when opted in.
* The call status reaches the incident contract and the debug overlay.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.emergency.call_record import DemoCallRecord  # noqa: E402
from ai_engine.emergency.dispatcher import DEMO_CALL_DISCLAIMER, DemoCallDispatcher  # noqa: E402
from ai_engine.emergency.envfile import env_flag, load_env_file, parse_env_text  # noqa: E402
from ai_engine.emergency.providers import (  # noqa: E402
    CallProvider,
    NullProvider,
    NotConfiguredCallProvider,
    ProviderResult,
    SimulationProvider,
    TwilioProvider,
    build_provider,
)
from ai_engine.emergency.recipient import (  # noqa: E402
    RecipientError,
    is_placeholder,
    mask_phone,
    normalize_phone,
)
from ai_engine.schemas.enums import CallStatus, IncidentStatus  # noqa: E402
from ai_engine.schemas.incident_result import DemoCallInfo, IncidentResult, validate_incident  # noqa: E402
from tests.helpers import blank_frame, test_config  # noqa: E402

RECIPIENT = "+15550100"  # fictional 555 number: tests never use a real number
MASKED = mask_phone(RECIPIENT)


# --------------------------------------------------------------------------- #
# a provider that records instead of dialling
# --------------------------------------------------------------------------- #
class RecordingProvider(CallProvider):
    """Counts every call attempt and reports a fixed status sequence."""

    name = "recording"

    def __init__(self, final: CallStatus = CallStatus.COMPLETED, usable: bool = True) -> None:
        self.attempts: List[Dict[str, Any]] = []
        self.final = final
        self._usable = usable

    def available(self) -> Any:
        return (self._usable, "" if self._usable else "provider disabled for this test")

    def place(self, recipient: str, message: str, on_status: Any) -> ProviderResult:
        self.attempts.append({"recipient": recipient, "message": message})
        on_status(CallStatus.RINGING, "ringing", "FAKE-SID-1", "")
        on_status(self.final, str(self.final).lower(), "FAKE-SID-1", "")
        return ProviderResult(
            ok=self.final is CallStatus.COMPLETED,
            status=self.final,
            provider_status=str(self.final).lower(),
            provider_call_sid="FAKE-SID-1",
        )


def make_dispatcher(**kwargs: Any) -> DemoCallDispatcher:
    """A dispatcher wired to the recording provider and a private environment."""
    provider = kwargs.pop("_provider", None) or RecordingProvider()
    config = test_config().emergency
    config.enabled = True
    config.demo_mode = True
    config.write_call_log = False
    config.async_calls = False
    for key, value in kwargs.items():
        setattr(config, key, value)
    env = {"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT}
    return DemoCallDispatcher(config, environ=env, provider=provider)


def incident(incident_id: str = "INC-TEST-0001", status: str = IncidentStatus.CONFIRMED_ACCIDENT.value) -> Dict[str, Any]:
    return {
        "incident_id": incident_id,
        "status": status,
        "camera": {"camera_id": "CAM-001"},
        "location": {"name": "MG Road, Bengaluru"},
        "accident": {"score": 0.52, "severity": "MEDIUM"},
    }


# --------------------------------------------------------------------------- #
class TestDotEnvParsing(unittest.TestCase):
    def test_parses_keys_values_comments_and_export(self) -> None:
        text = "\n".join(
            [
                "# a comment",
                "",
                "DEMO_MODE=true",
                "export EMERGENCY_CONTACT_NUMBER=+15550100",
                "QUOTED='single'",
                'DOUBLE="double"',
                "not a pair",
            ]
        )
        parsed = parse_env_text(text)
        self.assertEqual(parsed["DEMO_MODE"], "true")
        self.assertEqual(parsed["EMERGENCY_CONTACT_NUMBER"], "+15550100")
        self.assertEqual(parsed["QUOTED"], "single")
        self.assertEqual(parsed["DOUBLE"], "double")
        self.assertNotIn("not a pair", parsed)

    def test_existing_environment_wins_over_the_file(self) -> None:
        """A value exported in the shell must not be overwritten by `.env`."""
        env = {"DEMO_MODE": "false"}
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("DEMO_MODE=true\nEMERGENCY_CONTACT_NUMBER=+15550100\n", encoding="utf-8")
            applied = load_env_file(path, environ=env)
        self.assertEqual(env["DEMO_MODE"], "false", "the real environment must win")
        self.assertEqual(applied["EMERGENCY_CONTACT_NUMBER"], RECIPIENT)

    def test_env_flag_parsing(self) -> None:
        self.assertTrue(env_flag({"X": "true"}, "X"))
        self.assertTrue(env_flag({"X": "1"}, "X"))
        self.assertFalse(env_flag({"X": "false"}, "X"))
        self.assertFalse(env_flag({}, "X"))
        self.assertTrue(env_flag({}, "X", default=True))

    def test_the_projects_env_is_not_committed(self) -> None:
        """.gitignore must cover `.env` but keep `.env.example`."""
        rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", rules)
        self.assertIn("!*.env.example", rules)

    def test_env_example_uses_a_placeholder_not_a_real_number(self) -> None:
        """The committed template must never carry a callable number."""
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        parsed = parse_env_text(text)
        self.assertIn("EMERGENCY_CONTACT_NUMBER", parsed)
        self.assertTrue(is_placeholder(parsed["EMERGENCY_CONTACT_NUMBER"]))
        self.assertNotIn(RECIPIENT, text)

    def test_the_number_lives_only_in_the_env_file(self) -> None:
        """No source, config, doc or committed sample may contain a real number.

        This is the requirement that keeps the recipient out of version control
        and out of a screenshot of the code: only ``.env`` (git-ignored) holds it.

        The one carve-out is the North American ``555`` range, which is reserved
        for fiction: both the ``+1 555-0100``-``0199`` block and Twilio's
        documented non-ringing test numbers (``+1 500 555 xxxx``).  Tests and
        samples may use those, because they can never reach a phone.
        """
        import re

        real_number = re.compile(r"\+\d{8,}")

        def is_fictional(number: str) -> bool:
            digits = number[1:]
            return digits.startswith("1555") or digits.startswith("1500555")

        # Generated run artefacts (incidents.json, demo_report.json, ...) are
        # allowed to carry the number only because they are git-ignored - a
        # local file, never a committed one.
        ignored = {
            line.strip().rstrip("/")
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("!")
        }

        scanned = 0
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".venv" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            if path.suffix not in (".py", ".md", ".yaml", ".yml", ".json", ".example", ".txt"):
                continue
            if path.name == ".env":
                continue
            # any `.env*` variant is tolerated *only* because git ignores it
            if path.name.startswith(".env") or path.name.endswith(".env"):
                if path.name not in (".env.example",) and _is_env_pattern_ignored(path.name):
                    continue
            if path.name in ignored or any(part in ignored for part in relative.parts[:-1]):
                # tolerated *only* because it is ignored by git
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for found in real_number.finditer(text):
                if not is_fictional(found.group()):
                    self.fail(
                        f"{relative} contains a real phone number "
                        f"({found.group()}); it belongs in .env only"
                    )
        self.assertGreater(scanned, 20, "the scan should cover the whole project")

    def test_the_generated_artefacts_that_carry_a_number_are_git_ignored(self) -> None:
        """The files that may contain the number must all be ignored by git."""
        rules = {
            line.strip().rstrip("/")
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("!")
        }
        for name in ("incidents.json", "demo_report.json", "demo_calls.jsonl", "evidence", "*.log"):
            self.assertIn(name, rules, f"{name} must be git-ignored")

    def test_gitignore_covers_the_env_file(self) -> None:
        """`.env` must be ignored in every form, and `.env.example` must not be."""
        rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        ignored = [r.strip() for r in rules if r.strip() and not r.strip().startswith("#")]
        self.assertIn(".env", ignored)
        self.assertIn(".env.*", ignored)
        self.assertIn("*.env", ignored)
        self.assertIn("*.env.*", ignored)
        self.assertIn("!*.env.example", ignored)

    def test_every_env_variant_is_ignored(self) -> None:
        """A stray ``.env.txt`` must never be committable.

        Regression: a copy of `.env` landed in the project root as `.env.txt`,
        which the old `.env` / `*.env` patterns did not match - and it held the
        real recipient number.
        """
        import fnmatch

        patterns = [
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        def is_ignored(name: str) -> bool:
            ignored = False
            for pattern in patterns:
                if pattern.startswith("!"):
                    if fnmatch.fnmatch(name, pattern[1:]):
                        ignored = False
                elif fnmatch.fnmatch(name, pattern):
                    ignored = True
            return ignored

        for name in (
            ".env",
            ".env.txt",
            ".env.bak",
            ".env.local",
            ".env.production",
            "prod.env",
            "secrets.env",
            ".envrc",
        ):
            self.assertTrue(is_ignored(name), f"{name} must be git-ignored")

        # the committed template must stay visible
        self.assertFalse(is_ignored(".env.example"), ".env.example must be committable")


# --------------------------------------------------------------------------- #
class TestRecipientHandling(unittest.TestCase):
    def test_normalizes_pretty_forms(self) -> None:
        for raw in (RECIPIENT, "+1 555 0100", "+1-555-0100", "001 555 0100"):
            self.assertEqual(normalize_phone(raw), RECIPIENT)

    def test_rejects_placeholder_and_national_format(self) -> None:
        with self.assertRaises(RecipientError):
            normalize_phone("+91XXXXXXXXXX")
        with self.assertRaises(RecipientError):
            normalize_phone("5550100")
        with self.assertRaises(RecipientError):
            normalize_phone("")

    def test_masking_hides_the_middle(self) -> None:
        masked = mask_phone(RECIPIENT)
        self.assertNotIn("555", masked)
        self.assertTrue(masked.startswith("+15"))
        self.assertTrue(masked.endswith("0100"))
        self.assertIn("*", masked)

    def test_unparsable_value_is_never_leaked(self) -> None:
        self.assertEqual(mask_phone("not-a-number"), "(unconfigured)")
        self.assertEqual(mask_phone(None), "(none)")


# --------------------------------------------------------------------------- #
class TestProviderSelection(unittest.TestCase):
    def test_auto_without_credentials_cannot_place_a_real_call(self) -> None:
        """The core safety property of the whole feature.

        **Phase 4 change.** This used to assert that ``auto`` resolves to
        ``SimulationProvider`` when no credentials exist.  That was the
        highest-severity finding of the Phase 1 audit: with zero credentials the
        engine reported ``call=COMPLETED, sms=SENT`` and a dashboard could not
        tell that from a real notification.  With ``demo_mode=False`` ``auto``
        now resolves to a provider that reports ``NOT_CONFIGURED`` and notifies
        nobody.
        """
        provider = build_provider("auto", environ={}, demo_mode=False)
        self.assertIsInstance(provider, NotConfiguredCallProvider)
        self.assertFalse(provider.available()[0])

        # and it refuses to dial rather than faking success
        result = provider.place("+15550100", "hello", lambda *a, **k: None)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, CallStatus.NOT_CONFIGURED)
        self.assertIn("NOT_CONFIGURED", result.error)

    def test_auto_without_credentials_simulates_only_when_asked(self) -> None:
        """Simulation is the right answer to an *explicit* request to simulate."""
        provider = build_provider("auto", environ={}, demo_mode=True)
        self.assertIsInstance(provider, SimulationProvider)

    def test_not_configured_is_distinct_from_failed_and_completed(self) -> None:
        """Three different facts must not share one state."""
        self.assertNotEqual(CallStatus.NOT_CONFIGURED, CallStatus.FAILED)
        self.assertNotEqual(CallStatus.NOT_CONFIGURED, CallStatus.COMPLETED)
        self.assertTrue(CallStatus.NOT_CONFIGURED.is_terminal)
        self.assertFalse(CallStatus.NOT_CONFIGURED.was_placed,
                         "no telephony was placed, so was_placed must be False")
        self.assertFalse(CallStatus.NOT_CONFIGURED.reached_recipient)

    def test_explicit_twilio_never_falls_back(self) -> None:
        """An explicit choice must never be silently downgraded."""
        provider = build_provider("twilio", environ={}, demo_mode=True)
        self.assertIsInstance(provider, TwilioProvider)

    def test_auto_uses_twilio_when_credentials_are_present(self) -> None:
        env = {
            "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_FROM_NUMBER": "+15550000000",
        }
        provider = build_provider("auto", environ=env)
        self.assertIsInstance(provider, TwilioProvider)
        self.assertTrue(provider.available()[0])

    def test_twilio_reports_missing_credentials(self) -> None:
        provider = TwilioProvider(account_sid="", auth_token="", from_number="")
        usable, reason = provider.available()
        self.assertFalse(usable)
        self.assertIn("TWILIO_ACCOUNT_SID", reason)

    def test_none_provider_refuses(self) -> None:
        provider = NullProvider()
        self.assertFalse(provider.available()[0])

    def test_simulation_walks_the_full_lifecycle(self) -> None:
        seen: List[CallStatus] = []
        result = SimulationProvider(ring_after_seconds=0.0, duration_seconds=0.0).place(
            RECIPIENT, "hello", lambda status, *_: seen.append(status)
        )
        self.assertIn(CallStatus.RINGING, seen)
        self.assertIn(CallStatus.IN_PROGRESS, seen)
        self.assertIn(CallStatus.COMPLETED, seen)
        self.assertTrue(result.ok)


# --------------------------------------------------------------------------- #
class TestCallGates(unittest.TestCase):
    def test_no_call_when_demo_mode_is_off_and_real_sending_is_disabled(self) -> None:
        """Both switches off = nothing is dialled.

        **Updated for the real-send workflow.** `DEMO_MODE=false` alone now means
        "a real Twilio call is authorised"; the switch that stops it is
        `emergency.allow_real_call`. This test pins the fully-disabled path.
        """
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider, allow_real_call=False)
        dispatcher.environ["DEMO_MODE"] = "false"
        record = dispatcher.handle_incident(incident())
        self.assertEqual(provider.attempts, [])
        self.assertIsNotNone(record)
        self.assertEqual(record.status, CallStatus.SKIPPED)
        self.assertIn("allow_real_call", record.reason)

    def test_demo_mode_off_with_real_sending_enabled_actually_places_the_call(self) -> None:
        """`DEMO_MODE=false` is the REAL MODE the operator asked for.

        With `allow_real_call=true` and a usable provider, a confirmed incident
        must dial - this is the path that used to be unreachable.
        """
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider, allow_real_call=True)
        dispatcher.environ["DEMO_MODE"] = "false"
        record = dispatcher.handle_incident(incident())
        self.assertEqual(len(provider.attempts), 1)
        self.assertEqual(record.status, CallStatus.COMPLETED)

    def test_no_call_when_the_feature_is_disabled(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        dispatcher.config.enabled = False
        dispatcher.handle_incident(incident())
        self.assertEqual(provider.attempts, [])

    def test_no_call_for_an_unconfirmed_incident(self) -> None:
        """SUSPICIOUS / POSSIBLE must never trigger a call."""
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        for status in (IncidentStatus.NORMAL.value, IncidentStatus.POSSIBLE_ACCIDENT.value):
            dispatcher.handle_incident(incident("INC-X", status))
        self.assertEqual(provider.attempts, [])

    def test_no_call_without_a_recipient(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        dispatcher.environ.pop("EMERGENCY_CONTACT_NUMBER")
        record = dispatcher.handle_incident(incident())
        self.assertEqual(provider.attempts, [])
        self.assertEqual(record.status, CallStatus.BLOCKED)

    def test_no_call_when_the_recipient_is_still_a_placeholder(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        dispatcher.environ["EMERGENCY_CONTACT_NUMBER"] = "+91XXXXXXXXXX"
        record = dispatcher.handle_incident(incident())
        self.assertEqual(provider.attempts, [])
        self.assertEqual(record.status, CallStatus.BLOCKED)

    def test_no_call_when_the_provider_is_unusable(self) -> None:
        provider = RecordingProvider(usable=False)
        dispatcher = make_dispatcher(_provider=provider)
        record = dispatcher.handle_incident(incident())
        self.assertEqual(provider.attempts, [])
        self.assertEqual(record.status, CallStatus.BLOCKED)


# --------------------------------------------------------------------------- #
class TestExactlyOneCallPerIncident(unittest.TestCase):
    def test_one_call_per_confirmed_incident(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        dispatcher.handle_incident(incident("INC-A"))
        self.assertEqual(len(provider.attempts), 1)
        record = dispatcher.handle_incident(incident("INC-A"))
        self.assertEqual(len(provider.attempts), 1, "the same incident must not be dialled twice")
        self.assertEqual(record.deliveries, 2)

    def test_repeated_per_frame_deliveries_still_produce_one_call(self) -> None:
        """The requirement is 'not for every detected frame'."""
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        for _ in range(500):
            dispatcher.handle_incident(incident("INC-FRAMES"))
        self.assertEqual(len(provider.attempts), 1)
        self.assertEqual(dispatcher.call_count, 1)

    def test_distinct_incidents_each_get_one_call(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        for index in range(3):
            dispatcher.handle_incident(incident(f"INC-{index}"))
        self.assertEqual(len(provider.attempts), 3)
        self.assertEqual(dispatcher.call_count, 3)

    def test_per_run_limit_is_honoured(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(max_calls_per_run=1, _provider=provider)
        dispatcher.handle_incident(incident("INC-1"))
        dispatcher.handle_incident(incident("INC-2"))
        self.assertEqual(len(provider.attempts), 1)

    def test_per_run_limit_holds_with_async_dialling(self) -> None:
        """A call still in flight must count towards the limit."""
        provider = RecordingProvider()
        dispatcher = make_dispatcher(max_calls_per_run=1, async_calls=True, _provider=provider)
        dispatcher.handle_incident(incident("INC-1"))
        # the first call is on a worker thread and has not reached a terminal
        # state yet, but the limit must still refuse the second one
        second = dispatcher.handle_incident(incident("INC-2"))
        self.assertIsNotNone(second)
        self.assertEqual(second.status, CallStatus.SKIPPED)
        dispatcher.close(timeout=5.0)
        self.assertEqual(len(provider.attempts), 1)

    def test_min_gap_seconds_refuses_a_second_call(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(min_gap_seconds=30.0, _provider=provider)
        dispatcher.handle_incident(incident("INC-1"))
        second = dispatcher.handle_incident(incident("INC-2"))
        self.assertEqual(second.status, CallStatus.SKIPPED)
        self.assertIn("min_gap_seconds", second.reason)
        self.assertEqual(len(provider.attempts), 1)


# --------------------------------------------------------------------------- #
class TestNoManualTrigger(unittest.TestCase):
    def test_the_dispatcher_exposes_no_manual_call_method(self) -> None:
        """No 'call now' button exists anywhere in the API."""
        forbidden = (
            "call_now",
            "trigger_call",
            "place_call",
            "manual_call",
            "send_call",
            "dial_now",
        )
        public = [name for name in dir(DemoCallDispatcher) if not name.startswith("_")]
        for name in forbidden:
            self.assertNotIn(name, public, f"a manual trigger '{name}' must not exist")

    def test_the_trigger_is_always_automatic(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        record = dispatcher.handle_incident(incident())
        self.assertEqual(record.trigger, "automatic_on_confirmation")

    def test_the_voice_message_says_it_is_a_demonstration(self) -> None:
        """What the recipient hears must announce itself as a demo, once."""
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        dispatcher.handle_incident(incident())
        message = provider.attempts[0]["message"]
        self.assertEqual(message.count("SafeVision demonstration alert."), 1, message)
        self.assertIn("demonstration call", message.lower())
        self.assertIn("52 percent", message)
        self.assertIn("MEDIUM", message)

    def test_voice_message_never_claims_injury(self) -> None:
        """The engine has no injury detection, so the message must not imply it."""
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        dispatcher.handle_incident(incident())
        message = provider.attempts[0]["message"].lower()
        for word in ("injur", "casualt", "ambulance", "hospital", "dead"):
            self.assertNotIn(word, message, f"the spoken message must not say {word!r}")


# --------------------------------------------------------------------------- #
class TestCallStatusVisibility(unittest.TestCase):
    def test_status_feeds_the_overlay_payload(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        idle = dispatcher.status()
        self.assertEqual(idle["label"], "DEMO EMERGENCY CALL")
        self.assertTrue(idle["idle"])
        self.assertEqual(idle["recipient"], MASKED)

        dispatcher.handle_incident(incident())
        active = dispatcher.status()
        self.assertEqual(active["status"], CallStatus.COMPLETED.value)
        self.assertEqual(active["incident_id"], "INC-TEST-0001")

    def test_status_callback_streams_transitions(self) -> None:
        seen: List[Dict[str, Any]] = []
        provider = RecordingProvider()
        config = test_config().emergency
        config.enabled = True
        config.demo_mode = True
        config.write_call_log = False
        config.async_calls = False
        dispatcher = DemoCallDispatcher(
            config,
            environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=provider,
            on_status=seen.append,
        )
        dispatcher.handle_incident(incident())
        self.assertTrue(seen)
        self.assertTrue(any(s["status"] == CallStatus.RINGING.value for s in seen))
        self.assertEqual(seen[-1]["incident_id"], "INC-TEST-0001")
        # the call reaches COMPLETED, then the SMS follows it
        call_events = [s for s in seen if s.get("label") == "DEMO EMERGENCY CALL"]
        self.assertEqual(call_events[-1]["status"], CallStatus.COMPLETED.value)
        sms_events = [s for s in seen if s.get("label") == "DEMO SMS"]
        self.assertTrue(sms_events, "the SMS status must be streamed too")
        self.assertEqual(sms_events[-1]["status"], "SENT")
        self.assertLess(
            seen.index(call_events[-1]),
            seen.index(sms_events[-1]),
            "the SMS must be streamed after the call completed",
        )

    def test_the_recipient_is_masked_in_the_call_log_payload(self) -> None:
        provider = RecordingProvider()
        dispatcher = make_dispatcher(_provider=provider)
        record = dispatcher.handle_incident(incident())
        payload = record.to_dict(include_recipient=False)
        self.assertNotIn("recipient_full", payload)
        self.assertEqual(payload["recipient"], MASKED)
        self.assertNotIn("5550100", json.dumps(payload))

    def test_the_disclaimer_is_attached_everywhere(self) -> None:
        self.assertIn("not connected to any ambulance", DEMO_CALL_DISCLAIMER)
        info = DemoCallInfo.inactive().to_dict()
        self.assertEqual(info["disclaimer"], DEMO_CALL_DISCLAIMER)

    def test_call_log_is_appended_once_per_incident(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            provider = RecordingProvider()
            config = test_config().emergency
            config.enabled = True
            config.demo_mode = True
            config.write_call_log = False
            config.async_calls = False
            dispatcher = DemoCallDispatcher(
                config,
                environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
                provider=provider,
                log_path=log,
            )
            dispatcher.handle_incident(incident("INC-LOG"))
            dispatcher.handle_incident(incident("INC-LOG"))
            lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["incident_id"], "INC-LOG")
        self.assertNotIn("5550100", json.dumps(lines))


# --------------------------------------------------------------------------- #
class TestIncidentContract(unittest.TestCase):
    def test_every_incident_carries_the_demo_call_block(self) -> None:
        """Present whether or not demo mode is on, so the dashboard is uniform."""
        info = DemoCallInfo.inactive(reason="DEMO_MODE is not true").to_dict()
        self.assertFalse(info["attempted"])
        self.assertEqual(info["status"], CallStatus.SKIPPED.value)

    def test_round_trips_through_from_dict(self) -> None:
        payload = {
            "incident_id": "INC-RT",
            "status": IncidentStatus.CONFIRMED_ACCIDENT.value,
            "accident": {"score": 0.5, "severity": "MEDIUM", "frames_of_evidence": 5},
            "evidence": {"reasons": ["Sudden stop"]},
            "demo_emergency_call": DemoCallInfo(
                attempted=True,
                status=CallStatus.RINGING.value,
                call_id="CALL-1",
                provider="simulation",
            ).to_dict(),
        }
        restored = IncidentResult.from_dict(payload)
        self.assertEqual(restored.demo_emergency_call.status, CallStatus.RINGING.value)
        self.assertEqual(restored.to_dict()["demo_emergency_call"]["call_id"], "CALL-1")

    def test_a_payload_with_the_block_validates(self) -> None:
        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        config = test_config()
        provider = RecordingProvider()
        dispatcher = DemoCallDispatcher(
            config.emergency,
            environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=provider,
        )
        config.emergency.enabled = True
        config.emergency.demo_mode = True
        pipeline = AccidentPipeline(
            config,
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
            emergency=dispatcher,
        )
        frame = blank_frame()
        base = time.time()
        for index in range(80):
            pipeline.process_frame(frame, index, base + index / 15.0)
        pipeline.close()

        self.assertTrue(pipeline.incidents, "the scripted crash must confirm")
        incident_payload = pipeline.incidents[0].to_dict()
        call = incident_payload["demo_emergency_call"]
        self.assertTrue(call["attempted"])
        self.assertEqual(call["incident_id"] if "incident_id" in call else incident_payload["incident_id"],
                         incident_payload["incident_id"])
        self.assertEqual(call["trigger"], "automatic_on_confirmation")
        self.assertIn(call["status"], [s.value for s in CallStatus])
        self.assertEqual(validate_incident(incident_payload), [])


# --------------------------------------------------------------------------- #
def make_pipeline_for(config: Any, provider: CallProvider, scenario: str = "head_on_crash"):
    """A pipeline whose demo-call dispatcher uses ``provider``."""
    from ai_engine.detection.scenarios import SCENARIOS
    from ai_engine.detection.scripted import ScriptedDetector
    from ai_engine.pipeline import AccidentPipeline

    dispatcher = DemoCallDispatcher(
        config.emergency,
        environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
        provider=provider,
    )
    return AccidentPipeline(
        config,
        detector=ScriptedDetector(SCENARIOS[scenario]),
        camera_id="CAM-001",
        emergency=dispatcher,
    )


class TestPipelineIntegration(unittest.TestCase):
    def _run_scenario(self, scenario: str = "head_on_crash", **emergency_kwargs: Any):
        config = test_config()
        config.emergency.enabled = True
        config.emergency.demo_mode = True
        config.emergency.write_call_log = False
        config.emergency.async_calls = False
        for key, value in emergency_kwargs.items():
            setattr(config.emergency, key, value)

        provider = RecordingProvider()
        pipeline = make_pipeline_for(config, provider, scenario)
        frame = blank_frame()
        base = time.time()
        for index in range(90):
            pipeline.process_frame(frame, index, base + index / 15.0)
        pipeline.close()
        return pipeline, provider

    def test_a_confirmed_crash_places_exactly_one_call(self) -> None:
        pipeline, provider = self._run_scenario()
        self.assertEqual(len(pipeline.incidents), 1)
        self.assertEqual(len(provider.attempts), 1, "one call per confirmed incident")
        self.assertEqual(provider.attempts[0]["recipient"], RECIPIENT)
        # the message names the incident facts, not a medical claim
        self.assertIn("demonstration", provider.attempts[0]["message"].lower())

    def test_frames_after_the_confirmation_do_not_call_again(self) -> None:
        """The post-confirmation aftermath must stay silent."""
        pipeline, provider = self._run_scenario()
        self.assertGreater(len(pipeline.demo_calls), 0)
        self.assertEqual(len(provider.attempts), len(pipeline.incidents))

    def test_a_false_alarm_trap_places_no_call(self) -> None:
        pipeline, provider = self._run_scenario(scenario="hard_brake")
        self.assertEqual(len(provider.attempts), 0)
        self.assertEqual(pipeline.incidents, [])

    def test_quiet_traffic_places_no_call(self) -> None:
        """Calm scenarios must not reach the dispatcher at all."""
        for scenario in ("opposite_lanes", "following_close", "parked_cars"):
            with self.subTest(scenario=scenario):
                pipeline, provider = self._run_scenario(scenario=scenario)
                self.assertEqual(len(provider.attempts), 0)
                self.assertEqual(pipeline.incidents, [])

    def test_async_calls_do_not_stall_the_video_loop(self) -> None:
        """The dial happens on a worker thread; frames keep flowing."""
        import threading
        import time as _time

        class _SlowProvider(CallProvider):
            name = "slow"

            def available(self) -> Any:
                return True, ""

            def place(self, recipient: str, message: str, on_status: Any) -> ProviderResult:
                _time.sleep(0.6)
                on_status(CallStatus.COMPLETED, "completed", "SLOW", "")
                return ProviderResult(ok=True, status=CallStatus.COMPLETED, provider_call_sid="SLOW")

        config = test_config()
        config.emergency.enabled = True
        config.emergency.demo_mode = True
        config.emergency.write_call_log = False
        config.emergency.async_calls = True
        pipeline = make_pipeline_for(config, _SlowProvider())

        frame = blank_frame()
        base = _time.time()
        started = _time.perf_counter()
        for index in range(80):
            pipeline.process_frame(frame, index, base + index / 15.0)
        # the call was placed during the run, yet the loop did not wait for it
        self.assertLess(_time.perf_counter() - started, 2.0)
        dispatcher = pipeline.emergency
        self.assertIsNotNone(dispatcher)
        dispatcher.close(timeout=5.0)
        pipeline.close()
        self.assertEqual(dispatcher.call_count, 1)
        self.assertEqual(dispatcher.calls[0].status, CallStatus.COMPLETED)

    def test_close_waits_for_an_in_flight_call(self) -> None:
        import time as _time

        class _SlowProvider(CallProvider):
            name = "slow"

            def available(self) -> Any:
                return True, ""

            def place(self, recipient: str, message: str, on_status: Any) -> ProviderResult:
                _time.sleep(0.4)
                on_status(CallStatus.COMPLETED, "completed", "SLOW", "")
                return ProviderResult(ok=True, status=CallStatus.COMPLETED, provider_call_sid="SLOW")

        config = test_config()
        config.emergency.enabled = True
        config.emergency.demo_mode = True
        config.emergency.write_call_log = False
        config.emergency.async_calls = True
        pipeline = make_pipeline_for(config, _SlowProvider())
        frame = blank_frame()
        base = _time.time()
        for index in range(80):
            pipeline.process_frame(frame, index, base + index / 15.0)
        dispatcher = pipeline.emergency
        pipeline.close()  # must join the worker thread
        self.assertEqual(dispatcher.calls[0].status, CallStatus.COMPLETED)

    def test_pipeline_is_silent_when_demo_mode_is_off(self) -> None:
        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        config = test_config()
        config.emergency.enabled = True
        config.emergency.demo_mode = False
        provider = RecordingProvider()
        pipeline = AccidentPipeline(
            config,
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
            emergency=DemoCallDispatcher(
                config.emergency,
                environ={"DEMO_MODE": "false", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
                provider=provider,
            ),
        )
        frame = blank_frame()
        base = time.time()
        for index in range(80):
            pipeline.process_frame(frame, index, base + index / 15.0)
        pipeline.close()
        self.assertEqual(len(provider.attempts), 0, "demo mode off means no call")
        self.assertTrue(pipeline.incidents, "the incident itself is still reported")
        call = pipeline.incidents[0].to_dict()["demo_emergency_call"]
        self.assertFalse(call["attempted"])

    def test_frame_results_carry_the_call_status(self) -> None:
        """The overlay needs the state on every frame while a call is live."""
        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        config = test_config()
        config.visualization.enabled = False
        config.emergency.enabled = True
        config.emergency.demo_mode = True
        config.emergency.write_call_log = False
        config.emergency.async_calls = False
        dispatcher = DemoCallDispatcher(
            config.emergency,
            environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": RECIPIENT},
            provider=RecordingProvider(),
        )
        pipeline = AccidentPipeline(
            config,
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
            emergency=dispatcher,
        )
        frame = blank_frame()
        base = time.time()
        last: Any = None
        for index in range(80):
            last = pipeline.process_frame(frame, index, base + index / 15.0)
        pipeline.close()
        self.assertIn("DEMO EMERGENCY CALL", last.demo_call["label"])
        self.assertTrue(last.demo_call["demo_mode"])
        self.assertEqual(last.demo_call["status"], CallStatus.COMPLETED.value)
        self.assertEqual(last.demo_call["sms"]["status"], "SENT")
        self.assertIn("demo_emergency_call", last.to_dict())


# --------------------------------------------------------------------------- #
class TestOverlayBanner(unittest.TestCase):
    def _render(self, demo_call: Dict[str, Any]):
        from ai_engine.visualization.overlay import OverlayRenderer

        config = test_config().visualization
        config.enabled = True
        # isolate the demo-call banner from the boxes / HUD / state banner
        config.show_boxes = False
        config.show_trails = False
        config.show_collision = False
        config.show_hud = False
        config.show_perf = False
        config.state_banner = False
        renderer = OverlayRenderer(config)

        class _Stub:
            verification = None
            tracks: List[Any] = []
            collision = None
            severity = None
            accident_score = 0.0
            temporal_score = 0.0
            camera_id = "CAM-001"
            component_scores: Dict[str, float] = {}
            performance = None
            annotated_frame = None

            def __init__(self, call: Dict[str, Any]) -> None:
                self.demo_call = call

        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        return renderer.render(canvas, _Stub(demo_call))

    def test_banner_is_drawn_only_in_demo_mode(self) -> None:
        off = self._render({"demo_mode": False, "status": "COMPLETED"})
        on = self._render(
            {
                "demo_mode": True,
                "status": "RINGING",
                "recipient": MASKED,
                "provider": "simulation",
            }
        )
        self.assertEqual(off.sum(), 0, "nothing must be drawn when demo mode is off")
        self.assertGreater(on.sum(), 0, "the banner must be visible during a demo call")

    def test_the_banner_names_the_feature(self) -> None:
        """The literal label matters - it is what a judge looks for."""
        from ai_engine.emergency.call_record import DemoCallRecord

        record = DemoCallRecord(status=CallStatus.RINGING, recipient=MASKED)
        self.assertIn("DEMO EMERGENCY CALL", record.to_banner_text())


if __name__ == "__main__":
    unittest.main()
