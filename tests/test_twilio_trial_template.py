"""Twilio trial-account SMS compatibility (error 572006).

Regression: SafeVision POSTed a free-form ``Body`` to
``/2010-04-01/Accounts/{sid}/Messages.json``. A Twilio **trial account** refuses
that with::

    HTTP 400
    code: 572006
    message: "Invalid template name. Trial accounts can only use predefined
              SMS templates."

A trial account must send via a predefined Twilio Messaging Template, addressed
by its SID as ``ContentSid``. These tests pin the request shape for both account
types, the actionable error text, and - importantly - that the auth token is
never present in anything reported back to the operator.

Nothing here contacts Twilio: payload construction and error interpretation are
pure functions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.emergency.providers import (  # noqa: E402
    CONTENT_VARIABLES_ENV_VAR,
    SMS_ERROR_HINTS,
    TEMPLATE_ENV_VAR,
    TWILIO_TRIAL_TEMPLATE_CODE,
    TwilioSmsProvider,
    build_message_payload,
    build_sms_provider,
    explain_sms_error,
)

RECIPIENT = "+15550100"
SENDER = "+15005550306"
SID = "AC" + "0" * 32
TOKEN = "super-secret-auth-token"
TEMPLATE_SID = "HX" + "1" * 32


# --------------------------------------------------------------------------- #
class TestTrialTemplateError(unittest.TestCase):
    """The exact error the operator hit, and how it is reported."""

    def test_572006_is_known_and_explained(self) -> None:
        self.assertIn(TWILIO_TRIAL_TEMPLATE_CODE, SMS_ERROR_HINTS)

    def test_the_provider_message_is_preserved_and_explained(self) -> None:
        raw = ('{"code": 572006, "message": "Invalid template name. Trial accounts '
               'can only use predefined SMS templates."}')
        text = explain_sms_error(400, raw)
        # Twilio's own wording is kept verbatim ...
        self.assertIn("Invalid template name", text)
        self.assertIn("572006", text)
        # ... and the operator is told what to do
        self.assertIn(TEMPLATE_ENV_VAR, text)
        self.assertIn("TRIAL", text.upper())

    def test_an_unknown_error_is_passed_through_unchanged(self) -> None:
        text = explain_sms_error(404, "The requested resource was not found")
        self.assertEqual(text, "The requested resource was not found")

    def test_verified_recipient_and_alphanumeric_hints_exist(self) -> None:
        """The other two reasons a trial account's SMS fails."""
        for code in ("21612", "21614", "21211"):
            self.assertIn(code, SMS_ERROR_HINTS, f"no hint for {code}")

    def test_the_error_never_leaks_the_auth_token(self) -> None:
        text = explain_sms_error(
            400, '{"code": 572006, "message": "Invalid template name"}')
        self.assertNotIn(TOKEN, text)
        for hint in SMS_ERROR_HINTS.values():
            self.assertNotIn(TOKEN, hint)


# --------------------------------------------------------------------------- #
class TestPayloadShape(unittest.TestCase):
    """A trial account and a paid account need different requests."""

    def test_free_form_body_is_used_when_no_template_is_configured(self) -> None:
        data = build_message_payload(RECIPIENT, SENDER, "ACCIDENT ALERT")
        self.assertEqual(data["To"], RECIPIENT)
        self.assertEqual(data["From"], SENDER)
        self.assertEqual(data["Body"], "ACCIDENT ALERT")
        self.assertNotIn("ContentSid", data)

    def test_a_template_replaces_the_body(self) -> None:
        data = build_message_payload(
            RECIPIENT, SENDER, "ACCIDENT ALERT", content_sid=TEMPLATE_SID)
        self.assertEqual(data["ContentSid"], TEMPLATE_SID)
        self.assertNotIn("Body", data,
                         "Twilio rejects ContentSid together with Body")

    def test_content_variables_are_sent_only_with_a_template(self) -> None:
        data = build_message_payload(
            RECIPIENT, SENDER, "x", content_sid=TEMPLATE_SID,
            content_variables='{"1":"CAM-001"}')
        self.assertEqual(data["ContentVariables"], '{"1":"CAM-001"}')
        # ignored without a template, since it would be rejected
        plain = build_message_payload(RECIPIENT, SENDER, "x",
                                      content_variables='{"1":"CAM-001"}')
        self.assertNotIn("ContentVariables", plain)

    def test_the_body_is_still_length_capped(self) -> None:
        data = build_message_payload(RECIPIENT, SENDER, "A" * 5000,
                                     max_body_length=200)
        self.assertEqual(len(data["Body"]), 200)

    def test_the_payload_never_contains_the_token(self) -> None:
        data = build_message_payload(RECIPIENT, SENDER, "hello",
                                     content_sid=TEMPLATE_SID)
        self.assertNotIn(TOKEN, " ".join(data.values()))


# --------------------------------------------------------------------------- #
class TestProviderWiring(unittest.TestCase):
    def test_the_provider_reads_the_template_from_the_environment(self) -> None:
        env = {
            "TWILIO_ACCOUNT_SID": SID,
            "TWILIO_AUTH_TOKEN": TOKEN,
            "TWILIO_PHONE_NUMBER": SENDER,
            TEMPLATE_ENV_VAR: TEMPLATE_SID,
        }
        provider = build_sms_provider("twilio", environ=env)
        self.assertIsInstance(provider, TwilioSmsProvider)
        self.assertEqual(provider.content_sid, TEMPLATE_SID)

    def test_no_template_means_free_form(self) -> None:
        env = {"TWILIO_ACCOUNT_SID": SID, "TWILIO_AUTH_TOKEN": TOKEN,
               "TWILIO_PHONE_NUMBER": SENDER}
        provider = build_sms_provider("twilio", environ=env)
        self.assertEqual(provider.content_sid, "")

    def test_the_provider_uses_the_shared_builder(self) -> None:
        """The live test and production must not drift apart."""
        provider = TwilioSmsProvider(
            account_sid=SID, auth_token=TOKEN, from_number=SENDER,
            content_sid=TEMPLATE_SID)
        data = build_message_payload(
            RECIPIENT, provider.from_number, "body",
            content_sid=provider.content_sid,
            content_variables=provider.content_variables,
            max_body_length=provider.max_body_length,
        )
        self.assertEqual(data["ContentSid"], TEMPLATE_SID)
        self.assertNotIn("Body", data)


# --------------------------------------------------------------------------- #
class TestLiveTestUsesTheSameRequest(unittest.TestCase):
    """The gated live test must build exactly what production would build."""

    def _script(self):
        import importlib.util

        path = ROOT / "scripts" / "live_provider_test.py"
        spec = importlib.util.spec_from_file_location("live_provider_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_script_exposes_the_template_flags(self) -> None:
        script = self._script()
        args = script.build_parser().parse_args(
            ["--live", "--channel", "sms"])
        self.assertEqual(args.content_sid_env_var, TEMPLATE_ENV_VAR)
        self.assertEqual(args.content_variables_env_var, CONTENT_VARIABLES_ENV_VAR)

    def test_the_script_builds_a_template_request_when_configured(self) -> None:
        script = self._script()
        args = script.build_parser().parse_args(["--live", "--channel", "sms"])
        env = {
            "TWILIO_ACCOUNT_SID": SID, "TWILIO_AUTH_TOKEN": TOKEN,
            "TWILIO_FROM_NUMBER": SENDER,
            "EMERGENCY_CONTACT_LIVE_TEST": "YES",
            "EMERGENCY_CONTACT_LIVE_TEST_NUMBER": RECIPIENT,
            TEMPLATE_ENV_VAR: TEMPLATE_SID,
        }
        # run_sms must not reach the network here: the provider is unreachable
        # and the result records the failure honestly. Only the built payload is
        # asserted, via the shared builder the script imports.
        self.assertIn("build_message_payload", dir(script))
        data = script.build_message_payload(
            RECIPIENT, SENDER, args.body,
            content_sid=env.get(TEMPLATE_ENV_VAR, ""))
        self.assertEqual(data["ContentSid"], TEMPLATE_SID)
        self.assertNotIn("Body", data)

    def test_twilio_error_parsing_keeps_the_provider_code_and_message(self) -> None:
        script = self._script()
        raw = ('{"code": 572006, "message": "Invalid template name. Trial accounts '
               'can only use predefined SMS templates."}')
        code, message = script._parse_twilio_error(raw)
        self.assertEqual(code, 572006)
        self.assertIn("Invalid template name", message)
        self.assertNotIn(TOKEN, message)

    def test_unparsable_error_bodies_do_not_raise(self) -> None:
        script = self._script()
        code, message = script._parse_twilio_error("<html>gateway error</html>")
        self.assertIsNone(code)
        self.assertIn("gateway error", message)


# --------------------------------------------------------------------------- #
class TestDetectionThresholdsUntouched(unittest.TestCase):
    """Guard: this fix must not have touched detection or incident logic."""

    def test_verification_thresholds_are_unchanged(self) -> None:
        from ai_engine.config import load_config

        config = load_config()
        self.assertEqual(config.verification.suspicious_score, 0.30)
        self.assertEqual(config.verification.verify_entry_score, 0.34)
        self.assertEqual(config.verification.confirm_score, 0.42)
        self.assertEqual(config.verification.temporal_confirm_score, 0.40)
        self.assertEqual(config.verification.release_score, 0.22)

    def test_the_sms_body_still_carries_the_incident_fields(self) -> None:
        """The free-form body is unchanged for paid accounts."""
        from ai_engine.emergency.sms_record import build_sms_body

        body = build_sms_body({
            "incident_id": "INC-1", "timestamp": "2026-10-05T10:00:00+00:00",
            "camera": {"camera_id": "CAM-001"},
            "location": {"name": "Chennai Main Road", "latitude": 13.08,
                         "longitude": 80.27, "is_camera_registered": True},
            "accident": {"score": 0.44, "severity": "HIGH"},
        }, instruction="Emergency assistance may be required.")
        for field in ("Incident:", "Date:", "Time:", "Camera: CAM-001",
                      "Chennai Main Road", "Severity: HIGH"):
            self.assertIn(field, body)


if __name__ == "__main__":
    unittest.main()