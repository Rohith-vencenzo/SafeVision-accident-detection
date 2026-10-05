"""Phase 5: routing, dispatch decisions, operator override, claim boundary.

Routing is where a detection system can quietly become a harm: the wrong person
is told, or an emergency service is contacted without authorisation. These tests
pin the refusals down, and they pin down the fact that a *refusal is a first-class
outcome* rather than an empty list someone could mistake for "nothing to do".
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.dispatch import (  # noqa: E402
    ContactKind,
    DispatchMode,
    DispatchRefusal,
    Recipient,
    RecipientDirectory,
    Router,
    RoutingPolicy,
    RoutingTrace,
)
from ai_engine.emergency.webhook import CallbackApplier  # noqa: E402
from ai_engine.emergency.outbox import NotificationOutbox, OutboxState  # noqa: E402


def incident(**over):
    base = {
        "incident_id": "INC-ROUTE-0001",
        "status": "CONFIRMED_ACCIDENT",
        "timestamp": "2026-10-03T12:00:00+00:00",
        "camera": {"camera_id": "CAM-001"},
        "location": {"name": "Chennai Main Road", "latitude": 13.08, "longitude": 80.27},
        "accident": {"score": 0.7, "severity": "HIGH"},
        "extra": {"location_provenance": {"source": "CAMERA_REGISTERED",
                                          "latitude": 13.08, "longitude": 80.27}},
    }
    base.update(over)
    return base


def directory():
    return RecipientDirectory([
        Recipient(recipient_id="SAFETY-LEAD-01", kind=ContactKind.DESIGNATED_CONTACT,
                  roles=("safety-officer",), jurisdictions=("IN-TN",),
                  number_env_var="EMERGENCY_CONTACT_NUMBER", consented_by="operator"),
        Recipient(recipient_id="BACKUP-02", kind=ContactKind.DESIGNATED_CONTACT,
                  roles=("safety-officer",), jurisdictions=("IN-KA",),
                  number_env_var="EMERGENCY_CONTACT_NUMBER", consented_by="operator"),
    ])


# --------------------------------------------------------------------------- #
class TestRecipientDirectory(unittest.TestCase):
    def test_the_shipped_directory_loads(self) -> None:
        book = RecipientDirectory.from_file(ROOT / "dispatch" / "recipients.json")
        self.assertGreaterEqual(len(book), 1)
        self.assertIsNotNone(book.get("SAFETY-LEAD-01"))

    def test_no_phone_number_is_stored_in_the_directory(self) -> None:
        text = (ROOT / "dispatch" / "recipients.json").read_text(encoding="utf-8")
        import re
        self.assertEqual([m for m in re.findall(r"\+\d{8,}", text)], [],
                         "the directory must never contain a number")

    def test_a_missing_directory_is_empty_and_says_so(self) -> None:
        book = RecipientDirectory.from_file(ROOT / "no_such_directory.json")
        self.assertEqual(len(book), 0)
        decision = Router(directory=book).route(incident())
        self.assertEqual(decision.refused, DispatchRefusal.NO_DIRECTORY)

    def test_an_unvetted_emergency_contact_is_disabled_on_construction(self) -> None:
        contact = Recipient(recipient_id="POLICE", kind=ContactKind.EMERGENCY_SERVICE,
                            roles=("emergency-service",))
        self.assertFalse(contact.enabled,
                         "an emergency-service contact with no consent record is unusable")

    def test_eligibility_filters_by_role_and_jurisdiction(self) -> None:
        book = directory()
        self.assertEqual(len(book.eligible(role="safety-officer")), 2)
        self.assertEqual(len(book.eligible(role="safety-officer", jurisdiction="IN-TN")), 1)
        self.assertEqual(book.eligible(jurisdiction="IN-TN")[0].recipient_id, "SAFETY-LEAD-01")
        self.assertEqual(book.eligible(role="witness"), [])


# --------------------------------------------------------------------------- #
class TestRoutingRefusals(unittest.TestCase):
    """Each refusal is a decision, not an absence."""

    def _router(self, policy=None, book=None, trace=None):
        return Router(policy=policy or RoutingPolicy(mode=DispatchMode.LIVE),
                      directory=book if book is not None else directory(),
                      trace=trace)

    def test_an_unconfirmed_incident_is_never_routed(self) -> None:
        for status in ("SUSPICIOUS", "VERIFYING", "NORMAL", "FALSE_ALARM", "", None):
            decision = self._router().route(incident(status=status))
            self.assertEqual(decision.refused, DispatchRefusal.NOT_CONFIRMED,
                             f"status {status!r} must not be routed")
            self.assertIsNone(decision.recipient)

    def test_a_confirmed_incident_routes_to_an_eligible_contact(self) -> None:
        decision = self._router().route(incident())
        self.assertIsNone(decision.refused)
        self.assertIsNotNone(decision.recipient)
        self.assertTrue(decision.is_dispatchable)
        self.assertEqual(decision.rule, "eligible_contact")

    def test_emergency_service_routing_is_refused_not_attempted(self) -> None:
        decision = self._router().route(incident(request_emergency_service=True))
        self.assertEqual(decision.refused, DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED)
        self.assertIn("NOT APPROVED", decision.reason)

    def test_a_listed_emergency_contact_is_never_selected(self) -> None:
        book = RecipientDirectory([
            Recipient(recipient_id="POLICE-112", kind=ContactKind.EMERGENCY_SERVICE,
                      roles=("safety-officer",), jurisdictions=("IN-TN",),
                      consented_by="impossible", enabled=True),
            Recipient(recipient_id="SAFETY-LEAD-01", kind=ContactKind.DESIGNATED_CONTACT,
                      roles=("safety-officer",), jurisdictions=("IN-TN",),
                      consented_by="operator"),
        ])
        decision = self._router(book=book).route(incident())
        self.assertEqual(decision.recipient.recipient_id, "SAFETY-LEAD-01",
                         "an emergency-service contact must never be chosen")
        evaluated = [e for e in decision.evaluated if e["rule"] == "emergency_service_excluded"][0]
        self.assertIn("POLICE-112", evaluated["excluded"])

    def test_a_directory_of_only_emergency_contacts_is_refused(self) -> None:
        book = RecipientDirectory([
            Recipient(recipient_id="POLICE", kind=ContactKind.EMERGENCY_SERVICE,
                      roles=("safety-officer",), jurisdictions=("IN-TN",),
                      consented_by="x", enabled=True),
        ])
        decision = self._router(book=book).route(incident())
        self.assertEqual(decision.refused, DispatchRefusal.EMERGENCY_ROUTING_NOT_APPROVED)

    def test_no_jurisdiction_match_is_no_approved_recipient(self) -> None:
        decision = self._router(
            policy=RoutingPolicy(mode=DispatchMode.LIVE, required_jurisdiction="FR"),
        ).route(incident())
        self.assertEqual(decision.refused, DispatchRefusal.NO_APPROVED_RECIPIENT)

    def test_the_master_kill_switch_stops_everything(self) -> None:
        decision = self._router(
            policy=RoutingPolicy(mode=DispatchMode.LIVE, dispatch_enabled=False),
        ).route(incident())
        self.assertEqual(decision.refused, DispatchRefusal.SUPPRESSED_BY_OVERRIDE)
        self.assertIsNone(decision.recipient)

    def test_simulation_mode_never_becomes_a_dispatch(self) -> None:
        decision = self._router(
            policy=RoutingPolicy(mode=DispatchMode.SIMULATION)).route(incident())
        self.assertFalse(decision.is_dispatchable,
                         "simulation must record a decision without dispatching")
        self.assertEqual(decision.mode, "SIMULATION")

    def test_a_required_location_that_is_absent_is_refused(self) -> None:
        bare = incident()
        bare["extra"] = {"location_provenance": {"source": "LOCATION_UNAVAILABLE",
                                                 "latitude": None, "longitude": None}}
        decision = self._router(
            policy=RoutingPolicy(mode=DispatchMode.LIVE, require_location=True),
        ).route(bare)
        self.assertEqual(decision.refused, DispatchRefusal.LOCATION_UNAVAILABLE)

    def test_every_refusal_carries_a_human_explanation(self) -> None:
        for refusal in DispatchRefusal:
            self.assertTrue(refusal.explanation.strip())
            self.assertGreater(len(refusal.explanation), 20,
                               "a refusal an operator cannot act on is useless")


# --------------------------------------------------------------------------- #
class TestOperatorOverride(unittest.TestCase):
    def test_suppressing_an_incident_stops_its_dispatch(self) -> None:
        trace = RoutingTrace()
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE),
                        directory=directory(), trace=trace)
        trace.suppress("INC-ROUTE-0001", "operator: false positive, dismissing")
        decision = router.route(incident())
        self.assertEqual(decision.refused, DispatchRefusal.SUPPRESSED_BY_OVERRIDE)
        self.assertIn("false positive", decision.reason)
        self.assertFalse(decision.is_dispatchable)

    def test_a_suppression_does_not_affect_another_incident(self) -> None:
        trace = RoutingTrace()
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE),
                        directory=directory(), trace=trace)
        trace.suppress("INC-ROUTE-0001")
        other = router.route(incident(incident_id="INC-ROUTE-0002"))
        self.assertTrue(other.is_dispatchable)

    def test_suppression_can_be_lifted(self) -> None:
        trace = RoutingTrace()
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE),
                        directory=directory(), trace=trace)
        trace.suppress("INC-ROUTE-0001")
        trace.unsuppress("INC-ROUTE-0001")
        self.assertTrue(router.route(incident()).is_dispatchable)


# --------------------------------------------------------------------------- #
class TestDecisionLog(unittest.TestCase):
    def test_every_decision_names_the_rule_that_produced_it(self) -> None:
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE), directory=directory())
        router.route(incident())
        router.route(incident(incident_id="INC-2", status="SUSPICIOUS"))
        for entry in router.trace.entries:
            self.assertTrue(entry.rule, "a decision without a rule is unexplainable")
            self.assertTrue(entry.reason)

    def test_the_log_records_what_was_considered(self) -> None:
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE), directory=directory())
        decision = router.route(incident())
        considered = [e for e in decision.evaluated if e["rule"] == "eligibility"][0]
        self.assertEqual(len(considered["considered"]), 2)
        for row in considered["considered"]:
            self.assertIn("label", row)
            self.assertNotIn("number", row, "the log must not carry a phone number")

    def test_the_summary_separates_dispatchable_from_refused(self) -> None:
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE), directory=directory())
        router.route(incident())
        router.route(incident(incident_id="INC-2", status="VERIFYING"))
        summary = router.trace.summary()
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["dispatchable"], 1)
        self.assertEqual(summary["refusals_by_reason"]["NOT_CONFIRMED"], 1)

    def test_the_log_serialises_and_never_leaks_a_number(self) -> None:
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE), directory=directory())
        router.route(incident())
        blob = router.trace.to_json()
        json.loads(blob)
        self.assertNotIn("+91", blob)
        self.assertNotIn("15550100", blob)

    def test_refusal_explanations_survive_serialisation(self) -> None:
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE), directory=directory())
        router.route(incident(incident_id="X", request_emergency_service=True))
        payload = json.loads(router.trace.to_json())
        entry = payload["entries"][0]
        self.assertEqual(entry["refused"], "EMERGENCY_ROUTING_NOT_APPROVED")
        self.assertIn("NOT APPROVED", entry["refusal_explanation"])


# --------------------------------------------------------------------------- #
class TestFailSafeDefaults(unittest.TestCase):
    def test_the_default_policy_can_dispatch_nothing(self) -> None:
        policy = RoutingPolicy()
        self.assertIs(policy.mode, DispatchMode.SIMULATION)
        self.assertFalse(policy.allow_emergency_service)
        self.assertFalse(policy.require_location)
        self.assertEqual(policy.to_dict()["emergency_service_routing"], "NOT_APPROVED")

    def test_there_is_no_self_test_notification_path(self) -> None:
        """A 'send a test message to check it works' path must not exist."""
        router = Router(policy=RoutingPolicy(mode=DispatchMode.LIVE), directory=directory())
        # a synthetic non-incident is not CONFIRMED_ACCIDENT, so it is refused
        decision = router.route({"incident_id": "SELF-TEST", "status": "NORMAL"})
        self.assertEqual(decision.refused, DispatchRefusal.NOT_CONFIRMED)

    def test_routing_decisions_never_mutate_the_incident(self) -> None:
        original = incident()
        snapshot = json.dumps(original, sort_keys=True)
        Router(policy=RoutingPolicy(mode=DispatchMode.LIVE),
               directory=directory()).route(original)
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot)

    def test_a_restart_cannot_replay_a_dispatch(self) -> None:
        """No notification storm after a restart: the outbox row survives."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "outbox.sqlite3")
            first = NotificationOutbox(path)
            first.enqueue("INC-1", "call")
            first.claim("INC-1", "call")
            first.complete("INC-1", "call", state=OutboxState.DONE, provider_ref="CA1")
            first.close()
            for _ in range(5):        # simulate repeated restarts
                again = NotificationOutbox(path)
                self.assertIsNone(again.claim("INC-1", "call"))
                again.close()


# --------------------------------------------------------------------------- #
class TestEvidenceBoundary(unittest.TestCase):
    """The claim boundary is itself tested, so it cannot quietly rot."""

    def test_completed_is_not_reported_as_a_person_answering(self) -> None:
        box = NotificationOutbox(":memory:")
        box.enqueue("INC-1", "call")
        box.claim("INC-1", "call")
        result = CallbackApplier(box).apply("call", "INC-1", "CA1", "completed")
        self.assertEqual(result.status.value, "COMPLETED")
        self.assertFalse(hasattr(result, "human_answered"))
        box.close()

    def test_a_verified_callback_cannot_claim_an_emergency_dispatch(self) -> None:
        box = NotificationOutbox(":memory:")
        box.enqueue("INC-1", "call")
        box.claim("INC-1", "call")
        CallbackApplier(box).apply("call", "INC-1", "CA1", "completed")
        row = box.get("INC-1", "call")
        self.assertEqual(row.state, OutboxState.DONE)
        self.assertEqual(row.provider, "call",
                         "the outbox records a call channel, never an emergency dispatch")
        box.close()


if __name__ == "__main__":
    unittest.main()