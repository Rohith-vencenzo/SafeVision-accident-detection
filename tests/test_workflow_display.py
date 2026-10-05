"""The workflow display: what the user is shown, and when.

These lock down the requirement that each stage is *visible* and carries real
data - a bare "success" is not acceptable, because that is exactly what made the
pipeline look broken while it was in fact working.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.visualization.workflow import STAGES, WorkflowReporter  # noqa: E402


def capture(reporter: WorkflowReporter) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        reporter.summary()
    return buffer.getvalue()


def transition(to_state: str, **analysis) -> dict:
    base = {
        "detections": 4,
        "vehicles_involved": 2,
        "persons_involved": 0,
        "score": 0.44,
        "physical_score": 0.42,
        "temporal_score": 0.49,
        "components": {"motion": 0.55, "trajectory": 0.66},
        "agreement_signals": 2,
        "evidence_sufficient": True,
        "negative_evidence": [],
        "blocking_reasons": [],
        "frames_in_state": 12,
        "seconds_in_state": 1.8,
        "collision": {"tracks": [1, 3], "classes": ["car", "car"], "score": 0.31,
                      "reasons": ["impact"]},
    }
    base.update(analysis)
    return {
        "from": "NORMAL",
        "to": to_state,
        "score": base["score"],
        "reason": "because",
        "timestamp_iso": "2026-10-03T10:00:00+00:00",
        "analysis": base,
    }


INCIDENT = {
    "incident_id": "INC-UI-0001",
    "camera": {"camera_id": "CAM-001"},
    "accident": {"score": 0.44, "severity": "HIGH"},
    "objects": {"vehicles_involved": 2},
    "location": {"name": "Chennai Main Road", "latitude": 13.08, "longitude": 80.27},
    "media": {"detected_at": "2026-10-03T10:00:00+00:00",
              "evidence_dir": "evidence\\INC-UI-0001"},
    "extra": {"location_provenance": {"source": "CAMERA_REGISTERED"}},
}


# --------------------------------------------------------------------------- #
class TestAnalysisStageIsVisible(unittest.TestCase):
    def test_suspicious_shows_the_analysis_stage(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition("SUSPICIOUS"))
        text = buffer.getvalue()
        self.assertIn("ANALYZING ACCIDENT", text)
        self.assertIn("detection status", text)
        self.assertIn("accident confidence", text)
        self.assertIn("detected vehicles", text)
        self.assertIn("physical evidence", text)
        self.assertIn("why", text)

    def test_verifying_shows_temporal_status(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition("VERIFYING"))
        text = buffer.getvalue()
        self.assertIn("VERIFICATION", text)
        self.assertIn("temporal verification", text)
        self.assertIn("evidence so far", text)

    def test_confirmation_shows_evidence_severity_and_reason(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition("CONFIRMED_ACCIDENT"))
        text = buffer.getvalue()
        for required in ("ACCIDENT CONFIRMED", "accident confidence", "physical evidence",
                         "temporal evidence", "evidence / signals used",
                         "detected vehicles", "collision candidate",
                         "negative evidence", "why confirmed"):
            self.assertIn(required, text, f"confirmation output is missing {required!r}")

    def test_confirmation_shows_the_actual_signal_numbers(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition("CONFIRMED_ACCIDENT"))
        text = buffer.getvalue()
        self.assertIn("2 agreeing signal(s)", text)
        self.assertIn("44.0%", text)          # the score, as a percentage
        self.assertIn("car + car", text)      # the collision classes
        # the duration is quoted in the reason; the window line is only printed
        # when the counter is meaningful (it resets on the transition)
        self.assertIn("verification window", text)

    def test_the_verification_window_is_hidden_when_the_counter_is_empty(self) -> None:
        """A reset counter must not contradict the reason's duration."""
        reporter = WorkflowReporter()
        event = transition("CONFIRMED_ACCIDENT", frames_in_state=0, seconds_in_state=0.0)
        event["reason"] = ("multi-frame evidence confirmed: peak physical evidence 0.68, "
                           "temporal 0.49, 2 agreeing signals, 1.8s of verification")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(event)
        text = buffer.getvalue()
        self.assertNotIn("verification window", text)
        self.assertIn("1.8s of verification", text)

    def test_the_verification_window_is_shown_when_frames_were_held(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition("CONFIRMED_ACCIDENT"))
        self.assertIn("verification window", buffer.getvalue())

    def test_a_false_alarm_shows_why_it_was_rejected(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition(
                "FALSE_ALARM", blocking_reasons=["score fell back", "1 signal"]))
        text = buffer.getvalue()
        self.assertIn("FALSE ALARM", text)
        self.assertIn("blocking reasons", text)
        self.assertIn("score fell back", text)
        self.assertIn("SUPPRESSED", text)


# --------------------------------------------------------------------------- #
class TestIncidentStage(unittest.TestCase):
    def test_incident_details_are_shown(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_incident(INCIDENT)
        text = buffer.getvalue()
        for required in ("INCIDENT CREATED", "INC-UI-0001", "CAM-001",
                         "severity", "HIGH", "vehicles involved", "Chennai Main Road",
                         "CAMERA_REGISTERED", "evidence dir"):
            self.assertIn(required, text, f"incident output is missing {required!r}")

    def test_location_is_labelled_as_not_a_gps_fix(self) -> None:
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_incident(INCIDENT)
        self.assertIn("not a live GPS fix", buffer.getvalue())


# --------------------------------------------------------------------------- #
class TestNotificationStages(unittest.TestCase):
    def _run(self, events):
        reporter = WorkflowReporter()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            for payload in events:
                reporter.on_call_status(payload)
        return reporter, buffer.getvalue()

    def test_the_full_call_lifecycle_is_shown(self) -> None:
        reporter, text = self._run([
            {"label": "EMERGENCY CALL", "status": "INITIATED",
             "recipient": "+91*****3949", "provider": "twilio"},
            {"label": "EMERGENCY CALL", "status": "RINGING",
             "recipient": "+91*****3949", "provider": "twilio"},
            {"label": "EMERGENCY CALL", "status": "IN_PROGRESS",
             "recipient": "+91*****3949", "provider": "twilio"},
            {"label": "EMERGENCY CALL", "status": "COMPLETED",
             "recipient": "+91*****3949", "provider": "twilio"},
        ])
        for state in ("INITIATED", "RINGING", "IN_PROGRESS", "COMPLETED"):
            self.assertIn(state, text)
        self.assertIn("[CALL]", text)
        self.assertNotIn("[SIM]", text)
        self.assertEqual(reporter.stage_status["EMERGENCY CALL"], "done")
        self.assertEqual(reporter.stage_status["CALL COMPLETED"], "done")

    def test_a_simulated_call_is_tagged_so_it_cannot_be_mistaken_for_real(self) -> None:
        """`COMPLETED` for a simulated call must not read like a delivered call."""
        reporter, text = self._run([
            {"label": "DEMO EMERGENCY CALL", "status": "COMPLETED",
             "recipient": "+91*****3949", "provider": "simulation"},
            {"label": "DEMO SMS", "status": "SENT",
             "recipient": "+91*****3949", "provider": "simulation"},
        ])
        self.assertIn("[SIM]", text)
        self.assertIn("SIMULATED - no phone call was placed", text)
        self.assertIn("SIMULATED - no SMS was sent to any real number", text)
        self.assertTrue(reporter.simulated)

    def test_the_summary_reminds_the_user_a_simulation_sent_nothing(self) -> None:
        reporter = self._run([
            {"label": "DEMO EMERGENCY CALL", "status": "COMPLETED",
             "recipient": "+91*****3949", "provider": "simulation"},
        ])[0]
        text = capture(reporter)
        self.assertIn("SIMULATED", text)
        self.assertIn("Nothing was", text)

    def test_sms_generated_and_sent_are_both_shown(self) -> None:
        reporter, text = self._run([
            {"label": "DEMO SMS", "status": "SENT", "recipient": "+91*****3949"},
        ])
        self.assertIn("GENERATED", text)
        self.assertIn("SENT", text)
        self.assertEqual(reporter.stage_status["SMS GENERATED"], "done")
        self.assertEqual(reporter.stage_status["SMS SENT"], "done")

    def test_a_blocked_call_is_shown_as_refused_with_the_reason(self) -> None:
        reporter, text = self._run([
            {"label": "DEMO EMERGENCY CALL", "status": "NOT_CONFIGURED",
             "recipient": "(unconfigured)", "reason": "missing TWILIO_ACCOUNT_SID"},
        ])
        self.assertIn("NOT_CONFIGURED", text)
        self.assertIn("missing TWILIO_ACCOUNT_SID", text)
        self.assertNotEqual(reporter.stage_status["EMERGENCY CALL"], "done")

    def test_the_sms_body_is_printed_verbatim(self) -> None:
        reporter = WorkflowReporter()
        body = "ACCIDENT ALERT\nIncident: INC-1\nCamera: CAM-001"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.show_sms(body)
        text = buffer.getvalue()
        self.assertIn("SMS CONTENT", text)
        self.assertIn("Incident: INC-1", text)
        self.assertIn("Camera: CAM-001", text)


# --------------------------------------------------------------------------- #
class TestWorkflowSummary(unittest.TestCase):
    def _completed_run(self) -> WorkflowReporter:
        reporter = WorkflowReporter()
        reporter.on_transition(transition("SUSPICIOUS"))
        reporter.on_transition(transition("VERIFYING"))
        reporter.on_transition(transition("CONFIRMED_ACCIDENT"))
        reporter.on_incident(INCIDENT)
        for state in ("INITIATED", "RINGING", "IN_PROGRESS", "COMPLETED"):
            reporter.on_call_status({"label": "DEMO EMERGENCY CALL", "status": state,
                                     "recipient": "+91*****3949"})
        reporter.on_call_status({"label": "DEMO SMS", "status": "SENT",
                                 "recipient": "+91*****3949"})
        reporter.on_report_saved("incidents.json", "demo_calls.jsonl")
        return reporter

    def test_every_stage_is_marked_done_after_a_full_run(self) -> None:
        text = capture(self._completed_run())
        self.assertEqual(text.count("[x]"), len(STAGES),
                         f"not every stage was marked:\n{text}")
        self.assertNotIn("[ ]", text)
        self.assertNotIn("[!]", text)

    def test_the_summary_lists_the_stages_in_workflow_order(self) -> None:
        text = capture(self._completed_run())
        positions = [text.index(name) for name, _ in STAGES]
        self.assertEqual(positions, sorted(positions), "stages are out of order")

    def test_no_incident_leaves_the_notification_stages_pending(self) -> None:
        """A quiet clip must NOT claim a call was made."""
        reporter = WorkflowReporter()
        reporter.on_report_saved("incidents.json", None)
        text = capture(reporter)
        self.assertIn("[ ]", text)
        self.assertNotIn("[x] EMERGENCY CALL", text)
        self.assertNotIn("[x] SMS SENT", text)

    def test_a_quiet_reporter_prints_nothing(self) -> None:
        """--json mode must keep stdout a clean machine-readable stream."""
        reporter = WorkflowReporter(quiet=True)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            reporter.on_transition(transition("CONFIRMED_ACCIDENT"))
            reporter.on_incident(INCIDENT)
            reporter.on_call_status({"label": "DEMO SMS", "status": "SENT"})
            reporter.summary()
        self.assertEqual(buffer.getvalue(), "")


# --------------------------------------------------------------------------- #
class TestPipelineTransitionHook(unittest.TestCase):
    def test_the_pipeline_emits_transitions_with_an_analysis_block(self) -> None:
        """The hook must carry data, not just a state name."""
        import numpy as np

        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline
        from tests.helpers import test_config

        seen = []
        pipeline = AccidentPipeline(
            test_config(),
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
            on_transition=seen.append,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for index in range(80):
            pipeline.process_frame(frame, index, index / 15.0)
        pipeline.close()

        self.assertTrue(seen, "no transition was emitted")
        states = [event["to"] for event in seen]
        self.assertIn("SUSPICIOUS", states)
        self.assertIn("VERIFYING", states)
        self.assertIn("CONFIRMED_ACCIDENT", states)
        for event in seen:
            self.assertIn("analysis", event)
            for key in ("score", "vehicles_involved", "agreement_signals",
                        "physical_score", "temporal_score", "components"):
                self.assertIn(key, event["analysis"])

    def test_a_broken_transition_callback_cannot_stop_the_run(self) -> None:
        import numpy as np

        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline
        from tests.helpers import test_config

        def explode(_event):
            raise RuntimeError("UI bug")

        pipeline = AccidentPipeline(
            test_config(),
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
            on_transition=explode,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for index in range(80):
            pipeline.process_frame(frame, index, index / 15.0)
        pipeline.close()
        self.assertEqual(len(pipeline.incidents), 1,
                         "a failing UI callback must not lose the incident")


if __name__ == "__main__":
    unittest.main()