"""End-to-end pipeline tests (all 30 phases wired together)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import blank_frame, test_config

from ai_engine.detection.scripted import ScriptedDetector
from ai_engine.detection.scenarios import SCENARIOS, expectation_for, list_scenarios
from ai_engine.pipeline import AccidentPipeline
from ai_engine.pipeline.demo import DemoRunner
from ai_engine.reporting import IncidentReportGenerator
from ai_engine.schemas import SCHEMA_VERSION, IncidentStatus
from ai_engine.visualization import OverlayRenderer

FPS = 15.0


def run_scenario(name: str, frames: int = 70, config=None, root: Path | None = None):
    """Run one scripted scenario and return the pipeline (for assertions)."""
    config = config or test_config()
    if root is not None:
        config.evidence.enabled = True
        config.evidence.root = str(root)
    pipeline = AccidentPipeline(config, detector=ScriptedDetector(SCENARIOS[name]), camera_id="CAM-001")
    frame = blank_frame()
    for index in range(frames):
        pipeline.process_frame(frame, index, index / FPS)
    return pipeline


class TestPipelineMechanics(unittest.TestCase):
    def setUp(self) -> None:
        self.config = test_config()
        self.pipeline = AccidentPipeline(
            self.config, detector=ScriptedDetector(SCENARIOS["head_on_crash"]), camera_id="CAM-001"
        )

    def test_frame_result_is_complete(self) -> None:
        result = self.pipeline.process_frame(blank_frame(), 0, 0.0)
        for attribute in (
            "frame_index", "timestamp", "frame_size", "detections", "tracks", "motion",
            "trajectories", "trajectory_pairs", "collision_events", "temporal", "verification",
            "score_detail", "accident_score", "component_scores", "performance", "state", "status",
        ):
            self.assertTrue(hasattr(result, attribute), attribute)
        self.assertEqual(result.state, "NORMAL")
        self.assertEqual(result.status, "NORMAL")

    def test_frame_result_is_json_serialisable(self) -> None:
        result = self.pipeline.process_frame(blank_frame(), 0, 0.0)
        json.dumps(result.to_dict())

    def test_performance_fields(self) -> None:
        result = self.pipeline.process_frame(blank_frame(), 0, 0.0)
        perf = result.performance
        self.assertGreater(perf.total_ms, 0.0)
        self.assertGreaterEqual(perf.inference_ms, 0.0)
        self.assertGreaterEqual(perf.tracking_ms, 0.0)
        self.assertGreaterEqual(perf.analysis_ms, 0.0)
        self.assertEqual(perf.frame_size, (640, 360))

    def test_empty_frame_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.pipeline.process_frame(None, 0, 0.0)
        with self.assertRaises(ValueError):
            self.pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8), 0, 0.0)

    def test_closed_pipeline_refuses_frames(self) -> None:
        self.pipeline.close()
        with self.assertRaises(RuntimeError):
            self.pipeline.process_frame(blank_frame(), 0, 0.0)

    def test_reset_clears_state(self) -> None:
        for index in range(20):
            self.pipeline.process_frame(blank_frame(), index, index / FPS)
        self.pipeline.reset()
        self.assertEqual(self.pipeline.incidents, [])
        self.assertIs(self.pipeline.state, self.pipeline.verification.state)

    def test_detector_failure_does_not_kill_the_run(self) -> None:
        class BrokenDetector:
            model_name = "broken"

            def __init__(self) -> None:
                self.calls = 0

            def detect(self, *args, **kwargs):
                from ai_engine.detection.yolo_detector import DetectionError

                self.calls += 1
                raise DetectionError("simulated inference failure")

        pipeline = AccidentPipeline(test_config(), detector=BrokenDetector(), camera_id="CAM-001")
        for index in range(3):
            result = pipeline.process_frame(blank_frame(), index, index / FPS)
            self.assertTrue(result.warnings)
            self.assertEqual(len(result.tracks), 0)
        self.assertEqual(pipeline._detection_failures, 3)

    def test_no_evidence_means_no_incidents(self) -> None:
        self.assertEqual(self.pipeline.get_incident_result(), {})
        self.assertEqual(self.pipeline.generate_emergency_incident(), {})


class TestScenarioBehaviour(unittest.TestCase):
    """The false-alarm prevention contract, on all nine scenarios."""

    def test_every_scenario_behaves_as_configured(self) -> None:
        failures = []
        for name in list_scenarios():
            pipeline = run_scenario(name)
            confirmed = len(pipeline.incidents) > 0
            expected = expectation_for(name)
            if confirmed != expected:
                failures.append(
                    f"{name}: expected {'ALARM' if expected else 'quiet'}, got "
                    f"{'ALARM' if confirmed else 'quiet'} "
                    f"(peak {pipeline.verification.event_peak:.3f})"
                )
            pipeline.close()
        self.assertEqual(failures, [], "\n".join(failures))

    def test_confirmed_incident_is_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config()
            config.evidence.enabled = True
            config.evidence.root = tmp
            config.visualization.enabled = True
            pipeline = run_scenario("head_on_crash", root=Path(tmp), config=config)
            try:
                self.assertEqual(len(pipeline.incidents), 1)
                incident = pipeline.incidents[0]
                payload = incident.to_dict()

                self.assertEqual(payload["status"], "CONFIRMED_ACCIDENT")
                self.assertTrue(payload["incident_id"].startswith("INC-"))
                self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
                self.assertGreater(payload["accident"]["score"], 0.0)
                self.assertLessEqual(payload["accident"]["score"], 1.0)
                self.assertIn(payload["accident"]["severity"], ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
                self.assertGreaterEqual(payload["objects"]["vehicles_involved"], 2)
                self.assertGreaterEqual(payload["evidence"]["agreement_signals"], 2)
                self.assertTrue(payload["evidence"]["reasons"])
                self.assertEqual(incident.validate(), [])

                # camera location attached from cameras.yaml
                self.assertEqual(payload["camera"]["camera_id"], "CAM-001")
                self.assertTrue(payload["location"]["is_camera_registered"])
                self.assertIsNotNone(payload["location"]["latitude"])

                # evidence actually written to disk
                evidence_dir = Path(payload["media"]["evidence_dir"])
                self.assertTrue(evidence_dir.is_dir())
                files = {p.name for p in evidence_dir.iterdir()}
                self.assertIn("before.jpg", files)
                self.assertIn("collision.jpg", files)
                self.assertIn("after.jpg", files)
                self.assertIn("annotated.jpg", files)
                self.assertIn("meta.json", files)

                # the report is attached and honest
                self.assertIsNotNone(payload["report"])
                self.assertIn("what_happened", payload["report"])
                self.assertIn("markdown", payload["report"])

                json.dumps(payload)
            finally:
                pipeline.close()

    def test_incident_callback_is_invoked(self) -> None:
        received = []
        config = test_config()
        pipeline = AccidentPipeline(
            config,
            detector=ScriptedDetector(SCENARIOS["rear_end_crash"]),
            camera_id="CAM-001",
            on_incident=received.append,
        )
        frame = blank_frame()
        for index in range(70):
            pipeline.process_frame(frame, index, index / FPS)
        pipeline.close()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["status"], "CONFIRMED_ACCIDENT")

    def test_no_incident_is_emitted_twice_for_one_event(self) -> None:
        pipeline = run_scenario("head_on_crash", frames=120)
        try:
            self.assertLessEqual(len(pipeline.incidents), 1)
        finally:
            pipeline.close()


class TestVideoRun(unittest.TestCase):
    def _write_video(self, path: Path, frames: int = 40) -> None:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
        for index in range(frames):
            writer.write(blank_frame())
        writer.release()

    def test_process_video_over_a_file(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
            for index in range(30):
                writer.write(blank_frame())
            writer.release()

            pipeline = AccidentPipeline(
                test_config(), detector=ScriptedDetector(SCENARIOS["parked_cars"]), camera_id="CAM-001"
            )
            run = pipeline.process_video(str(path))
            pipeline.close()

        self.assertEqual(run.frames_processed, 30)
        self.assertEqual(run.incidents, [])
        self.assertEqual(run.confirmed, 0)
        self.assertIn("source", run.to_dict())
        self.assertEqual(run.video_stats["kind"], "file")

    def test_max_frames_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            self._write_video(path, 40)
            pipeline = AccidentPipeline(
                test_config(), detector=ScriptedDetector(SCENARIOS["parked_cars"]), camera_id="CAM-001"
            )
            run = pipeline.process_video(str(path), max_frames=5)
            pipeline.close()
        self.assertLessEqual(run.frames_processed, 6)

    def test_missing_video_does_not_raise(self) -> None:
        pipeline = AccidentPipeline(
            test_config(), detector=ScriptedDetector(SCENARIOS["parked_cars"]), camera_id="CAM-001"
        )
        run = pipeline.process_video("definitely_missing.mp4")
        pipeline.close()
        self.assertIn("error", run.video_stats)

    def test_debug_video_is_written(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
            for _ in range(12):
                writer.write(blank_frame())
            writer.release()

            config = test_config()
            config.runtime.save_output_video = True
            config.runtime.output_video = str(Path(tmp) / "out.mp4")
            pipeline = AccidentPipeline(
                config, detector=ScriptedDetector(SCENARIOS["parked_cars"]), camera_id="CAM-001"
            )
            pipeline.process_video(str(path))
            pipeline.close()
            self.assertTrue(Path(config.runtime.output_video).is_file())


class TestVisualisation(unittest.TestCase):
    def test_overlay_renders_without_error(self) -> None:
        pipeline = AccidentPipeline(
            test_config(), detector=ScriptedDetector(SCENARIOS["head_on_crash"]), camera_id="CAM-001"
        )
        config = pipeline.config
        config.visualization.enabled = True
        renderer = OverlayRenderer(config.visualization)
        frame = blank_frame()
        for index in range(20):
            result = pipeline.process_frame(frame, index, index / FPS)
        canvas = renderer.render(frame, result)
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape, frame.shape)
        self.assertTrue(canvas.any(), "the overlay must actually draw something")
        pipeline.close()

    def test_disabled_overlay_returns_a_copy(self) -> None:
        renderer = OverlayRenderer(test_config().visualization)
        frame = blank_frame()
        result = type("R", (), {"tracks": [], "verification": None})()
        canvas = renderer.render(frame, result)
        self.assertIsNot(canvas, frame)
        self.assertFalse(canvas.any())


class TestReporting(unittest.TestCase):
    def test_report_never_invents_facts(self) -> None:
        pipeline = run_scenario("head_on_crash", frames=70)
        try:
            incident = pipeline.incidents[0]
            report = IncidentReportGenerator().generate(incident.to_dict(include_report=False))
        finally:
            pipeline.close()

        self.assertIn("Collision detected between", report["what_happened"])
        self.assertTrue(report["why"])
        self.assertIn("AI-estimated", report["confidence"]["wording"])
        self.assertIn("not a medical assessment", report["severity"]["wording"])
        self.assertIn("camera", report["location"]["wording"].lower())
        self.assertEqual(report["confidence"]["percent"], int(round(incident.accident.score * 100)))
        # every person-count sentence must match the data
        text = report["what_happened"]
        self.assertEqual(
            f"{incident.objects.persons_detected} person(s)" in text,
            incident.objects.persons_detected > 0,
        )

    def test_markdown_renders(self) -> None:
        pipeline = run_scenario("head_on_crash", frames=70)
        try:
            incident = pipeline.incidents[0]
            markdown = IncidentReportGenerator().to_markdown(incident.to_dict(include_report=False))
        finally:
            pipeline.close()
        self.assertIn("# SafeVision incident", markdown)
        self.assertIn("## What happened", markdown)
        self.assertIn("## Confidence and severity", markdown)

    def test_report_of_a_normal_status(self) -> None:
        from ai_engine.schemas import AccidentInfo, IncidentResult

        report = IncidentReportGenerator().generate(IncidentResult(incident_id="X", status="NORMAL").to_dict())
        self.assertIn("Normal traffic", report["what_happened"])


class TestDemoRunner(unittest.TestCase):
    def test_demo_runs_and_reports(self) -> None:
        config = test_config()
        config.demo.write_report = False
        config.demo.scenarios = {
            "normal": "test_videos/normal_traffic.mp4",
            "accident": "test_videos/accident.mp4",
        }
        runner = DemoRunner(config, camera_id="CAM-001")
        results = runner.run(max_frames=40)
        # 2 scripted scenarios (always available) + 3 video clips
        self.assertGreaterEqual(len(results), 5)
        scripted = [r for r in results if r.source.startswith("scenario:")]
        self.assertTrue(scripted, "the scripted scenarios must always be part of the demo")
        for result in scripted:
            self.assertTrue(result.available)
        for result in results:
            payload = result.to_dict()
            self.assertIn("label", payload)
            self.assertIn("source", payload)
            if not payload["available"]:
                self.assertTrue(payload["note"], "a missing clip must be explained")

    def test_demo_confirms_a_scripted_crash_and_rejects_a_trap(self) -> None:
        """The demo must actually demonstrate the state machine."""
        config = test_config()
        config.demo.write_report = False
        config.evidence.enabled = False
        runner = DemoRunner(config, camera_id="CAM-001")
        results = runner.run(max_frames=70)
        by_source = {r.source: r for r in results}
        crash = by_source.get("scenario:head_on_crash")
        trap = by_source.get("scenario:hard_brake")
        self.assertIsNotNone(crash)
        self.assertIsNotNone(trap)
        self.assertEqual(len(crash.incidents), 1, "the scripted crash must be confirmed")
        self.assertIn("VERIFYING", crash.states_seen)
        self.assertIn("CONFIRMED_ACCIDENT", crash.states_seen)
        self.assertEqual(trap.incidents, [], "the false-alarm trap must stay quiet")
        self.assertEqual(trap.states_seen, ["NORMAL"])


if __name__ == "__main__":
    unittest.main()
