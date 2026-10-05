"""The open-evidence session must always be closed, even on an early exit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import blank_frame, test_config

from ai_engine.detection.scenarios import SCENARIOS
from ai_engine.detection.scripted import ScriptedDetector
from ai_engine.pipeline import AccidentPipeline

FPS = 15.0


def _pipeline(root: Path, name: str = "head_on_crash") -> AccidentPipeline:
    config = test_config()
    config.evidence.enabled = True
    config.evidence.root = str(root)
    config.visualization.enabled = False
    return AccidentPipeline(
        config, detector=ScriptedDetector(SCENARIOS[name]), camera_id="CAM-001"
    )


class TestOpenEvidenceIsAlwaysClosed(unittest.TestCase):
    def test_stream_ending_mid_verification_leaves_a_complete_record(self) -> None:
        """A clip that stops during VERIFYING must still write meta.json."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = _pipeline(root)
            frame = blank_frame()
            # stop just after the state machine enters VERIFYING
            for index in range(46):
                pipeline.process_frame(frame, index, index / FPS)
            self.assertIsNotNone(pipeline._open_evidence, "the test must stop during verification")
            pipeline.close()

            directories = [p for p in root.iterdir() if p.is_dir()]
            self.assertEqual(len(directories), 1)
            meta_path = directories[0] / "meta.json"
            self.assertTrue(meta_path.is_file(), "an interrupted run must still write meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["outcome"], "INCOMPLETE")
            self.assertIn("note", meta)
            self.assertTrue((directories[0] / "before.jpg").is_file())

    def test_reset_also_closes_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = _pipeline(root)
            frame = blank_frame()
            for index in range(46):
                pipeline.process_frame(frame, index, index / FPS)
            self.assertIsNotNone(pipeline._open_evidence)
            pipeline.reset()
            self.assertIsNone(pipeline._open_evidence)
            meta_files = list(root.glob("INC-*/meta.json"))
            self.assertEqual(len(meta_files), 1)
            self.assertEqual(
                json.loads(meta_files[0].read_text(encoding="utf-8"))["outcome"], "INCOMPLETE"
            )
            pipeline.close()

    def test_confirmed_incident_is_not_marked_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = _pipeline(root)
            pipeline.config.visualization.enabled = True   # annotated.jpg needs the overlay
            frame = blank_frame()
            for index in range(75):
                pipeline.process_frame(frame, index, index / FPS)
            pipeline.close()
            self.assertEqual(len(pipeline.incidents), 1)
            incident_dir = Path(pipeline.incidents[0].media.evidence_dir)
            meta = json.loads((incident_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["outcome"], "CONFIRMED_ACCIDENT")
            for name in ("before.jpg", "collision.jpg", "after.jpg", "annotated.jpg"):
                self.assertTrue((incident_dir / name).is_file(), f"{name} must be captured")


if __name__ == "__main__":
    unittest.main()
