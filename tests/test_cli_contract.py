"""The command-line contract, including the ``--json`` stdout contract.

``--json`` is the easiest way for the downstream 40% to consume the engine, so
stdout must stay machine-readable: exactly one JSON object per incident and
nothing else.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests.helpers import test_config

import main as cli
from ai_engine.detection.scenarios import SCENARIOS
from ai_engine.detection.scripted import ScriptedDetector
from ai_engine.pipeline import AccidentPipeline
from ai_engine.schemas import SCHEMA_VERSION
from ai_engine.utils.logging_utils import configure_logging

ROOT = Path(__file__).resolve().parents[1]


class TestCliArguments(unittest.TestCase):
    def setUp(self) -> None:
        configure_logging("ERROR")

    def test_parser_exposes_the_documented_flags(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--source", "clip.mp4", "--show", "--set", "model.conf=0.4"])
        self.assertEqual(args.source, "clip.mp4")
        self.assertTrue(args.show)
        self.assertEqual(args.overrides, ["model.conf=0.4"])

    def test_print_config_exits_cleanly(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--print-config"])
        self.assertEqual(code, cli.EXIT_OK)
        payload = json.loads(buffer.getvalue())
        self.assertIn("model", payload)
        self.assertIn("verification", payload)

    def test_print_schema_exits_cleanly(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--print-schema"])
        self.assertEqual(code, cli.EXIT_OK)
        schema = json.loads(buffer.getvalue())
        self.assertIn("incident_id", schema["properties"])

    def test_list_cameras(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--list-cameras"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("CAM-001", buffer.getvalue())

    def test_check_model_reports_the_loaded_checkpoint(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--check-model"])
        # 0 = weights available, 1 = not (an offline machine); never a crash
        self.assertIn(code, (cli.EXIT_OK, cli.EXIT_ERROR))

    def test_check_source_rejects_a_missing_file(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--check-source", "definitely_missing.mp4"])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("NOT USABLE", buffer.getvalue())

    def test_missing_source_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli.main([])
        self.assertEqual(ctx.exception.code, cli.EXIT_USAGE)

    def test_bad_override_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--source", "0", "--set", "model.conf=5.0"])
        self.assertEqual(ctx.exception.code, cli.EXIT_USAGE)


class TestJsonStdoutContract(unittest.TestCase):
    """``--json`` must emit exactly one JSON object per incident."""

    def setUp(self) -> None:
        configure_logging("ERROR")
        self.config = test_config()
        self.config.runtime.print_incident_json = True

    def _run(self) -> list:
        pipeline = AccidentPipeline(
            self.config,
            detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001",
            on_incident=cli._print_incident,
        )
        import numpy as np
        import time

        frame = np.zeros((360, 640, 3), np.uint8)
        base = time.time()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            for index in range(80):
                pipeline.process_frame(frame, index, base + index / 15.0)
        pipeline.close()
        return buffer.getvalue()

    def test_stdout_is_one_json_object_per_incident(self) -> None:
        text = self._run().strip()
        self.assertTrue(text, "the scripted crash must produce an incident")
        decoder = json.JSONDecoder()
        payload, index = decoder.raw_decode(text)
        self.assertIsInstance(payload, dict, "each stdout chunk must be a JSON object")
        self.assertEqual(text[index:].strip(), "", "nothing may follow an incident object")
        self.assertEqual(payload["status"], "CONFIRMED_ACCIDENT")

    def test_stdout_stream_is_parseable(self) -> None:
        text = self._run().strip()
        decoder = json.JSONDecoder()
        objects: list = []
        position = 0
        while position < len(text):
            obj, end = decoder.raw_decode(text[position:])
            objects.append(obj)
            position = end + len(text[position + end :].lstrip())
        self.assertEqual(len(objects), 1, "an incident must not be printed twice")
        self.assertEqual(objects[0]["schema_version"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
