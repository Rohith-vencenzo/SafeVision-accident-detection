"""A configured `video.source` must be honoured (no --source on the command line).

This is the counterpart of the "a bare run is a usage error" rule: the source
may come from either place, but it must come from *somewhere* deliberately.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import main as cli  # noqa: E402


def _write_clip(path: Path, frames: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    for _ in range(frames):
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
    writer.release()
    return path


class TestConfiguredSourceIsHonoured(unittest.TestCase):
    def test_config_source_runs_without_a_command_line_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = _write_clip(Path(tmp) / "clip.mp4")
            config = Path(tmp) / "config.yaml"
            config.write_text(
                "video:\n"
                f"  source: '{clip.as_posix()}'\n"
                "runtime:\n"
                "  write_incidents_file: false\n"
                "  max_frames: 5\n"
                "evidence:\n"
                "  enabled: false\n"
                "visualization:\n"
                "  enabled: false\n",
                encoding="utf-8",
            )
            code = cli.main(["--config", str(config), "--log-level", "ERROR"])
        self.assertEqual(code, cli.EXIT_OK, "a source configured in config.yaml must be used")

    def test_default_source_is_rejected(self) -> None:
        """`video.source: 0` is the built-in default, i.e. nobody chose one."""
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text("video:\n  source: '0'\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["--config", str(config), "--log-level", "ERROR"])
        self.assertEqual(ctx.exception.code, cli.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
