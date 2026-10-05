"""Evidence capture (Phase 15) and the video input layer (Phase 3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import FRAME_SIZE, blank_frame

from ai_engine.config.classes import EvidenceConfig, VideoConfig
from ai_engine.evidence import EvidenceCapture, sha256_file
from ai_engine.video import VideoReader, VideoSourceError, classify_source


class TestEvidenceCapture(unittest.TestCase):
    def _capture(self, root: Path, **overrides) -> EvidenceCapture:
        config = EvidenceConfig(root=str(root), **overrides)
        return EvidenceCapture(config, root=root)

    def test_creates_incident_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._capture(root)
            incident_id = capture.next_incident_id()
            session = capture.begin(incident_id)
            self.assertIsNotNone(session)
            self.assertTrue((root / incident_id).is_dir())
            self.assertTrue(incident_id.startswith("INC-"))

    def test_writes_frames_and_hashes_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._capture(root)
            session = capture.begin()
            session.save_frame("before", blank_frame())
            session.save_frame("collision", blank_frame())
            session.save_annotated(blank_frame())
            manifest = session.finalize({"accident_score": 0.5})

            self.assertTrue(Path(manifest.frames["before"]).is_file())
            self.assertTrue(Path(manifest.annotated).is_file())
            self.assertEqual(len(manifest.hashes), 3)
            for digest in manifest.hashes.values():
                self.assertEqual(len(digest), 64)
            self.assertTrue(Path(manifest.meta_file).is_file())
            meta = json.loads(Path(manifest.meta_file).read_text(encoding="utf-8"))
            self.assertEqual(meta["accident_score"], 0.5)
            self.assertEqual(meta["incident_id"], manifest.incident_id)

    def test_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.bin"
            path.write_bytes(b"safevision")
            first = sha256_file(path)
            second = sha256_file(path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_clip_written_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._capture(root, save_clip=True, clip_seconds=1.0, clip_fps=5.0)
            session = capture.begin()
            frames = [blank_frame() for _ in range(5)]
            clip = session.write_clip(frames)
            self.assertIsNotNone(clip)
            self.assertTrue(Path(clip).is_file())

    def test_disabled_capture_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._capture(root)
            capture.set_enabled(False)
            self.assertIsNone(capture.begin())
            self.assertEqual(list(root.iterdir()), [])

    def test_failure_does_not_raise(self) -> None:
        """A write failure must be recorded, never propagated to the caller."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._capture(root)
            session = capture.begin()
            # block the target path with a directory so cv2.imwrite cannot write
            (Path(session.directory) / "before.jpg").mkdir()
            self.assertIsNone(session.save_frame("before", blank_frame()))
            self.assertTrue(session.manifest.errors)
            # ... and the session still finalises cleanly afterwards
            session.save_frame("collision", blank_frame())
            manifest = session.finalize({"outcome": "test"})
            self.assertIn("collision", manifest.frames)

    def test_pre_event_buffer_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = self._capture(Path(tmp)).pre_event_buffer()
            for index in range(200):
                buffer.push(blank_frame(), index / 15.0)
            self.assertLessEqual(len(buffer), 45)
            self.assertGreaterEqual(len(buffer), 1)

    def test_prune_removes_old_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._capture(root, max_evidence_dirs=2)
            for _ in range(4):
                session = capture.begin()
                session.save_frame("before", blank_frame())
                session.finalize({})
            removed = capture.prune()
            remaining = [p for p in root.iterdir() if p.is_dir()]
            self.assertLessEqual(len(remaining), 2)
            self.assertTrue(removed)


class TestSourceClassification(unittest.TestCase):
    def test_classify(self) -> None:
        self.assertEqual(classify_source(0), "webcam")
        self.assertEqual(classify_source("0"), "webcam")
        self.assertEqual(classify_source("clip.mp4"), "file")
        self.assertEqual(classify_source(r"rtsp://cam/stream"), "rtsp")
        self.assertEqual(classify_source("video=Integrated Camera"), "webcam")


class TestVideoSourceErrors(unittest.TestCase):
    def test_missing_file_fails_fast(self) -> None:
        reader = VideoReader("definitely_not_here.mp4", VideoConfig())
        with self.assertRaises(VideoSourceError) as ctx:
            reader.open()
        self.assertIn("not found", str(ctx.exception))

    def test_directory_is_rejected(self) -> None:
        reader = VideoReader(".", VideoConfig())
        with self.assertRaises(VideoSourceError):
            reader.open()

    def test_empty_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.mp4"
            path.write_bytes(b"")
            reader = VideoReader(str(path), VideoConfig())
            with self.assertRaises(VideoSourceError) as ctx:
                reader.open()
            self.assertIn("empty", str(ctx.exception))

    def test_unreachable_rtsp_fails_with_a_hint(self) -> None:
        reader = VideoReader("rtsp://192.0.2.1:554/stream", VideoConfig(read_retries=1))
        with self.assertRaises(VideoSourceError) as ctx:
            reader.open()
        self.assertIn("SAFEVISION_RTSP_USERNAME", str(ctx.exception))

    def test_invalid_camera_index(self) -> None:
        reader = VideoReader(-1, VideoConfig())
        with self.assertRaises(VideoSourceError):
            reader.open()

    def test_credentials_are_injected_from_the_environment(self) -> None:
        from ai_engine.config.settings import build_source_url, redact_url

        environ = {"SAFEVISION_RTSP_USERNAME": "admin", "SAFEVISION_RTSP_PASSWORD": "secret"}
        url = build_source_url("rtsp://cam/stream", env=environ)
        self.assertIn("admin:secret@cam", url)
        # a URL that already carries credentials is left alone
        self.assertEqual(
            build_source_url("rtsp://bob:x@cam/s", env=environ), "rtsp://bob:x@cam/s"
        )
        # and it is redacted before logging
        self.assertNotIn("secret", redact_url(url))


class TestVideoRoundTrip(unittest.TestCase):
    def test_reads_generated_video_and_resizes(self) -> None:
        cv2 = __import__("cv2")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (FRAME_SIZE[0], FRAME_SIZE[1])
            )
            for index in range(12):
                frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), index * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            reader = VideoReader(str(path), VideoConfig(target_width=320, report_every=0))
            with reader:
                packets = list(reader.frames())
            self.assertGreaterEqual(len(packets), 8)
            self.assertEqual(packets[0].size, (320, 180))
            self.assertTrue(packets[0].resized)
            self.assertGreater(reader.fps, 0)
            stats = reader.stats()
            self.assertEqual(stats["kind"], "file")
            self.assertGreater(stats["frames_processed"], 0)

    def test_frame_stride(self) -> None:
        cv2 = __import__("cv2")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, FRAME_SIZE)
            for _ in range(20):
                writer.write(blank_frame())
            writer.release()
            reader = VideoReader(str(path), VideoConfig(frame_stride=5))
            with reader:
                packets = list(reader.frames())
            self.assertLess(len(packets), 20)
            self.assertTrue(all(p.dropped == 4 for p in packets))

    def test_closed_reader_refuses_to_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            cv2 = __import__("cv2")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, FRAME_SIZE)
            writer.write(blank_frame())
            writer.release()
            reader = VideoReader(str(path), VideoConfig())
            reader.open()
            reader.close()
            with self.assertRaises(VideoSourceError):
                reader.open()


if __name__ == "__main__":
    unittest.main()
