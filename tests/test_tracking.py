"""Tracking behaviour (Phase 5) - stable ids across the hard cases."""

from __future__ import annotations

import unittest

from tests.helpers import make_detection

from ai_engine.config.classes import TrackingConfig
from ai_engine.detection.detection_types import BBox, Detection
from ai_engine.tracking import ByteTracker, class_group
from ai_engine.utils.geometry import xyxy_iou

SIZE = (640, 360)
FPS = 15.0


def detections(boxes, confidence: float = 0.9, class_name: str = "car", frame: int = 0):
    """Build detections; ``class_name`` may be a single name or a per-box list."""
    names = class_name if isinstance(class_name, (list, tuple)) else [class_name] * len(boxes)
    return [
        Detection(
            class_id=-1,
            class_name=names[index] if index < len(names) else names[-1],
            confidence=confidence,
            bbox=BBox(*box),
            frame_index=frame,
            timestamp=frame / FPS,
        )
        for index, box in enumerate(boxes)
    ]


def class_group_check() -> None:
    assert class_group("car") == "vehicle"
    assert class_group("truck") == "vehicle"
    assert class_group("person") == "person"


class TestByteTrack(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ByteTracker(TrackingConfig())

    def test_stable_ids_for_constant_motion(self) -> None:
        seen = {}
        for frame in range(40):
            tracks = self.tracker.update(
                detections([(60 + 8 * frame, 120, 90 + 8 * frame, 142), (400 - 9 * frame, 220, 430 - 9 * frame, 242)]),
                SIZE,
                frame,
                frame / FPS,
            )
            for track in tracks:
                key = round(track.center[0])
                seen.setdefault(track.track_id, []).append(key)
        self.assertEqual(self.tracker.total_created, 2, "ids must not churn on smooth motion")
        self.assertEqual(len(seen), 2)

    def test_ids_survive_an_abrupt_stop(self) -> None:
        """The crash signature: Kalman overshoot must not create a new id."""
        for frame in range(20):
            self.tracker.update(detections([(60 + 8 * frame, 120, 90 + 8 * frame, 142)]), SIZE, frame, frame / FPS)
        x = 60 + 8 * 19
        for frame in range(20, 30):
            tracks = self.tracker.update(detections([(x, 120, x + 30, 142)]), SIZE, frame, frame / FPS)
            self.assertEqual([t.track_id for t in tracks], [1])
        self.assertEqual(self.tracker.total_created, 1)

    def test_new_object_gets_a_new_id(self) -> None:
        for frame in range(10):
            self.tracker.update(detections([(60 + 8 * frame, 120, 90 + 8 * frame, 142)]), SIZE, frame, frame / FPS)
        tracks = self.tracker.update(
            detections(
                [(140, 120, 170, 142), (300, 300, 318, 344)],
                class_name=["car", "person"],
            ),
            SIZE, 10, 10 / FPS,
        )
        self.assertEqual(len(self.tracker.active_tracks), 2)
        self.assertEqual(self.tracker.total_created, 2)
        # the new person only has one hit, so min_hits keeps it out of the output
        self.assertEqual([t.track_id for t in tracks], [1])

    def test_class_gate_prevents_id_theft(self) -> None:
        for frame in range(6):
            self.tracker.update(detections([(100, 100, 130, 122)]), SIZE, frame, frame / FPS)
        # a person detection exactly on top of the car must not take its id
        self.tracker.update(
            detections([(100, 100, 118, 144)], class_name="person", confidence=0.6), SIZE, 6, 6 / FPS
        )
        self.assertEqual(self.tracker.total_created, 2)

    def test_low_confidence_keeps_a_track_alive(self) -> None:
        for frame in range(8):
            self.tracker.update(detections([(100 + 5 * frame, 100, 130 + 5 * frame, 122)]), SIZE, frame, frame / FPS)
        track_id = self.tracker.active_tracks[0].track_id
        # blurry frame: below conf_high but above conf_low
        tracks = self.tracker.update(
            detections([(100 + 5 * 8, 100, 130 + 5 * 8, 122)], confidence=0.3), SIZE, 8, 8 / FPS
        )
        self.assertEqual([t.track_id for t in tracks], [track_id], "the second association stage must rescue this")
        self.assertEqual(self.tracker.total_created, 1)

    def test_confidence_below_floor_never_starts_a_track(self) -> None:
        tracks = self.tracker.update(detections([(100, 100, 130, 122)], confidence=0.05), SIZE, 0, 0.0)
        self.assertEqual(tracks, [])
        self.assertEqual(self.tracker.total_created, 0)

    def test_min_hits_suppresses_brand_new_tracks(self) -> None:
        tracks = self.tracker.update(detections([(100, 100, 130, 122)]), SIZE, 0, 0.0)
        self.assertEqual(tracks, [], "one hit is not enough with min_hits=2")
        tracks = self.tracker.update(detections([(101, 100, 131, 122)]), SIZE, 1, 1 / FPS)
        self.assertEqual(len(tracks), 1)

    def test_tracks_expire(self) -> None:
        for frame in range(6):
            self.tracker.update(detections([(100, 100, 130, 122)]), SIZE, frame, frame / FPS)
        self.assertTrue(self.tracker.active_tracks)
        for frame in range(6, 60):
            self.tracker.update([], SIZE, frame, frame / FPS)
        self.assertEqual(self.tracker.active_tracks, [])
        self.assertGreater(self.tracker.total_lost, 0)

    def test_accident_class_detections_are_not_tracked(self) -> None:
        detection = Detection(
            class_id=9, class_name="accident", confidence=0.9, bbox=BBox(0, 0, 200, 200), is_accident_class=True
        )
        tracks = self.tracker.update([detection], SIZE, 0, 0.0)
        self.assertEqual(tracks, [])

    def test_history_and_helpers(self) -> None:
        for frame in range(12):
            self.tracker.update(detections([(100 + 5 * frame, 100, 130 + 5 * frame, 122)]), SIZE, frame, frame / FPS)
        track = self.tracker.active_tracks[0]
        self.assertGreaterEqual(track.history_length, 10)
        self.assertGreater(track.speed_px_s, 0.0)
        self.assertGreater(track.speed_norm, 0.0)
        self.assertEqual(len(track.recent_points(3)), 3)
        self.assertGreater(track.displacement(0.5), 0.0)
        self.assertEqual(track.points_between(0.0, 0.5)[0].frame_index, 0)
        self.assertTrue(track.is_vehicle)
        self.assertEqual(track.class_group if hasattr(track, "class_group") else "vehicle", "vehicle")

    def test_stopping_object_reports_stopped_time(self) -> None:
        for frame in range(10):
            self.tracker.update(detections([(100 + 5 * frame, 100, 130 + 5 * frame, 122)]), SIZE, frame, frame / FPS)
        x = 100 + 5 * 9
        for frame in range(10, 25):
            self.tracker.update(detections([(x, 100, x + 30, 122)]), SIZE, frame, frame / FPS)
        track = self.tracker.active_tracks[0]
        self.assertGreater(track.stopped_for(0.5, 0.02, 24 / FPS), 0.0)
        self.assertTrue(track.has_been_stopped_for(0.5, 0.02, 24 / FPS))
        # ... but it was moving before, so the short window must not qualify
        self.assertFalse(track.has_been_stopped_for(5.0, 0.02, 24 / FPS))

    def test_always_moving_object_is_never_stopped(self) -> None:
        for frame in range(20):
            self.tracker.update(detections([(100 + 5 * frame, 100, 130 + 5 * frame, 122)]), SIZE, frame, frame / FPS)
        track = self.tracker.active_tracks[0]
        self.assertEqual(track.stopped_for(0.5, 0.02, 19 / FPS), 0.0)
        self.assertFalse(track.has_been_stopped_for(0.5, 0.02, 19 / FPS))

    def test_ultralytics_backend_adopts_external_ids(self) -> None:
        tracker = ByteTracker(TrackingConfig(backend="ultralytics"))
        # frame 0 creates the tracks, frame 1 confirms them (min_hits = 2)
        tracker.update(
            [
                make_detection("car", 0.9, (100, 100, 130, 122), track_id=7),
                make_detection("car", 0.9, (300, 200, 330, 222), track_id=9),
            ],
            SIZE, 0, 0.0,
        )
        tracks = tracker.update(
            [
                make_detection("car", 0.9, (103, 100, 133, 122), frame_index=1, timestamp=1 / FPS, track_id=7),
                make_detection("car", 0.9, (303, 200, 333, 222), frame_index=1, timestamp=1 / FPS, track_id=9),
            ],
            SIZE, 1, 1 / FPS,
        )
        self.assertEqual(sorted(t.track_id for t in tracks), [7, 9])
        self.assertEqual(tracker.total_created, 2, "external ids must not be renumbered")

    def test_reset(self) -> None:
        self.tracker.update(detections([(100, 100, 130, 122)]), SIZE, 0, 0.0)
        self.tracker.reset()
        self.assertEqual(self.tracker.track_count, 0)
        self.assertEqual(self.tracker.total_created, 0)

    def test_tiny_detections_are_dropped(self) -> None:
        tracks = self.tracker.update(detections([(100, 100, 102, 102)]), SIZE, 0, 0.0)
        self.assertEqual(tracks, [])

    def test_iou_geometry(self) -> None:
        self.assertAlmostEqual(xyxy_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0, places=6)
        self.assertEqual(xyxy_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(xyxy_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=6)


if __name__ == "__main__":
    class_group_check()
    unittest.main()
