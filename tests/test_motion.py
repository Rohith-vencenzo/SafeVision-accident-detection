"""Motion analysis (Phase 6)."""

from __future__ import annotations

import math
import unittest

from tests.helpers import FRAME_SIZE, deceleration_positions, make_track, moving_positions

from ai_engine.config.classes import MotionConfig
from ai_engine.motion import MotionAnalyzer
from ai_engine.utils.geometry import angle_between_deg, clamp, ema, smoothstep

FPS = 15.0
SCALE = math.hypot(*FRAME_SIZE)


class TestGeometryPrimitives(unittest.TestCase):
    def test_clamp(self) -> None:
        self.assertEqual(clamp(-1.0), 0.0)
        self.assertEqual(clamp(2.0), 1.0)
        self.assertEqual(clamp(0.4), 0.4)
        self.assertEqual(clamp(float("nan")), 0.0)

    def test_smoothstep_edges(self) -> None:
        self.assertEqual(smoothstep(0.0, 1.0, 0.0), 0.0)
        self.assertEqual(smoothstep(0.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(smoothstep(0.0, 1.0, 0.5), 0.5, places=6)
        self.assertEqual(smoothstep(1.0, 1.0, 0.5), 0.0)

    def test_ema(self) -> None:
        self.assertEqual(ema(None, 5.0, 0.5), 5.0)
        self.assertAlmostEqual(ema(0.0, 1.0, 0.5), 0.5, places=6)
        self.assertAlmostEqual(ema(0.0, 1.0, 1.0), 1.0, places=6)

    def test_angle_between_degrees(self) -> None:
        self.assertAlmostEqual(angle_between_deg((1, 0), (1, 0)), 0.0, places=6)
        self.assertAlmostEqual(angle_between_deg((1, 0), (-1, 0)), 180.0, places=6)
        self.assertAlmostEqual(angle_between_deg((1, 0), (0, 1)), 90.0, places=6)
        # a zero-length vector means "direction unknown" -> full change
        self.assertAlmostEqual(angle_between_deg((0, 0), (1, 0)), 180.0, places=6)


class TestMotionFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = MotionAnalyzer(MotionConfig())

    def test_not_enough_history_is_invalid(self) -> None:
        track = make_track(positions=[(0.0, 0.0), (5.0, 0.0)])
        features = self.analyzer._analyze_track(track, SCALE, 1 / FPS)
        self.assertFalse(features.valid)
        self.assertEqual(features.score, 0.0)
        self.assertIn("insufficient tracking history", features.reasons)

    def test_constant_motion_scores_low(self) -> None:
        track = make_track(positions=moving_positions((100.0, 100.0), (120.0, 0.0), 8, FPS))
        features = self.analyzer._analyze_track(track, SCALE, 7 / FPS)
        self.assertTrue(features.valid)
        self.assertLess(features.sudden_stop, 0.2)
        self.assertLess(features.sudden_turn, 0.2)
        self.assertLess(features.score, 0.25)
        self.assertTrue(features.moving)

    def test_sudden_stop_scores_high(self) -> None:
        positions = deceleration_positions((100.0, 100.0), (150.0, 0.0), stop_after=4, count=8, fps=FPS)
        track = make_track(positions=positions)
        analyzer = MotionAnalyzer(MotionConfig())
        peak = 0.0
        for index in range(8):
            features = analyzer._analyze_track(track, SCALE, index / FPS)
            peak = max(peak, features.sudden_stop, features.score)
        self.assertGreater(peak, 0.5, "a hard stop must produce strong motion evidence")
        self.assertIn("Sudden stop after sustained motion", self.analyzer._analyze_track(track, SCALE, 7 / FPS).reasons)

    def test_sudden_turn_scores_high(self) -> None:
        straight = moving_positions((100.0, 100.0), (150.0, 0.0), 5, FPS)
        after = moving_positions(straight[-1], (0.0, 150.0), 4, FPS)
        track = make_track(positions=straight + after[1:])
        analyzer = MotionAnalyzer(MotionConfig())
        peak = 0.0
        for index in range(5, 9):
            features = analyzer._analyze_track(track, SCALE, index / FPS)
            peak = max(peak, features.sudden_turn)
        self.assertGreater(peak, 0.4, "a 90 degree change of direction must be detected")

    def test_stationary_object_has_no_speed(self) -> None:
        track = make_track(positions=[(100.0, 100.0)] * 8)
        features = self.analyzer._analyze_track(track, SCALE, 7 / FPS)
        self.assertEqual(features.speed, 0.0)
        self.assertFalse(features.moving)
        self.assertLess(features.score, 0.2)

    def test_aggregate_uses_vehicle_tracks(self) -> None:
        """A pedestrian's erratic motion must not drive the vehicle aggregate."""
        car = make_track(
            track_id=1, positions=deceleration_positions((100.0, 200.0), (150.0, 0.0), 4, 8, FPS)
        )
        person = make_track(
            track_id=2,
            positions=deceleration_positions((300.0, 60.0), (0.0, 220.0), 2, 8, FPS),
            class_name="person",
        )
        tracks = [car, person]
        for index in range(8):
            motion = self.analyzer.analyze(tracks, FRAME_SIZE, index / FPS)
        self.assertEqual(self.analyzer.aggregate_ids(motion, tracks), [1], "only the vehicle counts")
        aggregated = self.analyzer.aggregate(motion, tracks)
        self.assertLessEqual(aggregated, motion[1].score + 1e-9)

    def test_empty_input(self) -> None:
        self.assertEqual(self.analyzer.analyze([], FRAME_SIZE, 0.0), {})
        self.assertEqual(self.analyzer.aggregate({}, []), 0.0)
        self.assertEqual(self.analyzer.aggregate_ids({}, []), [])

    def test_reset_clears_smoothing(self) -> None:
        track = make_track(positions=deceleration_positions((100.0, 100.0), (150.0, 0.0), 4, 8, FPS))
        self.analyzer.analyze([track], FRAME_SIZE, 7 / FPS)
        self.assertTrue(self.analyzer._ema)
        self.analyzer.reset()
        self.assertFalse(self.analyzer._ema)


if __name__ == "__main__":
    unittest.main()
