"""Trajectory analysis (Phase 7)."""

from __future__ import annotations

import math
import unittest

from tests.helpers import FRAME_SIZE, deceleration_positions, make_track, moving_positions

from ai_engine.config.classes import TrajectoryConfig
from ai_engine.trajectory import TrajectoryAnalyzer
from ai_engine.utils.geometry import segment_intersection

FPS = 15.0


class TestSegmentGeometry(unittest.TestCase):
    def test_crossing_segments_intersect(self) -> None:
        point = segment_intersection((0, 0), (10, 0), (5, -5), (5, 5))
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 5.0, places=6)
        self.assertAlmostEqual(point[1], 0.0, places=6)

    def test_touching_endpoint_is_not_a_crossing(self) -> None:
        self.assertIsNone(segment_intersection((0, 0), (10, 0), (10, 0), (10, 10)))

    def test_parallel_segments_do_not_intersect(self) -> None:
        self.assertIsNone(segment_intersection((0, 0), (10, 0), (0, 5), (10, 5)))


class TestTrajectoryFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = TrajectoryAnalyzer(TrajectoryConfig())

    def test_straight_path_is_boring(self) -> None:
        track = make_track(positions=moving_positions((100.0, 100.0), (120.0, 0.0), 10, FPS))
        features = self.analyzer._track_features(track, math.hypot(*FRAME_SIZE), 9 / FPS)
        self.assertTrue(features.valid)
        self.assertGreater(features.straightness, 0.95)
        self.assertLess(features.heading_deviation_deg, 5.0)
        self.assertLess(features.score, 0.25)

    def test_insufficient_points(self) -> None:
        track = make_track(positions=[(0.0, 0.0), (10.0, 0.0)])
        features = self.analyzer._track_features(track, math.hypot(*FRAME_SIZE), 1 / FPS)
        self.assertFalse(features.valid)
        self.assertIn("insufficient trajectory history", features.reasons)

    def test_sharp_deviation_raises_the_score(self) -> None:
        first = moving_positions((60.0, 100.0), (140.0, 0.0), 6, FPS)
        second = moving_positions(first[-1], (0.0, 140.0), 6, FPS)
        track = make_track(positions=first + second[1:])
        features = self.analyzer._track_features(track, math.hypot(*FRAME_SIZE), 11 / FPS)
        self.assertGreater(features.heading_deviation_deg, 40.0)
        self.assertGreater(features.lateral_deviation, 0.0)
        self.assertGreater(features.score, 0.25)
        self.assertIn("Path deviates from established heading", features.reasons)

    def test_path_ending_in_a_stop_is_flagged(self) -> None:
        positions = deceleration_positions((100.0, 100.0), (140.0, 0.0), stop_after=5, count=10, fps=FPS)
        track = make_track(positions=positions)
        features = self.analyzer._track_features(track, math.hypot(*FRAME_SIZE), 9 / FPS)
        self.assertGreater(features.score, 0.2)

    def test_aggregate_returns_the_worst_vehicle(self) -> None:
        calm = make_track(track_id=1, positions=moving_positions((60.0, 60.0), (100.0, 0.0), 10, FPS))
        wild = make_track(
            track_id=2,
            positions=deceleration_positions((300.0, 300.0), (0.0, -160.0), 5, 10, FPS),
        )
        tracks = [calm, wild]
        per_track, _ = self.analyzer.analyze(tracks, FRAME_SIZE, 9 / FPS)
        aggregate = self.analyzer.aggregate(per_track, tracks)
        self.assertGreaterEqual(aggregate, per_track[2].score)
        self.assertGreaterEqual(aggregate, per_track[1].score)


class TestTrajectoryPairs(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = TrajectoryAnalyzer(TrajectoryConfig())

    def test_crossing_paths_are_reported_but_capped(self) -> None:
        """A T-junction: one vehicle crosses the path of another."""
        # the horizontal path passes through x=330 around frame 8, while the
        # vertical one crosses y=180 around frame 5 -> the segments really meet
        first = make_track(track_id=1, positions=moving_positions((100.0, 180.0), (420.0, 0.0), 12, FPS))
        second = make_track(track_id=2, positions=moving_positions((330.0, 30.0), (0.0, 420.0), 12, FPS))
        _, pairs = self.analyzer.analyze([first, second], FRAME_SIZE, 11 / FPS)
        crossing = [p for p in pairs if p.crossing or p.converging]
        self.assertTrue(crossing, "crossing paths must be reported")
        for pair in crossing:
            # the whole point: a crossing alone can never be strong evidence
            self.assertLessEqual(pair.score, self.analyzer.config.crossing_weight + 1e-9)

    def test_distant_vehicles_produce_no_pair(self) -> None:
        first = make_track(track_id=1, positions=moving_positions((40.0, 60.0), (100.0, 0.0), 10, FPS))
        second = make_track(track_id=2, positions=moving_positions((600.0, 320.0), (0.0, 100.0), 10, FPS))
        _, pairs = self.analyzer.analyze([first, second], FRAME_SIZE, 9 / FPS)
        self.assertEqual([p for p in pairs if p.score > 0.0], [])

    def test_aggregate_pair_with_bonus(self) -> None:
        self.assertEqual(self.analyzer.aggregate_pair([]), 0.0)
        first = make_track(track_id=1, positions=moving_positions((100.0, 180.0), (420.0, 0.0), 12, FPS))
        second = make_track(track_id=2, positions=moving_positions((330.0, 30.0), (0.0, 420.0), 12, FPS))
        third = make_track(track_id=3, positions=moving_positions((100.0, 40.0), (420.0, 0.0), 12, FPS))
        _, pairs = self.analyzer.analyze([first, second, third], FRAME_SIZE, 11 / FPS)
        score = self.analyzer.aggregate_pair(pairs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
