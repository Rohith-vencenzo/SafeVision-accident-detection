"""Collision analysis (Phase 8) - the multi-signal gate."""

from __future__ import annotations

import math
import unittest

from tests.helpers import FRAME_SIZE, deceleration_positions, make_track, moving_positions

from ai_engine.collision import CollisionAnalyzer
from ai_engine.config.classes import CollisionConfig
from ai_engine.motion import MotionAnalyzer
from ai_engine.config.classes import MotionConfig

FPS = 15.0


def head_on_positions(impact: int = 8, count: int = 12, fps: float = FPS, speed: float = 150.0, gap: float = 190.0):
    """Two vehicles that close on each other and both stop, in contact.

    Physically plausible: constant approach, then a 3-frame braking ramp, so the
    last positions overlap and both vehicles come to rest.
    """

    def travel(start: float, sign: float):
        points = []
        for index in range(count):
            step = 1.0 if index < impact else max(0.0, 1.0 - (index - impact) / 3.0)
            start += sign * speed / fps * step
            points.append((start, 180.0))
        return points

    return travel(320.0 - gap / 2.0, +1.0), travel(320.0 + gap / 2.0, -1.0)


def crash_pair():
    """Two vehicles that close on each other and both stop: a real collision."""
    a_positions, b_positions = head_on_positions()
    return [
        make_track(track_id=1, positions=a_positions),
        make_track(track_id=2, positions=b_positions),
    ]


def separated_pair():
    """Two vehicles in different lanes, never interacting."""
    first = make_track(track_id=1, positions=moving_positions((40.0, 60.0), (120.0, 0.0), 12, FPS))
    second = make_track(track_id=2, positions=moving_positions((600.0, 300.0), (-120.0, 0.0), 12, FPS))
    return [first, second]


def parked_pair():
    """Two static vehicles, heavily overlapping in the image."""
    first = make_track(track_id=1, positions=[(200.0, 180.0)] * 12)
    second = make_track(track_id=2, positions=[(215.0, 182.0)] * 12)
    return [first, second]


class TestCollisionScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = CollisionAnalyzer(CollisionConfig())
        self.motion = MotionAnalyzer(MotionConfig())

    def _analyze(self, tracks):
        motion = self.motion.analyze(tracks, FRAME_SIZE, tracks[0].timestamp)
        best, events = self.analyzer.analyze(tracks, motion, [], timestamp=tracks[0].timestamp, frame_index=0)
        return best, events, motion

    def test_crash_produces_agreeing_signals(self) -> None:
        best, events, _ = self._analyze(crash_pair())
        self.assertIsNotNone(best, "a head-on collision must be detected")
        self.assertGreaterEqual(best.agreement, self.analyzer.config.min_signals)
        self.assertIn("deceleration", best.signals.active)
        self.assertGreater(best.score, 0.4)
        self.assertTrue(best.reasons)

    def test_overlap_alone_never_confirms(self) -> None:
        """The core false-alarm rule: parked, overlapping boxes -> no collision."""
        best, events, _ = self._analyze(parked_pair())
        for event in events:
            self.assertLessEqual(
                event.signals.closing, 0.0, "static objects cannot be closing"
            )
            self.assertLessEqual(event.signals.pair_deceleration, 0.0, "static objects cannot decelerate")
        if best is not None:
            self.assertLess(best.score, 0.3, "overlap of parked cars must stay a low score")

    def test_separated_vehicles_produce_no_candidate(self) -> None:
        best, events, _ = self._analyze(separated_pair())
        self.assertIsNone(best, "distant vehicles must never produce a best candidate")
        for event in events:
            self.assertLess(event.score, 0.1, "any residual signal must stay negligible")

    def test_agreement_gate_blocks_single_signal(self) -> None:
        """One signal must never be enough, whatever its strength."""
        tracks = crash_pair()
        # freeze the second vehicle: closing + deceleration can no longer agree
        tracks[1] = make_track(track_id=2, positions=[(400.0, 180.0)] * 12)
        best, events, _ = self._analyze(tracks)
        if best is not None:
            self.assertLess(best.agreement, 2)
            self.assertLess(best.score, 0.35)

    def test_signals_are_reported_in_the_payload(self) -> None:
        best, _, _ = self._analyze(crash_pair())
        self.assertIsNotNone(best)
        payload = best.to_dict()
        self.assertIn("signals", payload)
        self.assertIn("signals_active", payload["signals"])
        self.assertIn("agreement_signals", payload)
        self.assertEqual(payload["track_a"], 1)
        self.assertEqual(payload["track_b"], 2)

    def test_context_signals_are_capped(self) -> None:
        """Proximity/overlap alone can never reach a confirmable score."""
        cfg = self.analyzer.config
        proximity_only = cfg.physical_weight * 0.0 + cfg.context_weight * 1.0
        self.assertLess(proximity_only, 0.3)

    def test_physical_weight_dominates(self) -> None:
        cfg = self.analyzer.config
        self.assertGreater(cfg.physical_weight, cfg.context_weight)

    def test_reset_clears_impact_memory(self) -> None:
        self._analyze(crash_pair())
        self.assertTrue(self.analyzer._impact_memory)
        self.analyzer.reset()
        self.assertFalse(self.analyzer._impact_memory)

    def test_empty_input(self) -> None:
        best, events = self.analyzer.analyze([], {}, [], timestamp=0.0)
        self.assertIsNone(best)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
