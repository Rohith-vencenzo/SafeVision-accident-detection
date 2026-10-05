"""Temporal fusion (Phase 9)."""

from __future__ import annotations

import unittest

from ai_engine.config.classes import TemporalConfig
from ai_engine.temporal import EvidenceSnapshot, TemporalFusion

FPS = 15.0


def snapshot(index: int, collision: float = 0.0, motion: float = 0.0, trajectory: float = 0.0) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        frame_index=index,
        timestamp=index / FPS,
        collision_score=collision,
        motion_score=motion,
        trajectory_score=trajectory,
    )


class TestComposite(unittest.TestCase):
    def test_all_silent(self) -> None:
        self.assertEqual(snapshot(0).composite, 0.0)

    def test_only_collision(self) -> None:
        self.assertGreater(snapshot(0, collision=1.0).composite, 0.45)

    def test_single_signal_is_discounted(self) -> None:
        """A lone cue must not look as abnormal as agreeing evidence."""
        collision_only = snapshot(0, collision=0.9).composite
        motion_only = snapshot(0, motion=0.9).composite
        two_signals = snapshot(0, collision=0.9, motion=0.8).composite
        three_signals = snapshot(0, collision=0.9, motion=0.9, trajectory=0.9).composite

        # one strong signal is trusted at 60 %
        self.assertAlmostEqual(collision_only, 0.9 * 0.6, places=6)
        self.assertAlmostEqual(motion_only, collision_only, places=6, msg="one signal is one signal")
        # a second *corroborating* signal is rewarded ...
        self.assertGreater(two_signals, collision_only)
        self.assertGreater(three_signals, two_signals)
        # ... while a second *weak* signal legitimately dilutes the mean
        self.assertLess(snapshot(0, collision=0.9, motion=0.1).composite, collision_only)
        self.assertLessEqual(three_signals, 1.0)

    def test_composite_is_bounded(self) -> None:
        value = snapshot(0, collision=1.0, motion=1.0, trajectory=1.0).composite
        self.assertLessEqual(value, 1.0)


class TestTemporalFusion(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TemporalConfig()
        self.fusion = TemporalFusion(self.config)

    def test_empty_window(self) -> None:
        result = self.fusion.aggregate()
        self.assertEqual(result.temporal_score, 0.0)
        self.assertTrue(result.warmup)

    def test_quiet_sequence_stays_quiet(self) -> None:
        for index in range(30):
            result = self.fusion.push(snapshot(index, motion=0.05))
        self.assertLess(result.temporal_score, 0.2)
        self.assertEqual(result.frames_above, 0)

    def test_single_frame_spike_is_suppressed(self) -> None:
        """The headline false-alarm property of the temporal module."""
        for index in range(20):
            self.fusion.push(snapshot(index))
        spike = self.fusion.push(snapshot(20, collision=1.0, motion=1.0, trajectory=1.0))
        self.assertLess(spike.temporal_score, 0.5, "one frame must not look like an accident")
        self.assertTrue(spike.isolated)
        self.assertIn("Single-frame event (no temporal support)", spike.reasons)
        after = self.fusion.push(snapshot(21))
        self.assertLessEqual(after.temporal_score, spike.temporal_score, "the spike must not grow")
        self.assertTrue(after.isolated)

    def test_sustained_evidence_survives_the_same_gate(self) -> None:
        """...while the very first frame of a real burst is also 'isolated'."""
        for index in range(20):
            self.fusion.push(snapshot(index))
        first = self.fusion.push(snapshot(20, collision=0.8, motion=0.6))
        second = self.fusion.push(snapshot(21, collision=0.8, motion=0.6))
        self.assertTrue(first.isolated, "one frame alone has no temporal support")
        self.assertFalse(second.isolated, "two frames in a row start to count")
        self.assertGreater(second.temporal_score, first.temporal_score)

    def test_sustained_evidence_accumulates(self) -> None:
        for index in range(20):
            self.fusion.push(snapshot(index))
        scores = []
        for index in range(20, 40):
            result = self.fusion.push(snapshot(index, collision=0.8, motion=0.6, trajectory=0.4))
            scores.append(result.temporal_score)
        self.assertGreater(max(scores), 0.45, "sustained evidence must reach a confirmable score")
        self.assertGreater(scores[-1], scores[0], "the fused score must keep building")
        self.assertFalse(self.fusion.aggregate().isolated)

    def test_persistence_and_consecutive_counter(self) -> None:
        for index in range(20):
            self.fusion.push(snapshot(index))
        for index in range(20, 30):
            result = self.fusion.push(snapshot(index, collision=0.9, motion=0.7))
        self.assertGreaterEqual(result.consecutive_above, 9)
        self.assertGreaterEqual(result.persistence, 0.9)
        quiet = self.fusion.push(snapshot(30))
        self.assertEqual(quiet.consecutive_above, 0, "the run counter must reset when evidence stops")

    def test_consistency_is_high_for_steady_evidence(self) -> None:
        for index in range(20):
            self.fusion.push(snapshot(index, collision=0.6, motion=0.5))
        result = self.fusion.push(snapshot(20, collision=0.6, motion=0.5))
        self.assertGreater(result.consistency, 0.8)

    def test_window_is_time_bounded(self) -> None:
        for index in range(200):
            self.fusion.push(snapshot(index, motion=0.1))
        self.assertLessEqual(len(self.fusion), 200)
        window = self.fusion.aggregate()
        self.assertLessEqual(window.window_frames, int(self.config.window_seconds * FPS) + 2)

    def test_reset(self) -> None:
        for index in range(10):
            self.fusion.push(snapshot(index, collision=0.9))
        self.fusion.reset()
        self.assertEqual(len(self.fusion), 0)
        self.assertEqual(self.fusion.aggregate().temporal_score, 0.0)

    def test_history_is_serialisable(self) -> None:
        for index in range(5):
            self.fusion.push(snapshot(index, motion=0.2))
        history = self.fusion.snapshot_history()
        self.assertEqual(len(history), 5)
        self.assertIn("composite", history[0])

    def test_reasons_are_reported(self) -> None:
        for index in range(20):
            self.fusion.push(snapshot(index))
        for index in range(20, 30):
            result = self.fusion.push(snapshot(index, collision=0.9, motion=0.8))
        self.assertIn("Persistent abnormal motion across frames", result.reasons)


if __name__ == "__main__":
    unittest.main()
