"""False-alarm prevention and the verification state machine (Phases 10 + 12)."""

from __future__ import annotations

import unittest

from ai_engine.config.classes import VerificationConfig
from ai_engine.schemas.enums import DetectionState, IncidentStatus
from ai_engine.scoring import AccidentScorer
from ai_engine.config.classes import ScoringConfig
from ai_engine.temporal import TemporalResult
from ai_engine.verification import FalseAlarmPrevention, NegativeEvidence, assess_negative_evidence
from tests.helpers import make_track, moving_positions

FPS = 15.0


def score(value: float, physical: float = None):
    scorer = AccidentScorer(ScoringConfig())
    return scorer.score(
        {
            "collision": value,
            "motion": value,
            "trajectory": value,
            "temporal": value,
            "object_evidence": value,
            "scene_evidence": value,
        }
    )


def temporal(value: float, consecutive: int = 10, isolated: bool = False, persistence: float = 1.0) -> TemporalResult:
    return TemporalResult(
        temporal_score=value,
        consecutive_above=consecutive,
        persistence=persistence,
        window_frames=40,
        window_fill=0.9,
        frames_above=consecutive,
        isolated=isolated,
    )


def drive(machine: FalseAlarmPrevention, frames: int, value: float, temporal_value: float,
          start: int = 0, negative: NegativeEvidence = None, signals: int = 3, vehicles: int = 2):
    """Feed ``frames`` frames of constant evidence into the machine."""
    last = None
    for index in range(start, start + frames):
        last = machine.update(
            score=score(value),
            temporal=temporal(temporal_value),
            negative=negative,
            vehicles_involved=vehicles,
            agreement_signals=signals,
            timestamp=index / FPS,
        )
    return last


class TestStateMachine(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VerificationConfig()
        self.machine = FalseAlarmPrevention(self.config)
        self.machine.min_consecutive_frames = 3

    def test_starts_normal(self) -> None:
        self.assertIs(self.machine.state, DetectionState.NORMAL)
        self.assertEqual(self.machine.confirmed_count, 0)
        self.assertEqual(self.machine.false_alarm_count, 0)

    def test_quiet_traffic_never_leaves_normal(self) -> None:
        result = drive(self.machine, 40, 0.05, 0.05)
        self.assertIs(self.machine.state, DetectionState.NORMAL)
        self.assertEqual(self.machine.confirmed_count, 0)
        self.assertEqual(result.status, IncidentStatus.NORMAL)

    def test_full_confirmation_path(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        self.assertIs(self.machine.state, DetectionState.NORMAL)
        drive(self.machine, 2, 0.36, 0.40, start=20)
        self.assertIs(self.machine.state, DetectionState.SUSPICIOUS)
        drive(self.machine, 3, 0.60, 0.60, start=22)
        self.assertIs(self.machine.state, DetectionState.VERIFYING)
        result = drive(self.machine, 30, 0.45, 0.60, start=25)
        self.assertIs(self.machine.state, DetectionState.CONFIRMED_ACCIDENT)
        self.assertEqual(self.machine.confirmed_count, 1)
        self.assertEqual(result.status, IncidentStatus.CONFIRMED_ACCIDENT)
        states = [t.to_state for t in self.machine.transitions]
        self.assertEqual(
            states,
            [
                DetectionState.SUSPICIOUS.value,
                DetectionState.VERIFYING.value,
                DetectionState.CONFIRMED_ACCIDENT.value,
            ],
        )
        for transition in self.machine.transitions:
            self.assertTrue(transition.reason, "every transition must explain itself")

    def test_verification_releases_into_false_alarm(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.60, 0.60, start=20)
        result = drive(self.machine, 40, 0.05, 0.05, start=24)
        self.assertIs(self.machine.state, DetectionState.FALSE_ALARM)
        self.assertEqual(self.machine.false_alarm_count, 1)
        self.assertEqual(result.status, IncidentStatus.FALSE_ALARM)
        reasons = [t.reason for t in self.machine.transitions if t.to_state == DetectionState.FALSE_ALARM.value]
        self.assertTrue(reasons, "the false-alarm transition must carry a reason")

    def test_false_alarm_returns_to_normal(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.60, 0.60, start=20)
        drive(self.machine, 40, 0.05, 0.05, start=24)
        self.assertIs(self.machine.state, DetectionState.FALSE_ALARM)
        # the false-alarm cooldown (6 s) must elapse before NORMAL resumes
        result = drive(self.machine, 120, 0.05, 0.05, start=64)
        self.assertIs(self.machine.state, DetectionState.NORMAL)
        self.assertEqual(result.status, IncidentStatus.NORMAL)

    def test_single_vehicle_can_never_confirm(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        drive(self.machine, 40, 0.70, 0.70, start=24, vehicles=1, signals=3)
        self.assertIsNot(self.machine.state, DetectionState.CONFIRMED_ACCIDENT)
        self.assertEqual(self.machine.confirmed_count, 0)

    def test_single_signal_can_never_confirm(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        result = drive(self.machine, 40, 0.70, 0.70, start=24, signals=1)
        self.assertIsNot(self.machine.state, DetectionState.CONFIRMED_ACCIDENT)
        self.assertTrue(any("agreeing physical signal" in r for r in result.blocking_reasons))

    def test_parked_evidence_blocks_confirmation(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        result = drive(
            self.machine, 40, 0.70, 0.70, start=24,
            negative=NegativeEvidence(parked_vehicles=1.0), signals=1,
        )
        self.assertIsNot(self.machine.state, DetectionState.CONFIRMED_ACCIDENT)
        self.assertTrue(any("parked" in r for r in result.blocking_reasons))

    def test_single_frame_event_blocks_confirmation(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        result = drive(
            self.machine, 40, 0.70, 0.70, start=24,
            negative=NegativeEvidence(single_frame_event=1.0), signals=1,
        )
        self.assertIsNot(self.machine.state, DetectionState.CONFIRMED_ACCIDENT)
        self.assertIn("event is supported by a single frame only", result.blocking_reasons)

    def test_low_temporal_score_blocks_confirmation(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        drive(self.machine, 40, 0.70, 0.15, start=24)  # strong burst, no memory
        self.assertIsNot(self.machine.state, DetectionState.CONFIRMED_ACCIDENT)

    def test_cooldown_prevents_retriggering(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        drive(self.machine, 40, 0.70, 0.70, start=24)
        self.assertEqual(self.machine.confirmed_count, 1)
        first = self.machine.state
        # keep the scene abnormal past the cooldown: no second incident
        drive(self.machine, 60, 0.70, 0.70, start=64)
        self.assertEqual(self.machine.confirmed_count, 1, "one event must produce one incident")
        self.assertIsNotNone(first)

    def test_progress_is_reported(self) -> None:
        drive(self.machine, 20, 0.05, 0.05)
        drive(self.machine, 4, 0.70, 0.70, start=20)
        result = drive(self.machine, 3, 0.55, 0.55, start=24)
        if result.state is DetectionState.VERIFYING:
            self.assertGreater(result.progress, 0.0)
            self.assertLessEqual(result.progress, 1.0)

    def test_reset(self) -> None:
        drive(self.machine, 20, 0.70, 0.70)
        self.machine.reset()
        self.assertIs(self.machine.state, DetectionState.NORMAL)
        self.assertEqual(self.machine.transitions, [])

    def test_status_mapping(self) -> None:
        self.assertEqual(
            FalseAlarmPrevention.status_for(DetectionState.CONFIRMED_ACCIDENT),
            IncidentStatus.CONFIRMED_ACCIDENT,
        )
        self.assertEqual(
            FalseAlarmPrevention.status_for(DetectionState.VERIFYING),
            IncidentStatus.POSSIBLE_ACCIDENT,
        )
        self.assertEqual(
            FalseAlarmPrevention.status_for(DetectionState.SUSPICIOUS),
            IncidentStatus.POSSIBLE_ACCIDENT,
        )
        self.assertEqual(FalseAlarmPrevention.status_for(DetectionState.NORMAL), IncidentStatus.NORMAL)


class TestNegativeEvidence(unittest.TestCase):
    def test_parked_cars_are_negative_evidence(self) -> None:
        # enough history that "stopped for 2 s" is actually measurable
        tracks = [make_track(1, positions=[(200.0, 180.0)] * 40), make_track(2, positions=[(215.0, 182.0)] * 40)]
        for track in tracks:
            self.assertTrue(track.has_been_stopped_for(2.0, 0.012))
        evidence = assess_negative_evidence(tracks, {}, 0.9, temporal(0.05, consecutive=0))
        self.assertGreater(evidence.parked_vehicles, 0.5)
        self.assertIn("parked_vehicles", evidence.strengths())

    def test_moving_cars_are_not_negative_evidence(self) -> None:
        tracks = [make_track(1, positions=moving_positions((60.0, 180.0), (120.0, 0.0), 10, FPS))]
        evidence = assess_negative_evidence(tracks, {}, 0.9, temporal(0.2, consecutive=3))
        self.assertEqual(evidence.parked_vehicles, 0.0)

    def test_isolated_event_is_negative_evidence(self) -> None:
        tracks = [make_track(1, positions=moving_positions((60.0, 180.0), (120.0, 0.0), 10, FPS))]
        evidence = assess_negative_evidence(
            tracks, {}, 0.9, temporal(0.1, consecutive=0, isolated=True, persistence=0.0)
        )
        self.assertGreater(evidence.single_frame_event, 0.5)

    def test_crash_aftermath_is_not_treated_as_parked(self) -> None:
        """After a crash the vehicles ARE stationary - that must not veto it."""
        tracks = [make_track(1, positions=[(200.0, 180.0)] * 40), make_track(2, positions=[(215.0, 182.0)] * 40)]
        evidence_by_signals = assess_negative_evidence(
            tracks, {}, 0.9, temporal(0.1, consecutive=1), physical_signals=3
        )
        evidence_by_memory = assess_negative_evidence(
            tracks, {}, 0.9, temporal(0.6, consecutive=20), physical_signals=0
        )
        self.assertEqual(evidence_by_signals.parked_vehicles, 0.0)
        self.assertEqual(evidence_by_memory.parked_vehicles, 0.0)

    def test_no_vehicles_means_no_evidence(self) -> None:
        self.assertEqual(assess_negative_evidence([], {}, 0.0, temporal(0.0)).strengths(), {})

    def test_strengths_are_bounded(self) -> None:
        evidence = NegativeEvidence(parked_vehicles=5.0, stable_scene=-1.0)
        self.assertEqual(evidence.strengths()["parked_vehicles"], 1.0)
        self.assertNotIn("stable_scene", evidence.strengths())


if __name__ == "__main__":
    unittest.main()
