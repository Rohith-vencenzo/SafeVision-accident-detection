"""Accident score (Phase 11) and severity (Phase 13)."""

from __future__ import annotations

import unittest

from ai_engine.config.classes import ScoringConfig, SeverityConfig
from ai_engine.schemas.enums import SeverityLevel
from ai_engine.scoring import AccidentScorer
from ai_engine.severity import SeverityAnalyzer, SeverityContext

STRONG = {
    "collision": 0.9,
    "motion": 0.8,
    "trajectory": 0.6,
    "temporal": 0.7,
    "object_evidence": 0.5,
    "scene_evidence": 0.4,
}
WEAK = {key: 0.05 for key in STRONG}


class TestAccidentScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = AccidentScorer(ScoringConfig())

    def test_strong_evidence_scores_high(self) -> None:
        score = self.scorer.score(STRONG)
        self.assertGreater(score.total, 0.6)
        self.assertEqual(score.confidence_percent, int(round(score.total * 100)))
        self.assertEqual(score.penalty_factor, 1.0)

    def test_weak_evidence_scores_low(self) -> None:
        self.assertLess(self.scorer.score(WEAK).total, 0.15)

    def test_empty_input_is_zero(self) -> None:
        self.assertEqual(self.scorer.score({}).total, 0.0)

    def test_missing_component_is_excluded_not_zeroed(self) -> None:
        """A component that could not be measured must not drag the score down."""
        partial = dict(STRONG)
        partial.pop("trajectory")
        with_partial = self.scorer.score(partial)
        self.assertIn("trajectory", with_partial.missing_components)
        self.assertGreater(with_partial.total, self.scorer.score(STRONG).total - 1e-9)
        self.assertNotIn("trajectory", with_partial.components)

    def test_negative_evidence_reduces_the_score(self) -> None:
        clean = self.scorer.score(STRONG)
        penalised = self.scorer.score(STRONG, {"parked_vehicles": 1.0, "single_frame_event": 1.0})
        self.assertLess(penalised.total, clean.total)
        self.assertLess(penalised.penalty_factor, 1.0)
        self.assertIn("parked_vehicles", penalised.penalties)
        self.assertIn("Parked vehicles only (no meaningful movement change)", penalised.reasons)

    def test_penalties_are_capped(self) -> None:
        config = ScoringConfig(penalty_max=0.5)
        scorer = AccidentScorer(config)
        everything = {name: 1.0 for name in config.penalties}
        score = scorer.score(STRONG, everything)
        self.assertGreaterEqual(score.penalty_factor, 1.0 - config.penalty_max - 1e-9)
        self.assertGreaterEqual(score.total, 0.0)

    def test_negative_evidence_can_be_disabled(self) -> None:
        config = ScoringConfig(apply_negative_evidence=False)
        scorer = AccidentScorer(config)
        score = scorer.score(STRONG, {"parked_vehicles": 1.0})
        self.assertEqual(score.penalty_factor, 1.0)

    def test_score_is_bounded(self) -> None:
        everything = {key: 5.0 for key in STRONG}
        self.assertLessEqual(self.scorer.score(everything).total, 1.0)

    def test_physical_score_excludes_temporal(self) -> None:
        score = self.scorer.score(STRONG)
        self.assertLessEqual(score.physical, 1.0)
        without_temporal = dict(STRONG)
        without_temporal["temporal"] = 0.0
        self.assertEqual(self.scorer.score(without_temporal).physical, score.physical)

    def test_payload_is_json_safe(self) -> None:
        import json

        json.dumps(self.scorer.score(STRONG, {"parked_vehicles": 0.5}).to_dict())


class TestSeverity(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = SeverityAnalyzer(SeverityConfig())

    def test_empty_scene_is_low(self) -> None:
        result = self.analyzer.analyze(SeverityContext())
        self.assertEqual(result.level, SeverityLevel.LOW)
        self.assertEqual(result.score, 0.0)

    def test_high_energy_crash_is_critical_or_high(self) -> None:
        context = SeverityContext(
            vehicles_involved=3,
            persons_in_scene=1,
            impact_speed=0.5,
            deceleration=0.6,
            displacement=0.08,
            blocked_ratio=1.0,
            blocked_seconds=6.0,
            abnormal_duration=8.0,
        )
        result = self.analyzer.analyze(context)
        self.assertIn(result.level, (SeverityLevel.HIGH, SeverityLevel.CRITICAL))
        self.assertGreater(result.score, 0.6)

    def test_minor_fender_bender_is_low_or_medium(self) -> None:
        result = self.analyzer.analyze(
            SeverityContext(vehicles_involved=2, impact_speed=0.02, displacement=0.001)
        )
        self.assertIn(result.level, (SeverityLevel.LOW, SeverityLevel.MEDIUM))

    def test_person_presence_is_presence_only(self) -> None:
        """The wording must never imply an injury assessment."""
        result = self.analyzer.analyze(SeverityContext(vehicles_involved=2, persons_in_scene=1))
        joined = " ".join(result.reasons).lower()
        self.assertIn("presence only", joined)
        # these would be *claims* about injuries, which the system cannot make
        for forbidden in ("injured", "injury detected", "fatal", "casualty", "triage", "critical condition"):
            self.assertNotIn(forbidden, joined)
        self.assertIn("scene severity", result.disclaimer.lower())
        self.assertIn("not a medical assessment", result.disclaimer.lower())

    def test_smoke_signal_reported_as_unavailable_by_default(self) -> None:
        result = self.analyzer.analyze(SeverityContext(vehicles_involved=2))
        self.assertTrue(any("smoke_fire" in item for item in result.unavailable_signals))
        self.assertNotIn("smoke_fire", result.components)

    def test_external_smoke_signal_is_used_when_supplied(self) -> None:
        result = self.analyzer.analyze(
            SeverityContext(vehicles_involved=2, optional_signals={"smoke_fire": 0.9})
        )
        self.assertIn("smoke_fire", result.components)
        self.assertFalse([s for s in result.unavailable_signals if "smoke_fire" in s])
        self.assertTrue(any("smoke/fire" in r.lower() for r in result.reasons))

    def test_level_thresholds(self) -> None:
        self.assertEqual(self.analyzer.level_for(0.0), SeverityLevel.LOW)
        self.assertEqual(self.analyzer.level_for(0.3), SeverityLevel.MEDIUM)
        self.assertEqual(self.analyzer.level_for(0.6), SeverityLevel.HIGH)
        self.assertEqual(self.analyzer.level_for(0.9), SeverityLevel.CRITICAL)

    def test_payload(self) -> None:
        import json

        payload = self.analyzer.analyze(SeverityContext(vehicles_involved=2)).to_dict()
        json.dumps(payload)
        self.assertIn(payload["severity"], ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
        self.assertIn("disclaimer", payload)


if __name__ == "__main__":
    unittest.main()
