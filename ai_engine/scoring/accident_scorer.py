"""The transparent, weighted accident score (Phase 11).

    accident_score = sum(weight_i * component_i) / sum(weight_i)      [0, 1]
                     * penalty_factor
                     - small negative-evidence adjustments

Design decisions that matter:

* **Weights are renormalised over the components that exist.**  A component
  that could not be measured this frame (for example "no collision candidate at
  all") is dropped from both numerator and denominator instead of being treated
  as zero, which would silently drag every score down.
* **The weighting is readable.**  ``accident_score`` comes with the exact
  component values, the weights used, the penalties applied and the human
  readable reasons - see :class:`AccidentScore`.
* **Penalties are multiplicative and capped** so negative evidence can reduce
  confidence but can never flip the meaning of a strong multi-signal event on
  its own... it can, in fact, block confirmation - which is intentional: a
  confirmed "accident" among parked, static, low-confidence objects is exactly
  the false alarm we are trying to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ai_engine.config.classes import ScoringConfig
from ai_engine.utils.geometry import clamp

__all__ = ["AccidentScore", "AccidentScorer", "COMPONENT_NAMES"]

#: Canonical component order (also the order in the JSON payload).
COMPONENT_NAMES: Sequence[str] = (
    "collision",
    "motion",
    "trajectory",
    "temporal",
    "object_evidence",
    "scene_evidence",
)


@dataclass
class AccidentScore:
    """Score plus the full breakdown that produced it."""

    total: float = 0.0
    #: weighted mean of the *physical* components only (no temporal memory).
    #: A crash produces a burst of physical evidence followed by a quiet
    #: aftermath, so the state machine uses this to decide whether an event is
    #: *worth verifying*, and the temporal fusion to decide whether it is
    #: *confirmed*.
    physical: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    penalty_factor: float = 1.0
    bonus: float = 0.0
    missing_components: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)

    @property
    def confidence_percent(self) -> int:
        return int(round(self.total * 100))

    def component(self, name: str) -> float:
        return float(self.components.get(name, 0.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "accident_score": round(float(self.total), 4),
            "physical_score": round(float(self.physical), 4),
            "confidence_percent": self.confidence_percent,
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "penalties": {k: round(float(v), 4) for k, v in self.penalties.items()},
            "penalty_factor": round(float(self.penalty_factor), 4),
            "missing_components": list(self.missing_components),
            "reasons": list(self.reasons),
            "negative_evidence": list(self.negative_evidence),
        }


class AccidentScorer:
    """Combines component scores and negative evidence into one number."""

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        self.config = config or ScoringConfig()

    def score(
        self,
        components: Mapping[str, Optional[float]],
        negative_evidence: Optional[Mapping[str, float]] = None,
        reasons: Optional[Iterable[str]] = None,
    ) -> AccidentScore:
        """Compute the weighted score.

        Parameters
        ----------
        components:
            ``{"collision": 0.91, "motion": 0.84, ..., "trajectory": None}``.
            ``None`` (or a missing key) means "not measurable this frame" and is
            excluded from the weighted average.
        negative_evidence:
            ``{"parked_vehicles": 1.0, ...}`` - strengths in ``[0, 1]``; each is
            multiplied by its configured penalty weight.
        reasons:
            Positive evidence phrases contributed by the analyzers.
        """
        cfg = self.config
        weights = cfg.weights or {}
        used: Dict[str, float] = {}
        values: Dict[str, float] = {}
        missing: List[str] = []

        for name in COMPONENT_NAMES:
            raw = components.get(name, None)
            if raw is None:
                missing.append(name)
                continue
            weight = float(weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            values[name] = clamp(float(raw))
            used[name] = weight

        total_weight = sum(used.values())
        if total_weight <= 0.0:
            return AccidentScore(
                total=0.0,
                physical=0.0,
                components={},
                weights=used,
                missing_components=missing,
                reasons=list(reasons or []),
            )

        weighted = sum(values[name] * used[name] for name in values) / total_weight

        # Physical-only mean: identical maths without the temporal term, whose
        # weight is redistributed over the components that were measured now.
        physical_names = [n for n in used if n != "temporal"]
        physical_weight = sum(used[n] for n in physical_names)
        physical = (
            sum(values[n] * used[n] for n in physical_names) / physical_weight
            if physical_weight > 0.0
            else 0.0
        )

        penalties: Dict[str, float] = {}
        factor = 1.0
        if cfg.apply_negative_evidence and negative_evidence:
            for name, strength in negative_evidence.items():
                penalty_weight = float(cfg.penalties.get(name, 0.0))
                if penalty_weight <= 0.0:
                    continue
                strength_value = clamp(float(strength))
                if strength_value <= 0.0:
                    continue
                penalty = penalty_weight * strength_value
                penalties[name] = round(penalty, 4)
                factor -= penalty
            factor = max(1.0 - float(cfg.penalty_max), factor)

        total = clamp(weighted * max(0.0, factor))

        negative_names = [n for n, p in penalties.items() if p > 0.0]
        all_reasons = list(reasons or [])
        if penalties:
            all_reasons = all_reasons + [_NEGATIVE_REASON.get(n, n) for n in negative_names]

        return AccidentScore(
            total=total,
            physical=clamp(physical * max(0.0, factor)),
            components=values,
            weights=used,
            penalties=penalties,
            penalty_factor=round(max(0.0, factor), 4),
            missing_components=missing,
            reasons=all_reasons,
            negative_evidence=negative_names,
        )


#: Human readable phrasing for each negative-evidence key.
_NEGATIVE_REASON = {
    "parked_vehicles": "Parked vehicles only (no meaningful movement change)",
    "stable_scene": "Objects stable, no motion change",
    "no_movement_change": "No meaningful movement change detected",
    "single_frame_event": "Single-frame event (no temporal support)",
    "tracking_unstable": "Unstable tracking",
    "low_detection_confidence": "Low detection confidence",
    "insufficient_history": "Insufficient tracking history",
    "occlusion_conflict": "Occlusion / conflicting assignment",
}
