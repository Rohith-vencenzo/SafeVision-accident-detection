"""AI-estimated scene severity (Phase 13).

Terminology matters here.  What this module produces is an **AI-estimated scene
severity** derived from visible motion evidence:

* how many vehicles are involved
* collision intensity proxy (relative closing speed before the impact)
* how abruptly the vehicles changed velocity
* how far a vehicle was displaced
* whether people are present in the scene
* whether traffic is blocked
* how long the scene has stayed abnormal

What it does **not** do:

* it does not detect injuries, and it does not claim a medical triage level
* it does not measure damage
* it does not detect fire or smoke on its own - see
  ``severity.optional_signal_provider``: if the team supplies such a model the
  pipeline forwards its output, otherwise the signal is reported as
  ``not available`` and contributes nothing.

Every severity result therefore carries the wording ``AI-estimated scene
severity`` and an explicit list of the reasons behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from ai_engine.config.classes import SeverityConfig
from ai_engine.schemas.enums import SeverityLevel
from ai_engine.utils.geometry import clamp, smoothstep

__all__ = ["SeverityAnalysis", "SeverityAnalyzer", "SeverityContext", "SEVERITY_DISCLAIMER"]

SEVERITY_DISCLAIMER = (
    "AI-estimated scene severity from multi-frame video evidence. "
    "Not a medical assessment: no injury detection, triage or damage estimate is performed."
)


@dataclass
class SeverityContext:
    """Inputs for one severity assessment."""

    vehicles_involved: int = 0
    persons_in_scene: int = 0
    #: closing speed of the collision pair, frame-diagonals / second
    impact_speed: float = 0.0
    #: larger of the two decelerations, object-diagonals / s^2
    deceleration: float = 0.0
    #: largest unexplained displacement spike, object-diagonals
    displacement: float = 0.0
    #: stopped vehicles in the analysed area / total vehicles
    blocked_ratio: float = 0.0
    blocked_seconds: float = 0.0
    #: how long the scene has been abnormal
    abnormal_duration: float = 0.0
    #: optional outputs from an external model (e.g. a fire/smoke classifier)
    optional_signals: Dict[str, float] = field(default_factory=dict)
    optional_provider: Optional[str] = None


@dataclass
class SeverityAnalysis:
    """Result of :meth:`SeverityAnalyzer.analyze`."""

    level: SeverityLevel = SeverityLevel.LOW
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    unavailable_signals: List[str] = field(default_factory=list)
    disclaimer: str = SEVERITY_DISCLAIMER

    @property
    def label(self) -> str:
        return str(self.level)

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": str(self.level),
            "severity_score": round(float(self.score), 4),
            "severity_components": {k: round(float(v), 4) for k, v in self.components.items()},
            "severity_reasons": list(self.reasons),
            "unavailable_signals": list(self.unavailable_signals),
            "disclaimer": self.disclaimer,
        }


class SeverityAnalyzer:
    """Maps physical evidence onto LOW / MEDIUM / HIGH / CRITICAL."""

    def __init__(self, config: Optional[SeverityConfig] = None) -> None:
        self.config = config or SeverityConfig()

    # ------------------------------------------------------------------ #
    def analyze(self, context: SeverityContext) -> SeverityAnalysis:
        cfg = self.config
        components: Dict[str, float] = {}

        # 1. number of vehicles involved ----------------------------------
        scale = max(1, int(cfg.max_vehicles_for_scale))
        components["vehicles_involved"] = clamp(context.vehicles_involved / scale)

        # 2. collision intensity proxy ------------------------------------
        # Closing speed before impact is the only physically meaningful proxy
        # available from video; it is labelled as a *proxy* everywhere.
        components["impact_intensity"] = clamp(
            smoothstep(cfg.min_impact_speed, cfg.min_impact_speed * 3.0, context.impact_speed)
        )

        # 3. deceleration ---------------------------------------------------
        components["deceleration"] = clamp(
            smoothstep(cfg.min_decel, cfg.min_decel * 4.0, context.deceleration)
        )

        # 4. displacement --------------------------------------------------
        components["displacement"] = clamp(
            smoothstep(cfg.min_displacement, cfg.min_displacement * 5.0, context.displacement)
        )

        # 5. people present -------------------------------------------------
        # Presence only. This is *not* an injury signal.
        components["persons_present"] = clamp(context.persons_in_scene / 3.0)

        # 6. traffic blockage ------------------------------------------------
        components["traffic_blockage"] = clamp(
            smoothstep(0.0, 0.6, context.blocked_ratio) * smoothstep(0.0, cfg.blockage_seconds, context.blocked_seconds)
        )

        # 7. duration of the abnormal scene -----------------------------------
        components["abnormal_duration"] = clamp(
            smoothstep(0.0, max(1e-3, cfg.duration_weight_seconds), context.abnormal_duration)
        )

        unavailable: List[str] = []
        # 8. optional external signals (smoke / fire / debris) -----------------
        smoke = context.optional_signals.get("smoke_fire") if context.optional_signals else None
        if smoke is None:
            unavailable.append("smoke_fire (no optional signal provider configured)")
        else:
            components["smoke_fire"] = clamp(float(smoke))

        # ---- weighted total -------------------------------------------------
        weights = dict(cfg.weights)
        if "smoke_fire" in components:
            # the optional signal shares the weight of the blockage term
            weights["smoke_fire"] = weights.get("traffic_blockage", 0.10) * 0.5
        total_w = sum(float(w) for w in weights.values() if w and float(w) > 0.0)
        if total_w <= 0.0:
            return SeverityAnalysis(level=SeverityLevel.LOW, score=0.0, components=components, unavailable_signals=unavailable)

        score = sum(float(weights.get(name, 0.0)) * value for name, value in components.items()) / total_w
        score = clamp(score)
        level = self.level_for(score)
        reasons = self._reasons(context, components, level)
        return SeverityAnalysis(
            level=level,
            score=score,
            components=components,
            reasons=reasons,
            unavailable_signals=unavailable,
        )

    # ------------------------------------------------------------------ #
    def level_for(self, score: float) -> SeverityLevel:
        """Map a score onto a level using the configured thresholds."""
        cfg = self.config
        value = clamp(float(score))
        if value >= cfg.high_threshold:
            return SeverityLevel.CRITICAL
        if value >= cfg.medium_threshold:
            return SeverityLevel.HIGH
        if value >= cfg.low_threshold:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    @staticmethod
    def _reasons(context: SeverityContext, components: Mapping[str, float], level: SeverityLevel) -> List[str]:
        reasons: List[str] = []
        if context.vehicles_involved >= 2:
            reasons.append(f"{context.vehicles_involved} vehicles involved")
        if components.get("impact_intensity", 0.0) >= 0.5:
            reasons.append("High collision intensity proxy (large closing speed before impact)")
        if components.get("deceleration", 0.0) >= 0.5:
            reasons.append("Abrupt deceleration of the involved vehicles")
        if components.get("displacement", 0.0) >= 0.5:
            reasons.append("Vehicle displaced from its path")
        if context.persons_in_scene > 0:
            reasons.append(
                f"{context.persons_in_scene} person(s) present in the scene (presence only - "
                "no injury inference)"
            )
        if components.get("traffic_blockage", 0.0) >= 0.5:
            reasons.append("Traffic appears blocked by stopped vehicles")
        if components.get("abnormal_duration", 0.0) >= 0.5:
            reasons.append("Scene has remained abnormal for an extended period")
        if components.get("smoke_fire", 0.0) >= 0.5:
            reasons.append("Smoke/fire signal reported by the optional detector")
        if not reasons:
            reasons.append("Limited physical evidence; severity at the lower bound")
        reasons.append(f"AI-estimated scene severity: {str(level)}")
        return reasons
