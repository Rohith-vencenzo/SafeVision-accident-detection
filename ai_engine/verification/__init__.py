"""False-alarm prevention and the verification state machine (Phases 10 + 12)."""

from ai_engine.verification.false_alarm_prevention import (
    FalseAlarmPrevention,
    NegativeEvidence,
    StateTransition,
    VerificationResult,
    assess_negative_evidence,
)

__all__ = [
    "FalseAlarmPrevention",
    "NegativeEvidence",
    "StateTransition",
    "VerificationResult",
    "assess_negative_evidence",
]
