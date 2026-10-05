"""False-alarm prevention and the verification state machine (Phase 10 + 12).

State machine
-------------

    NORMAL ──(score > suspicious)──> SUSPICIOUS ──(score > verify_entry for N frames)──┐
       ^                                  │                                            │
       │                                  │ (score falls back)                        v
       │                                  └──> NORMAL                            VERIFYING
       │                                                                             │
       ├──(evidence disappeared)── FALSE_ALARM <──(score < release for N frames)───────┤
       │                                                                             │
       └──(cooldown)────────────────────────────────────────────────── CONFIRMED_ACCIDENT

Two rules make this a real defence rather than a formality:

1. **Confirmation needs agreement, not just a high score.**  ``min_vehicles``
   involved tracks, at least ``min_signals`` *independent* physical signals,
   a fused temporal score above ``confirm_score``, at least
   ``min_verification_seconds`` of evidence and at least
   ``temporal.min_consecutive_frames`` consecutive supporting frames.
2. **Negative evidence blocks confirmation.**  Parked objects, a stable scene,
   an isolated single-frame event, unstable tracking or very low detection
   confidence all reduce the score (see
   :mod:`ai_engine.scoring.accident_scorer`) and the state machine refuses to
   confirm while a single-frame event is the only support.

Nothing here decides *that* an accident happened from one frame: the only way
into ``CONFIRMED_ACCIDENT`` is through ``VERIFYING``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ai_engine.config.classes import VerificationConfig
from ai_engine.schemas.enums import DetectionState, IncidentStatus
from ai_engine.scoring.accident_scorer import AccidentScore
from ai_engine.temporal.temporal_fusion import TemporalResult
from ai_engine.utils.geometry import clamp
from ai_engine.utils.logging_utils import get_logger

__all__ = [
    "FalseAlarmPrevention",
    "NegativeEvidence",
    "StateTransition",
    "VerificationResult",
    "assess_negative_evidence",
]

_LOGGER = get_logger("verification")


# --------------------------------------------------------------------------- #
# negative evidence
# --------------------------------------------------------------------------- #
@dataclass
class NegativeEvidence:
    """Reasons the current frame should *not* be treated as an accident.

    Each field is a strength in ``[0, 1]``.
    """

    parked_vehicles: float = 0.0
    stable_scene: float = 0.0
    no_movement_change: float = 0.0
    single_frame_event: float = 0.0
    tracking_unstable: float = 0.0
    low_detection_confidence: float = 0.0
    insufficient_history: float = 0.0
    occlusion_conflict: float = 0.0

    def strengths(self) -> Dict[str, float]:
        """Only the non-zero entries, ready for :meth:`AccidentScorer.score`."""
        out: Dict[str, float] = {}
        for name in (
            "parked_vehicles",
            "stable_scene",
            "no_movement_change",
            "single_frame_event",
            "tracking_unstable",
            "low_detection_confidence",
            "insufficient_history",
            "occlusion_conflict",
        ):
            value = float(getattr(self, name, 0.0))
            if value > 0.0:
                out[name] = clamp(value)
        return out

    def total(self) -> float:
        return clamp(sum(self.strengths().values()) / 2.0)

    def labels(self) -> List[str]:
        from ai_engine.scoring.accident_scorer import _NEGATIVE_REASON

        return [_NEGATIVE_REASON.get(name, name) for name in self.strengths()]


def assess_negative_evidence(
    tracks: Sequence,
    motion: Mapping[int, object],
    detection_confidence: float,
    temporal: TemporalResult,
    involved_track_ids: Sequence[int] = (),
    min_history: int = 4,
    physical_signals: int = 0,
) -> NegativeEvidence:
    """Derive negative evidence from the current frame.

    Parameters
    ----------
    tracks:
        Live tracks (any object with ``track_id`` / ``history_length`` /
        ``speed_norm`` / ``is_vehicle``).
    motion:
        ``{track_id: MotionFeatures}`` - used for the "no movement change" test.
    detection_confidence:
        Mean confidence of the detections that produced these tracks.
    temporal:
        Result of the temporal fusion for this frame.
    involved_track_ids:
        Tracks that the collision analyzer flagged.  When empty, *all* vehicle
        tracks are considered, which is what makes "just parked cars" visible
        as negative evidence.
    physical_signals:
        How many independent physical collision signals fired this frame.

    Important subtlety
    ------------------
    A crash *leaves stopped vehicles behind*, so "the objects are static now" is
    only counter-evidence when nothing else is going on.  Once physical signals
    have fired, or once the fused temporal evidence is high, the stillness is
    the expected aftermath and the penalty is skipped - otherwise every real
    accident would be rejected two seconds after it happened.
    """
    evidence = NegativeEvidence()

    considered = [t for t in tracks if t.is_vehicle]
    if involved_track_ids:
        considered = [t for t in considered if t.track_id in set(involved_track_ids)]
    if not considered:
        return evidence

    aftermath = (
        physical_signals >= 2
        or temporal.temporal_score >= 0.30
    )

    # --- parked / static objects ---------------------------------------
    # A track counts as parked when it has been below the stop threshold for at
    # least 2 seconds.  ``speed_norm`` is the fallback for track objects that do
    # not expose ``stopped_for`` (e.g. lightweight test doubles).
    stationary = [
        t
        for t in considered
        if (
            t.has_been_stopped_for(2.0, 0.012)
            if hasattr(t, "has_been_stopped_for")
            else (
                t.stopped_for(2.0, 0.012) >= 1.8
                if hasattr(t, "stopped_for")
                else float(getattr(t, "speed_norm", 0.0)) < 0.012
            )
        )
    ]
    if not aftermath:
        if considered and len(stationary) == len(considered):
            evidence.parked_vehicles = 1.0
        elif stationary:
            evidence.parked_vehicles = 0.5

    # --- no meaningful movement change ----------------------------------
    features = [motion[t.track_id] for t in considered if t.track_id in motion and getattr(motion[t.track_id], "valid", False)]
    if features and not aftermath:
        max_speed = max((float(f.speed) for f in features), default=0.0)  # type: ignore[attr-defined]
        max_drop = max((float(f.speed_drop_ratio) for f in features), default=0.0)  # type: ignore[attr-defined]
        if max_speed < 0.02 and max_drop < 0.2:
            evidence.no_movement_change = 1.0
            evidence.stable_scene = 0.6
        elif max_speed < 0.04 and max_drop < 0.35:
            evidence.no_movement_change = 0.5

    # --- isolated single-frame event ------------------------------------
    if temporal.isolated or temporal.consecutive_above <= 0:
        evidence.single_frame_event = clamp(0.5 + 0.5 * (1.0 - temporal.persistence))

    # --- tracking quality ------------------------------------------------
    short = [t for t in considered if t.history_length < min_history]
    if short and len(short) == len(considered):
        evidence.insufficient_history = 1.0
    elif short:
        evidence.insufficient_history = 0.4
    lost = [t for t in tracks if getattr(t, "state", "") == "lost"]
    if tracks and len(lost) / len(tracks) > 0.4:
        evidence.tracking_unstable = clamp(len(lost) / len(tracks))
    if features and all(float(f.history_length) < min_history for f in features):  # type: ignore[attr-defined]
        evidence.insufficient_history = max(evidence.insufficient_history, 0.6)

    # --- detection confidence -------------------------------------------
    if detection_confidence <= 0.0:
        evidence.low_detection_confidence = 0.0
    elif detection_confidence < 0.4:
        evidence.low_detection_confidence = clamp((0.4 - detection_confidence) / 0.4)

    return evidence


# --------------------------------------------------------------------------- #
# transitions / results
# --------------------------------------------------------------------------- #
@dataclass
class StateTransition:
    """One state change, kept for the incident report and the demo log."""

    from_state: str
    to_state: str
    timestamp: float
    reason: str
    score: float = 0.0
    frames_in_previous_state: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "from": str(self.from_state),
            "to": str(self.to_state),
            "timestamp": round(float(self.timestamp), 3),
            "reason": self.reason,
            "score": round(float(self.score), 4),
            "frames_in_previous_state": int(self.frames_in_previous_state),
        }


@dataclass
class VerificationResult:
    """Output of one :meth:`FalseAlarmPrevention.update` call."""

    state: DetectionState = DetectionState.NORMAL
    status: IncidentStatus = IncidentStatus.NORMAL
    score: float = 0.0
    physical_score: float = 0.0
    event_score: float = 0.0
    temporal_score: float = 0.0
    progress: float = 0.0                 # 0..1 verification progress
    frames_in_state: int = 0
    seconds_in_state: float = 0.0
    verification_seconds: float = 0.0   # length of the state *before* this frame's transition
    consecutive_above: int = 0
    changed: bool = False
    is_new_incident: bool = False
    is_new_false_alarm: bool = False
    evidence_sufficient: bool = False
    blocking_reasons: List[str] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)
    vehicles_involved: int = 0
    persons_involved: int = 0
    agreement_signals: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "state": str(self.state),
            "status": str(self.status),
            "score": round(float(self.score), 4),
            "physical_score": round(float(self.physical_score), 4),
            "event_score": round(float(self.event_score), 4),
            "temporal_score": round(float(self.temporal_score), 4),
            "verification_progress": round(float(self.progress), 4),
            "frames_in_state": int(self.frames_in_state),
            "consecutive_above_threshold": int(self.consecutive_above),
            "changed": bool(self.changed),
            "evidence_sufficient": bool(self.evidence_sufficient),
            "blocking_reasons": list(self.blocking_reasons),
            "negative_evidence": list(self.negative_evidence),
            "vehicles_involved": int(self.vehicles_involved),
            "persons_involved": int(self.persons_involved),
            "agreement_signals": int(self.agreement_signals),
        }


# --------------------------------------------------------------------------- #
# the state machine
# --------------------------------------------------------------------------- #
class FalseAlarmPrevention:
    """Verification state machine + confirmation gate."""

    def __init__(self, config: Optional[VerificationConfig] = None) -> None:
        self.config = config or VerificationConfig()
        #: Consecutive supporting frames required.  The pipeline keeps this in
        #: sync with ``temporal.min_consecutive_frames`` (Phase 9); the default
        #: is a safe middle ground for a low-fps analysis stream.
        self.min_consecutive_frames: int = 3
        #: The confirmation gate judges the *event*, not a single frame: a crash
        #: produces a burst of evidence followed by a quiet aftermath, so the
        #: raw per-frame score must not be compared against the threshold once
        #: the burst is over.  ``event_score`` holds the peak and releases it
        #: with this time constant.
        self.event_hold_seconds: float = 4.0
        self._event_score = 0.0
        self._event_peak = 0.0
        self._event_peak_total = 0.0
        self._event_started = 0.0
        self._last_timestamp = 0.0
        self.state = DetectionState.NORMAL
        self._frames_in_state = 0
        self._state_since = 0.0
        self._suspicious_run = 0
        self._verify_run = 0
        self._release_run = 0
        self._cooldown_until = -1.0
        self._transitions: List[StateTransition] = []
        self._incident_open = False
        self.false_alarm_count = 0
        self.confirmed_count = 0
        self._event_peak = 0.0
        self._event_peak_total = 0.0
        self._event_started = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = DetectionState.NORMAL
        self._frames_in_state = 0
        self._suspicious_run = 0
        self._verify_run = 0
        self._release_run = 0
        self._cooldown_until = -1.0
        self._transitions.clear()
        self._incident_open = False
        self._event_score = 0.0
        self._event_peak = 0.0
        self._event_peak_total = 0.0
        self._event_started = 0.0
        self._last_timestamp = 0.0

    @property
    def transitions(self) -> List[StateTransition]:
        return list(self._transitions)

    @property
    def event_score(self) -> float:
        """Peak-holding score for the current event (see :attr:`event_hold_seconds`)."""
        return self._event_score

    @property
    def event_peak(self) -> float:
        """Highest *physical* evidence score reached during the current event."""
        return self._event_peak

    @property
    def event_peak_total(self) -> float:
        """Highest total score (incl. temporal) reached during the event."""
        return self._event_peak_total

    # ------------------------------------------------------------------ #
    def update(
        self,
        score: AccidentScore,
        temporal: TemporalResult,
        negative: Optional[NegativeEvidence] = None,
        vehicles_involved: int = 0,
        persons_involved: int = 0,
        agreement_signals: int = 0,
        timestamp: float = 0.0,
    ) -> VerificationResult:
        """Advance the state machine by one frame."""
        cfg = self.config
        negative = negative or NegativeEvidence()
        previous = self.state
        self._frames_in_state += 1
        dt = max(0.0, float(timestamp) - self._last_timestamp) if self._last_timestamp > 0 else 0.0

        # --- event score: hold the peak, release slowly -------------------
        # Entry is judged on the *physical* evidence (`score.physical`), because
        # the temporal memory lags the impact by design.  The accumulator decays
        # on every frame (that is what makes it a decaying memory) and is
        # re-raised immediately by fresh evidence.
        physical = float(score.physical)
        if self._frames_in_state > 1:
            release = math.exp(-dt / max(1e-3, self.event_hold_seconds))
        else:
            release = 0.0
        self._event_score *= release
        if physical >= cfg.suspicious_score:
            if self._event_started <= 0.0:
                self._event_started = float(timestamp)
            self._event_score = max(physical, self._event_score)
            self._event_peak = max(self._event_peak, self._event_score)
        self._event_peak_total = max(self._event_peak_total, float(score.total))
        self._last_timestamp = float(timestamp)

        result = VerificationResult(
            state=self.state,
            score=score.total,
            physical_score=physical,
            event_score=self._event_score,
            temporal_score=temporal.temporal_score,
            consecutive_above=temporal.consecutive_above,
            negative_evidence=negative.labels(),
            vehicles_involved=vehicles_involved,
            persons_involved=persons_involved,
            agreement_signals=agreement_signals,
        )
        # captured before any transition resets the state clock, so the incident
        # records how long verification actually took
        result.verification_seconds = self._seconds_in_state(timestamp)

        sufficient, blocking = self._evidence_sufficient(score, temporal, negative, vehicles_involved, agreement_signals)
        result.evidence_sufficient = sufficient
        result.blocking_reasons = blocking

        # --- run counters -------------------------------------------------
        if physical >= cfg.verify_entry_score:
            self._verify_run += 1
        else:
            self._verify_run = 0

        if physical >= cfg.suspicious_score:
            self._suspicious_run += 1
        else:
            self._suspicious_run = 0

        if score.total < cfg.release_score or temporal.temporal_score < cfg.release_score:
            self._release_run += 1
        else:
            self._release_run = 0

        new_state = self.state
        reason = ""

        if self.state is DetectionState.FALSE_ALARM:
            if self._seconds_in_state(timestamp) >= cfg.false_alarm_cooldown_seconds:
                new_state = DetectionState.NORMAL
                reason = "false-alarm cooldown elapsed"
                self._cooldown_until = timestamp + cfg.cooldown_seconds
            elif self._suspicious_run >= max(1, cfg.suspicious_frames) and physical >= cfg.suspicious_score:
                new_state = DetectionState.SUSPICIOUS
                reason = f"new physical evidence {physical:.2f} after a false alarm"
        elif self.state is DetectionState.NORMAL:
            if self._in_cooldown(timestamp):
                new_state = self.state
                reason = "cooldown active after a recent decision"
            elif self._suspicious_run >= max(1, cfg.suspicious_frames) and physical >= cfg.suspicious_score:
                new_state = DetectionState.SUSPICIOUS
                reason = f"physical evidence {physical:.2f} >= {cfg.suspicious_score:.2f} for {self._suspicious_run} frame(s)"

        elif self.state is DetectionState.SUSPICIOUS:
            if self._verify_run >= max(1, cfg.verify_frames) and physical >= cfg.verify_entry_score:
                new_state = DetectionState.VERIFYING
                reason = f"physical evidence held >= {cfg.verify_entry_score:.2f} for {self._verify_run} frame(s)"
            elif self._release_run >= max(1, cfg.false_alarm_frames):
                new_state = DetectionState.NORMAL
                reason = f"score fell below {cfg.release_score:.2f}"

        elif self.state is DetectionState.VERIFYING:
            can_confirm = (
                sufficient
                and temporal.temporal_score >= cfg.temporal_confirm_score
                and self._event_peak >= cfg.confirm_score
                and self._seconds_in_state(timestamp) >= cfg.min_verification_seconds
                and temporal.consecutive_above >= self.min_consecutive_frames
            )
            if can_confirm:
                new_state = DetectionState.CONFIRMED_ACCIDENT
                reason = (
                    f"multi-frame evidence confirmed: peak physical evidence {self._event_peak:.2f}, "
                    f"temporal {temporal.temporal_score:.2f}, "
                    f"{agreement_signals} agreeing signals, "
                    f"{self._seconds_in_state(timestamp):.1f}s of verification"
                )
            else:
                # A peak that is *only* carried by the event accumulator is not
                # enough: the scene must still look abnormal right now.
                evidence_gone = (
                    score.total < cfg.release_score and temporal.temporal_score < cfg.release_score
                )
                timed_out = (
                    cfg.verify_timeout_seconds > 0
                    and self._seconds_in_state(timestamp) >= cfg.verify_timeout_seconds
                )
                if evidence_gone or self._release_run >= max(1, cfg.false_alarm_frames):
                    new_state = DetectionState.FALSE_ALARM
                    reason = (
                        "verification timed out without sustained evidence"
                        if timed_out
                        else f"evidence disappeared (score < {cfg.release_score:.2f})"
                    )
                    self.false_alarm_count += 1
                    result.is_new_false_alarm = True

        elif self.state is DetectionState.CONFIRMED_ACCIDENT:
            # stay confirmed while the scene is still abnormal, then cool down
            if score.total < cfg.release_score and self._seconds_in_state(timestamp) >= cfg.cooldown_seconds:
                new_state = DetectionState.NORMAL
                reason = "incident closed, returning to NORMAL"
                self._cooldown_until = timestamp + cfg.cooldown_seconds
                self._incident_open = False
            elif self._seconds_in_state(timestamp) >= cfg.cooldown_seconds:
                new_state = DetectionState.NORMAL
                reason = "cooldown elapsed, returning to NORMAL"
                self._cooldown_until = timestamp + cfg.cooldown_seconds
                self._incident_open = False

        elif self.state is DetectionState.FALSE_ALARM:
            if self._seconds_in_state(timestamp) >= cfg.false_alarm_cooldown_seconds:
                new_state = DetectionState.NORMAL
                reason = "false-alarm cooldown elapsed"
                self._cooldown_until = timestamp + cfg.cooldown_seconds

        # --- apply transition ---------------------------------------------
        if new_state is not self.state:
            transition = StateTransition(
                from_state=str(self.state),
                to_state=str(new_state),
                timestamp=float(timestamp),
                reason=reason,
                score=score.total,
                frames_in_previous_state=self._frames_in_state,
            )
            self._transitions.append(transition)
            _LOGGER.info("STATE %s -> %s (%s)", transition.from_state, transition.to_state, reason)
            self.state = new_state
            self._frames_in_state = 0
            self._state_since = float(timestamp)
            result.changed = True
            result.transitions.append(transition)
            if new_state is DetectionState.CONFIRMED_ACCIDENT:
                result.is_new_incident = True
                self.confirmed_count += 1
                self._incident_open = True

        result.state = self.state
        result.status = self.status_for(self.state)
        result.frames_in_state = self._frames_in_state
        result.seconds_in_state = self._seconds_in_state(timestamp)
        result.progress = self._progress(score, temporal, result.seconds_in_state)
        result.transitions = list(self._transitions[-8:])
        return result

    # ------------------------------------------------------------------ #
    def _evidence_sufficient(
        self,
        score: AccidentScore,
        temporal: TemporalResult,
        negative: NegativeEvidence,
        vehicles_involved: int,
        agreement_signals: int,
    ) -> Tuple[bool, List[str]]:
        """The hard gate.  Returns ``(sufficient, blocking_reasons)``."""
        cfg = self.config
        blocking: List[str] = []

        if vehicles_involved < cfg.min_vehicles:
            blocking.append(f"only {vehicles_involved} vehicle(s) involved (need {cfg.min_vehicles})")
        if agreement_signals < cfg.min_signals:
            blocking.append(f"only {agreement_signals} agreeing physical signal(s) (need {cfg.min_signals})")
        if temporal.consecutive_above < self.min_consecutive_frames:
            blocking.append(
                f"evidence not continuous: {temporal.consecutive_above} consecutive frame(s) "
                f"(need {self.min_consecutive_frames})"
            )
        if negative.single_frame_event > 0.5:
            blocking.append("event is supported by a single frame only")
        if negative.parked_vehicles > 0.5:
            blocking.append("objects appear parked / static, not involved in a collision")
        if negative.tracking_unstable > 0.6:
            blocking.append("tracking unstable, ids cannot be trusted")
        if negative.insufficient_history > 0.8:
            blocking.append("not enough tracked history to judge")
        if negative.low_detection_confidence > 0.7:
            blocking.append("detection confidence too low")
        if temporal.window_fill < 0.2:
            blocking.append("temporal window not filled yet")

        return (not blocking), blocking

    def _progress(self, score: AccidentScore, temporal: TemporalResult, seconds_in_state: float) -> float:
        """How far through verification we are (drives the HUD countdown)."""
        cfg = self.config
        if self.state is not DetectionState.VERIFYING:
            return 0.0
        score_part = clamp(self._event_score / max(1e-3, cfg.confirm_score)) * 0.5
        temporal_part = clamp(temporal.temporal_score / max(1e-3, cfg.confirm_score)) * 0.3
        time_part = 0.0
        if cfg.min_verification_seconds > 0:
            time_part = clamp(seconds_in_state / cfg.min_verification_seconds) * 0.2
        return clamp(score_part + temporal_part + time_part)

    def _seconds_in_state(self, timestamp: float) -> float:
        if self._frames_in_state == 0:
            return 0.0
        return max(0.0, float(timestamp) - self._state_since)

    def _in_cooldown(self, timestamp: float) -> bool:
        return self._cooldown_until > float(timestamp)

    @staticmethod
    def status_for(state: DetectionState) -> IncidentStatus:
        """Map the internal state to the status reported downstream."""
        if state is DetectionState.CONFIRMED_ACCIDENT:
            return IncidentStatus.CONFIRMED_ACCIDENT
        if state is DetectionState.VERIFYING:
            return IncidentStatus.POSSIBLE_ACCIDENT
        if state is DetectionState.FALSE_ALARM:
            return IncidentStatus.FALSE_ALARM
        if state is DetectionState.SUSPICIOUS:
            return IncidentStatus.POSSIBLE_ACCIDENT
        return IncidentStatus.NORMAL
