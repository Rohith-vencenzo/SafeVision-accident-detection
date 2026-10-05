"""Typed configuration objects for the whole SafeVision engine.

Every threshold the pipeline uses lives here, so tuning during testing is a
matter of editing ``config.yaml`` (or passing ``--set key=value``) instead of
touching code.

Design rules
------------
1. All dataclass fields have defaults, so a missing config section is legal.
2. ``from_dict`` is strict: an unknown key raises :class:`ConfigError` instead
   of being silently ignored (a typo in a threshold must not disappear).
3. No hard-coded secret or device assumption; credentials come from env vars.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, get_origin, get_type_hints

__all__ = [
    "CollisionConfig",
    "Config",
    "ConfigError",
    "DemoConfig",
    "EmergencyConfig",
    "EvidenceConfig",
    "LocationConfig",
    "ModelConfig",
    "MotionConfig",
    "RuntimeConfig",
    "ScoringConfig",
    "SeverityConfig",
    "TemporalConfig",
    "TrackingConfig",
    "TrajectoryConfig",
    "VerificationConfig",
    "VideoConfig",
    "VisualizationConfig",
    "build_dataclass",
]

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for an invalid / unparsable configuration."""


_TYPE_HINTS: Dict[type, Dict[str, Any]] = {}


def _hints(cls: type) -> Dict[str, Any]:
    """Resolve (and cache) the annotations of ``cls``.

    ``from __future__ import annotations`` turns every annotation into a string,
    so ``dataclasses.fields(...).type`` cannot be compared to a real type.  We
    resolve them once per class with :func:`typing.get_type_hints`.
    """
    cached = _TYPE_HINTS.get(cls)
    if cached is None:
        cached = get_type_hints(cls)
        _TYPE_HINTS[cls] = cached
    return cached


# --------------------------------------------------------------------------- #
# generic dict -> dataclass builder
# --------------------------------------------------------------------------- #
def build_dataclass(cls: Type[T], data: Optional[Mapping[str, Any]], path: str = "config") -> T:
    """Instantiate dataclass ``cls`` from a mapping, validating unknown keys.

    Nested dataclasses are built recursively, so ``config.yaml`` mirrors the
    Python structure exactly.
    """
    if data is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    hints = _hints(cls)
    unknown = [key for key in data if key not in known]
    if unknown:
        allowed = ", ".join(sorted(known))
        raise ConfigError(
            f"{path}: unknown option(s) {sorted(unknown)}. Allowed options: {allowed}"
        )

    kwargs: Dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        child_path = f"{path}.{name}"
        annotation = hints.get(name, str)
        if is_dataclass(annotation) and isinstance(value, Mapping):
            kwargs[name] = build_dataclass(annotation, value, child_path)  # type: ignore[arg-type]
        else:
            kwargs[name] = _coerce(cls, name, annotation, value, child_path)
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(cls: Any, name: str, annotation: Any, value: Any, path: str) -> Any:
    """Light type coercion + range checks for scalar options."""
    origin = get_origin(annotation) or annotation

    # dataclass annotation that is itself a nested config
    if is_dataclass(origin):
        return build_dataclass(origin, value, path)

    if origin in (int, float) and isinstance(value, bool):
        raise ConfigError(f"{path}: expected {origin.__name__}, got bool")
    if origin is bool and isinstance(value, (int, float)) and not isinstance(value, bool):
        # YAML `1`/`0` for a boolean flag is a classic typo - be strict
        raise ConfigError(f"{path}: expected a boolean (true/false), got {value!r}")

    if value is None:
        if origin in (int, float) and annotation is not Optional:
            raise ConfigError(f"{path}: value must not be null")
        return None

    try:
        if origin is int:
            coerced: Any = int(value)
        elif origin is float:
            coerced = float(value)
        elif origin is bool:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "yes", "1", "on"):
                    coerced = True
                elif lowered in ("false", "no", "0", "off"):
                    coerced = False
                else:
                    raise ConfigError(f"{path}: expected a boolean, got {value!r}")
            else:
                coerced = bool(value)
        elif origin is str:
            coerced = str(value)
        else:
            coerced = value
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: invalid value {value!r} ({exc})") from exc

    _check_range(cls, name, origin, coerced, path)
    return coerced


def _check_range(cls: Any, name: str, origin: Any, value: Any, path: str) -> None:
    """Validate documented ranges; raise :class:`ConfigError` otherwise."""
    rng = _RANGES.get(f"{cls.__name__}.{name}")
    if not rng or not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    lo, hi = rng
    if value < lo or value > hi:
        raise ConfigError(f"{path}: value {value} is outside the allowed range [{lo}, {hi}]")


# name -> (min, max).  Documented in config.yaml; validated at load time so a
# typo like conf=1.5 fails immediately instead of silently disabling detection.
_RANGES: Dict[str, Tuple[float, float]] = {
    "ModelConfig.conf": (0.01, 1.0),
    "ModelConfig.iou": (0.01, 1.0),
    "ModelConfig.imgsz": (64, 4096),
    "ModelConfig.max_det": (1, 1000),
    "VideoConfig.process_fps": (0.5, 240.0),
    "VideoConfig.frame_stride": (1, 30),
    "VideoConfig.buffer_size": (1, 1000),
    "VideoConfig.max_reconnect_attempts": (0, 20),
    "VideoConfig.rtsp_transport": (0, 1),
    "TrackingConfig.match_threshold": (0.0, 1.0),
    "TrackingConfig.match_threshold_low": (0.0, 1.0),
    "TrackingConfig.min_hits": (1, 20),
    "TrackingConfig.max_age": (1, 300),
    "TrackingConfig.history_seconds": (0.2, 60.0),
    "TrackingConfig.conf_high": (0.0, 1.0),
    "TrackingConfig.conf_low": (0.0, 1.0),
    "TrackingConfig.ema_alpha": (0.01, 1.0),
    "TrackingConfig.min_box_area": (0.0, 1e7),
    "TrackingConfig.recovery_distance": (0.0, 10.0),
    "MotionConfig.ema_alpha": (0.01, 1.0),
    "MotionConfig.min_history": (2, 120),
    "MotionConfig.window_seconds": (0.1, 30.0),
    "MotionConfig.turn_window_seconds": (0.1, 30.0),
    "MotionConfig.baseline_seconds": (0.1, 30.0),
    "MotionConfig.baseline_lag_seconds": (0.0, 10.0),
    "MotionConfig.person_fallback_scale": (0.0, 1.0),
    "MotionConfig.camera_motion_min_speed": (0.0, 10.0),
    "MotionConfig.camera_motion_tolerance": (0.0, 5.0),
    "MotionConfig.camera_motion_agreement": (0.0, 1.0),
    "MotionConfig.camera_motion_min_tracks": (2, 100),
    "MotionConfig.stop_window_frames": (1, 120),
    "MotionConfig.stop_entry_speed": (0.0, 10.0),
    "MotionConfig.stop_exit_speed": (0.0, 10.0),
    "MotionConfig.turn_threshold_deg": (0.0, 180.0),
    "MotionConfig.min_turn_speed": (0.0, 10.0),
    "MotionConfig.accel_scale": (1e-4, 100.0),
    "MotionConfig.jitter_scale": (1e-4, 10.0),
    "TrajectoryConfig.window_seconds": (0.2, 30.0),
    "TrajectoryConfig.min_points": (2, 100),
    "TrajectoryConfig.stop_displacement": (0.0, 1.0),
    "TrajectoryConfig.heading_threshold_deg": (0.0, 180.0),
    "TrajectoryConfig.lateral_threshold": (0.0, 1.0),
    "TrajectoryConfig.max_ttc_seconds": (0.0, 10.0),
    "TrajectoryConfig.straightness_threshold": (0.0, 1.0),
    "CollisionConfig.proximity_radius": (0.1, 10.0),
    "CollisionConfig.physical_weight": (0.0, 1.0),
    "CollisionConfig.context_weight": (0.0, 1.0),
    "CollisionConfig.iou_threshold": (0.0, 1.0),
    "CollisionConfig.min_closing_speed": (0.0, 10.0),
    "CollisionConfig.closing_speed_scale": (1e-3, 100.0),
    "CollisionConfig.decel_ratio": (0.0, 1.0),
    "CollisionConfig.decel_scale": (1e-3, 100.0),
    "CollisionConfig.post_stop_speed": (0.0, 10.0),
    "CollisionConfig.post_stop_seconds": (0.0, 10.0),
    "CollisionConfig.pre_event_offset_seconds": (0.0, 10.0),
    "CollisionConfig.displacement_spike_ratio": (0.0, 100.0),
    "CollisionConfig.displacement_spike_scale": (1e-3, 100.0),
    "CollisionConfig.min_signals": (1, 8),
    "CollisionConfig.max_pairs": (1, 500),
    "TemporalConfig.window_seconds": (0.5, 60.0),
    "TemporalConfig.min_consecutive_frames": (1, 300),
    "TemporalConfig.hold_tau_seconds": (0.05, 60.0),
    "TemporalConfig.recency_halflife_seconds": (0.05, 60.0),
    "TemporalConfig.weight_recent_mean": (0.0, 1.0),
    "TemporalConfig.weight_accumulator": (0.0, 1.0),
    "TemporalConfig.weight_peak": (0.0, 1.0),
    "TemporalConfig.weight_consistency": (0.0, 1.0),
    "TemporalConfig.warmup_frames": (0, 300),
    "TemporalConfig.persistence_bonus": (0.0, 1.0),
    "TemporalConfig.isolated_penalty": (0.0, 1.0),
    "TemporalConfig.min_window_fill": (0.0, 1.0),
    "TemporalConfig.possible_threshold": (0.0, 1.0),
    "VerificationConfig.suspicious_score": (0.0, 1.0),
    "VerificationConfig.verify_entry_score": (0.0, 1.0),
    "VerificationConfig.confirm_score": (0.0, 1.0),
    "VerificationConfig.temporal_confirm_score": (0.0, 1.0),
    "VerificationConfig.release_score": (0.0, 1.0),
    "VerificationConfig.suspicious_frames": (1, 300),
    "VerificationConfig.verify_frames": (1, 300),
    "VerificationConfig.false_alarm_frames": (1, 300),
    "VerificationConfig.min_verification_seconds": (0.0, 60.0),
    "VerificationConfig.verify_timeout_seconds": (0.0, 300.0),
    "VerificationConfig.cooldown_seconds": (0.0, 600.0),
    "VerificationConfig.min_vehicles": (1, 20),
    "VerificationConfig.min_signals": (1, 8),
    "SeverityConfig.low_threshold": (0.0, 1.0),
    "SeverityConfig.medium_threshold": (0.0, 1.0),
    "SeverityConfig.high_threshold": (0.0, 1.0),
    "SeverityConfig.blockage_displacement": (0.0, 1.0),
    "SeverityConfig.blockage_seconds": (0.0, 120.0),
    "SeverityConfig.duration_weight_seconds": (0.1, 300.0),
    "SeverityConfig.min_impact_speed": (0.0, 10.0),
    "SeverityConfig.min_decel": (0.0, 100.0),
    "SeverityConfig.min_displacement": (0.0, 10.0),
    "EvidenceConfig.jpeg_quality": (10, 100),
    "EvidenceConfig.pre_event_seconds": (0.0, 30.0),
    "EvidenceConfig.post_event_seconds": (0.0, 30.0),
    "EvidenceConfig.clip_seconds": (0.0, 30.0),
    "EvidenceConfig.clip_fps": (1.0, 120.0),
    "EvidenceConfig.max_evidence_dirs": (0, 10000),
    "VisualizationConfig.trail_length": (1, 500),
    "VisualizationConfig.font_scale": (0.2, 4.0),
    "VisualizationConfig.line_thickness": (1, 20),
    "VisualizationConfig.hud_alpha": (0.0, 1.0),
    "VisualizationConfig.max_preview_width": (160, 7680),
    "RuntimeConfig.max_frames": (0, 10000000),
    "DemoConfig.hold_seconds": (0.0, 30.0),
    "EmergencyConfig.min_gap_seconds": (0.0, 3600.0),
    "EmergencyConfig.max_calls_per_run": (0, 1000),
    "EmergencyConfig.request_timeout_seconds": (1.0, 120.0),
    "EmergencyConfig.poll_attempts": (0, 200),
    "EmergencyConfig.poll_interval_seconds": (0.5, 120.0),
    "EmergencyConfig.simulated_ring_after_seconds": (0.0, 120.0),
    "EmergencyConfig.simulated_duration_seconds": (0.0, 300.0),
    "EmergencyConfig.sms_max_body_length": (160, 3200),
}


def _validate_cross(config: "Config") -> None:
    """Checks that only make sense across two sections."""
    tracking = config.tracking
    if tracking.conf_low > tracking.conf_high:
        raise ConfigError(
            "tracking.conf_low must be <= tracking.conf_high "
            f"(got {tracking.conf_low} > {tracking.conf_high})"
        )
    if tracking.match_threshold_low > tracking.match_threshold:
        raise ConfigError(
            "tracking.match_threshold_low must be <= tracking.match_threshold "
            f"(got {tracking.match_threshold_low} > {tracking.match_threshold})"
        )
    if tracking.backend not in ("safevision", "ultralytics"):
        raise ConfigError(
            f"tracking.backend must be 'safevision' or 'ultralytics' (got {tracking.backend!r})"
        )

    if config.motion.aggregate not in ("max", "mean"):
        raise ConfigError(f"motion.aggregate must be 'max' or 'mean' (got {config.motion.aggregate!r})")

    if config.video.rtsp_transport not in (0, 1):
        raise ConfigError("video.rtsp_transport must be 0 (UDP) or 1 (TCP)")

    motion = config.motion
    if motion.stop_exit_speed > motion.stop_entry_speed:
        raise ConfigError(
            "motion.stop_exit_speed must be <= motion.stop_entry_speed "
            f"(got {motion.stop_exit_speed} > {motion.stop_entry_speed})"
        )

    severity = config.severity
    if not (severity.low_threshold <= severity.medium_threshold <= severity.high_threshold):
        raise ConfigError(
            "severity thresholds must satisfy low <= medium <= high (got "
            f"{severity.low_threshold}, {severity.medium_threshold}, {severity.high_threshold})"
        )

    verification = config.verification
    if verification.release_score > verification.verify_entry_score:
        raise ConfigError(
            "verification.release_score must be <= verification.verify_entry_score "
            f"(got {verification.release_score} > {verification.verify_entry_score})"
        )

    weights = config.scoring.weights
    if not weights or all(w <= 0 for w in weights.values()):
        raise ConfigError("scoring.weights must contain at least one positive weight")

    emergency = config.emergency
    if emergency.provider not in ("auto", "twilio", "simulation", "none"):
        raise ConfigError(
            "emergency.provider must be 'auto', 'twilio', 'simulation' or 'none' "
            f"(got {emergency.provider!r})"
        )
    if emergency.sms_provider not in ("auto", "twilio", "simulation", "none"):
        raise ConfigError(
            "emergency.sms_provider must be 'auto', 'twilio', 'simulation' or 'none' "
            f"(got {emergency.sms_provider!r})"
        )
    if not emergency.contact_env_var.strip():
        raise ConfigError("emergency.contact_env_var must name the environment variable to read")
    if not emergency.voice_message.strip():
        raise ConfigError("emergency.voice_message must not be empty (it is spoken to the recipient)")
    if not emergency.emergency_instruction.strip():
        raise ConfigError(
            "emergency.emergency_instruction must not be empty (it is the SMS instruction line)"
        )

    # ---- Phase 4 cross-field checks ------------------------------------- #
    if emergency.both_channels_failed_policy not in ("none", "flag", "escalate"):
        raise ConfigError(
            "emergency.both_channels_failed_policy must be 'none', 'flag' or 'escalate' "
            f"(got {emergency.both_channels_failed_policy!r})"
        )
    if emergency.allow_emergency_service_routing:
        # Refused rather than ignored: a config that appears to enable routing to
        # police/ambulance/112 must fail loudly, because the project has no such
        # routing and no authorisation for it.
        raise ConfigError(
            "emergency.allow_emergency_service_routing is NOT_APPROVED in this project. "
            "SafeVision notifies a consented contact only; it has no public-emergency "
            "integration, no jurisdiction and no authorisation. Remove this setting."
        )
    if emergency.retry_max_attempts < 1:
        raise ConfigError(
            f"emergency.retry_max_attempts must be >= 1 (got {emergency.retry_max_attempts})"
        )
    if emergency.retry_base_delay_seconds < 0:
        raise ConfigError("emergency.retry_base_delay_seconds must be >= 0")
    if emergency.retry_max_delay_seconds < emergency.retry_base_delay_seconds:
        raise ConfigError(
            "emergency.retry_max_delay_seconds must be >= retry_base_delay_seconds"
        )
    if emergency.breaker_failure_threshold < 1:
        raise ConfigError("emergency.breaker_failure_threshold must be >= 1")
    if emergency.breaker_reset_timeout_seconds <= 0:
        raise ConfigError("emergency.breaker_reset_timeout_seconds must be > 0")
    if emergency.outbox_lease_seconds <= 0:
        raise ConfigError("emergency.outbox_lease_seconds must be > 0")
    if emergency.webhook_enabled and emergency.webhook_require_https and not emergency.webhook_path.startswith("/"):
        raise ConfigError(
            "emergency.webhook_path must be an absolute URL path beginning with '/' "
            f"(got {emergency.webhook_path!r})"
        )
    if emergency.webhook_tolerance_seconds <= 0:
        raise ConfigError("emergency.webhook_tolerance_seconds must be > 0")
    # A real send is gated at *runtime* rather than here: the dispatcher knows
    # whether DEMO_MODE is actually on for this run, which config.yaml alone
    # cannot tell (the flag can be overridden by .env or by --demo-call).
    # `DemoCallDispatcher._apply_gates` refuses a real send served by the
    # simulation provider, and reports BLOCKED with the reason.

    # ---- Phase 5 routing checks ------------------------------------------- #
    routing = config.routing
    if routing.mode not in ("simulation", "live"):
        raise ConfigError(
            f"routing.mode must be 'simulation' or 'live' (got {routing.mode!r})"
        )
    if routing.allow_emergency_service:
        # Refused, not ignored - see the note on the emergency flag above.
        raise ConfigError(
            "routing.allow_emergency_service is NOT_APPROVED. SafeVision has no "
            "public-emergency integration, no jurisdiction and no authorisation. "
            "Remove this setting."
        )
    if not str(routing.directory).strip():
        raise ConfigError("routing.directory must name the recipient directory file")

    class_names = config.model.class_names or {}
    for key in class_names:
        if not isinstance(key, str):
            raise ConfigError("model.class_names keys must be strings")


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """YOLO detector options (Phase 4)."""

    weights: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.50
    device: str = "auto"                 # "auto" | "cpu" | "0" | "0,1" | "mps"
    half: bool = False
    max_det: int = 100
    classes: Optional[List[int]] = None  # None -> use the road-scene class filter
    class_names: Optional[Dict[str, int]] = None  # COCO-style name -> id override
    verbose: bool = False
    warmup: bool = True
    torch_threads: Optional[int] = None  # None -> leave the torch default alone


@dataclass
class VideoConfig:
    """Unified video input options (Phase 3)."""

    source: str = "0"
    process_fps: Optional[float] = None    # None -> follow the source FPS
    frame_stride: int = 1                  # process every Nth decoded frame
    target_width: Optional[int] = None     # downscale before inference
    keep_aspect: bool = True
    buffer_size: int = 8
    rtsp_transport: int = 1                # 1 = cv2.CAP_FFMPEG over TCP, 0 = UDP
    open_timeout_seconds: float = 10.0
    read_retries: int = 3
    max_reconnect_attempts: int = 2
    reconnect_delay_seconds: float = 2.0
    reopen_on_zero_frames: int = 30        # N consecutive empty reads -> reopen
    report_every: int = 0                  # 0 = no periodic status line


@dataclass
class TrackingConfig:
    """ByteTrack-style association options (Phase 5)."""

    backend: str = "safevision"        # "safevision" (ours) | "ultralytics" (YOLO's ByteTrack)
    ultralytics_tracker: str = "bytetrack.yaml"
    match_threshold: float = 0.55      # high-confidence association IoU
    match_threshold_low: float = 0.25  # second association stage (low-conf dets)
    conf_high: float = 0.55            # split between association stages
    conf_low: float = 0.15
    min_hits: int = 2
    max_age: int = 30                  # frames a track survives without a detection
    history_seconds: float = 3.0
    max_history: int = 300
    ema_alpha: float = 0.60            # velocity smoothing (0.6 ~= 1 frame of lag)
    min_box_area: float = 120.0        # px^2, drop specks
    use_kalman: bool = True
    score_weighted_iou: bool = True    # associate on conf-weighted IoU (ByteTrack)
    class_gate: bool = True            # never match a "person" to a "vehicle" track
    use_recovery_association: bool = True  # distance stage for abruptly stopped tracks
    recovery_distance: float = 0.9     # gate in object diagonals


@dataclass
class MotionConfig:
    """Per-track motion features and anomaly score (Phase 6)."""

    min_history: int = 4
    ema_alpha: float = 0.4
    window_seconds: float = 1.0              # displacement window
    turn_window_seconds: float = 0.7          # window used for the "mean heading"
    baseline_seconds: float = 1.2             # how far back the "was moving" speed is measured
    baseline_lag_seconds: float = 0.2        # ignore the last moments when measuring the baseline
    stop_window_frames: int = 5
    stop_entry_speed: float = 0.045    # frame-diagonals / second, "was moving"
    stop_exit_speed: float = 0.012      # ... "now stopped"
    turn_threshold_deg: float = 55.0
    min_turn_speed: float = 0.030
    accel_scale: float = 0.35           # normaliser for |dv/dt|
    jitter_scale: float = 0.020         # normaliser for position jitter
    weights: Dict[str, float] = field(
        default_factory=lambda: {"sudden_stop": 0.45, "sudden_turn": 0.20, "acceleration": 0.25, "jitter": 0.10}
    )
    aggregate: str = "max"              # "max" | "mean" over involved tracks
    person_fallback_scale: float = 0.6  # used only when the scene has no vehicles
    # --- global (camera) motion compensation, for panning / PTZ cameras ---
    camera_motion_min_speed: float = 0.04   # frame-diagonals/s
    camera_motion_tolerance: float = 0.35   # spread allowed around the median
    camera_motion_agreement: float = 0.6    # fraction of tracks that must agree
    camera_motion_min_tracks: int = 3       # below this, agreement proves nothing


@dataclass
class TrajectoryConfig:
    """Path shape, deviation and crossing analysis (Phase 7)."""

    window_seconds: float = 3.0
    min_points: int = 4
    stop_displacement: float = 0.035     # frame-diagonals over the window
    straightness_threshold: float = 0.80
    heading_threshold_deg: float = 45.0
    lateral_threshold: float = 0.045
    max_ttc_seconds: float = 1.2
    crossing_min_speed: float = 0.020    # ignore crawling objects as "converging"
    crossing_max_speed: float = 1.20     # ... and ignore fly-by close calls
    crossing_weight: float = 0.30        # crossing alone is weak evidence (capped)
    weights: Dict[str, float] = field(
        default_factory=lambda: {"heading_deviation": 0.32, "lateral_deviation": 0.28, "sudden_stop": 0.25, "zigzag": 0.15}
    )


@dataclass
class CollisionConfig:
    """Multi-signal collision scoring (Phase 8)."""

    proximity_radius: float = 1.30       # in average object diagonals
    proximity_weight: float = 0.15       # documented: context signal (see collision_analyzer)
    iou_threshold: float = 0.10
    iou_weight: float = 0.18
    physical_weight: float = 0.75        # share of the score owned by physical signals
    context_weight: float = 0.25         # share owned by proximity / overlap / convergence
    min_closing_speed: float = 0.045     # normal closing speed, frame-diagonals / s
    closing_speed_scale: float = 0.55
    decel_ratio: float = 0.45            # speed drop fraction that counts as decel
    decel_scale: float = 0.45
    post_stop_speed: float = 0.020       # "afterwards both are essentially stopped"
    post_stop_seconds: float = 0.35
    post_stop_memory_seconds: float = 8.0   # how long "post-impact" stays true
    pre_event_offset_seconds: float = 0.5  # look back this far for "was it moving?"
    displacement_spike_ratio: float = 1.2  # unexplained residual / predicted step
    displacement_spike_scale: float = 0.9
    time_to_impact_window: float = 0.8
    min_signals: int = 2                 # a collision needs AGREEING signals
    max_pairs: int = 40
    person_collisions: bool = True
    person_signal_bonus: float = 0.05
    require_motion: bool = True          # both objects must show recent movement


@dataclass
class TemporalConfig:
    """Rolling multi-frame evidence fusion (Phase 9)."""

    window_seconds: float = 6.0
    min_consecutive_frames: int = 3
    hold_tau_seconds: float = 1.2        # time constant of the evidence "hold"
    recency_halflife_seconds: float = 1.2
    weight_recent_mean: float = 0.40
    weight_accumulator: float = 0.30
    weight_peak: float = 0.20            # peak hold inside the window
    weight_consistency: float = 0.10
    persistence_bonus: float = 0.18
    isolated_penalty: float = 0.45
    min_window_fill: float = 0.25
    possible_threshold: float = 0.30
    warmup_frames: int = 3


@dataclass
class ScoringConfig:
    """Transparent weighted accident score (Phase 11)."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "collision": 0.34,
            "motion": 0.16,
            "trajectory": 0.08,
            "temporal": 0.30,
            "object_evidence": 0.07,
            "scene_evidence": 0.05,
        }
    )
    object_evidence_per_vehicle: float = 0.45
    object_evidence_person_bonus: float = 0.12
    scene_stopped_weight: float = 0.35
    scene_disruption_weight: float = 0.30
    scene_occlusion_weight: float = 0.20
    scene_duration_weight: float = 0.15
    scene_duration_cap_seconds: float = 6.0
    apply_negative_evidence: bool = True
    penalty_max: float = 0.85            # max total multiplicative penalty
    penalties: Dict[str, float] = field(
        default_factory=lambda: {
            "parked_vehicles": 0.35,
            "stable_scene": 0.30,
            "no_movement_change": 0.20,
            "single_frame_event": 0.45,
            "tracking_unstable": 0.30,
            "low_detection_confidence": 0.25,
            "insufficient_history": 0.20,
            "occlusion_conflict": 0.15,
        }
    )


@dataclass
class VerificationConfig:
    """Verification state machine (Phase 12) / false-alarm prevention (Phase 10)."""

    suspicious_score: float = 0.30
    verify_entry_score: float = 0.34
    confirm_score: float = 0.42           # gate on the *peak physical* evidence
    temporal_confirm_score: float = 0.40   # separate gate on the fused evidence
    release_score: float = 0.22
    suspicious_frames: int = 2
    verify_frames: int = 3
    false_alarm_frames: int = 8
    min_verification_seconds: float = 1.0
    verify_timeout_seconds: float = 8.0
    cooldown_seconds: float = 12.0
    min_vehicles: int = 2
    min_signals: int = 2
    false_alarm_cooldown_seconds: float = 6.0
    log_false_alarms: bool = True


@dataclass
class SeverityConfig:
    """AI-estimated scene severity (Phase 13)."""

    low_threshold: float = 0.25
    medium_threshold: float = 0.50
    high_threshold: float = 0.75
    duration_weight_seconds: float = 8.0
    min_impact_speed: float = 0.10        # frame-diagonals / s, "real" impact proxy
    min_decel: float = 0.08
    min_displacement: float = 0.02
    blockage_displacement: float = 0.030
    blockage_seconds: float = 1.5
    max_vehicles_for_scale: int = 4
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "vehicles_involved": 0.20,
            "impact_intensity": 0.26,
            "deceleration": 0.16,
            "displacement": 0.10,
            "persons_present": 0.10,
            "traffic_blockage": 0.10,
            "abnormal_duration": 0.08,
        }
    )
    # Optional external detectors are NOT bundled. When a caller supplies one
    # (e.g. a fire/smoke classifier trained by the team) the pipeline forwards
    # its output; otherwise the signal is reported as "not available".
    optional_signal_provider: Optional[str] = None


@dataclass
class EvidenceConfig:
    """Evidence capture (Phase 15)."""

    enabled: bool = True
    root: str = "evidence"
    save_raw_frames: bool = True
    save_annotated: bool = True
    save_clip: bool = False
    clip_seconds: float = 4.0
    clip_fps: float = 12.0
    pre_event_seconds: float = 2.0
    post_event_seconds: float = 1.5
    jpeg_quality: int = 92
    hash_files: bool = True
    max_evidence_dirs: int = 200
    write_meta: bool = True


@dataclass
class LocationConfig:
    """Camera-registered location metadata (Phase 14)."""

    cameras_file: str = "cameras.yaml"
    default_camera_id: Optional[str] = None
    allow_unregistered: bool = True      # emit an incident with a warning instead of dropping it
    env_username: str = "SAFEVISION_RTSP_USERNAME"
    env_password: str = "SAFEVISION_RTSP_PASSWORD"


@dataclass
class VisualizationConfig:
    """Debug overlay (Phase 19)."""

    enabled: bool = True
    show_boxes: bool = True
    show_labels: bool = True
    show_trails: bool = True
    show_collision: bool = True
    show_hud: bool = True
    show_perf: bool = True
    trail_length: int = 60
    font_scale: float = 0.5
    line_thickness: int = 2
    hud_alpha: float = 0.55
    max_preview_width: int = 1280
    state_banner: bool = True


@dataclass
class DemoConfig:
    """Scripted presentation mode (Phase 20)."""

    normal_source: str = "test_videos/normal_traffic.mp4"
    accident_source: str = "test_videos/accident.mp4"
    suspicious_source: str = "test_videos/suspicious.mp4"
    camera_id: Optional[str] = None
    hold_seconds: float = 2.0
    print_every: int = 25
    write_report: bool = True
    report_file: str = "demo_report.json"
    scenarios: Dict[str, str] = field(default_factory=dict)  # label -> source override


@dataclass
class EmergencyConfig:
    """Demonstration emergency call (demo mode only).

    The recipient itself is **not** a config option: it is read at runtime from
    ``EMERGENCY_CONTACT_NUMBER`` in the environment / ``.env``, so a phone
    number can never end up committed in ``config.yaml``.

    Two switches, in order of precedence:

    ``demo_mode``
        The master switch.  Mirrors the ``DEMO_MODE`` environment variable
        (``.env``).  While it is false the dispatcher refuses to dial anything
        and the incident payload says why - it does not simply omit the field.
    ``enabled``
        Removes the whole feature (no dispatcher is constructed, so the
        recipient is never even read).  Set it to ``false`` to strip the
        demonstration call out of an engine that is not being demoed.
    """

    enabled: bool = True                   # false = strip the feature out entirely
    demo_mode: bool = False                # mirrors $DEMO_MODE (default: off)
    provider: str = "auto"                 # auto | twilio | simulation | none
    contact_env_var: str = "EMERGENCY_CONTACT_NUMBER"

    # call bookkeeping
    write_call_log: bool = True
    call_log: str = "demo_calls.jsonl"
    async_calls: bool = True               # dial on a worker thread (never stall the loop)
    expose_recipient: bool = True          # put the unmasked number in the incident JSON
    max_calls_per_run: int = 0             # 0 = unlimited; the per-incident_id dedup is the real guard
    min_gap_seconds: float = 0.0           # minimum time between two calls

    # spoken message / TwiML
    voice_message: str = (
        "SafeVision demonstration alert. This is a demonstration call for a "
        "project presentation, not an emergency notification."
    )

    # --- SMS: sent AFTER the call for the same incident -------------------
    send_sms: bool = True                 # false = call only, no SMS
    sms_provider: str = "auto"            # auto | twilio | simulation | none
    sms_body: str = ""                    # "" = build the standard alert body
    sms_max_body_length: int = 1500       # stays inside one SMS segment budget
    #: wording appended to the SMS; the body itself always names the incident
    emergency_instruction: str = (
        "A traffic accident has been confirmed by SafeVision from multi-frame "
        "video evidence. Emergency assistance may be required."
    )

    # provider transport
    request_timeout_seconds: float = 10.0
    poll_attempts: int = 15                # status polls while a call is live
    poll_interval_seconds: float = 4.0
    simulated_ring_after_seconds: float = 2.0   # simulation provider only
    simulated_duration_seconds: float = 4.0     # simulation provider only

    # Twilio credentials (names only - the values live in .env)
    twilio_sid_env_var: str = "TWILIO_ACCOUNT_SID"
    twilio_token_env_var: str = "TWILIO_AUTH_TOKEN"
    twilio_from_env_var: str = "TWILIO_FROM_NUMBER"
    twilio_phone_env_var: str = "TWILIO_PHONE_NUMBER"   # SMS sender
    # Twilio Messaging Template SID. **Required on a trial account**: Twilio
    # rejects free-form SMS with error 572006 ("Trial accounts can only use
    # predefined SMS templates"). Empty = free-form Body, which needs a paid
    # account. Only the *variable name* lives in config - never a value.
    twilio_content_sid_env_var: str = "TWILIO_CONTENT_SID"
    twilio_content_variables_env_var: str = "TWILIO_CONTENT_VARIABLES"

    # ---- Phase 4: fallback policy, durability, safety interlocks ----------
    #: Send an SMS when the call does not reach the recipient.  ``False`` (the
    #: default) preserves the original behaviour: one call, then one SMS only
    #: when the call actually reached the network.  Turning this on is an
    #: explicit operator decision, not an automatic safety net - an unverified
    #: number that could not be called may not be able to receive SMS either.
    sms_on_call_failure: bool = False
    #: Also send the primary SMS after a *successful* call.
    sms_on_call_success: bool = True
    #: What to do when both channels fail.  ``none`` | ``flag`` | ``escalate``.
    #: ``escalate`` only records the failure - it never contacts a public
    #: emergency service, which is NOT_APPROVED in this project.
    both_channels_failed_policy: str = "flag"

    # durable outbox.  Off by default in code, ON in config.yaml: durability
    # across restarts is a deployment decision, and a library default that
    # writes a database file into the caller's working directory is a trap.
    use_outbox: bool = False
    outbox_path: str = ""                 # "" = in-memory (lost on restart)
    outbox_lease_seconds: float = 120.0

    # retry / circuit breaker
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 30.0
    breaker_failure_threshold: int = 5
    breaker_reset_timeout_seconds: float = 60.0

    # provider callbacks
    webhook_enabled: bool = False         # opt-in; no server starts otherwise
    webhook_path: str = "/webhooks/twilio"
    webhook_tolerance_seconds: float = 300.0
    webhook_require_https: bool = True

    # kill switches.  These default to False *in the library* so importing
    # SafeVision can never notify anybody; `config.yaml` turns them on for the
    # shipped deployment, where the real gate is the presence of credentials.
    allow_real_call: bool = False
    allow_real_sms: bool = False
    #: How long the process waits for in-flight notifications to reach a
    #: terminal state before exiting.  Must exceed a real call's ring time:
    #: `request_timeout_seconds` is the HTTP *read* timeout and is far too short
    #: for this - reusing it truncated real calls mid-ring.
    notification_wait_seconds: float = 75.0
    #: Emergency-service routing.  Structurally not implemented; a true value
    #: is refused by validation rather than silently ignored.
    allow_emergency_service_routing: bool = False


@dataclass
class RoutingConfig:
    """Who gets told, and under what rules (Phase 5).

    Routing decides; dispatching acts. Keeping them separate is what lets a
    routing decision be reviewed, logged and rehearsed without sending anything.
    """

    #: master kill switch. false stops ALL sending, effective immediately.
    dispatch_enabled: bool = True
    #: simulation | live. ``live`` alone is not enough to send - see
    #: emergency.allow_real_call / allow_real_sms.
    mode: str = "simulation"
    directory: str = "dispatch/recipients.json"
    #: the role the chosen contact must hold
    required_role: str = ""
    #: the jurisdiction the contact must be authorised in
    required_jurisdiction: str = ""
    #: refuse a dispatch when no position is available
    require_location: bool = False
    #: Public-emergency routing. Always False in this project; an emergency
    #: contact listed in the directory is excluded on every decision.
    allow_emergency_service: bool = False


@dataclass
class RuntimeConfig:
    """Process-level behaviour (Phase 21)."""

    seed: int = 42
    log_level: str = "INFO"
    log_file: Optional[str] = None
    max_frames: int = 0                  # 0 = unlimited
    save_output_video: bool = False
    output_video: str = "output_debug.mp4"
    output_video_fps: Optional[float] = None
    incidents_file: str = "incidents.json"
    write_incidents_file: bool = True
    print_incident_json: bool = False
    stop_on_incident: bool = False


@dataclass
class Config:
    """Root configuration object."""

    name: str = "safevision-default"
    version: str = "1.0.0"
    source_file: str = "<defaults>"
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    location: LocationConfig = field(default_factory=LocationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> "Config":
        """Run cross-section validation. Returns ``self`` for chaining."""
        _validate_cross(self)
        return self
