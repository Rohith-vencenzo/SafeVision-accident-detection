"""Configuration package for SafeVision.

    from ai_engine.config import load_config
    cfg = load_config("config.yaml", overrides=["model.conf=0.4"])
"""

from ai_engine.config.classes import (
    CollisionConfig,
    Config,
    ConfigError,
    DemoConfig,
    EmergencyConfig,
    EvidenceConfig,
    LocationConfig,
    ModelConfig,
    MotionConfig,
    RuntimeConfig,
    ScoringConfig,
    SeverityConfig,
    TemporalConfig,
    TrackingConfig,
    TrajectoryConfig,
    VerificationConfig,
    VideoConfig,
    VisualizationConfig,
    build_dataclass,
)
from ai_engine.config.settings import (
    build_source_url,
    deep_merge,
    describe_config,
    find_project_root,
    load_config,
    parse_override,
    project_path,
    redact_url,
)

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
    "build_source_url",
    "deep_merge",
    "describe_config",
    "find_project_root",
    "load_config",
    "parse_override",
    "project_path",
    "redact_url",
]
