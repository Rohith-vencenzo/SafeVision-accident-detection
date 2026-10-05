"""SafeVision - AI / Computer-Vision accident detection engine.

This package is the 60% AI subsystem of the SafeVision project.  It is
completely independent of any backend, database or dashboard: its only
contract is a JSON-serialisable incident object (see
:mod:`ai_engine.schemas.incident_result`).

Quick start::

    from ai_engine import AccidentPipeline, load_config

    config = load_config("config.yaml")
    pipeline = AccidentPipeline(config)

    for incident in pipeline.process_video("test_videos/accident.mp4"):
        print(incident["incident_id"], incident["accident"]["score"])
"""

__version__ = "1.0.0"

from ai_engine.config import Config, ConfigError, load_config
from ai_engine.schemas.incident_result import (
    SCHEMA_VERSION,
    IncidentResult,
    generate_emergency_incident,
)
from ai_engine.utils.logging_utils import configure_logging, get_logger

__all__ = [
    "AccidentPipeline",
    "Config",
    "ConfigError",
    "IncidentResult",
    "SCHEMA_VERSION",
    "__version__",
    "configure_logging",
    "get_incident_result",
    "generate_emergency_incident",
    "load_config",
]


def __getattr__(name: str):
    """Lazy re-exports so ``import ai_engine`` stays cheap and cycle-free."""
    if name == "AccidentPipeline":
        from ai_engine.pipeline.accident_pipeline import AccidentPipeline

        return AccidentPipeline
    if name == "get_incident_result":
        from ai_engine.schemas.incident_result import get_incident_result

        return get_incident_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
