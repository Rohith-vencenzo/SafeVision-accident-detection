"""Output contract: versioned incident schema + enums shared across modules."""

from ai_engine.schemas.enums import DetectionState, IncidentStatus, SeverityLevel
from ai_engine.schemas.incident_result import (
    DISCLAIMER,
    INCIDENT_JSON_SCHEMA,
    SCHEMA_VERSION,
    AccidentInfo,
    CameraRef,
    EvidenceInfo,
    IncidentResult,
    LocationInfo,
    MediaInfo,
    ObjectsInfo,
    PerformanceInfo,
    generate_emergency_incident,
    get_incident_result,
    validate_incident,
)

__all__ = [
    "AccidentInfo",
    "CameraRef",
    "DISCLAIMER",
    "DetectionState",
    "EvidenceInfo",
    "INCIDENT_JSON_SCHEMA",
    "IncidentResult",
    "IncidentStatus",
    "LocationInfo",
    "MediaInfo",
    "ObjectsInfo",
    "PerformanceInfo",
    "SCHEMA_VERSION",
    "SeverityLevel",
    "generate_emergency_incident",
    "get_incident_result",
    "validate_incident",
]
