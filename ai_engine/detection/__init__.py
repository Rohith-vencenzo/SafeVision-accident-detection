"""Detection layer: YOLO inference + structured detection types."""

from ai_engine.detection.detection_types import (
    ACCIDENT_CLASS_ALIASES,
    BBox,
    CLASS_ALIASES,
    Detection,
    DetectionResult,
    PERSON,
    ROAD_CLASS_NAMES,
    VEHICLE_CLASS_NAMES,
    detections_from_specs,
    is_vehicle_class,
    normalize_class_name,
)
from ai_engine.detection.scripted import ScriptedDetector
from ai_engine.detection.yolo_detector import (
    BaseDetector,
    DetectionError,
    ModelNotAvailableError,
    YoloDetector,
    resolve_device,
)

__all__ = [
    "ACCIDENT_CLASS_ALIASES",
    "BBox",
    "BaseDetector",
    "CLASS_ALIASES",
    "Detection",
    "DetectionError",
    "DetectionResult",
    "ModelNotAvailableError",
    "PERSON",
    "ROAD_CLASS_NAMES",
    "ScriptedDetector",
    "VEHICLE_CLASS_NAMES",
    "YoloDetector",
    "detections_from_specs",
    "is_vehicle_class",
    "normalize_class_name",
    "resolve_device",
]
