"""Structured detection types (Phase 4).

Two things matter here:

1. Every detection carries the metadata the rest of the pipeline needs
   (class, confidence, box, centre, frame index, timestamp) so no downstream
   module has to touch raw ultralytics structures.
2. The class vocabulary is normalised. COCO-trained YOLO models know
   ``car``/``truck``/``person``; other public checkpoints use ``auto``,
   ``lorry``, ``pedestrian``... We map them onto one internal vocabulary.

Honesty note
------------
A stock COCO detector has **no** "accident" class. If a model *does* expose an
accident-like class (``accident``, ``crash``, ``accident_scene``) we surface it
as a separate signal (:attr:`DetectionResult.accident_class_score`) instead of
pretending plain object detection proves an accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "ACCIDENT_CLASS_ALIASES",
    "BBox",
    "CLASS_ALIASES",
    "Detection",
    "DetectionResult",
    "PERSON",
    "ROAD_CLASS_NAMES",
    "VEHICLE_CLASS_NAMES",
    "is_vehicle_class",
    "normalize_class_name",
]

PERSON = "person"

#: Internal road-scene vocabulary.
VEHICLE_CLASS_NAMES: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck", "bicycle")
ROAD_CLASS_NAMES: Tuple[str, ...] = VEHICLE_CLASS_NAMES + (PERSON,)

#: Alias -> internal name.  Covers the common public checkpoint vocabularies.
CLASS_ALIASES: Dict[str, str] = {
    "person": PERSON,
    "people": PERSON,
    "pedestrian": PERSON,
    "human": PERSON,
    "car": "car",
    "auto": "car",
    "automobile": "car",
    "van": "car",
    "minivan": "car",
    "sedan": "car",
    "suv": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor": "motorcycle",
    "moped": "motorcycle",
    "scooter": "motorcycle",
    "tricycle": "motorcycle",
    "bus": "bus",
    "coach": "bus",
    "truck": "truck",
    "lorry": "truck",
    "pickup": "truck",
    "pickup truck": "truck",
    "trailer": "truck",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
}

#: Classes that *indicate* an accident in accident-specific checkpoints.
ACCIDENT_CLASS_ALIASES: Dict[str, str] = {
    "accident": "accident",
    "accident_scene": "accident",
    "crash": "accident",
    "collision": "accident",
    "vehicle_collision": "accident",
    "accident_scene/accident": "accident",
}


def normalize_class_name(raw: str) -> str:
    """Map a raw model label to the internal vocabulary (lowercased fallback)."""
    key = str(raw).strip().lower()
    if key in ACCIDENT_CLASS_ALIASES:
        return ACCIDENT_CLASS_ALIASES[key]
    return CLASS_ALIASES.get(key, key)


def is_vehicle_class(name: str) -> bool:
    return normalize_class_name(name) in VEHICLE_CLASS_NAMES


class BBox(NamedTuple):
    """Axis-aligned box in absolute pixels: ``(x1, y1, x2, y2)``."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def diagonal(self) -> float:
        """Diagonal length - our per-object scale reference."""
        return float(((self.width ** 2) + (self.height ** 2)) ** 0.5)

    def as_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def clamped(self, width: int, height: int) -> "BBox":
        return BBox(
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    def to_list(self) -> List[float]:
        return [round(float(v), 2) for v in self]


@dataclass
class Detection:
    """One detected road object in one frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    frame_index: int = 0
    timestamp: float = 0.0

    # normalised fields, filled in by the detector
    normalized_name: str = ""
    is_accident_class: bool = False
    #: Track id supplied by an external tracker (``tracking.backend=ultralytics``).
    #: ``None`` when the detector does not track.
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.normalized_name:
            self.normalized_name = normalize_class_name(self.class_name)

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center

    @property
    def area(self) -> float:
        return self.bbox.area

    @property
    def is_vehicle(self) -> bool:
        return self.normalized_name in VEHICLE_CLASS_NAMES

    @property
    def is_person(self) -> bool:
        return self.normalized_name == PERSON

    def to_dict(self) -> Dict[str, Any]:
        cx, cy = self.center
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "class": self.normalized_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_list(),
            "center": [round(cx, 2), round(cy, 2)],
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class DetectionResult:
    """All detections for a single frame, plus the timing metadata."""

    detections: List[Detection] = field(default_factory=list)
    frame_index: int = 0
    timestamp: float = 0.0
    inference_ms: float = 0.0
    model_name: str = ""
    frame_size: Tuple[int, int] = (0, 0)  # (width, height)
    raw_total: int = 0                   # detections before the road-class filter

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def __bool__(self) -> bool:
        return bool(self.detections)

    @property
    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections if d.is_vehicle]

    @property
    def persons(self) -> List[Detection]:
        return [d for d in self.detections if d.is_person]

    @property
    def accident_class_score(self) -> float:
        """Max confidence of any *accident-class* detection (0.0 when absent)."""
        scores = [d.confidence for d in self.detections if d.is_accident_class]
        return float(max(scores)) if scores else 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(sum(d.confidence for d in self.detections) / len(self.detections))

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(max(d.confidence for d in self.detections))

    def filter_classes(self, names: Sequence[str]) -> List[Detection]:
        wanted = {normalize_class_name(n) for n in names}
        return [d for d in self.detections if d.normalized_name in wanted]

    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.normalized_name] = counts.get(det.normalized_name, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 4),
            "inference_ms": round(float(self.inference_ms), 3),
            "frame_size": list(self.frame_size),
            "counts": self.count_by_class(),
            "detections": [d.to_dict() for d in self.detections],
        }


def detections_from_specs(
    specs: Iterable[Sequence[Any]],
    frame_index: int = 0,
    timestamp: float = 0.0,
) -> List[Detection]:
    """Build detections from compact specs - used by tests and scripted demos.

    ``specs`` items look like ``(class_name, confidence, x1, y1, x2, y2)``.
    """
    out: List[Detection] = []
    for spec in specs:
        name, conf = spec[0], float(spec[1])
        box = BBox(*(float(v) for v in spec[2:6]))
        out.append(
            Detection(
                class_id=-1,
                class_name=str(name),
                confidence=conf,
                bbox=box,
                frame_index=frame_index,
                timestamp=timestamp,
            )
        )
    return out


def optional(value: Optional[float], default: float = 0.0) -> float:
    return default if value is None else float(value)
