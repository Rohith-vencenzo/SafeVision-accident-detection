"""A scripted, dependency-free detector.

Purpose: exercise the *whole* pipeline (tracking, motion, trajectory, collision,
temporal fusion, verification, scoring, incident JSON) on machines or CI jobs
where downloading YOLO weights is not possible, and give the demo mode a
reproducible "synthetic crash" clip.

This is **test/demo scaffolding, not a detector**: it has no vision model
behind it. The production path is :class:`~ai_engine.detection.yolo_detector.YoloDetector`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from ai_engine.detection.detection_types import BBox, Detection, DetectionResult
from ai_engine.detection.yolo_detector import BaseDetector

__all__ = ["ScriptedDetector", "moving_boxes"]

# A spec is (class_name, confidence, x1, y1, x2, y2)
Spec = Sequence[Any]
ScenarioFn = Callable[[int, int, int], List[Spec]]


def moving_boxes(start: Sequence[float], velocity: Sequence[float], size: Sequence[float]) -> Spec:
    """Build a box spec for a body of ``size`` whose top-left is ``start``."""
    return ("car", 0.9, float(start[0]), float(start[1]), float(start[0] + size[0]), float(start[1] + size[1]))


class ScriptedDetector(BaseDetector):
    """Returns detections from a script instead of a neural network.

    Parameters
    ----------
    scenario:
        ``fn(frame_index, width, height) -> list of specs``.  Called every frame.
    frames:
        Alternative to ``scenario``: a list of per-frame spec lists.  The last
        entry repeats once the list is exhausted.
    model_name:
        Reported in the detection result.
    inference_ms:
        Fake timing, so performance displays stay meaningful in tests.
    """

    def __init__(
        self,
        scenario: Optional[ScenarioFn] = None,
        frames: Optional[Sequence[Sequence[Spec]]] = None,
        model_name: str = "scripted",
        inference_ms: float = 12.0,
    ) -> None:
        if scenario is None and frames is None:
            raise ValueError("ScriptedDetector needs either `scenario` or `frames`")
        self._scenario = scenario
        self._frames: List[Sequence[Spec]] = [list(f) for f in (frames or [])]
        self.model_name = model_name
        self.inference_ms = float(inference_ms)
        self.calls = 0

    def detect(self, frame: Any, frame_index: int = 0, timestamp: float = 0.0) -> DetectionResult:
        height = width = 0
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])

        if self._scenario is not None:
            specs = self._scenario(frame_index, width, height) or []
        elif self._frames:
            index = min(frame_index, len(self._frames) - 1)
            specs = self._frames[index] or []
        else:  # pragma: no cover - guarded in __init__
            specs = []

        detections: List[Detection] = []
        for spec in specs:
            detections.append(
                Detection(
                    class_id=-1,
                    class_name=str(spec[0]),
                    confidence=float(spec[1]),
                    bbox=BBox(*(float(v) for v in spec[2:6])),
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
        self.calls += 1
        return DetectionResult(
            detections=detections,
            frame_index=frame_index,
            timestamp=timestamp,
            inference_ms=self.inference_ms,
            model_name=self.model_name,
            frame_size=(width, height),
            raw_total=len(detections),
        )

    def close(self) -> None:
        return None
