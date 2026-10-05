"""Shared helpers for the test-suite (no YOLO weights, no network)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# make the project importable when tests are run from the repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ai_engine.config import Config, load_config  # noqa: E402
from ai_engine.detection.detection_types import BBox, Detection, DetectionResult  # noqa: E402
from ai_engine.pipeline.accident_pipeline import AccidentPipeline  # noqa: E402
from ai_engine.tracking.byte_tracker import Track, TrackPoint  # noqa: E402

FRAME_SIZE = (640, 360)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config(**overrides: Any) -> Config:
    """A config with evidence/visualisation switched off (tests stay fast).

    The demonstration emergency call is disabled here on purpose: no test may
    ever reach a call provider, even if ``DEMO_MODE=true`` happens to be set in
    the developer's environment.
    """
    config = load_config()
    config.evidence.enabled = False
    config.visualization.enabled = False
    config.runtime.write_incidents_file = False
    config.location.cameras_file = "cameras.yaml"
    # no test may place a call
    config.emergency.enabled = False
    config.emergency.demo_mode = False
    # Phase 4: never let a test write the durable outbox. `config.yaml` enables
    # it for real deployments; a test run must not create a database in the
    # repository or share one across test cases (a shared outbox would make the
    # second test of an incident_id see the first test's DONE row and skip).
    # Tests that exercise the outbox build their own in-memory instance.
    config.emergency.use_outbox = False
    config.emergency.outbox_path = ""
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        if name:
            setattr(getattr(config, section), name, value)
        else:
            setattr(config, section, value)
    return config.validate()


def blank_frame(size: Tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# detections / tracks
# --------------------------------------------------------------------------- #
def make_detection(
    class_name: str = "car",
    confidence: float = 0.9,
    box: Tuple[float, float, float, float] = (100.0, 100.0, 130.0, 122.0),
    frame_index: int = 0,
    timestamp: float = 0.0,
    track_id: Optional[int] = None,
) -> Detection:
    return Detection(
        class_id=-1,
        class_name=class_name,
        confidence=confidence,
        bbox=BBox(*box),
        frame_index=frame_index,
        timestamp=timestamp,
        track_id=track_id,
    )


def make_track(
    track_id: int = 1,
    positions: Sequence[Tuple[float, float]] = ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
    fps: float = 15.0,
    class_name: str = "car",
    size: Tuple[float, float] = (30.0, 22.0),
    scale: float = math.hypot(*FRAME_SIZE),
    start_frame: int = 0,
    confidence: float = 0.9,
) -> Track:
    """Build a :class:`Track` with a real history (velocities included).

    ``positions`` are box *centres*; the box is centred on each of them.
    """
    track = Track(
        track_id=track_id,
        class_name=class_name,
        bbox=(positions[0][0] - size[0] / 2, positions[0][1] - size[1] / 2,
              positions[0][0] + size[0] / 2, positions[0][1] + size[1] / 2),
        confidence=confidence,
        normalized_name=class_name,
        scale=scale,
        first_frame=start_frame,
        first_timestamp=start_frame / fps,
    )
    previous = positions[0]
    for index, (cx, cy) in enumerate(positions):
        timestamp = (start_frame + index) / fps
        if index == 0:
            velocity = (0.0, 0.0)
        else:
            velocity = ((cx - previous[0]) * fps, (cy - previous[1]) * fps)
        track.history.append(
            TrackPoint(
                frame_index=start_frame + index,
                timestamp=timestamp,
                center=(cx, cy),
                bbox=(cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2),
                velocity=velocity,
                speed=math.hypot(*velocity),
                instant_velocity=velocity,
                instant_speed=math.hypot(*velocity),
            )
        )
        previous = (cx, cy)
    track.frame_index = start_frame + len(positions) - 1
    track.timestamp = track.frame_index / fps
    track.hits = len(positions)
    track.state = "tracked"
    track.last_measured_center = previous
    track.last_measured_time = track.timestamp
    track.velocity = (track.history[-1].velocity[0], track.history[-1].velocity[1])
    # the live box must be the *newest* observation, not the first one
    cx, cy = previous
    track.bbox = (cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2)
    return track


def moving_positions(
    start: Tuple[float, float],
    velocity: Tuple[float, float],
    count: int,
    fps: float = 15.0,
) -> List[Tuple[float, float]]:
    """Centres of an object moving at ``velocity`` px/second."""
    return [
        (start[0] + velocity[0] * i / fps, start[1] + velocity[1] * i / fps)
        for i in range(count)
    ]


def deceleration_positions(
    start: Tuple[float, float],
    velocity: Tuple[float, float],
    stop_after: int,
    count: int,
    fps: float = 15.0,
    ramp: int = 3,
) -> List[Tuple[float, float]]:
    """Positions that move at constant speed, then brake to a full stop."""
    points: List[Tuple[float, float]] = []
    x, y = start
    for i in range(count):
        if i < stop_after:
            step = 1.0
        else:
            step = max(0.0, 1.0 - (i - stop_after) / max(1, ramp))
        x += velocity[0] / fps * step
        y += velocity[1] / fps * step
        points.append((x, y))
    return points


# --------------------------------------------------------------------------- #
# scripted detector
# --------------------------------------------------------------------------- #
class ScriptDetector:
    """Minimal detector stub: returns a fixed detection list every frame."""

    def __init__(self, boxes: Sequence[Sequence[Any]], model_name: str = "test") -> None:
        self.boxes = list(boxes)
        self.model_name = model_name
        self.calls = 0

    def detect(self, frame: Any, frame_index: int = 0, timestamp: float = 0.0) -> DetectionResult:
        self.calls += 1
        detections = [
            Detection(
                class_id=-1,
                class_name=str(item[0]),
                confidence=float(item[1]),
                bbox=BBox(*(float(v) for v in item[2:6])),
                frame_index=frame_index,
                timestamp=timestamp,
            )
            for item in self.boxes
        ]
        return DetectionResult(
            detections=detections,
            frame_index=frame_index,
            timestamp=timestamp,
            inference_ms=1.0,
            model_name=self.model_name,
            frame_size=(int(frame.shape[1]), int(frame.shape[0])),
        )

    def close(self) -> None:
        return None


def make_pipeline(boxes_per_frame: Optional[Sequence[Sequence[Sequence[Any]]]] = None, **config_overrides: Any):
    """A pipeline wired to a scripted detector (no YOLO weights needed)."""
    from ai_engine.detection.scripted import ScriptedDetector

    if boxes_per_frame is None:
        boxes_per_frame = []
    detector = ScriptedDetector(frames=boxes_per_frame)
    return AccidentPipeline(test_config(**config_overrides), detector=detector, camera_id="CAM-TEST")


def run_frames(pipeline: AccidentPipeline, frames: int = 30, fps: float = 15.0):
    """Drive a pipeline with blank frames; returns the list of FrameResults."""
    frame = blank_frame()
    results = []
    for index in range(frames):
        results.append(pipeline.process_frame(frame, index, index / fps))
    return results
