"""Small, dependency-free helpers shared by every SafeVision module.

Nothing in here knows about detection, tracking or accidents - it is pure
numeric / bookkeeping code so the analysis modules stay testable in isolation.
"""

from ai_engine.utils.geometry import (
    angle_between_deg,
    clamp,
    ema,
    normalize_point,
    rescale,
    smoothstep,
    xyxy_center,
    xyxy_iou,
    xyxy_to_xywh,
    xywh_to_xyxy,
)
from ai_engine.utils.jsonio import atomic_write_json, read_json, to_jsonable
from ai_engine.utils.logging_utils import configure_logging, get_logger
from ai_engine.utils.ringbuffer import TimeRingBuffer
from ai_engine.utils.timeutil import FrameClock, iso_timestamp, now_iso

__all__ = [
    "FrameClock",
    "TimeRingBuffer",
    "angle_between_deg",
    "atomic_write_json",
    "clamp",
    "configure_logging",
    "ema",
    "get_logger",
    "iso_timestamp",
    "normalize_point",
    "now_iso",
    "read_json",
    "rescale",
    "smoothstep",
    "to_jsonable",
    "xywh_to_xyxy",
    "xyxy_center",
    "xyxy_iou",
    "xyxy_to_xywh",
]
