"""Geometry helpers.

Convention used across SafeVision: bounding boxes are ``(x1, y1, x2, y2)`` in
absolute pixel coordinates, ``x2 >= x1`` and ``y2 >= y1``.

Frame-relative quantities (scores, thresholds) are expressed in *normalised*
units so that the same configuration works for a 320x240 preview and a
1920x1080 stream:

* ``unit``   -> 1.0 == one frame diagonal (or one frame side, see call sites)
* ``u_per_s``-> normalised units per second
* angles    -> degrees in ``[0, 180]``
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]

__all__ = [
    "BBox",
    "Point",
    "angle_between_deg",
    "bbox_diagonal",
    "bbox_height",
    "bbox_width",
    "clamp",
    "cross_product",
    "ema",
    "interpolate_bbox",
    "iou_matrix",
    "normalize_point",
    "point_in_bbox",
    "rescale",
    "segment_intersection",
    "smoothstep",
    "unit_vector",
    "xywh_to_xyxy",
    "xyxy_center",
    "xyxy_iou",
    "xyxy_to_xywh",
]


# --------------------------------------------------------------------------- #
# scalars
# --------------------------------------------------------------------------- #
def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]`` (order independent, NaN safe)."""
    if value is None:
        return low
    if math.isnan(value):
        return low
    if value < low:
        return low
    if value > high:
        return high
    return float(value)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Classic smoothstep in ``[0, 1]``.

    Below ``edge0`` -> 0.0, above ``edge1`` -> 1.0, smooth in between.  Used to
    turn raw measurements (distance, speed, angle) into soft evidence scores
    instead of hard binary thresholds.
    """
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def rescale(x: float, in_lo: float, in_hi: float, out_lo: float = 0.0, out_hi: float = 1.0) -> float:
    """Linearly rescale ``x`` from ``[in_lo, in_hi]`` to ``[out_lo, out_hi]`` and clamp."""
    if in_hi == in_lo:
        return out_hi if x >= in_hi else out_lo
    t = (x - in_lo) / (in_hi - in_lo)
    return clamp(t, 0.0, 1.0) * (out_hi - out_lo) + out_lo


def ema(previous: float | None, current: float, alpha: float) -> float:
    """Exponential moving average. ``alpha`` in ``(0, 1]``; 1.0 == no smoothing."""
    if previous is None:
        return float(current)
    a = clamp(alpha, 1e-3, 1.0)
    return float(previous * (1.0 - a) + current * a)


def angle_between_deg(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Unsigned angle between two 2-D vectors, in ``[0, 180]``.

    Returns ``180.0`` when either vector has (near) zero magnitude, which is the
    safe answer for "direction unknown / changed completely".
    """
    x1, y1 = float(v1[0]), float(v1[1])
    x2, y2 = float(v2[0]), float(v2[1])
    m1 = math.hypot(x1, y1)
    m2 = math.hypot(x2, y2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 180.0
    cos = (x1 * x2 + y1 * y2) / (m1 * m2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def cross_product(a: Sequence[float], b: Sequence[float]) -> float:
    """2-D cross product (z component). Sign tells which side ``b`` is on."""
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def unit_vector(v: Sequence[float]) -> Point:
    x, y = float(v[0]), float(v[1])
    m = math.hypot(x, y)
    if m < 1e-9:
        return (0.0, 0.0)
    return (x / m, y / m)


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> BBox:
    return (float(x), float(y), float(x + w), float(y + h))


def xyxy_to_xywh(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, x2 - x1, y2 - y1)


def xyxy_center(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_width(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0]))


def bbox_height(box: Sequence[float]) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def bbox_area(box: Sequence[float]) -> float:
    return bbox_width(box) * bbox_height(box)


def bbox_diagonal(box: Sequence[float]) -> float:
    """Diagonal length - our main scale reference for a single object."""
    return math.hypot(bbox_width(box), bbox_height(box))


def interpolate_bbox(box: Sequence[float], dx: float, dy: float) -> BBox:
    """Translate a box by ``(dx, dy)`` (used after a Kalman prediction)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def normalize_point(point: Sequence[float], frame_shape: Tuple[int, int]) -> Point:
    """Normalise a pixel point to ``[0, 1]`` using ``(width, height)``."""
    w, h = float(frame_shape[0]), float(frame_shape[1])
    if w <= 0 or h <= 0:
        return (0.0, 0.0)
    return (clamp(float(point[0]) / w, -1.0, 2.0), clamp(float(point[1]) / h, -1.0, 2.0))


def point_in_bbox(point: Sequence[float], box: Sequence[float], margin: float = 0.0) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        float(box[0]) - margin <= x <= float(box[2]) + margin
        and float(box[1]) - margin <= y <= float(box[3]) + margin
    )


def xyxy_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``xyxy`` boxes (0.0 if disjoint)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return clamp(inter / union)


def iou_matrix(boxes_a: Iterable[Sequence[float]], boxes_b: Iterable[Sequence[float]]) -> list[list[float]]:
    """Pairwise IoU matrix, ``len(a) x len(b)`` (used by the tracker)."""
    a = list(boxes_a)
    b = list(boxes_b)
    return [[xyxy_iou(x, y) for y in b] for x in a]


# --------------------------------------------------------------------------- #
# trajectory geometry
# --------------------------------------------------------------------------- #
def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segment_intersection(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float], p4: Sequence[float]) -> Point | None:
    """Intersection point of segments ``p1p2`` and ``p3p4``, or ``None``.

    Colinear / touching-only cases return ``None`` on purpose: a shared end
    point is "objects close", not "trajectories crossing".
    """
    x1, y1, x2, y2 = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
    x3, y3, x4, y4 = (float(p3[0]), float(p3[1]), float(p4[0]), float(p4[1]))
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < 1e-9:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    eps = 1e-6
    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None
