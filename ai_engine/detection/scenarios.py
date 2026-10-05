"""Named synthetic detection scenarios.

These are **scripted detection sequences**, not videos and not detections from a
model.  They exist so that the *decision logic* (tracking, motion, trajectory,
collision, temporal fusion, verification) can be validated on any machine,
offline, and so that a change in the thresholds can be regression-tested.

Each scenario is a function ``fn(frame_index, width, height) -> list of specs``
that a :class:`ScriptedDetector` consumes.

The set is deliberately built around the *confusion pairs* that matter for false
alarm prevention:

======================================  ======  ==============================
scenario                                expect  what it models
======================================  ======  ==============================
``head_on_crash``                       ALARM   two vehicles, same lane,
                                                impact, both stop
``rear_end_crash``                      ALARM   follower stops abruptly,
                                                lead vehicle pushed
``t_bone_crash``                        ALARM   perpendicular impact
``hard_brake``                          quiet   sudden stop, no other vehicle
``opposite_lanes``                      quiet   vehicles cross in the image
                                                plane but never interact
``following_close``                     quiet   tailgating, constant distance
``parked_cars``                         quiet   static vehicles (overlap in
                                                the image, no motion at all)
``pedestrian_crossing``                 quiet   person walking across
``single_frame_ghost``                  quiet   one-frame false detection
======================================  ======  ==============================

Honesty note: these scenarios validate the *pipeline logic*, not detection
accuracy.  Real accuracy must be measured on real accident footage - see
``scripts/calibrate.py`` and the "Limitations" section of the README.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

__all__ = ["SCENARIOS", "Scenario", "list_scenarios", "expectation_for"]

#: spec = (class_name, confidence, x1, y1, x2, y2)
Spec = Sequence[object]
Scenario = Callable[[int, int, int], List[Spec]]


def _car(x: float, y: float, w: float = 30.0, h: float = 22.0, conf: float = 0.9) -> Spec:
    return ("car", conf, x, y, x + w, y + h)


def _person(x: float, y: float, w: float = 16.0, h: float = 44.0, conf: float = 0.85) -> Spec:
    return ("person", conf, x, y, x + w, y + h)


def _truck(x: float, y: float, w: float = 46.0, h: float = 28.0, conf: float = 0.9) -> Spec:
    return ("truck", conf, x, y, x + w, y + h)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _travel(start: float, speed: float, frame: int, f_decel: int, ramp: int = 5) -> float:
    """Position of an object after ``frame`` frames.

    Constant speed until ``f_decel``, then a **linear** deceleration to a full
    stop over ``ramp`` frames.  This is the physical shape of a braking vehicle:
    a genuine, sustained speed drop rather than a one-frame teleport, which is
    exactly what the motion and collision analyzers are supposed to measure.
    """
    if frame <= f_decel:
        return start + speed * frame
    ramp = max(1, int(ramp))
    elapsed = min(frame - f_decel, ramp)
    ramp_distance = speed * elapsed - 0.5 * speed * elapsed * elapsed / ramp
    return start + speed * f_decel + ramp_distance


# --------------------------------------------------------------------------- #
# expected alarms
# --------------------------------------------------------------------------- #
def head_on_crash(i: int, w: int, h: int) -> List[Spec]:
    """Two vehicles in one lane close on each other, collide, then stop."""
    speed = 7.0
    a = _travel(30.0, speed, i, f_decel=34, ramp=3)
    b = _travel(560.0, -speed, i, f_decel=34, ramp=3)
    # the impact shoves both vehicles a little further into each other
    push = 3.0 * min(6, max(0, i - 36))
    return [_car(a + push, 150), _car(b - push, 152)]


def rear_end_crash(i: int, w: int, h: int) -> List[Spec]:
    """A fast following vehicle brakes hard and hits the rear of a slower one."""
    impact = 16
    shove = 2.5 * min(6, max(0, i - impact))
    lead = _travel(260.0, 6.0, i, f_decel=impact, ramp=4) + shove
    follower = _travel(120.0, 14.0, i, f_decel=impact, ramp=3)
    return [_car(lead, 120), _car(follower, 122)]


def t_bone_crash(i: int, w: int, h: int) -> List[Spec]:
    """A car turning across the path is hit sideways and both stop."""
    impact = 26
    shove = 2.0 * min(6, max(0, i - impact))
    straight = _travel(20.0, 7.0, i, f_decel=impact, ramp=3) + shove
    turning = _travel(560.0, -13.0, i, f_decel=impact, ramp=3)
    # Contact constraint: a vehicle cannot drive through the one it hit, so the
    # turning car stays against the struck car's flank and both come to rest.
    turning = max(turning, straight - 4.0)
    return [_car(straight, 78), _car(turning, 88), _truck(60, 280)]


# --------------------------------------------------------------------------- #
# expected quiet (false-alarm traps)
# --------------------------------------------------------------------------- #
def hard_brake(i: int, w: int, h: int) -> List[Spec]:
    """One vehicle brakes at a junction, nothing else in the scene."""
    stop = 15
    x = 90 + 10.0 * min(i, stop)
    y = 100 + 3.0 * (min(i, stop) // 6)
    return [_car(x, y), _car(500, 260)]


def opposite_lanes(i: int, w: int, h: int) -> List[Spec]:
    """Vehicles travelling in opposite directions cross the image plane.

    Their boxes pass within a few dozen pixels but the objects never interact -
    the classic "overlap means nothing" case.
    """
    a = 60 + 9.0 * i
    b = 500 - 9.0 * i
    return [_car(a, 110), _car(b, 200)]


def following_close(i: int, w: int, h: int) -> List[Spec]:
    """Tailgating: constant gap, both moving, boxes nearly touching."""
    lead = 150 + 7.0 * i
    return [_car(lead, 140), _car(lead - 46.0, 142)]


def parked_cars(i: int, w: int, h: int) -> List[Spec]:
    """Static vehicles, heavily overlapping in the image, no motion at all."""
    return [_car(120, 130), _car(140, 133), _truck(360, 120)]


def pedestrian_crossing(i: int, w: int, h: int) -> List[Spec]:
    """A person walks across the road while traffic flows normally."""
    px = 80 + 5.0 * i
    return [_person(px, 220), _car(60 + 8.0 * i, 110), _car(520 - 8.0 * i, 200)]


def single_frame_ghost(i: int, w: int, h: int) -> List[Spec]:
    """Two cars overlap for exactly one frame - a pure false detection."""
    specs: List[Spec] = [_car(70 + 7.0 * i, 150), _car(470 - 7.0 * i, 150)]
    if i == 22:
        specs.append(_car(275, 150, conf=0.45))
    return specs


# --------------------------------------------------------------------------- #
SCENARIOS: Dict[str, Scenario] = {
    "head_on_crash": head_on_crash,
    "rear_end_crash": rear_end_crash,
    "t_bone_crash": t_bone_crash,
    "hard_brake": hard_brake,
    "opposite_lanes": opposite_lanes,
    "following_close": following_close,
    "parked_cars": parked_cars,
    "pedestrian_crossing": pedestrian_crossing,
    "single_frame_ghost": single_frame_ghost,
}

#: what a correct configuration must do with each scenario
EXPECTATIONS: Dict[str, bool] = {
    "head_on_crash": True,
    "rear_end_crash": True,
    "t_bone_crash": True,
    "hard_brake": False,
    "opposite_lanes": False,
    "following_close": False,
    "parked_cars": False,
    "pedestrian_crossing": False,
    "single_frame_ghost": False,
}


def list_scenarios() -> List[str]:
    return sorted(SCENARIOS)


def expectation_for(name: str) -> bool:
    """``True`` when the scenario is supposed to raise an alarm."""
    try:
        return EXPECTATIONS[name]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"unknown scenario {name!r}; available: {list_scenarios()}") from exc
