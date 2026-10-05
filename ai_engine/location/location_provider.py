"""Location provenance: where a position came from and how much to trust it.

The rule this module enforces: a camera's registered coordinates are **not** a
GPS fix, and nothing in this package will invent a fix.  If no device supplies
one, the answer is ``LOCATION_UNAVAILABLE`` - not a guess, not an IP lookup.

    from ai_engine.location.location_provider import LocationProviderRegistry

    registry = LocationProviderRegistry.from_config(config.location)
    fix = registry.resolve(camera_id="CAM-001", at_unix=time.time())
    fix.source          # CAMERA_REGISTERED | LIVE_GPS | LOCATION_UNAVAILABLE
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai_engine.utils.logging_utils import get_logger

__all__ = [
    "GpsFix",
    "LocationFix",
    "LocationProvider",
    "CameraRegisteredProvider",
    "GpsDeviceProvider",
    "FixPolicy",
    "LocationProviderRegistry",
    "LOCATION_CAMERA_REGISTERED",
    "LOCATION_LIVE_GPS",
    "LOCATION_UNAVAILABLE",
]

_LOGGER = get_logger("location")

LOCATION_CAMERA_REGISTERED = "CAMERA_REGISTERED"
LOCATION_LIVE_GPS = "LIVE_GPS"
LOCATION_UNAVAILABLE = "LOCATION_UNAVAILABLE"
#: Rejected by policy. Coarse, frequently city-level, and routinely misread as
#: a fix.  Never produced by this package.
LOCATION_IP_GEOLOCATION = "IP_GEOLOCATION"


@dataclass
class GpsFix:
    """A raw fix as reported by a device or geolocation service."""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    radius_m: Optional[float] = None
    fix_timestamp: Optional[float] = None       # unix seconds
    provider: str = ""
    raw_status: str = ""                        # ok | no_fix | permission_denied | error

    @property
    def has_position(self) -> bool:
        return (self.latitude is not None and self.longitude is not None
                and self.raw_status == "ok")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy_m": self.accuracy_m,
            "radius_m": self.radius_m,
            "fix_timestamp": self.fix_timestamp,
            "provider": self.provider,
            "status": self.raw_status,
        }


@dataclass
class LocationFix:
    """The accepted position for an incident, with its provenance.

    ``source`` is the field that must never be guessed, and ``confidence``
    tells a responder how much to lean on it.
    """

    source: str = LOCATION_UNAVAILABLE
    camera_id: Optional[str] = None
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    radius_m: Optional[float] = None
    fix_timestamp: Optional[float] = None
    captured_at: Optional[float] = None
    fix_age_seconds: Optional[float] = None
    confidence: str = "NONE"
    reason: str = ""
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def is_live_gps(self) -> bool:
        return self.source == LOCATION_LIVE_GPS

    @property
    def is_camera_registered(self) -> bool:
        return self.source == LOCATION_CAMERA_REGISTERED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "camera_id": self.camera_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy_m": self.accuracy_m,
            "radius_m": self.radius_m,
            "fix_timestamp": self.fix_timestamp,
            "captured_at": self.captured_at,
            "fix_age_seconds": self.fix_age_seconds,
            "confidence": self.confidence,
            "reason": self.reason or None,
            "rejected_fixes": list(self.rejected),
            "note": ("Coordinates are the camera's registered position and are "
                     "APPROXIMATE. They are not a live GPS fix."
                     if self.source == LOCATION_CAMERA_REGISTERED else None),
        }

    def summary_line(self) -> str:
        """The wording a responder sees."""
        if self.source == LOCATION_LIVE_GPS:
            return (f"LIVE_GPS fix from {self.camera_id or 'device'} "
                    f"(+/-{self.accuracy_m}m, age {self.fix_age_seconds:.0f}s)")
        if self.source == LOCATION_CAMERA_REGISTERED:
            if not self.has_position:
                # registered, but the registry entry has no coordinates
                return (f"LOCATION_UNAVAILABLE - camera {self.camera_id or 'unknown'} is "
                        "registered but has no coordinates, and no live GPS fix")
            return (f"CAMERA_REGISTERED position of {self.camera_id or 'camera'} - "
                    "approximate, not a live GPS fix")
        return f"LOCATION_UNAVAILABLE ({self.reason or 'no position source'})"


# --------------------------------------------------------------------------- #
class FixPolicy:
    """Rejects a live fix that is stale, imprecise or out of bounds.

    Defaults are deliberately strict: a 500 m fix from a parked phone is worse
    for a responder than the camera's registered position, which at least tells
    them which junction to go to.
    """

    def __init__(
        self,
        max_fix_age_seconds: float = 30.0,
        max_accuracy_m: float = 100.0,
        max_radius_m: float = 200.0,
        max_clock_skew_seconds: float = 5.0,
        latitude_bounds: Tuple[float, float] = (-90.0, 90.0),
        longitude_bounds: Tuple[float, float] = (-180.0, 180.0),
    ) -> None:
        self.max_fix_age_seconds = max_fix_age_seconds
        self.max_accuracy_m = max_accuracy_m
        self.max_radius_m = max_radius_m
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.latitude_bounds = latitude_bounds
        self.longitude_bounds = longitude_bounds

    def check(self, fix: GpsFix, now: float) -> Tuple[bool, str, Optional[float]]:
        """``(accepted, reason, age_seconds)``."""
        if fix.raw_status and fix.raw_status != "ok":
            return False, f"device reported status={fix.raw_status}", None
        if not fix.has_position:
            return False, "device returned no position", None

        lat, lon = float(fix.latitude), float(fix.longitude)
        if not (self.latitude_bounds[0] <= lat <= self.latitude_bounds[1]):
            return False, f"latitude {lat} out of bounds", None
        if not (self.longitude_bounds[0] <= lon <= self.longitude_bounds[1]):
            return False, f"longitude {lon} out of bounds", None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False, "coordinates are not finite", None

        stamp = fix.fix_timestamp if fix.fix_timestamp is not None else now
        age = max(0.0, float(now) - float(stamp))
        if age > self.max_fix_age_seconds:
            return False, (f"fix is {age:.0f}s old, limit is "
                           f"{self.max_fix_age_seconds:.0f}s"), age
        if float(now) - float(stamp) < -self.max_clock_skew_seconds:
            return False, "fix timestamp is in the future", age
        if fix.accuracy_m is not None and float(fix.accuracy_m) > self.max_accuracy_m:
            return False, (f"accuracy {fix.accuracy_m:.0f}m exceeds "
                           f"{self.max_accuracy_m:.0f}m"), age
        if fix.radius_m is not None and float(fix.radius_m) > self.max_radius_m:
            return False, (f"uncertainty radius {fix.radius_m:.0f}m exceeds "
                           f"{self.max_radius_m:.0f}m"), age
        return True, "accepted", age

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_fix_age_seconds": self.max_fix_age_seconds,
            "max_accuracy_m": self.max_accuracy_m,
            "max_radius_m": self.max_radius_m,
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
        }


# --------------------------------------------------------------------------- #
class LocationProvider:
    """Base class.  ``resolve`` returns a :class:`LocationFix`, never an exception."""

    name = "base"
    source = LOCATION_UNAVAILABLE

    def available(self) -> Tuple[bool, str]:
        return False, "not implemented"

    def resolve(self, camera_id: Optional[str], at_unix: float) -> LocationFix:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        usable, reason = self.available()
        return {"provider": self.name, "source": self.source,
                "usable": usable, "reason": reason or None}


class CameraRegisteredProvider(LocationProvider):
    """Positions configured for a camera.  Approximate, always labelled."""

    name = "camera_registry"
    source = LOCATION_CAMERA_REGISTERED

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def available(self) -> Tuple[bool, str]:
        return True, "camera registry is always readable"

    def resolve(self, camera_id: Optional[str], at_unix: float) -> LocationFix:
        # Mirror LocationManager.get(): a missing camera id falls back to the
        # registry's default camera rather than reporting "unregistered". Getting
        # this wrong made the provenance say LOCATION_UNAVAILABLE while the
        # incident's own location block correctly named the registered camera -
        # two parts of the same incident contradicting each other.
        camera = self.manager.get(camera_id) if camera_id else self.manager.get(None)
        resolved_id = camera_id or getattr(self.manager, "default_camera_id", None)
        if camera is None:
            return LocationFix(
                source=LOCATION_UNAVAILABLE,
                camera_id=resolved_id,
                captured_at=at_unix,
                confidence="NONE",
                reason=f"camera {resolved_id!r} is not registered and no GPS fix is available",
            )
        if camera is None:
            return LocationFix(
                source=LOCATION_UNAVAILABLE,
                camera_id=resolved_id,
                captured_at=at_unix,
                confidence="NONE",
                reason=f"camera {resolved_id!r} is not registered and no GPS fix is available",
            )
        has_coords = camera.latitude is not None and camera.longitude is not None
        return LocationFix(
            # a registered camera with no coordinates has no position to give,
            # so it must not advertise itself as CAMERA_REGISTERED
            source=self.source if has_coords else LOCATION_UNAVAILABLE,
            camera_id=camera.camera_id or resolved_id,
            name=camera.location_name or camera.name,
            latitude=camera.latitude if has_coords else None,
            longitude=camera.longitude if has_coords else None,
            captured_at=at_unix,
            confidence="APPROXIMATE_CAMERA_POSITION" if has_coords else "NONE",
            reason="" if has_coords else "camera is registered but has no coordinates",
        )


class GpsDeviceProvider(LocationProvider):
    """Wraps a real fix source (a phone, a vehicle tracker, a GNSS API).

    ``reader`` is injected.  On this project no reader is configured, so
    ``available()`` is False and the registry reports ``LOCATION_UNAVAILABLE``
    rather than falling back to anything invented.

    Deliberately absent: IP geolocation.  It is coarse, and labelling it as a
    position is how "the accident is at this junction" becomes wrong.
    """

    name = "gps_device"
    source = LOCATION_LIVE_GPS

    def __init__(
        self,
        reader: Optional[Callable[[], GpsFix]] = None,
        policy: Optional[FixPolicy] = None,
    ) -> None:
        self.reader = reader
        self.policy = policy or FixPolicy()

    def available(self) -> Tuple[bool, str]:
        if self.reader is None:
            return False, ("no GPS fix source is configured (no GNSS receiver, "
                           "vehicle tracker or geolocation API is wired in)")
        return True, "a GPS fix reader is configured"

    def resolve(self, camera_id: Optional[str], at_unix: float) -> LocationFix:
        if self.reader is None:
            return LocationFix(
                source=LOCATION_UNAVAILABLE, camera_id=camera_id,
                captured_at=at_unix, confidence="NONE",
                reason="no GPS fix source configured",
            )
        try:
            raw = self.reader()
        except Exception as exc:  # noqa: BLE001 - a device fault is not fatal
            _LOGGER.warning("GPS reader raised: %s", exc)
            return LocationFix(
                source=LOCATION_UNAVAILABLE, camera_id=camera_id,
                captured_at=at_unix, confidence="NONE",
                reason=f"GPS reader failed: {exc}",
            )
        accepted, why, age = self.policy.check(raw, at_unix)
        if not accepted:
            return LocationFix(
                source=LOCATION_UNAVAILABLE, camera_id=camera_id,
                captured_at=at_unix, confidence="NONE",
                reason=f"GPS fix rejected: {why}",
                rejected=[raw.to_dict()],
            )
        return LocationFix(
            source=LOCATION_LIVE_GPS,
            camera_id=camera_id,
            latitude=raw.latitude,
            longitude=raw.longitude,
            accuracy_m=raw.accuracy_m,
            radius_m=raw.radius_m,
            fix_timestamp=raw.fix_timestamp,
            captured_at=at_unix,
            fix_age_seconds=round(age or 0.0, 2),
            confidence="DEVICE_REPORTED",
            reason=f"provider={raw.provider}",
        )


# --------------------------------------------------------------------------- #
class LocationProviderRegistry:
    """Resolves the best available position and records what it could not use.

    Order: a live GPS fix wins when one is available **and** passes policy;
    otherwise the registered camera position, explicitly labelled; otherwise
    nothing.  A rejected fix is recorded in ``rejected`` so an operator can see
    that a fix existed and was refused, rather than wondering why.
    """

    def __init__(
        self,
        camera_provider: Optional[LocationProvider] = None,
        gps_provider: Optional[LocationProvider] = None,
        policy: Optional[FixPolicy] = None,
        allow_camera_fallback: bool = True,
    ) -> None:
        self.camera_provider = camera_provider
        self.gps_provider = gps_provider
        self.policy = policy or FixPolicy()
        self.allow_camera_fallback = allow_camera_fallback

    @classmethod
    def from_config(cls, location_config: Any, gps_reader=None) -> "LocationProviderRegistry":
        from ai_engine.location.location_manager import LocationManager

        manager = LocationManager.from_config(location_config)
        return cls(
            camera_provider=CameraRegisteredProvider(manager),
            gps_provider=GpsDeviceProvider(gps_reader),
        )

    def resolve(self, camera_id: Optional[str], at_unix: Optional[float] = None) -> LocationFix:
        now = float(at_unix if at_unix is not None else time.time())
        rejected: List[Dict[str, Any]] = []

        if self.gps_provider is not None:
            usable, _ = self.gps_provider.available()
            if usable:
                fix = self.gps_provider.resolve(camera_id, now)
                if fix.is_live_gps:
                    return fix
                rejected.extend(fix.rejected)
                reason = fix.reason

        if self.allow_camera_fallback and self.camera_provider is not None:
            fix = self.camera_provider.resolve(camera_id, now)
            fix.rejected.extend(rejected)
            if fix.has_position:
                if rejected:
                    fix.reason = (fix.reason + "; " if fix.reason else "") + \
                        "no usable live GPS fix; falling back to the registered position"
                return fix
            fix.rejected.extend(rejected)
            return fix

        return LocationFix(
            source=LOCATION_UNAVAILABLE, camera_id=camera_id, captured_at=now,
            confidence="NONE",
            reason="no GPS fix source configured and the camera has no registered position",
            rejected=rejected,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "providers": [
                p.describe() for p in (self.gps_provider, self.camera_provider) if p
            ],
            "allow_camera_fallback": self.allow_camera_fallback,
            "ip_geolocation": "rejected by policy - never used",
        }