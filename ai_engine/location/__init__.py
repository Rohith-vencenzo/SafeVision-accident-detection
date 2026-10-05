"""Location metadata and provenance (Phase 14 + Phase 3).

Two distinct questions, deliberately kept apart:

* where is the **camera**?  -> the registry (:mod:`location_manager`)
* where is the **accident**? -> a location provider (:mod:`location_provider`),
  which reports ``CAMERA_REGISTERED``, ``LIVE_GPS`` or ``LOCATION_UNAVAILABLE``
  and never blurs the first into the second.
"""

from ai_engine.location.location_manager import (
    CameraLocation,
    LocationError,
    LocationManager,
)
from ai_engine.location.location_provider import (
    LOCATION_CAMERA_REGISTERED,
    LOCATION_LIVE_GPS,
    LOCATION_UNAVAILABLE,
    CameraRegisteredProvider,
    FixPolicy,
    GpsDeviceProvider,
    GpsFix,
    LocationFix,
    LocationProvider,
    LocationProviderRegistry,
)

__all__ = [
    "CameraLocation",
    "CameraRegisteredProvider",
    "FixPolicy",
    "GpsDeviceProvider",
    "GpsFix",
    "LOCATION_CAMERA_REGISTERED",
    "LOCATION_LIVE_GPS",
    "LOCATION_UNAVAILABLE",
    "LocationError",
    "LocationFix",
    "LocationManager",
    "LocationProvider",
    "LocationProviderRegistry",
]