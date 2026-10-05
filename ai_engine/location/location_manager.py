"""Camera-registered location metadata (Phase 14).

A fixed CCTV camera cannot tell you where it is - that information is installed
*with* the camera.  So SafeVision does not pretend to derive GPS: it looks the
camera up in a registry and attaches whatever was configured.

    cameras:
      CAM-001:
        name: Main Road Camera
        latitude: 13.0827
        longitude: 80.2707
        location_name: Chennai Main Road
        road: Anna Salai
        address: 112, Anna Salai, Chennai
        zone: TN-01

The module is deliberately independent of any database so the teammate's 40%
can either (a) point ``location.cameras_file`` at a YAML/JSON export, or
(b) push cameras in at runtime with :meth:`LocationManager.register` from his
own database.

Honesty note: ``is_camera_registered`` is ``False`` for unregistered cameras and
the incident then carries ``latitude=None``.  We never invent coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ai_engine.config.classes import LocationConfig
from ai_engine.utils.logging_utils import get_logger

__all__ = ["CameraLocation", "LocationError", "LocationManager"]

_LOGGER = get_logger("location")


class LocationError(ValueError):
    """Invalid camera metadata."""


@dataclass
class CameraLocation:
    """Registered metadata for one camera."""

    camera_id: str
    name: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: str = ""
    road: Optional[str] = None
    address: Optional[str] = None
    zone: Optional[str] = None
    source: str = "camera_metadata"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.camera_id = str(self.camera_id).strip()
        if not self.camera_id:
            raise LocationError("camera_id must not be empty")
        if self.latitude is not None:
            self.latitude = float(self.latitude)
            if not -90.0 <= self.latitude <= 90.0:
                raise LocationError(f"camera {self.camera_id}: latitude {self.latitude} outside [-90, 90]")
        if self.longitude is not None:
            self.longitude = float(self.longitude)
            if not -180.0 <= self.longitude <= 180.0:
                raise LocationError(f"camera {self.camera_id}: longitude {self.longitude} outside [-180, 180]")
        if (self.latitude is None) != (self.longitude is None):
            raise LocationError(
                f"camera {self.camera_id}: latitude and longitude must be provided together"
            )
        if not self.name:
            self.name = self.camera_id

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_name": self.location_name or self.name,
            "road": self.road,
            "address": self.address,
            "zone": self.zone,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, camera_id: str, data: Mapping[str, Any]) -> "CameraLocation":
        known = {
            "name",
            "latitude",
            "longitude",
            "location_name",
            "road",
            "address",
            "zone",
            "source",
        }
        kwargs = {key: value for key, value in (data or {}).items() if key in known}
        extra = {key: value for key, value in (data or {}).items() if key not in known}
        if "camera_id" in (data or {}):
            extra.pop("camera_id", None)
        return cls(camera_id=str(camera_id), extra=extra, **kwargs)


class LocationManager:
    """Registry of camera locations.

    Parameters
    ----------
    cameras:
        ``{camera_id: CameraLocation}``.
    default_camera_id:
        Used when the caller does not pass a camera id explicitly.
    allow_unregistered:
        When ``True`` (default) an incident for an unknown camera is still
        emitted, with ``location.is_camera_registered = False`` and a warning.
        When ``False`` the pipeline logs an error and attaches no location.
    """

    def __init__(
        self,
        cameras: Optional[Mapping[str, CameraLocation]] = None,
        default_camera_id: Optional[str] = None,
        allow_unregistered: bool = True,
    ) -> None:
        self._cameras: Dict[str, CameraLocation] = dict(cameras or {})
        self.default_camera_id = default_camera_id
        self.allow_unregistered = bool(allow_unregistered)
        if default_camera_id and default_camera_id not in self._cameras:
            _LOGGER.warning(
                "default camera '%s' is not in the camera registry (%d camera(s) known)",
                default_camera_id,
                len(self._cameras),
            )

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._cameras)

    def __contains__(self, camera_id: object) -> bool:
        return str(camera_id) in self._cameras

    @property
    def cameras(self) -> Dict[str, CameraLocation]:
        return dict(self._cameras)

    def register(self, camera: CameraLocation) -> None:
        """Add or replace a camera at runtime (e.g. from the backend)."""
        self._cameras[camera.camera_id] = camera
        _LOGGER.debug("Registered camera %s", camera.camera_id)

    def register_many(self, cameras: Iterable[CameraLocation]) -> None:
        for camera in cameras:
            self.register(camera)

    def get(self, camera_id: Optional[str] = None) -> Optional[CameraLocation]:
        """Look a camera up.  Falls back to the default camera."""
        key = str(camera_id) if camera_id else (self.default_camera_id or "")
        if not key:
            return None
        return self._cameras.get(key)

    def require(self, camera_id: Optional[str] = None) -> Optional[CameraLocation]:
        """Like :meth:`get` but warns loudly when metadata is missing."""
        camera = self.get(camera_id)
        if camera is None:
            _LOGGER.warning(
                "No registered location for camera %r - the incident will be reported without coordinates",
                camera_id or self.default_camera_id or "<unspecified>",
            )
        return camera

    def to_payload(self) -> Dict[str, Any]:
        """Serialisable registry (handy for a dashboard camera dropdown)."""
        return {cid: camera.to_dict() for cid, camera in self._cameras.items()}

    def describe(self) -> str:
        if not self._cameras:
            return "no cameras registered"
        return ", ".join(
            f"{cid} ({camera.name}{'' if camera.has_coordinates else ', no coordinates'})"
            for cid, camera in sorted(self._cameras.items())
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "LocationManager":
        """Load ``cameras.yaml`` / ``cameras.json``.

        Accepted layouts::

            cameras:
              CAM-001: {name: ..., latitude: ..., longitude: ...}

            CAM-001: {name: ..., latitude: ..., longitude: ...}      # top level

            {"CAM-001": {...}}                                        # json
        """
        from ai_engine.utils.jsonio import read_json

        file_path = Path(path).expanduser()
        if not file_path.is_file():
            _LOGGER.warning("Camera metadata file not found: %s", file_path)
            return cls(**kwargs)

        if file_path.suffix.lower() in (".json",):
            raw: Any = read_json(file_path)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise LocationError("PyYAML is required to read camera metadata") from exc
            with file_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)

        if raw is None:
            return cls(**kwargs)
        if not isinstance(raw, Mapping):
            raise LocationError(f"{file_path}: expected a mapping at the top level")

        block = raw.get("cameras", raw)
        if not isinstance(block, Mapping):
            raise LocationError(f"{file_path}: 'cameras' must be a mapping of camera_id -> metadata")

        cameras: Dict[str, CameraLocation] = {}
        for camera_id, data in block.items():
            if not isinstance(data, Mapping):
                raise LocationError(f"{file_path}: camera {camera_id!r} must be a mapping")
            try:
                cameras[str(camera_id)] = CameraLocation.from_dict(str(camera_id), data)
            except LocationError as exc:
                _LOGGER.error("Skipping invalid camera entry in %s: %s", file_path, exc)

        default_id = kwargs.pop("default_camera_id", None) or raw.get("default_camera_id")
        _LOGGER.info("Loaded %d camera location(s) from %s", len(cameras), file_path)
        return cls(cameras=cameras, default_camera_id=default_id, **kwargs)

    @classmethod
    def from_config(cls, config: LocationConfig, root: Optional[Path] = None) -> "LocationManager":
        """Build a manager from the ``location`` config section."""
        from ai_engine.config.settings import project_path

        path = project_path(config.cameras_file, root)
        return cls.from_file(
            path,
            default_camera_id=config.default_camera_id,
            allow_unregistered=config.allow_unregistered,
        )
