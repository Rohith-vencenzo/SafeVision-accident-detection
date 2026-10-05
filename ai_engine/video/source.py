"""Unified video input (Phase 3).

One class handles all three sources:

===================================  =========================================
``0`` / ``"0"``                     laptop webcam (index or device node)
``"accident.mp4"``                  local file
``"rtsp://user@host/stream"``       IP / CCTV camera
===================================  =========================================

What it does for you:

* validates the source *before* opening (so a typo fails in 5 ms, not 30 s)
* injects RTSP credentials from environment variables (never hard-coded)
* reports the real source FPS and frame size
* reconnects a broken stream instead of dying
* counts empty / corrupt reads so the pipeline can report them
* releases the handle deterministically (``with`` block, ``close()``)
* downsizes frames and skips frames *before* the model ever sees them
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union

from ai_engine.config.classes import VideoConfig
from ai_engine.utils.logging_utils import get_logger

__all__ = ["FramePacket", "VideoReader", "VideoSourceError", "classify_source"]

_LOGGER = get_logger("video")

SourceType = Union[int, str]


class VideoSourceError(RuntimeError):
    """The video source could not be opened or read."""


def classify_source(source: SourceType) -> str:
    """Return ``"webcam"``, ``"rtsp"``, ``"stream"`` or ``"file"``."""
    if isinstance(source, int):
        return "webcam"
    text = str(source).strip()
    if text.isdigit():
        return "webcam"
    lowered = text.lower()
    for scheme in ("rtsp://", "rtsps://", "rtmp://", "http://", "https://"):
        if lowered.startswith(scheme):
            return "rtsp" if "rtsp" in scheme or "rtmp" in scheme else "stream"
    if lowered.startswith("video"):  # DirectShow device, e.g. "video=Integrated Camera"
        return "webcam"
    return "file"


@dataclass
class FramePacket:
    """One decoded frame plus its timing metadata."""

    frame: Any                      # numpy.ndarray (BGR)
    index: int                      # index of the frame *processed* (after skipping)
    decoded_index: int              # index as decoded from the source
    timestamp: float
    read_ms: float = 0.0
    dropped: int = 0                # frames skipped since the previous packet
    source_fps: float = 0.0
    resized: bool = False

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.frame.shape) if self.frame is not None else ()

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.frame.shape[1]), int(self.frame.shape[0])) if self.frame is not None else (0, 0)


class VideoReader:
    """Frame source with reconnect logic and optional resizing / skipping.

    Usage::

        with VideoReader("test_videos/accident.mp4", VideoConfig()) as reader:
            for packet in reader:
                ...  # packet.frame
    """

    def __init__(
        self,
        source: SourceType,
        config: Optional[VideoConfig] = None,
        credentials: Optional[dict] = None,
    ) -> None:
        self.source = source
        self.config = config or VideoConfig()
        self.credentials = credentials or {}
        self.kind = classify_source(source)

        self.cap: Any = None
        self.fps: float = 0.0
        self.frame_size: Tuple[int, int] = (0, 0)
        self.total_frames: int = 0
        self.is_live: bool = self.kind in ("webcam", "rtsp", "stream")

        # diagnostics
        self.frames_decoded = 0
        self.frames_returned = 0
        self.read_failures = 0
        self.reconnects = 0
        self.started_at = 0.0

        self._decoded_index = 0
        self._process_index = 0
        self._closed = False
        self._last_return: Optional[float] = None
        self._empty_run = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> "VideoReader":
        """Open the source.  Raises :class:`VideoSourceError` with a clear message."""
        if self._closed:
            raise VideoSourceError("this VideoReader has already been closed")
        if self.cap is not None:
            return self

        self._validate()
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise VideoSourceError("opencv-python is required for video input") from exc

        target = self._resolved_source()
        backend_hint = ""
        if self.kind == "rtsp":
            transport = "tcp" if int(self.config.rtsp_transport) == 1 else "udp"
            backend_hint = f" (transport={transport})"
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{transport}")

        cap = None
        errors: list[str] = []
        attempts = max(1, int(self.config.read_retries))
        for attempt in range(attempts):
            try:
                cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG if self.kind == "rtsp" else cv2.CAP_ANY)
                if cap is not None and cap.isOpened():
                    break
                if cap is not None:
                    cap.release()
                cap = None
                errors.append(f"attempt {attempt + 1}/{attempts}: could not open")
            except Exception as exc:  # noqa: BLE001 - backend specific failures
                errors.append(f"attempt {attempt + 1}/{attempts}: {exc}")
                cap = None
            if attempt + 1 < attempts:
                time.sleep(0.4)

        if cap is None or not cap.isOpened():
            raise VideoSourceError(
                f"could not open video source {self._describe_source()}{backend_hint}. "
                + "; ".join(errors[-3:])
                + self._hint()
            )

        self.cap = cap
        self.fps = self._read_fps(cap)
        self.frame_size = self._read_size(cap)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not self.is_live else 0
        self.started_at = time.monotonic()
        _LOGGER.info(
            "opened %s (%s): %dx%d @ %.1f fps%s",
            self._describe_source(),
            self.kind,
            self.frame_size[0],
            self.frame_size[1],
            self.fps,
            f", {self.total_frames} frames" if self.total_frames else "",
        )
        return self

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:  # pragma: no cover
                _LOGGER.debug("error releasing capture: %s", exc)
            self.cap = None
        self._closed = True

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FramePacket]:
        return self.frames()

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.kind == "file":
            path = Path(str(self.source)).expanduser()
            if not path.exists():
                raise VideoSourceError(f"video file not found: {path}{self._hint()}")
            if path.is_dir():
                raise VideoSourceError(f"expected a video file but {path} is a directory")
            if path.stat().st_size == 0:
                raise VideoSourceError(f"video file is empty: {path}")
        elif self.kind == "webcam":
            index = self.source if isinstance(self.source, int) else int(str(self.source))
            if index < 0:
                raise VideoSourceError(f"invalid camera index: {index}")

    def _resolved_source(self) -> SourceType:
        """Apply RTSP credentials from the environment, if configured."""
        source = self.source
        if self.kind not in ("rtsp", "stream"):
            return source
        from ai_engine.config.settings import build_source_url

        return build_source_url(
            source,
            username_env=str(self.credentials.get("env_username", "SAFEVISION_RTSP_USERNAME")),
            password_env=str(self.credentials.get("env_password", "SAFEVISION_RTSP_PASSWORD")),
        )

    def _describe_source(self) -> str:
        if isinstance(self.source, int) or str(self.source).isdigit():
            return f"webcam index {self.source}"
        from ai_engine.config.settings import redact_url

        return redact_url(self.source)

    def _hint(self) -> str:
        if self.kind == "file":
            return " (check the path, or use --source 0 for the webcam)"
        if self.kind == "rtsp":
            return (
                " (check the URL, that the camera is reachable, and that the port is allowed; "
                "set SAFEVISION_RTSP_USERNAME / SAFEVISION_RTSP_PASSWORD if the stream needs credentials)"
            )
        return " (check that the camera is connected and not in use by another application)"

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #
    def read(self) -> Optional[FramePacket]:
        """Read the next *processed* frame, or ``None`` at the end / on failure."""
        if self.cap is None:
            self.open()

        stride = max(1, int(self.config.frame_stride))
        skipped = 0
        started = time.perf_counter()

        for _ in range(stride):
            ok, frame = self.cap.read()
            self._decoded_index += 1
            if not ok or frame is None:
                self.read_failures += 1
                self._empty_run += 1
                if self._should_reopen():
                    self._reopen()
                    return None
                return None
            skipped += 1
            if stride == 1:
                break

        read_ms = (time.perf_counter() - started) * 1000.0
        self.frames_decoded += 1
        self._empty_run = 0

        resized = False
        if self.config.target_width:
            frame, resized = self._resize(frame, int(self.config.target_width))

        self.frames_returned += 1
        index = self._process_index
        self._process_index += 1
        timestamp = self._timestamp()
        self._last_return = timestamp

        return FramePacket(
            frame=frame,
            index=index,
            decoded_index=self._decoded_index,
            timestamp=timestamp,
            read_ms=read_ms,
            dropped=skipped - 1,
            source_fps=self.fps,
            resized=resized,
        )

    def frames(self, max_frames: int = 0) -> Iterator[FramePacket]:
        """Yield packets until the source ends or ``max_frames`` is reached."""
        produced = 0
        empty_streak = 0
        while not self._closed:
            if max_frames and produced >= max_frames:
                break
            packet = self.read()
            if packet is None:
                empty_streak += 1
                if self.cap is None:  # reopened: give it another chance next call
                    continue
                if empty_streak >= 1 and not self.is_live:
                    break
                if empty_streak > 30:
                    _LOGGER.warning("giving up after 30 consecutive failed reads")
                    break
                time.sleep(0.05)
                continue
            empty_streak = 0
            produced += 1
            yield packet

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        elapsed = max(1e-6, time.monotonic() - self.started_at) if self.started_at else 1e-6
        return {
            "source": self._describe_source(),
            "kind": self.kind,
            "fps": round(self.fps, 2),
            "frame_size": list(self.frame_size),
            "total_frames": self.total_frames,
            "frames_decoded": self.frames_decoded,
            "frames_processed": self.frames_returned,
            "read_failures": self.read_failures,
            "reconnects": self.reconnects,
            "elapsed_seconds": round(elapsed, 2),
            "processing_fps": round(self.frames_returned / elapsed, 2),
        }

    def _timestamp(self) -> float:
        """Media-timeline timestamp for a file, wall clock for a live stream."""
        import cv2

        if not self.is_live and self.fps > 0:
            pos = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pos > 0.0:
                return pos / 1000.0
            return self._process_index / self.fps
        return time.time()

    def _resize(self, frame: Any, target_width: int) -> Tuple[Any, bool]:
        import cv2

        height, width = frame.shape[:2]
        if width <= 0 or target_width <= 0 or width == target_width:
            return frame, False
        if self.config.keep_aspect:
            scale = target_width / float(width)
            new_size = (target_width, max(2, int(round(height * scale))))
        else:
            new_size = (target_width, height)
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), True

    def _should_reopen(self) -> bool:
        return (
            self.is_live
            and self.reconnects < int(self.config.max_reconnect_attempts)
            and self._empty_run >= int(self.config.reopen_on_zero_frames)
        )

    def _reopen(self) -> None:
        self.reconnects += 1
        _LOGGER.warning(
            "no frames for %d reads - reopening %s (attempt %d/%d)",
            self._empty_run,
            self._describe_source(),
            self.reconnects,
            self.config.max_reconnect_attempts,
        )
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:  # pragma: no cover
            pass
        self.cap = None
        self._empty_run = 0
        time.sleep(max(0.0, float(self.config.reconnect_delay_seconds)))
        try:
            self.open()
        except VideoSourceError as exc:
            _LOGGER.error("reconnect failed: %s", exc)
            self.cap = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_fps(cap: Any) -> float:
        import cv2

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.1 or fps > 1000.0:
            return 30.0
        return fps

    @staticmethod
    def _read_size(cap: Any) -> Tuple[int, int]:
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (max(0, width), max(0, height))

    def __repr__(self) -> str:  # pragma: no cover
        return f"VideoReader({self._describe_source()!r}, kind={self.kind}, open={self.cap is not None})"
