"""Time helpers: timestamps and a frame clock.

The engine must produce sensible timestamps for *live* streams too, where
``cv2.CAP_PROP_POS_MSEC`` is unreliable.  :class:`FrameClock` therefore
estimates time from the nominal source FPS and, when a real media clock is
available, anchors the offset to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

__all__ = ["FrameClock", "iso_timestamp", "now_iso", "utc_now"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Current UTC time as ``2026-09-28T09:15:00.123456+00:00``."""
    return utc_now().isoformat()


def iso_timestamp(epoch_seconds: float) -> str:
    """Convert a UNIX timestamp to the same ISO-8601 UTC representation."""
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat()


@dataclass
class FrameClock:
    """Assigns a monotonic-ish timestamp to every processed frame.

    Parameters
    ----------
    nominal_fps:
        FPS reported by the source (falls back to 30.0 when unknown).
    anchor:
        Optional real media timestamp of frame 0 (``CAP_PROP_POS_MSEC`` based).
        When given, the clock advances from it so file playback lines up with
        the video timeline; otherwise it starts at "now".
    live:
        Live sources (webcam / RTSP) can drop frames.  When ``live`` is True the
        clock simply follows wall time, so a stall shows up as a genuine gap in
        time instead of being hidden.
    """

    nominal_fps: float = 30.0
    anchor: Optional[float] = None
    live: bool = False
    start_wall: float = field(default_factory=time.monotonic)
    start_perf: float = field(default_factory=time.perf_counter)
    frame_index: int = 0

    def __post_init__(self) -> None:
        self.nominal_fps = float(self.nominal_fps) if self.nominal_fps and self.nominal_fps > 0 else 30.0

    @property
    def dt(self) -> float:
        """Nominal frame duration in seconds."""
        return 1.0 / self.nominal_fps

    def reset(self) -> None:
        self.start_wall = time.monotonic()
        self.start_perf = time.perf_counter()
        self.frame_index = 0

    def tick(self, media_timestamp: Optional[float] = None) -> float:
        """Return the timestamp for the current frame and advance the clock."""
        if media_timestamp is not None and not self.live:
            if self.anchor is None:
                self.anchor = float(media_timestamp)
            stamp = self.anchor + self.frame_index * self.dt
        elif self.anchor is not None and not self.live:
            stamp = self.anchor + self.frame_index * self.dt
        else:
            stamp = time.time() - (time.monotonic() - self.start_wall)
        self.frame_index += 1
        return stamp

    def elapsed(self) -> float:
        """Seconds of video processed so far (nominal, frame-count based)."""
        return self.frame_index * self.dt

    def nominal_frame_index(self, seconds: float) -> int:
        """Frames in ``seconds`` at the nominal rate (used for window sizes)."""
        return max(1, int(round(float(seconds) * self.nominal_fps)))
