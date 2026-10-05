"""A time-bounded ring buffer.

Both the trajectory history and the pre-event evidence buffer need "the last N
seconds", not "the last N items", because the processing FPS changes with the
machine load.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Iterator, List, Optional, Tuple

__all__ = ["TimeRingBuffer"]


class TimeRingBuffer:
    """Keeps ``(timestamp, item)`` pairs newer than ``window_seconds``.

    Parameters
    ----------
    window_seconds:
        Retention window. ``None`` means unbounded.
    max_items:
        Hard cap (memory safety). ``None`` means unbounded.
    """

    def __init__(self, window_seconds: Optional[float] = None, max_items: Optional[int] = None) -> None:
        self.window_seconds = float(window_seconds) if window_seconds else None
        self.max_items = int(max_items) if max_items else None
        self._data: Deque[Tuple[float, Any]] = deque(maxlen=self.max_items)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Tuple[float, Any]]:
        return iter(self._data)

    @property
    def items(self) -> List[Any]:
        return [item for _, item in self._data]

    @property
    def span_seconds(self) -> float:
        if len(self._data) < 2:
            return 0.0
        return self._data[-1][0] - self._data[0][0]

    def push(self, timestamp: float, item: Any) -> None:
        self._data.append((float(timestamp), item))
        self._prune(float(timestamp))

    def extend(self, pairs) -> None:
        for timestamp, item in pairs:
            self._data.append((float(timestamp), item))
        if self._data:
            self._prune(self._data[-1][0])

    def window(self, now: Optional[float] = None) -> List[Any]:
        """Items inside the retention window relative to ``now`` (or the newest)."""
        reference = float(now) if now is not None else (self._data[-1][0] if self._data else 0.0)
        self._prune(reference)
        return [item for _, item in self._data]

    def copy(self) -> "TimeRingBuffer":
        clone = TimeRingBuffer(self.window_seconds, self.max_items)
        clone._data = deque(self._data, maxlen=self.max_items)
        return clone

    def clear(self) -> None:
        self._data.clear()

    def _prune(self, reference: float) -> None:
        if self.window_seconds is None or not self._data:
            return
        cutoff = reference - self.window_seconds
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()
