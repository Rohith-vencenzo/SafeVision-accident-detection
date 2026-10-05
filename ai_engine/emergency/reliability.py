"""Reliability primitives for outbound notification.  Phase 4.

Three concerns, deliberately separate so each can be tested and reasoned about
alone:

* :class:`RetryPolicy` - which failures are worth repeating, and how long to
  wait.  Retrying a rejected request wastes the recipient's patience and can
  duplicate an SMS, so *permanent* errors must never be retried.
* :class:`CircuitBreaker` - stop hammering a provider that is already failing.
  Without it, a provider outage turns every confirmed incident into a burst of
  doomed requests and a stuck notification thread.
* :func:`idempotency_key` - a stable key per ``(incident, channel, attempt)``,
  stored **before** the request so a crash mid-request cannot lose it.

Nothing in this module contacts a provider.  It is pure policy, which is what
makes it cheap to test exhaustively.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "CircuitBreaker",
    "CircuitOpen",
    "RetryPolicy",
    "TransientFailure",
    "PermanentFailure",
    "classify_http_status",
    "idempotency_key",
]


# --------------------------------------------------------------------------- #
# error classification
# --------------------------------------------------------------------------- #
class TransientFailure(Exception):
    """Worth retrying: rate limited, 5xx, or a connection that failed early.

    ``retry_after`` carries a provider-supplied hint (Twilio's ``Retry-After``
    header).  When present it wins over our own backoff, because the provider is
    telling us exactly when it will be ready.
    """

    def __init__(self, message: str = "", retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentFailure(Exception):
    """Not worth retrying: bad credentials, bad number, malformed request."""


#: HTTP statuses worth trying again.  429 is rate limiting and is the single most
#: common transient on a messaging provider.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def classify_http_status(code: int, *, after: Optional[float] = None) -> None:
    """Raise :class:`TransientFailure` or :class:`PermanentFailure` for a status.

    ``after`` carries a provider-supplied ``Retry-After`` in seconds so the
    caller can honour it instead of guessing.
    """
    if code in RETRYABLE_STATUS:
        raise TransientFailure(f"HTTP {code} is retryable", retry_after=after)
    raise PermanentFailure(f"HTTP {code} is permanent")


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def idempotency_key(incident_id: str, channel: str, attempt: int = 1) -> str:
    """A deterministic key for one outbound attempt.

    Deterministic on purpose: the same incident retried after a crash must
    produce the same key, otherwise the key protects nothing.  The SHA-256 keeps
    the value a safe length for a header and avoids leaking the incident id to
    third parties.
    """
    raw = f"safevision|{incident_id}|{channel}|{attempt}".encode("utf-8")
    return f"sv-{hashlib.sha256(raw).hexdigest()[:32]}"


# --------------------------------------------------------------------------- #
# retry
# --------------------------------------------------------------------------- #
@dataclass
class RetryPolicy:
    """Bounded exponential backoff with full jitter.

    ``max_attempts`` counts the *first* try, so ``max_attempts=3`` means at most
    three provider requests - not two retries plus a try.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    multiplier: float = 2.0
    #: Deterministic jitter for tests; ``None`` uses the global RNG.
    jitter_source: Optional[Callable[[], float]] = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def should_retry(self, attempt: int, error: BaseException) -> bool:
        """Retry only transient failures, and only while attempts remain."""
        if attempt >= self.max_attempts:
            return False
        if isinstance(error, TransientFailure):
            return True
        return False

    def delay_for(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Seconds to wait before attempt ``attempt + 1``.

        A provider-supplied ``Retry-After`` wins over our own backoff: it is the
        provider telling us exactly when it will be ready.
        """
        if retry_after is not None and retry_after >= 0:
            return float(min(retry_after, self.max_delay_seconds))
        raw = self.base_delay_seconds * (self.multiplier ** max(0, attempt - 1))
        capped = min(raw, self.max_delay_seconds)
        if capped <= 0:
            return 0.0
        source = self.jitter_source or random.random
        # full jitter: uniform in [0, capped].  Spreads a thundering herd of
        # retries from many cameras after a provider outage.
        return float(capped * source())

    def plan(self, error: BaseException, retry_after: Optional[float] = None) -> List[float]:
        """The full delay schedule for an error, for logging and for tests."""
        delays: List[float] = []
        retry_after_value = retry_after
        for attempt in range(1, self.max_attempts):
            if not self.should_retry(attempt, error):
                break
            delay = self.delay_for(attempt, retry_after_value)
            delays.append(round(delay, 3))
            retry_after_value = None   # honour Retry-After once only
        return delays

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "multiplier": self.multiplier,
            "retryable_http": sorted(RETRYABLE_STATUS),
            "note": "permanent errors (4xx other than 408/425/429) are never retried",
        }


# --------------------------------------------------------------------------- #
# circuit breaker
# --------------------------------------------------------------------------- #
class CircuitOpen(Exception):
    """The breaker is open; the provider must not be contacted."""


@dataclass
class CircuitBreaker:
    """Trip after consecutive failures, half-open after a cool-off.

    States: ``closed`` (normal) -> ``open`` (refusing) -> ``half_open`` (one
    trial) -> ``closed`` on success or back to ``open`` on failure.
    """

    failure_threshold: int = 5
    reset_timeout_seconds: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _opened_at: Optional[float] = field(default=None, init=False)
    _state: str = field(default="closed", init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open" and self._opened_at is not None:
                if self.clock() - self._opened_at >= self.reset_timeout_seconds:
                    self._state = "half_open"
            return self._state

    def allow(self) -> bool:
        return self.state != "open"

    def check(self) -> None:
        """Raise :class:`CircuitOpen` when the breaker refuses a call."""
        if not self.allow():
            raise CircuitOpen(
                f"circuit open after {self._failures} consecutive failures; "
                f"retrying in {self._remaining():.0f}s"
            )

    def _remaining(self) -> float:
        if self._opened_at is None:
            return 0.0
        elapsed = self.clock() - self._opened_at
        return max(0.0, self.reset_timeout_seconds - elapsed)

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "half_open":
                # the trial failed: straight back to open, do not ramp
                self._state = "open"
                self._opened_at = self.clock()
            elif self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self.clock()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._state = "closed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self._failures,
            "failure_threshold": self.failure_threshold,
            "reset_timeout_seconds": self.reset_timeout_seconds,
            "retry_in_seconds": round(self._remaining(), 1) if self._state == "open" else 0.0,
        }


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def run_with_retry(
    operation: Callable[[int], Any],
    policy: RetryPolicy,
    breaker: Optional[CircuitBreaker] = None,
    on_attempt: Optional[Callable[[int, Optional[BaseException], Optional[float]], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[bool, Any, Optional[BaseException], List[float]]:
    """Call ``operation(attempt)`` until it succeeds or the policy gives up.

    Returns ``(succeeded, value, error, delays)``.  ``value`` is whatever the
    operation returned on success; on failure it is ``None``.

    A :class:`PermanentFailure` short-circuits immediately - no sleeping, no
    second request.
    """
    delays: List[float] = []
    last_error: Optional[BaseException] = None
    attempt = 0
    while attempt < policy.max_attempts:
        attempt += 1
        if breaker is not None:
            breaker.check()
        try:
            value = operation(attempt)
        except PermanentFailure as exc:
            last_error = exc
            if on_attempt:
                on_attempt(attempt, exc, None)
            if breaker is not None:
                breaker.record_failure()
            return False, None, exc, delays
        except TransientFailure as exc:
            last_error = exc
            if on_attempt:
                on_attempt(attempt, exc, getattr(exc, "retry_after", None))
            if breaker is not None:
                breaker.record_failure()
            if not policy.should_retry(attempt, exc):
                return False, None, exc, delays
            delay = policy.delay_for(attempt, getattr(exc, "retry_after", None))
            delays.append(round(delay, 3))
            sleep(delay)
            continue
        except CircuitOpen:
            raise
        except Exception as exc:  # noqa: BLE001 - an unknown fault is permanent
            last_error = exc
            if on_attempt:
                on_attempt(attempt, exc, None)
            if breaker is not None:
                breaker.record_failure()
            return False, None, exc, delays
        else:
            if breaker is not None:
                breaker.record_success()
            if on_attempt:
                on_attempt(attempt, None, None)
            return True, value, None, delays
    return False, None, last_error, delays