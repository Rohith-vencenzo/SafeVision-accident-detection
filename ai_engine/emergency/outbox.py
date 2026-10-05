"""Durable notification outbox.  Phase 4.

The in-memory de-duplication this project started with has two fatal flaws for a
system whose whole job is to alert somebody:

1. **It is lost on restart.**  A crash between "call placed" and "record
   written" means the incident is either re-notified or silently never
   notified, and nothing on disk can say which.
2. **It cannot be shared.**  Two processes guarding the same camera race.

This module persists one row per ``(incident_id, channel)`` in SQLite with a
``UNIQUE`` constraint, so exactly-once is a property of the *schema* rather than
of a code path remembering to check.  Duplicate triggers, worker restarts and
provider retries all collapse onto the same row.

Safety properties:

* ``claim`` is atomic - two workers cannot hold the same row.
* A lease reclaims rows abandoned by a crashed worker, so a kill -9 mid-notification
  self-heals instead of stranding the row in ``IN_FLIGHT`` forever.
* The idempotency key is written **before** the provider request, never after.

The outbox stores no credentials and no message bodies beyond what is needed to
retry; the recipient is stored masked.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_engine.utils.logging_utils import get_logger

__all__ = [
    "CHANNELS",
    "DeliveryRow",
    "NotificationOutbox",
    "OutboxState",
]

_LOGGER = get_logger("emergency.outbox")

#: The two channels this project sends on.
CHANNELS = ("call", "sms")


class OutboxState:
    """Row states.  Deliberately strings so the DB and the API agree."""

    #: created, not yet attempted
    PENDING = "PENDING"
    #: claimed by a worker; provider request may be in flight
    IN_FLIGHT = "IN_FLIGHT"
    #: terminal success (call reached the network / SMS accepted by provider)
    DONE = "DONE"
    #: terminal failure after retries
    FAILED = "FAILED"
    #: deliberately not attempted
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    #: no provider configured - **Phase 4**
    NOT_CONFIGURED = "NOT_CONFIGURED"

    TERMINAL = frozenset({DONE, FAILED, SKIPPED, BLOCKED, NOT_CONFIGURED})


_SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_outbox (
    incident_id     TEXT    NOT NULL,
    channel         TEXT    NOT NULL,
    state           TEXT    NOT NULL,
    idempotency_key TEXT    NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 0,
    provider        TEXT,
    provider_ref    TEXT,
    provider_status TEXT,
    recipient_masked TEXT,
    correlation_id  TEXT,
    detail          TEXT,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    claimed_at      REAL,
    lease_seconds   REAL,
    UNIQUE (incident_id, channel)
);
CREATE INDEX IF NOT EXISTS ix_outbox_state ON notification_outbox (state, claimed_at);
"""


@dataclass
class DeliveryRow:
    """One (incident, channel) delivery."""

    incident_id: str
    channel: str
    state: str = OutboxState.PENDING
    idempotency_key: str = ""
    attempt: int = 0
    provider: str = ""
    provider_ref: str = ""
    provider_status: str = ""
    recipient_masked: str = ""
    correlation_id: str = ""
    detail: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    claimed_at: Optional[float] = None
    lease_seconds: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return self.state in OutboxState.TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "channel": self.channel,
            "state": self.state,
            "idempotency_key": self.idempotency_key,
            "attempt": self.attempt,
            "provider": self.provider or None,
            "provider_ref": self.provider_ref or None,
            "provider_status": self.provider_status or None,
            "recipient_masked": self.recipient_masked or None,
            "correlation_id": self.correlation_id or None,
            "detail": self.detail or None,
            "is_terminal": self.is_terminal,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DeliveryRow":
        return cls(
            incident_id=row["incident_id"],
            channel=row["channel"],
            state=row["state"],
            idempotency_key=row["idempotency_key"],
            attempt=row["attempt"],
            provider=row["provider"] or "",
            provider_ref=row["provider_ref"] or "",
            provider_status=row["provider_status"] or "",
            recipient_masked=row["recipient_masked"] or "",
            correlation_id=row["correlation_id"] or "",
            detail=row["detail"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            claimed_at=row["claimed_at"],
            lease_seconds=row["lease_seconds"],
        )


class NotificationOutbox:
    """Transactional, exactly-once delivery store.

    Usage::

        with outbox as box:
            row = box.enqueue(incident_id, "call", recipient_masked)
            if row.attempted:          # already claimed before
                return                  # do not send a second time
            claim = box.claim(incident_id, "call")
            ... place the call ...
            box.complete(incident_id, "call", provider_ref=..., provider_status=...)

    Thread-safe for in-process use (``check_same_thread=False`` plus a lock) and
    safe across processes for the atomic claim, which relies on SQLite's write
    lock rather than the Python lock.
    """

    def __init__(
        self,
        path: str | os.PathLike = ":memory:",
        lease_seconds: float = 120.0,
        clock=time.time,
    ) -> None:
        self.path = str(path)
        self.lease_seconds = float(lease_seconds)
        self.clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "NotificationOutbox":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #
    def enqueue(
        self,
        incident_id: str,
        channel: str,
        recipient_masked: str = "",
        correlation_id: str = "",
        idempotency_key: str = "",
        provider: str = "",
    ) -> DeliveryRow:
        """Create the row if absent.  Returns the row that already existed, too.

        The caller distinguishes the two cases by checking ``state``: a fresh row
        is ``PENDING`` with ``attempt == 0``.  Using INSERT OR IGNORE (rather
        than read-then-write) is what makes this safe under concurrency.
        """
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
        now = self.clock()
        key = idempotency_key or f"sv-{incident_id}-{channel}"
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """INSERT OR IGNORE INTO notification_outbox
                       (incident_id, channel, state, idempotency_key, attempt, provider,
                        recipient_masked, correlation_id, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (incident_id, channel, OutboxState.PENDING, key, 0,
                     provider, recipient_masked, correlation_id, now, now),
                )
        row = self.get(incident_id, channel)
        assert row is not None  # just inserted or already present
        return row

    def claim(self, incident_id: str, channel: str, lease_seconds: Optional[float] = None) -> Optional[DeliveryRow]:
        """Atomically take ownership of a row, or return ``None``.

        ``None`` means: the row is already terminal, or another worker holds it
        with a live lease.  Either way the caller must **not** send.
        """
        now = self.clock()
        lease = float(self.lease_seconds if lease_seconds is None else lease_seconds)
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    """UPDATE notification_outbox
                          SET state = ?, attempt = attempt + 1, claimed_at = ?,
                              lease_seconds = ?, updated_at = ?
                        WHERE incident_id = ? AND channel = ?
                          AND ( state = ?
                                OR (state = ? AND claimed_at IS NOT NULL
                                    AND claimed_at + COALESCE(lease_seconds, 0) < ?) )""",
                    (OutboxState.IN_FLIGHT, now, lease, now,
                     incident_id, channel,
                     OutboxState.PENDING,      # never attempted
                     OutboxState.IN_FLIGHT,     # a worker crashed; lease expired
                     now),
                )
                claimed = cursor.rowcount > 0
        return self.get(incident_id, channel) if claimed else None

    def complete(
        self,
        incident_id: str,
        channel: str,
        state: str = OutboxState.DONE,
        provider: str = "",
        provider_ref: str = "",
        provider_status: str = "",
        detail: str = "",
    ) -> Optional[DeliveryRow]:
        """Write a terminal outcome.  Rejects a non-terminal state by mistake."""
        if state not in OutboxState.TERMINAL:
            raise ValueError(f"{state!r} is not terminal; use claim() then complete()")
        return self._update(
            incident_id, channel,
            state=state, provider=provider, provider_ref=provider_ref,
            provider_status=provider_status, detail=detail,
        )

    def release(self, incident_id: str, channel: str, detail: str = "") -> Optional[DeliveryRow]:
        """Return an ``IN_FLIGHT`` row to ``PENDING`` so a retry can pick it up.

        Used when the provider request was never made (breaker open, transient
        failure with attempts left).
        """
        return self._update(incident_id, channel, state=OutboxState.PENDING, detail=detail)

    def _update(
        self,
        incident_id: str,
        channel: str,
        state: Optional[str] = None,
        provider: str = "",
        provider_ref: str = "",
        provider_status: str = "",
        detail: str = "",
    ) -> Optional[DeliveryRow]:
        now = self.clock()
        sets = ["updated_at = ?"]
        args: List[Any] = [now]
        if state is not None:
            sets.append("state = ?")
            args.append(state)
        for column, value in (
            ("provider", provider),
            ("provider_ref", provider_ref),
            ("provider_status", provider_status),
            ("detail", detail),
        ):
            if value:
                sets.append(f"{column} = ?")
                args.append(value)
        args.extend([incident_id, channel])
        with self._lock:
            with self._conn:
                self._conn.execute(
                    f"UPDATE notification_outbox SET {', '.join(sets)} "
                    "WHERE incident_id = ? AND channel = ?",
                    args,
                )
        return self.get(incident_id, channel)

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #
    def get(self, incident_id: str, channel: str) -> Optional[DeliveryRow]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM notification_outbox WHERE incident_id = ? AND channel = ?",
                (incident_id, channel),
            )
            row = cursor.fetchone()
        return DeliveryRow.from_row(row) if row else None

    def state_of(self, incident_id: str, channel: str) -> Optional[str]:
        row = self.get(incident_id, channel)
        return row.state if row else None

    def for_incident(self, incident_id: str) -> List[DeliveryRow]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM notification_outbox WHERE incident_id = ? ORDER BY channel",
                (incident_id,),
            )
            return [DeliveryRow.from_row(row) for row in cursor.fetchall()]

    def pending(self, limit: int = 100) -> List[DeliveryRow]:
        """Rows still to attempt - the worker's queue after a restart."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM notification_outbox WHERE state IN (?, ?) ORDER BY created_at LIMIT ?",
                (OutboxState.PENDING, OutboxState.IN_FLIGHT, int(limit)),
            )
            return [DeliveryRow.from_row(row) for row in cursor.fetchall()]

    def reclaim_expired(self, now: Optional[float] = None) -> int:
        """Release rows whose lease expired (a crashed worker).  Returns the count."""
        moment = self.clock() if now is None else now
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    """UPDATE notification_outbox
                          SET state = ?, claimed_at = NULL, updated_at = ?,
                              detail = 'lease expired; reclaimed'
                        WHERE state = ?
                          AND claimed_at IS NOT NULL
                          AND claimed_at + COALESCE(lease_seconds, 0) < ?""",
                    (OutboxState.PENDING, moment, OutboxState.IN_FLIGHT, moment),
                )
                return cursor.rowcount

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM notification_outbox GROUP BY state"
            )
            by_state = {row["state"]: row["n"] for row in cursor.fetchall()}
            cursor = self._conn.execute("SELECT COUNT(*) AS n FROM notification_outbox")
            total = cursor.fetchone()["n"]
        return {
            "path": self.path,
            "total_rows": total,
            "by_state": by_state,
            "exactly_once": "UNIQUE(incident_id, channel)",
        }

    def export_json(self) -> str:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM notification_outbox ORDER BY created_at")
            return json.dumps([dict(row) for row in cursor.fetchall()], indent=2)