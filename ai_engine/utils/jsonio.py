"""JSON helpers.

The incident payload is the contract with the rest of the project, so every
value that leaves the engine goes through :func:`to_jsonable` first: numpy
scalars/arrays become plain Python types and floats are rounded, which keeps
the output diff-friendly and makes it stable across runs.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_json", "dumps", "read_json", "to_jsonable"]


def to_jsonable(obj: Any, ndigits: int = 6) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable primitives.

    * ``numpy`` scalars / arrays -> ``int`` / ``float`` / ``list``
    * ``pathlib.Path``  -> ``str``
    * ``set`` / ``tuple`` -> ``list``
    * ``float('nan')`` / ``inf`` -> ``None`` (invalid JSON, never emitted)
    * dataclass instances -> ``dataclasses.asdict``
    * objects exposing ``to_dict()`` -> that dict
    * everything else -> unchanged
    """
    # numpy is an optional import at module scope for startup speed
    try:  # pragma: no cover - trivial
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits)
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return None if (math.isnan(value) or math.isinf(value)) else round(value, ndigits)
        if isinstance(obj, np.ndarray):
            return [to_jsonable(v, ndigits) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, ndigits) for v in obj]

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict(), ndigits)

    try:  # dataclasses (avoid importing dataclasses cost on the hot path)
        import dataclasses

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                f.name: to_jsonable(getattr(obj, f.name), ndigits)
                for f in dataclasses.fields(obj)
            }
    except Exception:  # pragma: no cover
        pass

    return obj


def dumps(obj: Any, indent: int | None = 2, ndigits: int = 6) -> str:
    """``json.dumps`` on a sanitised copy of ``obj`` (never emits NaN)."""
    return json.dumps(to_jsonable(obj, ndigits), indent=indent, allow_nan=False, default=str)


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 2) -> str:
    """Write JSON via a temp file + atomic replace.

    A crash mid-write can therefore never leave a half-written incident file
    that the downstream dashboard would try to parse.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps(obj, indent=indent)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return str(target)


def read_json(path: str | os.PathLike) -> Any:
    """Read JSON, raising a clear error instead of a bare ``FileNotFoundError``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)
