"""A tiny ``.env`` reader - no third-party dependency, no side effects.

``.env`` is the **only** place the demonstration recipient is stored.  It is
listed in ``.gitignore`` and must never be committed.

Rules
-----
* Real environment variables always win over ``.env`` - a value exported in the
  shell or set by CI overrides the file, so a developer can always run without
  dialling anybody.
* Loading is explicit (:func:`load_env_file`), never at import time, so the
  test-suite stays hermetic.
* Only ``KEY=VALUE`` lines are understood: ``#`` comments, blank lines, an
  optional ``export`` prefix and matching single/double quotes around the
  value.  Interpolation, nesting and multi-line values are deliberately not
  supported.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

__all__ = ["DOTENV_NAME", "env_file_path", "load_env_file", "parse_env_text"]

DOTENV_NAME = ".env"


def parse_env_text(text: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines into a dict.  Never raises on malformed lines."""
    result: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result


def env_file_path(root: Optional[Path] = None) -> Optional[Path]:
    """Return the project's ``.env``, or ``None`` when there is none."""
    from ai_engine.config.settings import find_project_root

    base = Path(root) if root is not None else find_project_root()
    candidate = base / DOTENV_NAME
    return candidate if candidate.is_file() else None


def load_env_file(
    path: Optional[str | os.PathLike] = None,
    environ: Optional[MutableMapping[str, str]] = None,
    root: Optional[Path] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Load ``.env`` into ``environ`` and return the keys that were applied.

    Parameters
    ----------
    path:
        Explicit ``.env`` path.  Defaults to ``<project root>/.env``.
    environ:
        Target mapping (defaults to :data:`os.environ`).
    override:
        When ``False`` (the default) an existing environment variable is never
        replaced by the file.
    """
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    source = Path(path).expanduser() if path is not None else env_file_path(root)
    if source is None or not source.is_file():
        return {}

    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return {}

    applied: Dict[str, str] = {}
    for key, value in parse_env_text(text).items():
        if override or key not in target:
            target[key] = value
            applied[key] = value
    return applied


def env_flag(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (``true/1/yes/on`` are truthy)."""
    raw = (environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("true", "yes", "1", "on"):
        return True
    if raw in ("false", "no", "0", "off"):
        return False
    return default
