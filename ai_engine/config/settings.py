"""Configuration loading: ``config.yaml`` + CLI overrides + environment.

Priority (lowest to highest)::

    dataclass defaults  <  config.yaml  <  SAFEVISION_* env vars  <  --set overrides

Nothing in the engine reads a threshold from anywhere else, and no credential is
ever stored in a file: RTSP credentials come from environment variables.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

from ai_engine.config.classes import Config, ConfigError, build_dataclass

__all__ = [
    "CONFIG_ENV_VAR",
    "ConfigError",
    "build_source_url",
    "deep_merge",
    "describe_config",
    "find_project_root",
    "load_config",
    "parse_override",
    "project_path",
]

CONFIG_ENV_VAR = "SAFEVISION_CONFIG"
DEFAULT_CONFIG_NAMES = ("config.yaml", "config.yml")

# env var -> dotted config path
_ENV_OVERRIDES: Dict[str, str] = {
    "SAFEVISION_MODEL": "model.weights",
    "SAFEVISION_DEVICE": "model.device",
    "SAFEVISION_CONFIDENCE": "model.conf",
    "SAFEVISION_IOU": "model.iou",
    "SAFEVISION_SOURCE": "video.source",
    "SAFEVISION_PROCESS_FPS": "video.process_fps",
    "SAFEVISION_CAMERA_ID": "location.default_camera_id",
    "SAFEVISION_EVIDENCE_DIR": "evidence.root",
    "SAFEVISION_LOG_LEVEL": "runtime.log_level",
    "SAFEVISION_LOG_FILE": "runtime.log_file",
    "SAFEVISION_OUTPUT": "runtime.output_video",
    # Demonstration emergency call.  `DEMO_MODE` is the documented master switch
    # (it lives in .env, next to EMERGENCY_CONTACT_NUMBER).
    "DEMO_MODE": "emergency.demo_mode",
    "SAFEVISION_EMERGENCY_ENABLED": "emergency.enabled",
    "SAFEVISION_EMERGENCY_PROVIDER": "emergency.provider",
}

_REDACT_KEYS = {"password", "pass", "secret", "token", "api_key"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins)."""
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(assignment: str) -> Tuple[str, Any]:
    """Parse a ``--set section.key=value`` string into ``("section.key", value)``.

    Values are interpreted as JSON when possible so ``true``, ``12``, ``0.4``,
    ``[1,2]`` and ``{"a":1}`` all work; otherwise the raw string is used.
    """
    if "=" not in assignment:
        raise ConfigError(f"invalid override {assignment!r}: expected key=value")
    key, _, raw = assignment.partition("=")
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise ConfigError(f"invalid override {assignment!r}: empty key")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _set_dotted(target: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the project root (the folder holding ``config.yaml`` / ``.git``).

    Falls back to the parent of the ``ai_engine`` package, which is the
    repository layout we ship.
    """
    env_root = os.environ.get("SAFEVISION_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(start) if start else Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if candidate.is_dir():
            for name in DEFAULT_CONFIG_NAMES:
                if (candidate / name).is_file():
                    return candidate
            if (candidate / ".git").exists():
                return candidate
    return Path(__file__).resolve().parents[2]


def project_path(relative: str | os.PathLike, root: Optional[Path] = None) -> Path:
    """Resolve a project-relative path (absolute paths pass through)."""
    p = Path(relative).expanduser()
    if p.is_absolute():
        return p
    base = root or find_project_root()
    return (base / p).resolve()


# --------------------------------------------------------------------------- #
# config loading
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to read config files (pip install PyYAML)") from exc

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except Exception as exc:  # yaml.YAMLError and friends
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a top-level mapping")
    return data


def _env_overrides(env: Mapping[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for env_name, dotted in _ENV_OVERRIDES.items():
        if env_name in env and env[env_name] != "":
            raw = env[env_name]
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            _set_dotted(out, dotted, value)
    return out


def load_config(
    path: Optional[str | os.PathLike] = None,
    overrides: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    use_env: bool = True,
    validate: bool = True,
    root: Optional[Path] = None,
) -> Config:
    """Build a validated :class:`Config`.

    Parameters
    ----------
    path:
        Explicit path to a YAML config. When omitted, ``$SAFEVISION_CONFIG`` is
        used, else ``<project root>/config.yaml`` if present, else defaults.
    overrides:
        Iterable of ``"section.key=value"`` strings (CLI ``--set``).
    env:
        Environment mapping (defaults to ``os.environ``).
    use_env:
        Apply the ``SAFEVISION_*`` overrides.
    validate:
        Run range / cross-section validation.
    """
    environ = os.environ if env is None else env
    project_root = root or find_project_root()

    config_path: Optional[Path] = None
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
    elif environ.get(CONFIG_ENV_VAR):
        config_path = Path(environ[CONFIG_ENV_VAR]).expanduser()
        if not config_path.is_file():
            raise ConfigError(f"{CONFIG_ENV_VAR} points to a missing file: {config_path}")
    else:
        for name in DEFAULT_CONFIG_NAMES:
            candidate = project_root / name
            if candidate.is_file():
                config_path = candidate
                break

    raw: Dict[str, Any] = {}
    if config_path is not None:
        raw = _read_yaml(config_path)

    if use_env:
        raw = deep_merge(raw, _env_overrides(environ))

    for assignment in overrides or []:
        key, value = parse_override(assignment)
        _set_dotted(raw, key, value)

    data = raw.pop("name", "safevision-config") if "name" in raw else "safevision-default"
    version = str(raw.pop("version", "1.0.0")) if "version" in raw else "1.0.0"

    config = build_dataclass(Config, raw, path="config")
    config.name = data
    config.version = version
    # remember where this config came from (handy for logs / --print-config)
    config.source_file = str(config_path) if config_path else "<defaults>"
    if validate:
        config.validate()
    return config


def describe_config(config: Config) -> Dict[str, Any]:
    """Config as a plain dict, with any secret-looking value redacted."""
    data = config.to_dict()

    def _redact(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: ("***" if k.lower() in _REDACT_KEYS else _redact(v)) for k, v in node.items()}
        return node

    data["source_file"] = getattr(config, "source_file", "<defaults>")
    return _redact(data)


# --------------------------------------------------------------------------- #
# video source helpers
# --------------------------------------------------------------------------- #
def build_source_url(
    source: str | int,
    username_env: str = "SAFEVISION_RTSP_USERNAME",
    password_env: str = "SAFEVISION_RTSP_PASSWORD",
    env: Optional[Mapping[str, str]] = None,
) -> str | int:
    """Return the source, injecting RTSP credentials from the environment.

    ``rtsp://user:pass@host/stream`` is built only when the URL has no
    credentials *and* the env vars are set.  Credentials are never written to
    config files, logs or incident JSON.
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return int(source)

    text = str(source)
    parsed = urlparse(text)
    if parsed.scheme not in ("rtsp", "rtsps", "rtmp"):
        return text

    environ = os.environ if env is None else env
    if parsed.username is not None:
        return text  # already authenticated in the URL

    username = (environ.get(username_env) or "").strip()
    password = environ.get(password_env) or ""
    if not username:
        return text

    netloc = f"{username}:{password}@{parsed.hostname or ''}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def iter_env_keys() -> Iterable[str]:
    """Env var names SafeVision understands (for ``--help`` / docs)."""
    return sorted(_ENV_OVERRIDES)


def redact_url(url: str | int) -> str:
    """Strip credentials from a URL so it is safe to log or store."""
    if isinstance(url, int):
        return f"webcam:{url}"
    parsed = urlparse(str(url))
    if parsed.username is None:
        return str(url)
    netloc = f"***@{parsed.hostname or ''}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
