"""Configuration loading for the AgriBot stack.

One YAML file (``config/robot.yaml``) is the single source of truth for every
tunable in the system. This module loads it, applies environment-variable
overrides, and exposes it as a dotted-access mapping so call sites read as
``cfg.navigation.pid.kp`` rather than ``cfg["navigation"]["pid"]["kp"]``.

Overrides are applied in increasing order of precedence:

1. ``config/robot.yaml``          (checked-in defaults)
2. ``config/robot.local.yaml``    (per-machine, git-ignored, optional)
3. ``AGRIBOT_<SECTION>__<KEY>``   (environment, double underscore = nesting)

The environment layer exists so that a systemd unit or a CI job can flip a
single value (e.g. ``AGRIBOT_SPRAY__ENABLED=false``) without editing files.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import yaml

__all__ = ["Config", "load_config", "find_config", "PROJECT_ROOT"]

# src/agribot/config.py -> src/agribot -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_PREFIX = "AGRIBOT_"
_ENV_NEST = "__"


class Config(Mapping):
    """Read-only, dotted-access view over a nested mapping.

    Behaves like a ``dict`` (so ``**cfg`` and ``cfg["k"]`` work) while also
    supporting attribute access for nested sections. Nested dicts are wrapped
    lazily on access; lists of dicts are wrapped element-wise.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # -- Attribute access ---------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return _wrap(self._data[key])
        except KeyError:
            raise AttributeError(
                f"config has no key {key!r}; available: {sorted(self._data)}"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("Config is read-only; edit robot.yaml or use .merged()")

    # -- Helpers ------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return _wrap(self._data.get(key, default))

    def dotted(self, path: str, default: Any = None) -> Any:
        """Fetch by dotted path, e.g. ``cfg.dotted("navigation.pid.kp")``."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return _wrap(node)

    def to_dict(self) -> Dict[str, Any]:
        """Deep copy as plain Python containers (safe to mutate or serialise)."""
        return copy.deepcopy(self._data)

    def merged(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with ``overrides`` deep-merged on top."""
        return Config(_deep_merge(self.to_dict(), overrides))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({sorted(self._data)})"


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning ``base``."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    return base


def _coerce(text: str) -> Any:
    """Parse an environment string into the most specific YAML scalar."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _env_overrides(environ: Mapping[str, str]) -> Dict[str, Any]:
    """Build a nested override dict from ``AGRIBOT_A__B__C=value`` variables."""
    out: Dict[str, Any] = {}
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(_ENV_PREFIX):
            continue
        path = raw_key[len(_ENV_PREFIX):].lower().split(_ENV_NEST)
        if not path or not path[0]:
            continue
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # conflicting scalar already set
                break
        else:
            node[path[-1]] = _coerce(raw_value)
    return out


def find_config(explicit: Optional[os.PathLike] = None) -> Path:
    """Resolve the config path: explicit arg, then $AGRIBOT_CONFIG, then default."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_path = os.environ.get("AGRIBOT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return PROJECT_ROOT / "config" / "robot.yaml"


def load_config(
    path: Optional[os.PathLike] = None,
    *,
    use_local: bool = True,
    use_env: bool = True,
    environ: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Load the master configuration with all override layers applied.

    Args:
        path: explicit YAML path; defaults to ``config/robot.yaml``.
        use_local: also merge a sibling ``*.local.yaml`` if present.
        use_env: apply ``AGRIBOT_*`` environment overrides.
        environ: environment mapping to read (defaults to ``os.environ``).
        overrides: final programmatic overrides, highest precedence of all.

    Raises:
        FileNotFoundError: if the base config file is missing.
        ValueError: if the YAML does not parse to a mapping.
    """
    cfg_path = find_config(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"AgriBot config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a YAML mapping at the top level")

    if use_local:
        local_path = cfg_path.with_name(cfg_path.stem + ".local" + cfg_path.suffix)
        if local_path.is_file():
            with local_path.open("r", encoding="utf-8") as fh:
                local_data = yaml.safe_load(fh) or {}
            if not isinstance(local_data, dict):
                raise ValueError(f"{local_path} must contain a YAML mapping")
            _deep_merge(data, local_data)

    if use_env:
        _deep_merge(data, _env_overrides(os.environ if environ is None else environ))

    if overrides:
        _deep_merge(data, overrides)

    data.setdefault("_meta", {})["config_path"] = str(cfg_path)
    return Config(data)
