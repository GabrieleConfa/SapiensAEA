"""Loading of config.toml, the single configuration file of this application."""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_ROOT / "config.toml"
TODO_HW = "TODO_HW"


class HardwareParamMissing(RuntimeError):
    """A config value that depends on unknown hardware is still TODO_HW."""


class Config:
    """Read-only view of the TOML config with dotted-key access."""

    def __init__(self, data: dict[str, Any], path: Path, raw: bytes):
        self._data = data
        self.path = path
        self.raw = raw
        self.sha256 = hashlib.sha256(raw).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(f"config key '{key}' not found in {self.path}")
        return value

    def hw(self, key: str) -> Any:
        """Return a hardware-dependent value, refusing placeholders."""
        value = self[key]
        if value == TODO_HW:
            raise HardwareParamMissing(
                f"config key '{key}' is still {TODO_HW} in {self.path}; set it (or pass it on "
                "the command line) once the hardware detail is known")
        return value

    def section(self, key: str) -> dict[str, Any]:
        return dict(self[key])

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def path_of(self, key: str) -> Path:
        """A path from the config, relative to the application folder."""
        p = Path(self[key])
        return p if p.is_absolute() else APP_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    """Load the config from `path`, $EMGACQ_CONFIG, or config.toml next to the package."""
    path = Path(path or os.environ.get("EMGACQ_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = path.read_bytes()
    return Config(tomllib.loads(raw.decode("utf-8")), path, raw)
