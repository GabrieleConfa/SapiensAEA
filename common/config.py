"""Loading of config/config.toml, the single configuration file."""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.toml"
TODO_HW = "TODO_HW"


class HardwareParamMissing(RuntimeError):
    """A config value that depends on unknown hardware is still TODO_HW."""


class Config:
    """Read-only view of the TOML config with dotted-key access."""

    def __init__(self, data: dict[str, Any], path: Path, raw: bytes):
        self._data = data
        self.path = path
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
                f"config key '{key}' is still {TODO_HW} in {self.path}; "
                "set it once the hardware detail is known"
            )
        return value

    def is_set(self, key: str) -> bool:
        return self.get(key, TODO_HW) != TODO_HW

    def section(self, key: str) -> dict[str, Any]:
        return dict(self[key])

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    """Load the config from `path`, $EMG_CONFIG, or config/config.toml."""
    path = Path(path or os.environ.get("EMG_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = path.read_bytes()
    return Config(tomllib.loads(raw.decode("utf-8")), path, raw)


def todo_hw_keys(cfg: Config) -> list[str]:
    """List every dotted key whose value is still TODO_HW."""
    found: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif node == TODO_HW:
            found.append(prefix)

    walk(cfg.as_dict(), "")
    return found
