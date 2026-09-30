"""Factory for the live EMG sources: the GIGA, the simulator, or the Cyton via BrainFlow."""

from __future__ import annotations

from common.config import Config
from common.sources.base import EmgSource, RawSink

SOURCE_KINDS = ("giga", "sim", "cyton")


def make_source(cfg: Config, kind: str, port: str | None = None) -> EmgSource:
    if kind in ("giga", "sim"):
        from common.sources.giga import GigaSerialSource

        url = port or (cfg["serial.simulator_url"] if kind == "sim" else cfg.hw("serial.port"))
        src = GigaSerialSource(url, int(cfg["emg.n_channels"]), float(cfg["emg.fs_hz"]),
                               int(cfg["serial.baudrate"]))
        src.name = kind
        return src
    if kind == "cyton":
        from common.sources.cyton import CytonSource

        return CytonSource(port or cfg.hw("cyton.serial_port"), int(cfg["emg.n_channels"]),
                           float(cfg["cyton.fs_hz"]), cfg["cyton.channel_command"])
    raise ValueError(f"unknown source {kind!r}, expected one of {SOURCE_KINDS}")


__all__ = ["EmgSource", "RawSink", "SOURCE_KINDS", "make_source"]
