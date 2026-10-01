"""Factory of the data sources. Adding a source = a new SampleSource subclass + a branch here."""

from __future__ import annotations

from emgacq.config import Config
from emgacq.sources.base import Chunk, SampleSource, SourceError, check_port, list_serial_ports

SOURCE_KINDS = ("cyton", "synthetic", "giga", "giga_sim")


def make_source(cfg: Config, kind: str | None = None, port: str | None = None) -> SampleSource:
    kind = kind or cfg["board.source"]
    b = cfg.section("board")
    if kind in ("cyton", "synthetic"):
        from emgacq.sources.brainflow_source import BrainFlowSource

        serial_port = (port or cfg.hw("board.serial_port")) if kind == "cyton" else None
        return BrainFlowSource(kind, serial_port, b["channels"], b["sampling_rate_hz"],
                               b["channel_command"], b["power_down_command"], b["full_scale_uv"],
                               synthetic=cfg.section("synthetic") if kind == "synthetic" else None)
    if kind in ("giga", "giga_sim"):
        from emgacq.sources.giga_source import GigaSource

        g = cfg.section("giga")
        url = port or (g["simulator_url"] if kind == "giga_sim" else cfg.hw("giga.url"))
        return GigaSource(url, b["channels"], g["sampling_rate_hz"], int(g["n_channels"]),
                          g["repo_root"], simulator=kind == "giga_sim")
    raise ValueError(f"unknown source {kind!r}, expected one of {SOURCE_KINDS}")


__all__ = ["Chunk", "SampleSource", "SourceError", "SOURCE_KINDS", "check_port",
           "list_serial_ports", "make_source"]
