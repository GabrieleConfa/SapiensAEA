"""Common interface of the live EMG sources used by the acquisition and the quality check."""

from __future__ import annotations

import abc
import time

from common.stream import StreamBuffer


class RawSink(abc.ABC):
    """Receives everything a source reads, before any processing, for lossless recording."""

    @abc.abstractmethod
    def write_bytes(self, data: bytes, host_time_ns: int) -> None: ...

    def close(self) -> None:
        pass


class EmgSource(abc.ABC):
    name: str = "source"
    supports_flash: bool = False

    def __init__(self, n_channels: int, fs: float, buffer_s: float = 30.0):
        self.n_channels = n_channels
        self.fs = fs
        self.buffer = StreamBuffer(n_channels, int(buffer_s * fs))
        self.info: dict = {}
        self.uv_per_lsb = 1.0
        self.saturation_uv = float("inf")   # |value| at or above this counts as saturated
        self.messages: list[tuple[int, str]] = []   # (host_time_ns, text) from the device
        self.error: str | None = None
        self.started_ns: int | None = None

    @abc.abstractmethod
    def open(self) -> dict:
        """Connect, check the device and return its info dict."""

    @abc.abstractmethod
    def start(self, raw_sink: RawSink | None = None) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    def flash(self) -> bool:
        """Request an LED flash. Returns False if the source cannot drive the LED."""
        return False

    def send(self, command: str) -> None:
        pass

    def stats(self) -> dict:
        elapsed = (time.perf_counter_ns() - self.started_ns) / 1e9 if self.started_ns else 0.0
        return {"samples": self.buffer.total,
                "rate_hz": self.buffer.total / elapsed if elapsed > 0 else 0.0}
