"""Interface of the data sources.

The rest of the application only sees `SampleSource`: samples in µV of the channels in use, a
timestamp and a package number per sample, and a marker column. A marker inserted with
`insert_marker` lands on the next sample the source delivers (BrainFlow semantics), one marker
per sample if several are queued.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

GUI_HINT = ("Close the OpenBCI GUI (or stop its session) first: two programs cannot hold the "
            "dongle's serial port at the same time.")


class SourceError(RuntimeError):
    pass


@dataclass
class Chunk:
    timestamp: np.ndarray      # (n,) seconds; BrainFlow: host wall clock when the packet arrived
    package_num: np.ndarray    # (n,) board package counter
    data_uv: np.ndarray        # (n, channels in use) raw µV
    marker: np.ndarray         # (n,) 0 or a marker code

    def __len__(self) -> int:
        return len(self.timestamp)

    @staticmethod
    def empty(n_channels: int) -> "Chunk":
        z = np.zeros(0)
        return Chunk(z, z, np.zeros((0, n_channels)), z)


class SampleSource(abc.ABC):
    kind: str = "source"
    package_modulus: int | None = 256

    def __init__(self, channels: list[int], fs: float, full_scale_uv: float):
        self.channels = list(channels)
        self.fs = float(fs)
        self.full_scale_uv = float(full_scale_uv)
        self.info: dict = {}
        self.command_log: list[dict] = []   # every configuration command and the board reply

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @abc.abstractmethod
    def connect(self) -> dict:
        """Open the device, configure it and return an info dict. Raises SourceError."""

    @abc.abstractmethod
    def start(self) -> None:
        """Start streaming."""

    @abc.abstractmethod
    def read(self) -> Chunk:
        """All samples received since the previous call (possibly none)."""

    @abc.abstractmethod
    def insert_marker(self, code: int) -> None: ...

    @abc.abstractmethod
    def release(self) -> None:
        """Stop the stream and free the device. Safe to call more than once."""

    def reconnect(self) -> dict:
        self.release()
        info = self.connect()
        self.start()
        return info

    def simulate_stall(self, seconds: float) -> bool:
        """Testing aid: drop all data for `seconds`. Only the synthetic source supports it."""
        return False


def list_serial_ports() -> list[tuple[str, str]]:
    """(device, description) of the serial ports present."""
    from serial.tools import list_ports

    return [(p.device, p.description or "") for p in list_ports.comports()]


def check_port(port: str) -> None:
    """Raise SourceError with the list of ports if `port` does not exist."""
    if port.startswith(("socket://", "loop://")):
        return
    ports = list_serial_ports()
    devices = {d.upper() for d, _ in ports}
    if port.upper() not in devices:
        listing = "\n".join(f"  {d}  {desc}" for d, desc in ports) or "  (no serial ports found)"
        raise SourceError(f"serial port {port} not found. Ports present:\n{listing}\n"
                          "Is the dongle plugged in? The OpenBCI dongle usually shows up as an "
                          "FTDI 'USB Serial Port'.")
