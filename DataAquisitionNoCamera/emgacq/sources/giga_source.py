"""Arduino GIGA stream of the main project (or its simulator), behind the same interface.

Uses the GIGA serial source of the main project (`common.sources.giga`, in `[giga] repo_root`).
Differences from the Cyton, hidden from the rest of the application:
- package_num is the GIGA's sample counter (never wraps, so no modulus);
- the device cannot stamp markers, so a marker lands on the first sample read after the call;
- timestamp = laptop wall clock of the first sample + counter / fs (the GIGA clock).
With the simulator, every gesture marker also sets the simulated grip, so the signal follows
the protocol.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from emgacq import markers as mk
from emgacq.config import APP_ROOT
from emgacq.sources.base import Chunk, SampleSource, SourceError

SIM_GRIPS = {"OPEN": "OPEN", "CLOSE": "FIST", "INDEX": "POINT"}


class GigaSource(SampleSource):
    package_modulus = None

    def __init__(self, url: str, channels: list[int], fs: float, n_board_channels: int,
                 repo_root: str, simulator: bool = False):
        super().__init__(channels, fs, full_scale_uv=float("inf"))
        self.kind = "giga_sim" if simulator else "giga"
        self.url = url
        self.n_board_channels = n_board_channels
        root = Path(repo_root)
        self.repo_root = root if root.is_absolute() else (APP_ROOT / root).resolve()
        self.simulator = simulator
        self._src = None
        self._consumed = 0
        self._pending: deque[int] = deque()
        self._lock = threading.Lock()
        self._t0: tuple[float, int] | None = None
        self._rng = np.random.default_rng(0)

    def connect(self) -> dict:
        if str(self.repo_root) not in sys.path:
            sys.path.append(str(self.repo_root))
        try:
            from common.sources.giga import GigaSerialSource
        except ImportError as e:
            raise SourceError(f"the GIGA source needs the main project at {self.repo_root}: {e}") from e
        src = GigaSerialSource(self.url, self.n_board_channels, self.fs)
        try:
            info = src.open()
        except Exception as e:
            raise SourceError(f"cannot open the GIGA on {self.url}: {e}") from e
        self._src = src
        self.full_scale_uv = float(src.saturation_uv)
        self.info = {"source": self.kind, "url": self.url, "fs": self.fs, "channels": self.channels,
                     "full_scale_uv": self.full_scale_uv, "package_modulus": None, "device": info}
        self.command_log.append({"command": "INFO", "reply": info, "time": time.time()})
        return self.info

    def start(self) -> None:
        self._consumed = self._src.buffer.total
        self._t0 = None
        self._src.start()

    def read(self) -> Chunk:
        if self._src is None:
            return Chunk.empty(self.n_channels)
        buf = self._src.buffer
        samples, counters, _flags = buf.since(self._consumed)
        self._consumed += len(counters)
        n = len(counters)
        if n == 0:
            return Chunk.empty(self.n_channels)
        if self._t0 is None:
            self._t0 = (time.time() - (n - 1) / self.fs, int(counters[0]))
        ts = self._t0[0] + (counters - self._t0[1]) / self.fs
        marker = np.zeros(n)
        with self._lock:
            k = 0
            while self._pending and k < n:
                marker[k] = self._pending.popleft()
                k += 1
        uv = samples[:, [c - 1 for c in self.channels]].astype(float)
        return Chunk(ts, counters.astype(float), uv, marker)

    def insert_marker(self, code: int) -> None:
        with self._lock:
            self._pending.append(int(code))
        if self.simulator and self._src is not None:
            info = mk.decode(code)
            if info.label in SIM_GRIPS:
                self._src.send(f"SIM GRIP {SIM_GRIPS[info.label]} {self._rng.uniform(0.5, 1.3):.2f}")
            elif info.phase:
                self._src.send("SIM GRIP REST 0")

    def release(self) -> None:
        src, self._src = self._src, None
        if src is not None:
            try:
                src.stop()
            except Exception:
                pass
