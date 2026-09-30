"""Thread-safe ring buffer shared between a stream reader thread and its consumers."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from common.packet import FLAG_LED_ON, FLAG_LED_RISE


@dataclass(frozen=True)
class LedEdge:
    counter: int          # sample counter of the LED rising edge (EMG time base)
    host_time_ns: int     # laptop perf_counter_ns when the packet was received


class StreamBuffer:
    """Keeps the last `capacity` samples (µV) with their counters, flags and receive times."""

    def __init__(self, n_channels: int, capacity: int):
        self.n_channels = n_channels
        self.capacity = capacity
        self._lock = threading.Lock()
        self._samples = np.zeros((capacity, n_channels), np.float32)
        self._counters = np.zeros(capacity, np.int64)
        self._flags = np.zeros(capacity, np.uint8)
        self._host_ns = np.zeros(capacity, np.int64)
        self.total = 0                   # samples appended since creation
        self.led_edges: list[LedEdge] = []
        self.last_flags = 0

    def append(self, counters, flags, samples_uv, host_time_ns: int) -> None:
        n = len(counters)
        if n == 0:
            return
        counters = np.asarray(counters, np.int64)
        flags = np.asarray(flags, np.uint8)
        samples_uv = np.asarray(samples_uv, np.float32)
        if n > self.capacity:
            counters, flags, samples_uv = counters[-self.capacity:], flags[-self.capacity:], samples_uv[-self.capacity:]
            skipped = n - self.capacity
        else:
            skipped = 0
        with self._lock:
            m = len(counters)
            idx = (self.total + skipped + np.arange(m)) % self.capacity
            self._samples[idx] = samples_uv
            self._counters[idx] = counters
            self._flags[idx] = flags
            self._host_ns[idx] = host_time_ns
            self.total += n
            self.last_flags = int(flags[-1])
            for c in counters[(flags & FLAG_LED_RISE) != 0]:
                self.led_edges.append(LedEdge(int(c), host_time_ns))

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Last n samples (fewer if not available yet): samples (n, ch), counters, flags."""
        with self._lock:
            n = min(n, self.total, self.capacity)
            idx = (self.total - n + np.arange(n)) % self.capacity
            return self._samples[idx].copy(), self._counters[idx].copy(), self._flags[idx].copy()

    def since(self, start_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples appended after the moment `total` was `start_total` (bounded by capacity)."""
        with self._lock:
            n = min(self.total - start_total, self.capacity)
            idx = (self.total - n + np.arange(n)) % self.capacity
            return self._samples[idx].copy(), self._counters[idx].copy(), self._flags[idx].copy()

    @property
    def led_on(self) -> bool:
        return bool(self.last_flags & FLAG_LED_ON)

    def last_counter(self) -> int | None:
        with self._lock:
            if self.total == 0:
                return None
            return int(self._counters[(self.total - 1) % self.capacity])

    def edges_since(self, host_time_ns: int) -> list[LedEdge]:
        with self._lock:
            return [e for e in self.led_edges if e.host_time_ns >= host_time_ns]
