"""Files written during a recording. Everything is flushed as it goes, so a crash loses at most the
last few rows."""

from __future__ import annotations

import csv
import threading
from pathlib import Path

from common.sources.base import RawSink


class RawStreamSink(RawSink):
    """Byte-exact copy of the device stream + one (byte offset, host time) row per read.

    emg_raw.bin         everything received from the serial port, unmodified
    emg_host_times.csv  end_offset,host_time_ns  (end_offset = stream length after that read)
    """

    def __init__(self, data_path: Path, times_path: Path, header: bytes = b""):
        self._data = open(data_path, "wb")
        self._times = open(times_path, "w", newline="", encoding="ascii")
        self._times.write("end_offset,host_time_ns\n")
        self._offset = 0
        self._lock = threading.Lock()
        if header:
            self._data.write(header)
            self._offset = len(header)

    def write_bytes(self, data: bytes, host_time_ns: int) -> None:
        with self._lock:
            self._data.write(data)
            self._offset += len(data)
            self._times.write(f"{self._offset},{host_time_ns}\n")
            if self._offset % 65536 < len(data):  # flush roughly every 64 kB
                self._data.flush()
                self._times.flush()

    @property
    def bytes_written(self) -> int:
        return self._offset

    def close(self) -> None:
        with self._lock:
            self._data.close()
            self._times.close()


class CsvLog:
    """Append-only CSV with a fixed header, flushed after every row (thread-safe)."""

    def __init__(self, path: Path, columns: list[str]):
        self.columns = columns
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=columns, extrasaction="raise")
        self._w.writeheader()
        self._lock = threading.Lock()
        self.rows = 0

    def write(self, **row) -> None:
        with self._lock:
            self._w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in self.columns})
            self._f.flush()
            self.rows += 1

    def close(self) -> None:
        with self._lock:
            self._f.close()
