"""Incremental writers of emg.csv and events.csv, and the bad-trial update at session close."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HEAD_COLUMNS = ["sample_index", "timestamp", "package_num"]
TAIL_COLUMNS = ["marker", "phase", "label", "block", "trial_id", "trial_bad"]
EVENT_COLUMNS = ["wall_time", "unix_time", "sample_index", "marker", "event", "phase", "gesture",
                 "block", "trial_id", "detail"]


def emg_columns(channel_columns: list[str]) -> list[str]:
    return HEAD_COLUMNS + list(channel_columns) + TAIL_COLUMNS


class EmgCsvWriter:
    """Writes one row per sample and flushes (with fsync) at least every `flush_interval_s`.

    A crash therefore loses at most that much data. trial_bad is written as 0; the trials flagged
    bad are updated once, when the session closes (mark_bad_trials).
    """

    def __init__(self, path: Path, channel_columns: list[str], flush_interval_s: float = 1.0):
        self.path = Path(path)
        self.columns = emg_columns(channel_columns)
        self.flush_interval_s = flush_interval_s
        self._f = open(self.path, "w", newline="\n", encoding="ascii")
        self._f.write(",".join(self.columns) + "\n")
        self.rows = 0
        self._last_flush = time.monotonic()
        self.flush(force=True)

    def write(self, timestamp: np.ndarray, package_num: np.ndarray, data_uv: np.ndarray,
              marker: np.ndarray, phase: np.ndarray, label: np.ndarray, block: np.ndarray,
              trial_id: np.ndarray) -> range:
        """Append rows; returns their sample_index range."""
        n = len(timestamp)
        first = self.rows
        lines = []
        for i in range(n):
            ch = ",".join(f"{v:.4f}" for v in data_uv[i])
            lines.append(f"{first + i},{timestamp[i]:.6f},{int(package_num[i])},{ch},"
                         f"{int(round(marker[i]))},{phase[i]},{label[i]},{int(block[i])},"
                         f"{int(trial_id[i])},0\n")
        self._f.write("".join(lines))
        self.rows += n
        self.flush()
        return range(first, first + n)

    def flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_flush >= self.flush_interval_s:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._last_flush = now

    def close(self) -> None:
        if not self._f.closed:
            self.flush(force=True)
            self._f.close()


def mark_bad_trials(path: Path, trial_ids: set[int] | list[int],
                    ranges: list[tuple[int, int]] = ()) -> int:
    """Set trial_bad = 1 on the rows of the given trials and sample ranges [start, end).

    The other bytes of the file are kept as they are. Returns the number of rows set to bad.
    """
    path = Path(path)
    trial_ids = {int(t) for t in trial_ids}
    if not trial_ids and not ranges:
        return 0
    tmp = path.with_suffix(".tmp")
    n_bad = 0
    with open(path, encoding="ascii", newline="") as src, \
            open(tmp, "w", encoding="ascii", newline="\n") as dst:
        header = src.readline()
        cols = header.rstrip("\n").split(",")
        i_trial = cols.index("trial_id")
        if cols[-1] != "trial_bad":
            raise ValueError(f"{path}: last column is {cols[-1]!r}, expected 'trial_bad'")
        dst.write(header)
        for line in src:
            fields = line.rstrip("\n").split(",")
            sample = int(fields[0])
            bad = (int(fields[i_trial]) in trial_ids
                   or any(a <= sample < b for a, b in ranges))
            if bad:
                fields[-1] = "1"
                n_bad += 1
            dst.write(",".join(fields) + "\n")
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, path)
    return n_bad


class EventLog:
    """events.csv: one row per event, flushed immediately. Rows are also kept in memory."""

    def __init__(self, path: Path | None):
        self.rows: list[dict] = []
        self._f = None
        if path is not None:
            self._f = open(path, "w", newline="", encoding="utf-8")
            self._w = csv.DictWriter(self._f, fieldnames=EVENT_COLUMNS)
            self._w.writeheader()
            self._f.flush()

    def log(self, event: str, marker: int | None = None, phase: str = "", gesture: str = "",
            block: int = 0, trial_id: int = 0, sample_index: int | None = None,
            detail: str = "") -> dict:
        now = time.time()
        row = {"wall_time": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
               "unix_time": f"{now:.3f}",
               "sample_index": "" if sample_index is None else sample_index,
               "marker": "" if marker is None else marker, "event": event, "phase": phase,
               "gesture": gesture, "block": block, "trial_id": trial_id, "detail": detail}
        self.rows.append(row)
        if self._f is not None and not self._f.closed:
            self._w.writerow(row)
            self._f.flush()
        return row

    def close(self) -> None:
        if self._f is not None and not self._f.closed:
            self._f.close()
