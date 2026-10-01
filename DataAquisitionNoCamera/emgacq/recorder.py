"""Acquisition thread: pulls samples from the source, labels them from the markers, writes emg.csv.

The stream runs (and feeds the live view) from the moment the source is connected. Writing starts
with the first marker after `start_recording`, so emg.csv begins exactly at the first protocol
step, and ends with the SESSION_END / SESSION_ABORT marker row.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from emgacq.labels import Labeler
from emgacq.packets import PacketTracker
from emgacq.sources.base import Chunk, SampleSource
from emgacq.writer import EmgCsvWriter, EventLog


@dataclass
class LiveData:
    """New raw samples for the live view since the last call."""
    data_uv: np.ndarray
    total: int


@dataclass
class QcCapture:
    """Samples of the quality-check rest and fist steps, one array per step."""
    rest: list[np.ndarray] = field(default_factory=list)
    fist: list[np.ndarray] = field(default_factory=list)


class Recorder:
    def __init__(self, source: SampleSource, read_interval_s: float = 0.02):
        self.source = source
        self.read_interval_s = read_interval_s
        self.packets = PacketTracker(source.package_modulus)
        self.received = 0                   # samples received since connect (written or not)
        self.last_sample_time = time.monotonic()
        self.error: str | None = None
        self.read_errors = 0
        self._lock = threading.RLock()
        self._source_lock = threading.Lock()
        self._live: list[np.ndarray] = []
        self._thread: threading.Thread | None = None
        self._running = False
        # recording state
        self.writer: EmgCsvWriter | None = None
        self.labeler: Labeler | None = None
        self.events: EventLog | None = None
        self._writing = False
        self.finished = False               # end marker row written
        self.qc = QcCapture()
        self._qc_step = -1
        self._qc_parts: list[np.ndarray] = []
        self._qc_kind = ""
        self._loss_event_cursor = 0
        self._loss_log_window = (0.0, 0)    # (window start, lost count) for rate-limited logging

    # -- thread -----------------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self.last_sample_time = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="emg-recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(3.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                with self._source_lock:
                    chunk = self.source.read()
                if len(chunk):
                    self.process(chunk)
            except Exception as e:      # keep the thread alive; the gap detector reports it
                self.read_errors += 1
                self.error = f"{type(e).__name__}: {e}"
                if self.read_errors <= 3:
                    traceback.print_exc()
            dt = time.monotonic() - t0
            time.sleep(max(0.0, self.read_interval_s - dt))

    # -- recording --------------------------------------------------------------------------
    def start_recording(self, writer: EmgCsvWriter, labeler: Labeler, events: EventLog) -> None:
        with self._lock:
            self.writer, self.labeler, self.events = writer, labeler, events
            self._writing = False
            self.finished = False

    def stop_recording(self) -> None:
        with self._lock:
            if self.writer is not None:
                self.writer.close()
            self._writing = False

    @property
    def rows_written(self) -> int:
        return self.writer.rows if self.writer is not None else 0

    @property
    def markers_seen(self) -> int:
        return self.labeler.markers_seen if self.labeler is not None else 0

    def seconds_since_last_sample(self) -> float:
        return time.monotonic() - self.last_sample_time

    def reset_qc(self) -> None:
        with self._lock:
            self.qc = QcCapture()
            self._qc_step = -1
            self._qc_parts = []

    def process(self, chunk: Chunk) -> None:
        with self._lock:
            n = len(chunk)
            self.last_sample_time = time.monotonic()
            first_row = self.rows_written
            self.packets.update(chunk.package_num, first_row)
            self._log_packet_loss()
            self.received += n
            self._live.append(np.asarray(chunk.data_uv, float))
            if len(self._live) > 500:           # nobody is reading the live data (headless)
                del self._live[:-50]
            if self.writer is None or self.finished:
                return
            start = 0
            if not self._writing:
                nz = np.flatnonzero(chunk.marker)
                if nz.size == 0:
                    return
                start = int(nz[0])
                self._writing = True
            sl = slice(start, n)
            markers = chunk.marker[sl]
            lab = self.labeler.label_chunk(markers, self.writer.rows)
            stop = len(markers)
            if self.labeler.ended:
                stop = self.labeler.end_sample - self.writer.rows + 1
            cut = slice(start, start + stop)
            self.writer.write(chunk.timestamp[cut], chunk.package_num[cut], chunk.data_uv[cut],
                              chunk.marker[cut], lab.phase[:stop], lab.label[:stop],
                              lab.block[:stop], lab.trial_id[:stop])
            self._collect_qc(chunk.data_uv[cut], lab.phase[:stop], lab.label[:stop], lab.step[:stop])
            if self.labeler.ended:
                self.finished = True
                self.writer.flush(force=True)

    def _collect_qc(self, data: np.ndarray, phase: np.ndarray, label: np.ndarray,
                    step: np.ndarray) -> None:
        """Keep the samples of each quality-check rest / fist step, step by step."""
        for s in np.unique(step):
            rows = step == s
            if s != self._qc_step:
                self._close_qc_step()
                self._qc_step = int(s)
                ph, lb = phase[rows][0], label[rows][0]
                self._qc_kind = ("rest" if lb == "REST" else "fist") if ph == "QUALITY" and lb != "TRANSITION" else ""
            if self._qc_kind:
                self._qc_parts.append(np.asarray(data[rows], float))

    def _close_qc_step(self) -> None:
        if self._qc_kind and self._qc_parts:
            getattr(self.qc, self._qc_kind).append(np.concatenate(self._qc_parts))
        self._qc_parts = []
        self._qc_kind = ""

    def close_qc_capture(self) -> QcCapture:
        with self._lock:
            self._close_qc_step()
            self._qc_step = -1
            return self.qc

    def _log_packet_loss(self) -> None:
        """Lost packets go to events.csv, at most one row per second."""
        events = self.packets.loss_events
        new = events[self._loss_event_cursor:]
        self._loss_event_cursor = len(events)
        if not new:
            return
        now = time.monotonic()
        start, count = self._loss_log_window
        count += sum(lost for _, lost in new)
        if self.events is not None and now - start >= 1.0:
            self.events.log("PACKET_LOSS", sample_index=new[-1][0],
                            detail=f"{count} samples lost (total {self.packets.lost})")
            self._loss_log_window = (now, 0)
        else:
            self._loss_log_window = (start, count)

    # -- live view --------------------------------------------------------------------------
    def take_live(self) -> LiveData:
        with self._lock:
            parts, self._live = self._live, []
            total = self.received
        if parts:
            return LiveData(np.concatenate(parts), total)
        return LiveData(np.zeros((0, self.source.n_channels)), total)

    # -- reconnection -----------------------------------------------------------------------
    def reconnect(self) -> dict:
        """Release and reopen the source; the recording continues in the same files."""
        with self._source_lock:
            info = self.source.reconnect()
        with self._lock:
            self.packets.reset()
            self.error = None
        return info
