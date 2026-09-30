"""OpenBCI Cyton through BrainFlow (stock firmware + USB dongle), used as a clean reference.

The OpenBCI GUI and a BrainFlow script cannot hold the dongle's serial port at the same time:
close the GUI (or stop its session) before running this.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from common.sources.base import EmgSource, RawSink

PORT_HINT = ("Is the OpenBCI GUI still connected? The GUI and a BrainFlow script cannot hold the "
             "serial port at the same time: stop the GUI session first.")


class CytonSource(EmgSource):
    name = "cyton"
    supports_flash = False

    def __init__(self, serial_port: str, n_channels: int = 8, fs: float = 250.0,
                 channel_command: str = "x{ch}060100X", buffer_s: float = 30.0):
        super().__init__(n_channels, fs, buffer_s)
        self.serial_port = serial_port
        self.channel_command = channel_command
        self._board = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._raw_sink: RawSink | None = None
        self._counter = 0
        # ADS1299 full scale at gain 24: ±(4.5 V / 24)
        self.saturation_uv = 4.5 / 24 * 1e6 * 0.999

    def open(self) -> dict:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        board_id = BoardIds.CYTON_BOARD.value
        self._board = BoardShim(board_id, params)
        try:
            self._board.prepare_session()
        except Exception as e:  # BrainFlowError
            raise RuntimeError(f"cannot open Cyton on {self.serial_port}: {e}. {PORT_HINT}") from e
        replies = []
        for ch in range(1, self.n_channels + 1):
            cmd = self.channel_command.format(ch=ch)
            replies.append((cmd, self._board.config_board(cmd)))
        self._eeg_rows = BoardShim.get_eeg_channels(board_id)[: self.n_channels]
        self._ts_row = BoardShim.get_timestamp_channel(board_id)
        fs = BoardShim.get_sampling_rate(board_id)
        if fs != self.fs:
            raise RuntimeError(f"Cyton samples at {fs} Hz, config says {self.fs} Hz")
        self.info = {"source": "cyton_brainflow", "fs": fs, "n_ch": self.n_channels, "bits": 24,
                     "uv_per_lsb": 1.0, "channel_commands": [c for c, _ in replies],
                     "serial_port": self.serial_port}
        return self.info

    def start(self, raw_sink: RawSink | None = None) -> None:
        self._raw_sink = raw_sink
        self._board.start_stream(45000)
        self._running = True
        self.started_ns = time.perf_counter_ns()
        # BrainFlow timestamps are wall clock (time.time); keep an anchor to map to perf_counter.
        self._anchor = (time.time(), time.perf_counter_ns())
        self._thread = threading.Thread(target=self._reader, name="cyton-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            time.sleep(0.02)
            data = self._board.get_board_data()
            n = data.shape[1]
            if n == 0:
                continue
            t = time.perf_counter_ns()
            uv = data[self._eeg_rows].T
            counters = np.arange(self._counter, self._counter + n)
            self._counter += n
            if self._raw_sink is not None:
                ts = data[self._ts_row]
                host_ns = self._anchor[1] + ((ts - self._anchor[0]) * 1e9).astype(np.int64)
                lines = "".join(
                    f"{c},{h}," + ",".join(f"{v:.3f}" for v in row) + "\n"
                    for c, h, row in zip(counters, host_ns, uv))
                self._raw_sink.write_bytes(lines.encode("ascii"), t)
            self.buffer.append(counters, np.zeros(n, np.uint8), uv, t)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)
        if self._board is not None:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            self._board.release_session()
            self._board = None
