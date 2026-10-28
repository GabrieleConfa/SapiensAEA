"""OpenBCI Cyton (RF dongle, stock firmware) and the BrainFlow synthetic board."""

from __future__ import annotations

import threading
import time

import numpy as np

from emgacq.sources.base import GUI_HINT, Chunk, SampleSource, SourceError, check_port
from emgacq.sources.synthetic_emg import SyntheticEmg


class BrainFlowSource(SampleSource):
    package_modulus = 256

    def __init__(self, board: str, serial_port: str | None, channels: list[int], fs: float,
                 channel_command: str, power_down_command: str, full_scale_uv: float,
                 synthetic: dict | None = None):
        super().__init__(channels, fs, full_scale_uv)
        if board not in ("cyton", "synthetic"):
            raise ValueError(f"unknown BrainFlow board {board!r}")
        if not all(1 <= c <= 8 for c in channels) or len(set(channels)) != len(channels):
            raise ValueError(f"channels must be distinct numbers 1..8, got {channels}")
        self.kind = board
        self.serial_port = serial_port
        self.channel_command = channel_command
        self.power_down_command = power_down_command
        self.synthetic = synthetic or {}
        self._board = None
        self._streaming = False
        self._lock = threading.Lock()
        self._sim: SyntheticEmg | None = None
        self._rng = np.random.default_rng(int(self.synthetic.get("seed", 0)) + 1)
        self._stall_until = 0.0
        self._stalled_markers: list[float] = []
        self.zero_rows_dropped = 0

    # -- connection -------------------------------------------------------------------------
    def connect(self) -> dict:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams

        BoardShim.disable_board_logger()
        self._stalled_markers = []     # a new board session starts with an empty marker queue
        board_id = (BoardIds.CYTON_BOARD if self.kind == "cyton" else BoardIds.SYNTHETIC_BOARD).value
        params = BrainFlowInputParams()
        if self.kind == "cyton":
            if not self.serial_port:
                raise SourceError("no serial port given for the Cyton dongle")
            check_port(self.serial_port)
            params.serial_port = self.serial_port
        board = BoardShim(board_id, params)
        try:
            board.prepare_session()
        except BrainFlowError as e:
            raise SourceError(f"cannot open the {self.kind} board on {self.serial_port}: {e}. "
                              f"{GUI_HINT} Is the board switched on (PC position)?") from e
        self._board = board
        try:
            desc = BoardShim.get_board_descr(board_id)
            fs = float(desc["sampling_rate"])
            if fs != self.fs:
                raise SourceError(f"{self.kind} samples at {fs} Hz, config says {self.fs} Hz")
            eeg = desc["eeg_channels"]
            self._rows_uv = [eeg[c - 1] for c in self.channels]
            self._row_ts = desc["timestamp_channel"]
            self._row_pkg = desc["package_num_channel"]
            self._row_marker = desc["marker_channel"]
            self._configure_channels(board, n_board_channels=min(8, len(eeg)))
        except Exception:
            self.release()
            raise
        if self.kind == "synthetic" and self.synthetic.get("emg_model", True):
            self._sim = SyntheticEmg(self.fs, self.channels, seed=int(self.synthetic.get("seed", 0)))
        self.info = {"source": self.kind, "board_id": board_id, "serial_port": self.serial_port,
                     "fs": self.fs, "channels": self.channels, "full_scale_uv": self.full_scale_uv,
                     "package_modulus": self.package_modulus,
                     "commands": list(self.command_log),
                     "synthetic_emg_model": self._sim is not None,
                     "zero_rows_dropped": self.zero_rows_dropped}
        return self.info

    def _configure_channels(self, board, n_board_channels: int) -> None:
        """Channels in use: bipolar settings; the others: powered down. Every reply is logged."""
        failures = []
        for ch in range(1, n_board_channels + 1):
            in_use = ch in self.channels
            cmd = (self.channel_command if in_use else self.power_down_command).format(ch=ch)
            reply = board.config_board(cmd)
            entry = {"channel": ch, "in_use": in_use, "command": cmd, "reply": reply.strip(),
                     "time": time.time()}
            if self.kind == "cyton" and in_use:
                entry["ok"] = "Success" in reply and "Failure" not in reply
                if not entry["ok"]:
                    failures.append(f"{cmd} -> {reply.strip()!r}")
            self.command_log.append(entry)
        if failures:
            raise SourceError("the Cyton did not confirm the channel settings: " + "; ".join(failures))

    # -- streaming --------------------------------------------------------------------------
    def start(self) -> None:
        self._board.start_stream(45000)
        self._streaming = True

    def read(self) -> Chunk:
        with self._lock:
            if self._board is None or not self._streaming:
                return Chunk.empty(self.n_channels)
            data = self._board.get_board_data()
        if time.monotonic() < self._stall_until:
            # simulated radio stall: the samples are lost, but BrainFlow keeps the markers queued
            # and stamps them on the next samples that arrive
            m = data[self._row_marker] if data.shape[1] else []
            self._stalled_markers += [float(x) for x in m if x != 0]
            return Chunk.empty(self.n_channels)
        if data.shape[1] == 0:
            return Chunk.empty(self.n_channels)
        marker = data[self._row_marker]
        if self._stalled_markers:
            k = min(len(self._stalled_markers), len(marker))
            marker[:k] = self._stalled_markers[:k]
            del self._stalled_markers[:k]
        if self.kind == "synthetic":
            p = float(self.synthetic.get("packet_loss_fraction", 0.0))
            if p > 0:
                keep = (self._rng.random(len(marker)) >= p) | (marker != 0)
                data, marker = data[:, keep], marker[keep]
        uv = data[self._rows_uv].T
        if self.kind == "cyton":
            # after a (re)start the Cyton stream can deliver a sample with every channel at exactly
            # 0 µV (seen after both reconnections of 2026-10-21): not a measurement - the electrode
            # offsets alone are mV - so it is dropped (unless it carries a marker)
            invalid = (uv == 0).all(axis=1) & (marker == 0)
            if invalid.any():
                self.zero_rows_dropped += int(invalid.sum())
                self.info["zero_rows_dropped"] = self.zero_rows_dropped
                keep = ~invalid
                data, marker, uv = data[:, keep], marker[keep], uv[keep]
        if self._sim is not None:
            uv = self._sim.generate(marker)
        return Chunk(data[self._row_ts], data[self._row_pkg], uv, marker)

    def insert_marker(self, code: int) -> None:
        with self._lock:
            if self._board is not None and self._streaming:
                self._board.insert_marker(float(code))

    def release(self) -> None:
        with self._lock:
            board, self._board = self._board, None
            streaming, self._streaming = self._streaming, False
        if board is None:
            return
        if streaming:
            try:
                board.stop_stream()
            except Exception:
                pass
        try:
            board.release_session()
        except Exception:
            pass

    def simulate_stall(self, seconds: float) -> bool:
        if self.kind != "synthetic":
            return False
        self._stall_until = time.monotonic() + seconds
        return True
