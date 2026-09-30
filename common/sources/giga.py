"""GIGA stream over USB serial (also the simulator, through a socket:// URL)."""

from __future__ import annotations

import json
import threading
import time

import serial

from common.packet import PacketParser
from common.sources.base import EmgSource, RawSink


class DeviceError(RuntimeError):
    pass


class GigaSerialSource(EmgSource):
    name = "giga"
    supports_flash = True

    def __init__(self, url: str, n_channels: int, fs: float, baudrate: int = 2_000_000,
                 buffer_s: float = 30.0):
        super().__init__(n_channels, fs, buffer_s)
        self.url = url
        self.baudrate = baudrate
        self.parser = PacketParser(n_channels)
        self._ser: serial.Serial | None = None
        self._write_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_ack = threading.Event()
        self._raw_sink: RawSink | None = None

    # -- connection -------------------------------------------------------------------------
    def open(self) -> dict:
        try:
            self._ser = serial.serial_for_url(self.url, baudrate=self.baudrate, timeout=0.05)
        except (serial.SerialException, OSError) as e:
            raise DeviceError(f"cannot open {self.url}: {e}") from e
        # the device may still be streaming from an interrupted run
        self._write("STOP")
        time.sleep(0.3)
        self._ser.reset_input_buffer()
        info = self._query("INFO", "#INFO")
        self.info = json.loads(info[len("#INFO"):].strip())
        if self.info.get("n_ch") != self.n_channels:
            raise DeviceError(f"device has {self.info.get('n_ch')} channels, config says {self.n_channels}")
        if abs(float(self.info.get("fs", 0)) - self.fs) > 1e-6:
            raise DeviceError(f"device samples at {self.info.get('fs')} Hz, config says {self.fs} Hz")
        self.uv_per_lsb = float(self.info["uv_per_lsb"])
        max_code = int(self.info["max_code"])
        min_code = int(self.info["min_code"])
        self._sat_codes = (min_code, max_code)
        # distance from the mid code, in µV; codes at the limits are saturated
        mid = (max_code + min_code) / 2
        self.saturation_uv = (max_code - mid) * self.uv_per_lsb
        self._mid_code = mid if min_code >= 0 else 0.0
        return self.info

    def _query(self, command: str, expect: str, timeout_s: float = 2.0) -> str:
        self._write(command)
        deadline = time.monotonic() + timeout_s
        pending = b""
        while time.monotonic() < deadline:
            pending += self._ser.read(max(1, self._ser.in_waiting))
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                line = raw.decode("ascii", "replace").strip()
                if line.startswith(expect):
                    return line
                if line.startswith("#ERR"):
                    raise DeviceError(f"{command}: {line}")
        raise DeviceError(f"no reply to {command} from {self.url} within {timeout_s} s")

    def _write(self, command: str) -> None:
        with self._write_lock:
            self._ser.write((command + "\n").encode("ascii"))

    # -- streaming --------------------------------------------------------------------------
    def start(self, raw_sink: RawSink | None = None) -> None:
        self._raw_sink = raw_sink
        self._running = True
        self._stop_ack.clear()
        self.parser.reset_counter_tracking()
        self._thread = threading.Thread(target=self._reader, name="giga-reader", daemon=True)
        self._thread.start()
        self.started_ns = time.perf_counter_ns()
        self._write("START")

    def _reader(self) -> None:
        ser = self._ser
        try:
            while self._running:
                data = ser.read(max(1, ser.in_waiting))
                if not data:
                    continue
                t = time.perf_counter_ns()
                if self._raw_sink is not None:
                    self._raw_sink.write_bytes(data, t)
                block = self.parser.feed(data)
                for text in block.texts:
                    self.messages.append((t, text))
                    if text.startswith("#OK STOP"):
                        self._stop_ack.set()
                if len(block):
                    uv = (block.samples - self._mid_code) * self.uv_per_lsb
                    self.buffer.append(block.counters, block.flags, uv, t)
        except (serial.SerialException, OSError) as e:
            self.error = f"serial read failed: {e}"
            self._running = False

    def flash(self) -> bool:
        self._write("FLASH")
        return True

    def send(self, command: str) -> None:
        self._write(command)

    def stop(self) -> None:
        if self._ser is None:
            return
        if self._running:
            try:
                self._write("STOP")
                self._stop_ack.wait(1.0)   # keep reading until the last packet is in
            except (serial.SerialException, OSError):
                pass
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)
        self._ser.close()
        self._ser = None

    def stats(self) -> dict:
        s = super().stats()
        p = self.parser.stats
        s.update(packets=p.packets, crc_errors=p.crc_errors, bytes_skipped=p.bytes_skipped,
                 gaps=p.gaps, lost_samples=p.lost_samples, overrun_flags=p.overrun_flags,
                 led_edges=len(self.buffer.led_edges))
        return s
