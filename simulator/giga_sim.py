"""GIGA stream simulator: synthetic EMG + LED events over TCP, same protocol as the firmware.

The laptop programs connect with the pyserial URL `socket://localhost:<port>` instead of a COM
port, so the whole laptop side runs without hardware:

    python -m simulator.giga_sim                 # then e.g.
    python -m acquisition --subject A --session 1 --source sim --camera synthetic

Synthetic EMG: band-limited noise whose per-channel amplitude follows the current grip (set by
the `SIM GRIP <grip> <level>` command that the acquisition sends in simulator mode), plus a
baseline noise floor, 50 Hz interference and an electrode DC offset per channel.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import threading
import time

import numpy as np
from scipy import signal

from common.config import Config, load_config
from common.packet import FLAG_LED_ON, FLAG_LED_RISE, encode_packets

FW_VERSION = "sim-0.1.0"

# Amplitude (µV RMS at level 1) of each grip on the 8 channels. Arbitrary but distinct patterns.
GRIP_PATTERNS_UV = {
    "REST": [0, 0, 0, 0, 0, 0, 0, 0],
    "OPEN": [20, 15, 30, 80, 90, 70, 30, 20],
    "EXTENSION": [25, 20, 35, 110, 120, 95, 35, 25],
    "FIST": [90, 100, 80, 45, 25, 20, 35, 70],
    "PINCH": [30, 60, 20, 15, 40, 20, 75, 35],
    "POINT": [65, 20, 75, 55, 30, 65, 15, 25],
}


class SyntheticEmg:
    def __init__(self, fs: float, n_channels: int, uv_per_lsb: float, seed: int = 0,
                 line_uv: float = 5.0, floor_uv: float = 3.0, tau_s: float = 0.06):
        self.fs, self.n = fs, n_channels
        self.uv_per_lsb = uv_per_lsb
        self.rng = np.random.default_rng(seed)
        high = min(450.0, 0.45 * fs)
        self.sos = signal.butter(4, [20.0, high], btype="bandpass", fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, n_channels))
        # scale so that the filtered noise has unit RMS
        probe = signal.sosfilt(self.sos, self.rng.standard_normal(200_000))
        self.unit = 1.0 / probe[20_000:].std()
        self.alpha = 1.0 - np.exp(-1.0 / (tau_s * fs))
        self.activation = np.zeros(n_channels)
        self.target = np.zeros(n_channels)
        self.line_uv, self.floor_uv = line_uv, floor_uv
        self.line_phase = self.rng.uniform(0, 2 * np.pi, n_channels)
        self.offset_uv = self.rng.uniform(-5000, 5000, n_channels)
        self.t = 0

    def set_grip(self, grip: str, level: float) -> None:
        pattern = np.asarray(GRIP_PATTERNS_UV.get(grip, GRIP_PATTERNS_UV["REST"]), float)
        self.target = pattern[: self.n] * float(np.clip(level, 0.0, 1.5))

    def generate(self, n: int) -> np.ndarray:
        """Next n samples as int ADC codes, shape (n, channels)."""
        noise = self.rng.standard_normal((n, self.n))
        filtered, self.zi = signal.sosfilt(self.sos, noise, axis=0, zi=self.zi)
        # first-order smoothing of the activation towards the target, sample by sample
        k = np.arange(1, n + 1)[:, None]
        decay = (1.0 - self.alpha) ** k
        act = self.target + (self.activation - self.target) * decay
        self.activation = act[-1]
        t = (self.t + np.arange(n)) / self.fs
        self.t += n
        uv = (filtered * self.unit * act
              + self.floor_uv * self.rng.standard_normal((n, self.n))
              + self.line_uv * np.sin(2 * np.pi * 50.0 * t[:, None] + self.line_phase)
              + self.offset_uv)
        codes = np.round(uv / self.uv_per_lsb)
        return np.clip(codes, -(1 << 23), (1 << 23) - 1).astype(np.int64)


class GigaSimulator:
    def __init__(self, cfg: Config, host: str = "127.0.0.1", port: int = 8765,
                 drop_rate: float = 0.0, corrupt_rate: float = 0.0, seed: int = 0):
        self.fs = float(cfg["emg.fs_hz"])
        self.n_ch = int(cfg["emg.n_channels"])
        gain = float(cfg["emg.ads1299.gain"])
        vref = float(cfg["emg.ads1299.vref_v"])
        self.uv_per_lsb = vref / gain / ((1 << 23) - 1) * 1e6
        self.flash_samples = max(1, round(float(cfg["sync.flash_ms"]) * self.fs / 1000))
        self.flash_ms = float(cfg["sync.flash_ms"])
        self.host, self.port = host, port
        self.drop_rate, self.corrupt_rate = drop_rate, corrupt_rate
        self.rng = np.random.default_rng(seed + 1)
        self.seed = seed
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.flash_counters: list[int] = []   # ground truth for tests

    # -- server -----------------------------------------------------------------------------
    def start(self) -> int:
        """Start serving in a background thread; returns the bound port."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._running = True
        self._thread = threading.Thread(target=self._serve, name="giga-sim", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.close()
        if self._thread is not None:
            self._thread.join(2.0)

    def _serve(self) -> None:
        while self._running:
            try:
                self._server.settimeout(0.2)
                conn, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            with conn:
                self._session(conn)

    # -- one client connection ----------------------------------------------------------------
    def _session(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setblocking(False)
        emg = SyntheticEmg(self.fs, self.n_ch, self.uv_per_lsb, seed=self.seed)
        inbuf = b""
        streaming = False
        counter = 0
        t0 = 0.0
        flash_pending = False
        flash_left = 0
        out = bytearray()

        def reply(text: str) -> None:
            out.extend((text + "\n").encode("ascii"))

        while self._running:
            r, _, _ = select.select([conn], [], [], 0.002)
            if r:
                try:
                    data = conn.recv(4096)
                except BlockingIOError:
                    data = None
                except (ConnectionResetError, ConnectionAbortedError):
                    return
                if data == b"":
                    return  # client closed the connection
                inbuf += data or b""
                *lines, inbuf = inbuf.split(b"\n")
                for raw in lines:
                    parts = raw.decode("ascii", "replace").strip().split()
                    if not parts:
                        continue
                    cmd = parts[0].upper()
                    if cmd == "PING":
                        reply("#PONG")
                    elif cmd == "INFO":
                        reply("#INFO " + json.dumps(self._info(streaming)))
                    elif cmd == "START":
                        reply("#OK START")
                        streaming, counter, t0 = True, 0, time.perf_counter()
                    elif cmd == "STOP":
                        streaming = False
                        reply("#OK STOP")
                    elif cmd == "FLASH":
                        flash_pending = True
                        if not streaming:
                            reply("#OK FLASH")
                    elif cmd == "MODE" and len(parts) > 1 and parts[1].upper() == "RECORD":
                        reply("#OK MODE RECORD")
                    elif cmd == "SIM" and len(parts) >= 3 and parts[1].upper() == "GRIP":
                        level = float(parts[3]) if len(parts) > 3 else 1.0
                        emg.set_grip(parts[2].upper(), level)
                    else:
                        reply(f"#ERR unknown command: {' '.join(parts)[:40]}")
            if streaming:
                due = int((time.perf_counter() - t0) * self.fs) - counter
                if due > 0:
                    counters = np.arange(counter, counter + due)
                    flags = np.zeros(due, np.uint8)
                    for i in range(due):
                        if flash_pending and flash_left == 0:
                            flash_pending, flash_left = False, self.flash_samples
                            flags[i] |= FLAG_LED_RISE
                            self.flash_counters.append(counter + i)
                        if flash_left > 0:
                            flags[i] |= FLAG_LED_ON
                            flash_left -= 1
                    codes = emg.generate(due)
                    counter += due
                    keep = self.rng.random(due) >= self.drop_rate
                    if keep.any():
                        pkt = bytearray(encode_packets(counters[keep], flags[keep], codes[keep]))
                        if self.corrupt_rate > 0:
                            nbad = self.rng.binomial(len(pkt), self.corrupt_rate / 33)
                            for pos in self.rng.integers(0, len(pkt), nbad):
                                pkt[pos] ^= 0x5A
                        out.extend(pkt)
            if out:
                try:
                    sent = conn.send(bytes(out))
                    del out[:sent]
                except BlockingIOError:
                    pass
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    return

    def _info(self, streaming: bool) -> dict:
        return {"fw": FW_VERSION, "mode": "RECORD", "source": "simulator", "fs": self.fs,
                "n_ch": self.n_ch, "bits": 24, "uv_per_lsb": self.uv_per_lsb,
                "min_code": -(1 << 23), "max_code": (1 << 23) - 1, "flash_ms": self.flash_ms,
                "pins_configured": True, "streaming": streaming}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=None, help="TCP port (default from serial.simulator_url)")
    ap.add_argument("--drop-rate", type=float, default=0.0, help="fraction of packets dropped")
    ap.add_argument("--corrupt-rate", type=float, default=0.0, help="fraction of packets corrupted")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    port = args.port or int(cfg["serial.simulator_url"].rsplit(":", 1)[1])
    sim = GigaSimulator(cfg, port=port, drop_rate=args.drop_rate, corrupt_rate=args.corrupt_rate,
                        seed=args.seed)
    sim.start()
    print(f"GIGA simulator on socket://localhost:{sim.port}  "
          f"({sim.fs:.0f} Hz, {sim.n_ch} ch). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
