"""EMG-like test signal for the synthetic board, driven by the markers in the stream.

Band-limited noise whose per-channel amplitude follows the gesture of the current step, with a
random force level per contraction, a noise floor, some 50 Hz interference and an electrode DC
offset. It only exists to exercise the application without hardware.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from emgacq import markers as mk

# µV RMS at force level 1, channels 1-8. Arbitrary but distinct patterns.
PATTERNS_UV = {
    "OPEN": [20, 15, 30, 80, 90, 70, 30, 20],
    "CLOSE": [90, 100, 80, 45, 25, 20, 35, 70],
    "INDEX": [65, 20, 75, 55, 30, 65, 15, 25],
}


class SyntheticEmg:
    def __init__(self, fs: float, channels: list[int], seed: int = 0, floor_uv: float = 1.5,
                 line_uv: float = 0.5, tau_s: float = 0.08):
        self.fs = fs
        self.idx = [c - 1 for c in channels]
        n = len(channels)
        self.rng = np.random.default_rng(seed)
        self.sos = signal.butter(4, [20.0, min(150.0, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, n))
        probe = signal.sosfilt(self.sos, self.rng.standard_normal(100_000))
        self.unit = 1.0 / probe[10_000:].std()
        self.alpha = 1.0 - np.exp(-1.0 / (tau_s * fs))
        self.activation = np.zeros(n)
        self.target = np.zeros(n)
        self.floor_uv, self.line_uv = floor_uv, line_uv
        self.line_phase = self.rng.uniform(0, 2 * np.pi, n)
        self.offset_uv = self.rng.uniform(-3000, 3000, n)
        self.t = 0

    def _set_from_marker(self, code: float) -> None:
        try:
            info = mk.decode(code)
        except ValueError:
            return
        if info.label in PATTERNS_UV:
            level = self.rng.uniform(0.5, 1.3)          # light to firm
            self.target = np.asarray(PATTERNS_UV[info.label], float)[self.idx] * level
        elif info.phase:
            self.target = np.zeros_like(self.target)

    def generate(self, markers: np.ndarray) -> np.ndarray:
        """Samples (n, channels) in µV; each marker updates the target from its sample on."""
        n = len(markers)
        out = np.empty((n, len(self.idx)))
        start = 0
        for i in list(np.flatnonzero(markers)) + [n]:
            if i > start:
                out[start:i] = self._run(i - start)
            if i < n:
                self._set_from_marker(markers[i])
            start = i
        return out

    def _run(self, n: int) -> np.ndarray:
        noise = self.rng.standard_normal((n, len(self.idx)))
        filtered, self.zi = signal.sosfilt(self.sos, noise, axis=0, zi=self.zi)
        k = np.arange(1, n + 1)[:, None]
        act = self.target + (self.activation - self.target) * (1.0 - self.alpha) ** k
        self.activation = act[-1]
        t = (self.t + np.arange(n)) / self.fs
        self.t += n
        return (filtered * self.unit * act
                + self.floor_uv * self.rng.standard_normal((n, len(self.idx)))
                + self.line_uv * np.sin(2 * np.pi * 50.0 * t[:, None] + self.line_phase)
                + self.offset_uv)
