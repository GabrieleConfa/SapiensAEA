"""Filters and per-channel signal checks.

Display filters are causal (sosfilt with state) and used only on screen; the saved data stay raw.
The zero-phase versions are for offline use (quality check, inspect tool).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _band(band: tuple[float, float], fs: float) -> tuple[float, float]:
    """Clip a band-pass to what the sampling rate allows (120 Hz is close to Nyquist at 250 Hz)."""
    return float(band[0]), float(min(band[1], 0.48 * fs))


def filter_sos(fs: float, band=(20.0, 120.0), notches=(50.0, 100.0), q: float = 30.0) -> np.ndarray:
    """Band-pass (Butterworth, order 4) followed by notches, as one SOS cascade."""
    lo, hi = _band(band, fs)
    parts = [signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")]
    for f0 in notches:
        if f0 < 0.48 * fs:
            b, a = signal.iirnotch(f0, Q=q, fs=fs)
            parts.append(signal.tf2sos(b, a))
    return np.vstack(parts)


def filtfilt(x: np.ndarray, fs: float, band=(20.0, 120.0), notches=(50.0, 100.0)) -> np.ndarray:
    """Zero-phase band-pass + notches along axis 0 (offline only)."""
    x = np.asarray(x, float)
    sos = filter_sos(fs, band, notches)
    padlen = min(len(x) - 1, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, x - x.mean(axis=0), axis=0, padlen=max(padlen, 0))


class LiveFilter:
    """Causal multichannel filter that keeps its state between chunks."""

    def __init__(self, fs: float, n_channels: int, band=(20.0, 120.0), notches=(50.0, 100.0)):
        self.sos = filter_sos(fs, band, notches)
        self.n_channels = n_channels
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if len(x) == 0:
            return x.reshape(0, self.n_channels)
        if self.zi is None:
            # start from the steady state of the first sample to avoid a large DC transient
            zi0 = signal.sosfilt_zi(self.sos)
            self.zi = zi0[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class Ring:
    """Fixed-size buffer of the last `capacity` rows."""

    def __init__(self, capacity: int, n_channels: int):
        self.data = np.zeros((capacity, n_channels))
        self.capacity = capacity
        self.count = 0

    def extend(self, x: np.ndarray) -> None:
        x = np.asarray(x, float)
        n = len(x)
        if n == 0:
            return
        if n >= self.capacity:
            self.data[:] = x[-self.capacity:]
        else:
            self.data = np.roll(self.data, -n, axis=0)
            self.data[-n:] = x
        self.count = min(self.capacity, self.count + n)

    def last(self, n: int) -> np.ndarray:
        n = min(n, self.count)
        return self.data[self.capacity - n:]


def rms(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.zeros(x.shape[1] if x.ndim == 2 else 0)
    return np.sqrt(np.mean(np.square(x), axis=0))


def is_flat(raw_uv: np.ndarray, std_max_uv: float) -> np.ndarray:
    raw_uv = np.asarray(raw_uv, float)
    return (raw_uv - raw_uv.mean(axis=0)).std(axis=0) < std_max_uv


def railed_fraction(raw_uv: np.ndarray, full_scale_uv: float, level: float = 0.9) -> np.ndarray:
    return np.mean(np.abs(np.asarray(raw_uv, float)) >= level * full_scale_uv, axis=0)


class ChannelMonitor:
    """Live per-channel state for the operator view: filtered trace, RMS bar, flat/railed flags."""

    def __init__(self, fs: float, n_channels: int, mon: dict, q: dict, full_scale_uv: float):
        self.fs = fs
        self.filter = LiveFilter(fs, n_channels, tuple(mon["display_band_hz"]), tuple(mon["notch_hz"]))
        self.display = Ring(int(mon["display_window_s"] * fs), n_channels)
        self.raw = Ring(int(mon["flag_window_s"] * fs), n_channels)
        self.rms_n = max(1, int(mon["rms_window_s"] * fs))
        self.flat_std = float(q["flat_std_uv_max"])
        self.railed_level = float(q["railed_level"])
        self.railed_max = float(q["railed_fraction_max"])
        self.full_scale_uv = full_scale_uv

    def update(self, raw_uv: np.ndarray) -> None:
        if len(raw_uv) == 0:
            return
        self.raw.extend(raw_uv)
        self.display.extend(self.filter.process(raw_uv))

    def rms(self) -> np.ndarray:
        return rms(self.display.last(self.rms_n))

    def flags(self) -> tuple[np.ndarray, np.ndarray]:
        """(flat, railed) per channel over the flag window; all False until the window is full."""
        n = self.raw.data.shape[1]
        if self.raw.count < self.raw.capacity:
            return np.zeros(n, bool), np.zeros(n, bool)
        raw = self.raw.last(self.raw.capacity)
        return (is_flat(raw, self.flat_std),
                railed_fraction(raw, self.full_scale_uv, self.railed_level) > self.railed_max)
