"""Signal-quality metrics for the pre-session check (pure functions, tested on synthetic data)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def usable_band(band: tuple[float, float], fs: float) -> tuple[float, float]:
    """Clip a band-pass to what the sampling rate allows (e.g. Cyton at 250 Hz)."""
    return band[0], min(band[1], 0.45 * fs)


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase band-pass + 50/100 Hz notches, along axis 0 (QC only, not the causal pipeline)."""
    lo, hi = usable_band(band, fs)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=0)
    for f0 in (50.0, 100.0):
        if f0 < 0.45 * fs:
            b, a = signal.iirnotch(f0, Q=30.0, fs=fs)
            y = signal.filtfilt(b, a, y, axis=0)
    return y


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel: (freqs, psd[freq, channel])."""
    nper = int(min(len(x), fs))  # 1 Hz resolution when possible
    f, p = signal.welch(x - x.mean(axis=0), fs=fs, nperseg=nper, axis=0)
    return f, p


def line_share(x: np.ndarray, fs: float, line_band=(48.0, 52.0), low_hz: float = 10.0) -> np.ndarray:
    """Share of the power in the mains band over the power from low_hz to Nyquist, per channel."""
    f, p = spectrum(x, fs)
    total = p[(f >= low_hz)].sum(axis=0)
    line = p[(f >= line_band[0]) & (f <= line_band[1])].sum(axis=0)
    return np.divide(line, total, out=np.zeros_like(line), where=total > 0)


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=0))


def flat(raw_uv: np.ndarray, std_max_uv: float) -> np.ndarray:
    return (raw_uv - raw_uv.mean(axis=0)).std(axis=0) < std_max_uv


def saturation_fraction(raw_uv: np.ndarray, saturation_uv: float) -> np.ndarray:
    return np.mean(np.abs(raw_uv) >= saturation_uv, axis=0)


def evaluate(rest_uv: np.ndarray, fist_uv: np.ndarray, fs: float, q: dict, saturation_uv: float,
             led: dict | None, channel_labels: list[str]) -> dict:
    """Combine the metrics into a per-channel table and an overall PASS/FAIL.

    q is the [quality] config section; led is the LED test result or None if not applicable.
    """
    band = tuple(q["bandpass_hz"])
    line_band = tuple(q["line_band_hz"])
    n_ch = rest_uv.shape[1]
    # Judged during the fist: at rest the signal is only the noise floor, so even a few µV of mains
    # dominate the rest spectrum. The rest values are reported for information.
    share = line_share(fist_uv, fs, line_band)
    share_rest = line_share(rest_uv, fs, line_band)
    f, p = spectrum(rest_uv, fs)
    df = f[1] - f[0]
    line_rms_rest = np.sqrt(p[(f >= line_band[0]) & (f <= line_band[1])].sum(axis=0) * df)
    rest_rms = rms(bandpass(rest_uv, fs, band))
    fist_rms = rms(bandpass(fist_uv, fs, band))
    ratio = np.divide(fist_rms, rest_rms, out=np.zeros_like(fist_rms), where=rest_rms > 0)
    both = np.concatenate([rest_uv, fist_uv])
    is_flat = flat(rest_uv, q["flat_std_uv_max"]) | flat(fist_uv, q["flat_std_uv_max"])
    sat = saturation_fraction(both, saturation_uv)

    channels = []
    reasons = []
    for c in range(n_ch):
        ch_ok = True
        issues = []
        if is_flat[c]:
            issues.append("flat")
        if sat[c] > q["saturation_fraction_max"]:
            issues.append(f"saturated {100 * sat[c]:.2f}%")
        if share[c] > q["line_share_max"]:
            issues.append(f"50 Hz share {share[c]:.2f}")
        if issues:
            ch_ok = False
            reasons.append(f"{channel_labels[c]}: {', '.join(issues)}")
        channels.append({
            "channel": channel_labels[c],
            "line_share_48_52_fist": round(float(share[c]), 4),
            "line_share_48_52_rest": round(float(share_rest[c]), 4),
            "line_rms_rest_uv": round(float(line_rms_rest[c]), 3),
            "rest_rms_uv": round(float(rest_rms[c]), 3),
            "fist_rms_uv": round(float(fist_rms[c]), 3),
            "rms_ratio_fist_rest": round(float(ratio[c]), 3),
            "rms_ratio_ok": bool(ratio[c] >= q["rms_ratio_min"]),
            "flat": bool(is_flat[c]),
            "saturation_fraction": round(float(sat[c]), 6),
            "ok": ch_ok,
            "issues": issues,
        })

    n_ratio_ok = sum(ch["rms_ratio_ok"] for ch in channels)
    if n_ratio_ok < q["min_channels_rms_ok"]:
        reasons.append(f"only {n_ratio_ok}/{n_ch} channels reach fist/rest RMS ratio "
                       f">= {q['rms_ratio_min']} (need {q['min_channels_rms_ok']})")
    if led is not None and not led.get("ok", False):
        reasons.append(f"LED: {led.get('message', 'not detected')}")
    passed = not reasons
    return {"result": "PASS" if passed else "FAIL", "reasons": reasons, "channels": channels,
            "led": led, "fs": fs, "thresholds": q}
