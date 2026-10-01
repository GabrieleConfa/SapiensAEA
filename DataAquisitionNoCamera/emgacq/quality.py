"""Quality-check metrics and PASS/FAIL (pure functions of the captured rest and fist samples)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from emgacq.signals import filtfilt, is_flat, railed_fraction


def mean_psd(runs: list[np.ndarray], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel, averaged over the runs (weighted by length): (freqs, psd[f, ch])."""
    acc, weight, f = None, 0, None
    for x in runs:
        x = np.asarray(x, float)
        if len(x) < 16:
            continue
        nper = int(min(len(x), fs))          # 1 Hz resolution when possible
        f_run, p = signal.welch(x - x.mean(axis=0), fs=fs, nperseg=nper, axis=0)
        if f is None:
            f = f_run
        elif len(f_run) != len(f):           # shorter run, coarser grid: interpolate
            p = np.stack([np.interp(f, f_run, p[:, c]) for c in range(p.shape[1])], axis=1)
        acc = p * len(x) if acc is None else acc + p * len(x)
        weight += len(x)
    if acc is None:
        raise ValueError("no capture long enough for a spectrum")
    return f, acc / weight


def line_share(runs: list[np.ndarray], fs: float, line_band=(48.0, 52.0), low_hz: float = 10.0) -> np.ndarray:
    """Share of the power in the mains band over the power from low_hz to Nyquist, per channel."""
    f, p = mean_psd(runs, fs)
    total = p[f >= low_hz].sum(axis=0)
    line = p[(f >= line_band[0]) & (f <= line_band[1])].sum(axis=0)
    return np.divide(line, total, out=np.zeros_like(line), where=total > 0)


def filtered_rms(runs: list[np.ndarray], fs: float, band, notches=(50.0, 100.0)) -> np.ndarray:
    """RMS of the band-passed runs, pooled over all their samples."""
    sq, n = 0.0, 0
    for x in runs:
        y = filtfilt(x, fs, band, notches)
        sq = sq + np.sum(np.square(y), axis=0)
        n += len(y)
    return np.sqrt(sq / n)


def evaluate(rest_runs: list[np.ndarray], fist_runs: list[np.ndarray], fs: float, q: dict,
             full_scale_uv: float, channel_names: list[str], packet_loss_pct: float) -> dict:
    """Per-channel metrics and the overall PASS/FAIL. `q` is the [quality] config section."""
    d = int(round(q["discard_s"] * fs))
    rest = [r[d:] for r in rest_runs if len(r) > d]
    fist = [r[d:] for r in fist_runs if len(r) > d]
    if not rest or not fist:
        raise ValueError("quality check: missing rest or fist capture")
    band = tuple(q["band_hz"])
    lb = tuple(q["line_band_hz"])
    share_rest = line_share(rest, fs, lb, q["line_share_low_hz"])
    share_fist = line_share(fist, fs, lb, q["line_share_low_hz"])
    judged_on = q["line_share_judged_on"]
    share = share_rest if judged_on == "rest" else share_fist
    rest_rms = filtered_rms(rest, fs, band)
    fist_rms = filtered_rms(fist, fs, band)
    ratio = np.divide(fist_rms, rest_rms, out=np.zeros_like(fist_rms), where=rest_rms > 0)
    both = np.concatenate(rest + fist)
    flat = np.zeros(len(channel_names), bool)
    for r in rest + fist:
        flat |= is_flat(r, q["flat_std_uv_max"])
    railed = railed_fraction(both, full_scale_uv, q["railed_level"])

    channels, reasons = [], []
    for c, name in enumerate(channel_names):
        issues = []
        if flat[c]:
            issues.append("flat")
        if railed[c] > q["railed_fraction_max"]:
            issues.append(f"railed {100 * railed[c]:.2f}%")
        if share[c] > q["line_share_max"]:
            issues.append(f"50 Hz share {share[c]:.2f}")
        if rest_rms[c] > q["rest_rms_max_uv"]:
            issues.append(f"rest RMS {rest_rms[c]:.1f} µV")
        ratio_ok = bool(ratio[c] >= q["rms_ratio_min"])
        if not ratio_ok:
            issues.append(f"fist/rest {ratio[c]:.1f}")
        hard = [i for i in issues if not i.startswith("fist/rest")]
        if hard:
            reasons.append(f"{name}: {', '.join(hard)}")
        channels.append({
            "channel": name,
            "line_share_48_52_rest": round(float(share_rest[c]), 4),
            "line_share_48_52_fist": round(float(share_fist[c]), 4),
            "rest_rms_uv": round(float(rest_rms[c]), 3),
            "fist_rms_uv": round(float(fist_rms[c]), 3),
            "rms_ratio_fist_rest": round(float(ratio[c]), 3),
            "rms_ratio_ok": ratio_ok,
            "flat": bool(flat[c]),
            "railed_fraction": round(float(railed[c]), 6),
            "ok": not issues,
            "issues": issues,
        })

    n_ratio_ok = sum(ch["rms_ratio_ok"] for ch in channels)
    need = int(np.ceil(q["min_fraction_channels_ratio_ok"] * len(channels)))
    if n_ratio_ok < need:
        reasons.append(f"only {n_ratio_ok}/{len(channels)} channels reach fist/rest RMS ratio "
                       f">= {q['rms_ratio_min']} (need {need})")
    if packet_loss_pct > q["packet_loss_max_pct"]:
        reasons.append(f"packet loss {packet_loss_pct:.2f}% > {q['packet_loss_max_pct']}%")
    return {"result": "FAIL" if reasons else "PASS", "reasons": reasons, "channels": channels,
            "packet_loss_pct": round(float(packet_loss_pct), 3),
            "line_share_judged_on": judged_on, "fs": fs,
            "rest_s": round(sum(len(r) for r in rest) / fs, 2),
            "fist_s": round(sum(len(r) for r in fist) / fs, 2),
            "thresholds": dict(q)}
