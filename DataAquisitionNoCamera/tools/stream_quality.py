"""Stream quality of a recorded session: is the data stream good, noisy or disturbed, and by what?

    python -m tools.stream_quality                     # latest session in data/
    python -m tools.stream_quality data/<folder>
    python -m tools.stream_quality --all               # one line per session in data/

The notebook tools/stream_quality.ipynb runs the same checks with plots and explanations.

Works on protocol sessions, free recordings and sessions that crashed (status "recording").
Everything is computed from emg.csv (raw µV) and the side files; nothing is written.

Checks:
- link: sample rate from the timestamps, lost packets, duplicates, gaps, reconnections and
  all-zero rows (an invalid sample the Cyton stream delivers right after a reconnection);
- per channel: electrode offset (DC) and baseline wander, railed and flat samples, pops (sudden
  jumps), mains at rest (50 Hz amplitude), noise floor at rest (20-120 Hz with the mains notched),
  low-frequency artifacts at rest (1-15 Hz), EMG strength (gesture RMS / rest RMS);
- between channels: correlation of the filtered signals at rest;
- 1 s windows: which disturbance dominates each channel, and when.

The stream is cut into segments at every timestamp jump and invalid row, and every filter runs
per segment, so a reconnection (new DC offset, restarted counter) does not leak into the metrics.
Rest = rows labelled REST outside the pauses (first second of each rest skipped); a free
recording has no labels, so its quietest 20% of the 1 s windows stand in for rest and the
loudest 10% for activity.

The thresholds are first guesses from the first Cyton sessions (2026-10-21); see THRESHOLDS.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from emgacq import analysis as an
from emgacq.config import load_config
from emgacq.markers import GESTURES
from emgacq.quality import mean_psd
from emgacq.signals import filtfilt

THRESHOLDS = {
    "mains_hz": 50.0,
    # link
    "split_gap_s": 0.5,           # a timestamp jump longer than this cuts the stream (a gap)
    "loss_warn_pct": 0.1,
    "loss_bad_pct": 1.0,
    "rate_tol_pct": 2.0,          # samples/s (lost ones included) vs nominal
    # per channel (µV unless stated)
    "dc_warn_frac": 0.25,         # electrode offset / full scale (187.5 mV on the Cyton)
    "dc_bad_frac": 0.8,
    "railed_level": 0.9,          # |raw| >= this fraction of full scale
    "railed_bad_frac": 0.001,
    "flat_std_uv": 0.5,           # a 1 s window with a raw std below this is flat
    "flat_bad_frac": 0.05,        # fraction of flat windows
    "pop_uv": 1000.0,             # a sample-to-sample jump above this is a pop
    "pop_warn_per_min": 1.0,
    "pop_bad_per_min": 5.0,
    "wander_warn_uv": 1000.0,     # baseline excursion: p95 |1 s mean - segment median|
    "wander_bad_uv": 5000.0,
    "mains_warn_uv": 15.0,        # 50 Hz RMS at rest, raw
    "mains_bad_uv": 50.0,
    "noise_warn_uv": 5.0,         # RMS at rest, 20-120 Hz with 50/100 Hz notched
    "noise_bad_uv": 10.0,
    "lowfreq_warn_uv": 20.0,      # RMS at rest, 1-15 Hz (movement, cables, sweat)
    "lowfreq_bad_uv": 50.0,
    "snr_warn": 3.0,              # best gesture RMS / rest RMS, filtered
    "snr_bad": 1.5,
    "corr_warn": 0.6,             # |correlation| of two filtered channels at rest
    # selections
    "window_s": 1.0,
    "rest_settle_s": 1.0,         # skipped at the start of each rest
    "active_settle_s": 0.5,       # skipped at the start of each contraction
    "min_rest_s": 4.0,            # less labelled rest than this: use the quietest windows
}

EMG_BAND = (20.0, 120.0)
LOWFREQ_BAND = (1.0, 15.0)
MIN_SEGMENT_S = 1.0               # shorter segments are not filtered (too short for the filters)
GRADES = ("GOOD", "WARN", "BAD")

ADVICE = {
    "loss": "Lost packets: put the dongle close to the board with a clear line of sight, use a USB "
            "port without a hub (or a USB extension cable), check the board battery.",
    "gap": "Gaps in the stream: the radio link dropped. The protocol repeats the interrupted "
           "trials; if gaps keep coming, see the packet-loss advice.",
    "zero_rows": "All-zero rows: invalid samples (typically the first sample after a "
                 "reconnection). Exclude them in the offline analysis (this tool already does).",
    "rate": "Sample rate off: the stream did not deliver the nominal rate (laptop busy or the "
            "clock jumped). Close other programs; check the laptop clock.",
    "clock": "The laptop clock jumped during the session (time synchronisation?). The "
             "timestamps around the jump are unreliable; sample_index is not affected.",
    "railed": "Railed: the input is out of range. An electrode is off or the offset is huge: "
              "re-attach the electrodes of this channel.",
    "flat": "Flat: no signal. The channel is not connected (cable on the N/P pins) or is off.",
    "dc": "Large electrode offset: poor or unsettled skin contact (dry gel, mismatched electrodes). "
          "Re-prepare the skin, use fresh gel electrodes, wait a few minutes for them to settle.",
    "wander": "Baseline wander: slow drifts (movement, sweat, cable pulled). Fix the cables to the "
              "arm with tape so they cannot pull on the electrodes.",
    "pops": "Pops (sudden jumps): an electrode is losing contact intermittently. Press it on or "
            "replace it; strain-relieve the cable.",
    "mains": "Mains (50 Hz): the two electrodes of this pair see different impedances, which turns "
             "common-mode mains into signal. Check the BIAS (reference) electrode first, re-prepare "
             "the skin of this pair, keep the cables short and together, keep away from power "
             "supplies, run the laptop on battery. (The notch removes 50 Hz offline, but a large "
             "mains level means poor contact.)",
    "noise": "High noise floor at rest: poor contact, or the subject is not fully relaxed. Re-seat "
             "the electrodes and ask the subject to let the arm rest completely.",
    "lowfreq": "Low-frequency artifacts at rest (1-15 Hz): movement or cable sway. Support the arm, "
               "tape the cables down.",
    "snr": "Weak EMG: the gestures barely raise this channel above rest. Check that the electrodes "
           "lie over an active muscle (a channel may legitimately stay quiet for some gestures).",
    "corr": "Channels correlated at rest: most often a common-mode signal (weak BIAS) leaking "
            "through unequal electrode impedances; the sign (+ or -) only says which electrode of "
            "each pair has the higher impedance. Check the BIAS electrode first. If the same pair "
            "correlates in every session, even after re-applying the electrodes, suspect the "
            "wiring or gel bridging two sites instead.",
}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def data_dir() -> Path:
    return load_config().path_of("paths.data_dir")


def list_sessions(folder: str | Path | None = None) -> list[Path]:
    """Session folders (with an emg.csv), oldest first."""
    root = Path(folder) if folder else data_dir()
    found = [p for p in root.iterdir() if (p / "emg.csv").exists()] if root.exists() else []
    return sorted(found, key=lambda p: (p / "emg.csv").stat().st_mtime)


def latest_session(folder: str | Path | None = None) -> Path:
    sessions = list_sessions(folder)
    if not sessions:
        raise FileNotFoundError(f"no session folder with an emg.csv in {folder or data_dir()}")
    return sessions[-1]


@dataclass
class Recording:
    """A session prepared for the checks: raw data, segments and the filtered versions."""
    session: an.Session
    th: dict
    fs: float
    channels: list[str]
    full_scale_uv: float
    raw: np.ndarray           # (n, c) µV as saved
    t: np.ndarray             # (n,) s from the first sample (timestamps)
    valid: np.ndarray         # (n,) False on all-zero rows
    segments: list[tuple[int, int]]   # [a, b) row ranges of uninterrupted valid data
    seg: np.ndarray           # (n,) segment number, -1 on invalid rows
    phase: np.ndarray
    label: np.ndarray
    bad: np.ndarray           # trial_bad
    centered: np.ndarray      # raw minus its segment median (plots)
    notched: np.ndarray       # raw with the mains notched, nothing else (pops: a large mains wave
                              # plus strong EMG would otherwise look like jumps)
    emg: np.ndarray          # 20-120 Hz band-pass + mains notches (NaN outside long segments)
    low: np.ndarray           # 1-15 Hz
    mains_pow: np.ndarray     # 48-52 Hz power minus the neighbouring bands (instantaneous)

    @property
    def folder(self) -> Path:
        return self.session.folder

    @property
    def meta(self) -> dict:
        return self.session.meta

    @property
    def mode(self) -> str:
        return self.meta.get("mode", "protocol")

    @property
    def n(self) -> int:
        return len(self.raw)


def _key_runs(keys: list[np.ndarray], n: int) -> list[tuple[int, int]]:
    """[a, b) runs over which all the key arrays stay constant."""
    if n == 0:
        return []
    change = np.zeros(n, bool)
    change[0] = True
    for k in keys:
        change[1:] |= k[1:] != k[:-1]
    starts = np.flatnonzero(change)
    return list(zip(starts.tolist(), np.r_[starts[1:], n].tolist()))


def _segments(valid: np.ndarray, ts: np.ndarray, split_s: float) -> list[tuple[int, int]]:
    n = len(valid)
    if n == 0:
        return []
    jump = np.r_[False, np.abs(np.diff(ts)) > split_s]
    runs = _key_runs([valid, np.cumsum(jump)], n)
    return [(a, b) for a, b in runs if valid[a]]


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [lo, min(hi, 0.48 * fs)], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x - x.mean(axis=0), axis=0)


def load(folder: str | Path | None = None, thresholds: dict | None = None) -> Recording:
    """Load a session folder (default: the latest one) and prepare it for the checks."""
    folder = Path(folder) if folder else latest_session()
    th = {**THRESHOLDS, **(thresholds or {})}
    s = an.load_session(folder)
    if s.emg.empty:
        raise ValueError(f"{folder}: emg.csv has no samples")
    fs = s.fs
    ch = s.channel_columns
    raw = s.emg[ch].to_numpy(float)
    ts = s.emg["timestamp"].to_numpy(float)
    valid = ~(raw == 0).all(axis=1)
    segments = _segments(valid, ts, th["split_gap_s"])
    seg = np.full(len(raw), -1)
    for k, (a, b) in enumerate(segments):
        seg[a:b] = k
    full_scale = float(s.meta.get("source", {}).get("full_scale_uv") or np.inf)

    m = th["mains_hz"]
    notches = tuple(f for f in (m, 2 * m) if f < 0.48 * fs)
    nan = np.full_like(raw, np.nan)
    centered, notched, emg, low, mains_pow = (nan.copy() for _ in range(5))
    notch_sos = np.vstack([signal.tf2sos(*signal.iirnotch(f0, Q=30.0, fs=fs)) for f0 in notches])
    for a, b in segments:
        x = raw[a:b]
        centered[a:b] = x - np.median(x, axis=0)
        if b - a < MIN_SEGMENT_S * fs:
            continue
        notched[a:b] = signal.sosfiltfilt(notch_sos, x - x.mean(axis=0), axis=0)
        emg[a:b] = filtfilt(x, fs, EMG_BAND, notches)
        low[a:b] = _bandpass(x, fs, *LOWFREQ_BAND)
        line = _bandpass(x, fs, m - 2, m + 2, 2)
        side = (_bandpass(x, fs, m - 8, m - 4, 2) ** 2 + _bandpass(x, fs, m + 4, m + 8, 2) ** 2) / 2
        mains_pow[a:b] = line ** 2 - side

    return Recording(
        session=s, th=th, fs=fs, channels=ch, full_scale_uv=full_scale, raw=raw, t=ts - ts[0],
        valid=valid, segments=segments, seg=seg,
        phase=s.emg["phase"].astype(str).to_numpy(dtype=str),
        label=s.emg["label"].astype(str).to_numpy(dtype=str),
        bad=s.emg["trial_bad"].to_numpy(int), centered=centered, notched=notched, emg=emg, low=low,
        mains_pow=mains_pow)


# ---------------------------------------------------------------------------------------------
# Selections (rest, gestures) and windows
# ---------------------------------------------------------------------------------------------
def _labelled_runs(rec: Recording, mask: np.ndarray, settle_s: float) -> list[tuple[int, int]]:
    """Runs of `mask` (constant phase, label, segment, bad flag), minus their first `settle_s`."""
    skip = int(round(settle_s * rec.fs))
    out = []
    for a, b in _key_runs([rec.phase, rec.label, rec.seg, rec.bad], rec.n):
        if mask[a] and rec.seg[a] >= 0 and np.isfinite(rec.emg[a, 0]) and b - (a + skip) >= rec.fs / 2:
            out.append((a + skip, b))
    return out


def _rows(runs: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([np.arange(a, b) for a, b in runs]) if runs else np.zeros(0, int)


@dataclass
class Windows:
    """Consecutive windows (default 1 s) inside the segments, with per-channel metrics."""
    info: pd.DataFrame        # a, b, t0, t1, phase, label (at the window centre), rest (bool)
    metrics: dict[str, np.ndarray]   # name -> (n_windows, channels)
    cls: np.ndarray           # (n_windows, channels) dominant problem, see WINDOW_CLASSES


WINDOW_CLASSES = ("ok", "mains", "noisy", "low-freq", "pop", "flat", "railed")


def windows(rec: Recording) -> Windows:
    th, fs = rec.th, rec.fs
    w = max(2, int(round(th["window_s"] * fs)))
    rows, metric_rows = [], []
    for a0, b0 in rec.segments:
        if not np.isfinite(rec.emg[a0, 0]):
            continue
        for a in range(a0, b0, w):
            b = min(a + w, b0)
            if b - a < w // 2:
                continue
            mid = (a + b) // 2
            x = rec.raw[a:b]
            rows.append({"a": a, "b": b, "t0": rec.t[a], "t1": rec.t[b - 1] + 1 / fs,
                         "segment": int(rec.seg[a]), "phase": rec.phase[mid], "label": rec.label[mid],
                         "rest": rec.label[mid] == "REST" and rec.phase[mid] != "PAUSE"})
            metric_rows.append((
                np.sqrt(np.mean(rec.emg[a:b] ** 2, axis=0)),
                np.sqrt(np.mean(rec.low[a:b] ** 2, axis=0)),
                np.sqrt(np.maximum(0.0, np.mean(rec.mains_pow[a:b], axis=0))),
                x.std(axis=0),
                x.mean(axis=0),
                np.abs(np.diff(rec.notched[a:b], axis=0)).max(axis=0) if b - a > 1 else np.zeros(x.shape[1]),
                np.mean(np.abs(x) >= th["railed_level"] * rec.full_scale_uv, axis=0),
            ))
    names = ("emg_rms", "lowfreq_rms", "mains_rms", "raw_std", "raw_mean", "max_jump", "railed")
    nc = len(rec.channels)
    metrics = {k: (np.array([r[i] for r in metric_rows]) if metric_rows else np.zeros((0, nc)))
               for i, k in enumerate(names)}
    info = pd.DataFrame(rows, columns=["a", "b", "t0", "t1", "segment", "phase", "label", "rest"])
    cls = np.full(metrics["emg_rms"].shape, "ok", dtype=object)
    rest = info["rest"].to_numpy(bool)[:, None] if len(info) else np.zeros((0, 1), bool)
    if rec.mode == "free":
        rest = np.zeros_like(rest)          # no labels: the noise floor is not judged per window
    # lowest priority first, so the most serious problem of a window wins
    cls[rest & (metrics["emg_rms"] > th["noise_bad_uv"])] = "noisy"
    cls[metrics["mains_rms"] > th["mains_bad_uv"]] = "mains"
    cls[metrics["lowfreq_rms"] > th["lowfreq_bad_uv"]] = "low-freq"
    cls[metrics["max_jump"] > th["pop_uv"]] = "pop"
    cls[metrics["raw_std"] < th["flat_std_uv"]] = "flat"
    cls[metrics["railed"] > 0] = "railed"
    return Windows(info, metrics, cls)


def selections(rec: Recording, win: Windows) -> tuple[dict[str, list[tuple[int, int]]], str]:
    """Row runs of REST and of each gesture's contractions, and where the rest comes from."""
    th = rec.th
    rest_mask = (rec.label == "REST") & (rec.phase != "PAUSE") & (rec.bad == 0)
    sel = {"REST": _labelled_runs(rec, rest_mask, th["rest_settle_s"])}
    active_phase = np.isin(rec.phase, ["QUALITY", "CALIBRATION", "HOLD"])
    for g in GESTURES:
        runs = _labelled_runs(rec, (rec.label == g) & active_phase & (rec.bad == 0),
                              th["active_settle_s"])
        if runs:
            sel[g] = runs
    source = "labelled rest"
    rest_s = sum(b - a for a, b in sel["REST"]) / rec.fs
    if rest_s < th["min_rest_s"] and len(win.info) >= 5:
        level = np.median(win.metrics["emg_rms"], axis=1)
        order = np.argsort(level)
        k = max(1, int(round(0.2 * len(order))))
        sel["REST"] = [(int(win.info["a"].iloc[i]), int(win.info["b"].iloc[i])) for i in order[:k]]
        source = "quietest 20% of the windows (no labelled rest)"
        if not any(g in sel for g in GESTURES):
            top = order[-max(1, int(round(0.1 * len(order)))):]
            sel["ACTIVE"] = [(int(win.info["a"].iloc[i]), int(win.info["b"].iloc[i])) for i in top]
    return sel, source


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------
def _grade(value: float | None, warn: float, bad: float, low_is_bad: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    if low_is_bad:
        return "BAD" if value < bad else "WARN" if value < warn else "GOOD"
    return "BAD" if value > bad else "WARN" if value > warn else "GOOD"


def _worst(grades) -> str:
    known = [g for g in grades if g in GRADES]
    return max(known, key=GRADES.index) if known else "N/A"


def _clock_jump(events: pd.DataFrame) -> str | None:
    """A pre-start event logged later than the session start: the laptop clock went back."""
    if events.empty or "detail" not in events:
        return None
    start = events.loc[events["event"] == "SESSION_START", "wall_time"]
    if start.empty:
        return None
    t_start = datetime.fromisoformat(start.iloc[0])
    for d in events["detail"]:
        m = re.match(r"\[before start (\S+)\]", str(d))
        if m and datetime.fromisoformat(m.group(1)) > t_start:
            late = (datetime.fromisoformat(m.group(1)) - t_start).total_seconds()
            return f"an event before the start is stamped {late / 60:.0f} min after it"
    return None


def link_report(rec: Recording) -> dict:
    th, fs, s = rec.th, rec.fs, rec.session
    pk = s.emg["package_num"].to_numpy(np.int64)
    ts = s.emg["timestamp"].to_numpy(float)
    mod = s.meta.get("source", {}).get("package_modulus", 256)
    lost = dups = loss_events = 0
    span = 0.0
    for a, b in rec.segments:
        d = np.diff(pk[a:b])
        d = d % int(mod) if mod else np.where(d < 0, 1, d)
        dups += int(np.count_nonzero(d == 0))
        lost_each = np.where(d > 1, d - 1, 0)
        lost += int(lost_each.sum())
        loss_events += int(np.count_nonzero(lost_each))
        span += ts[b - 1] - ts[a] + 1 / fs if b > a else 0.0
    n_valid = int(rec.valid.sum())
    dt = np.diff(ts)
    gaps = [{"t_s": round(float(rec.t[i]), 2), "gap_s": round(float(dt[i]), 2),
             "phase_after": rec.phase[i + 1]} for i in np.flatnonzero(dt > th["split_gap_s"])]
    ev = s.events
    count = (lambda name: int((ev["event"] == name).sum())) if not ev.empty else (lambda name: 0)
    out = {
        "samples": rec.n,
        "duration_s": round(float(ts[-1] - ts[0]), 1),
        "segments": len(rec.segments),
        "rate_hz": round((n_valid + lost) / span, 2) if span > 0 else float("nan"),
        "nominal_hz": fs,
        "lost": lost,
        "loss_pct": round(100.0 * lost / (n_valid + lost), 3) if n_valid + lost else 0.0,
        "loss_events": loss_events,
        "duplicates": dups,
        "gaps": gaps,
        "gap_total_s": round(sum(g["gap_s"] for g in gaps), 1),
        "zero_rows": int((~rec.valid).sum()),
        "backward_timestamps": int(np.count_nonzero(dt < 0)),
        "reconnects": count("RECONNECT_OK"),
        "reconnects_failed": count("RECONNECT_FAILED"),
        "clock": _clock_jump(ev),
    }
    problems = []
    g = _grade(out["loss_pct"], th["loss_warn_pct"], th["loss_bad_pct"])
    if g != "GOOD":
        problems.append((g, "loss", f"{lost} samples lost ({out['loss_pct']:.2f}%) in {loss_events} events"))
    if gaps:
        problems.append(("WARN", "gap", f"{len(gaps)} gap(s) in the stream, {out['gap_total_s']} s in total"))
    if out["zero_rows"]:
        problems.append(("WARN", "zero_rows", f"{out['zero_rows']} all-zero row(s) (invalid samples)"))
    if np.isfinite(out["rate_hz"]) and abs(out["rate_hz"] - fs) > fs * th["rate_tol_pct"] / 100:
        problems.append(("WARN", "rate", f"{out['rate_hz']:.1f} samples/s instead of {fs:g}"))
    if out["backward_timestamps"]:
        problems.append(("WARN", "clock", f"{out['backward_timestamps']} timestamp(s) going backwards"))
    if out["clock"]:
        problems.append(("WARN", "clock", f"laptop clock jump: {out['clock']}"))
    out["problems"] = problems
    out["status"] = _worst([p[0] for p in problems] or ["GOOD"])
    return out


def _narrowband_uv(f: np.ndarray, p: np.ndarray, f0: float) -> np.ndarray:
    """RMS of a narrow line at f0 above the local floor (the median of the neighbouring bins)."""
    if f0 > f[-1] - 2:
        return np.full(p.shape[1], np.nan)
    df = f[1] - f[0]
    line = np.abs(f - f0) <= 2
    side = (np.abs(f - f0) >= 4) & (np.abs(f - f0) <= 10)
    floor = np.median(p[side], axis=0)
    return np.sqrt(np.maximum(0.0, (p[line] - floor).sum(axis=0)) * df)


def spectrum(rec: Recording, runs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    parts = [rec.raw[a:b] for a, b in runs if b - a >= 16]
    if not parts:
        return None
    return mean_psd(parts, rec.fs)


def spurs(rec: Recording, f: np.ndarray, p: np.ndarray, min_db: float = 10.0) -> pd.DataFrame:
    """Narrow spectral peaks standing out of the floor at rest (interference lines)."""
    df = f[1] - f[0]
    k = max(3, int(round(10 / df)) | 1)
    m = rec.th["mains_hz"]
    rows = []
    for c, name in enumerate(rec.channels):
        floor = signal.medfilt(p[:, c], k) + 1e-12
        excess_db = 10 * np.log10(p[:, c] / floor + 1e-12)
        peaks, _ = signal.find_peaks(excess_db, height=min_db)
        for i in peaks:
            if not 5 <= f[i] <= f[-1] - 3:
                continue
            j = slice(max(0, i - 1), i + 2)
            amp = np.sqrt(max(0.0, float((p[j, c] - floor[j]).sum())) * df)
            near_mains = min(abs(f[i] - m), abs(f[i] - 2 * m)) <= 2
            rows.append({"channel": name, "freq_hz": round(float(f[i]), 1),
                         "above_floor_db": round(float(excess_db[i]), 1), "rms_uv": round(amp, 2),
                         "source": "mains" if near_mains else "other interference"})
    out = pd.DataFrame(rows, columns=["channel", "freq_hz", "above_floor_db", "rms_uv", "source"])
    return out.sort_values("rms_uv", ascending=False).reset_index(drop=True)


@dataclass
class Report:
    rec: Recording
    win: Windows
    link: dict
    channels: pd.DataFrame        # one row per channel: metrics, status, problems
    problems: dict[str, list[tuple[str, str, str]]]   # channel -> [(grade, code, text)]
    gesture_rms: pd.DataFrame     # rows REST + gestures (or ACTIVE), columns channels
    rest_source: str
    rest_s: float
    psd_rest: tuple[np.ndarray, np.ndarray] | None
    psd_active: tuple[np.ndarray, np.ndarray] | None
    spurs: pd.DataFrame
    corr: np.ndarray | None
    trials: pd.DataFrame | None
    verdict: str
    verdict_text: str


def analyze(rec_or_folder: Recording | str | Path | None = None, thresholds: dict | None = None) -> Report:
    rec = rec_or_folder if isinstance(rec_or_folder, Recording) else load(rec_or_folder, thresholds)
    th, fs, nc = rec.th, rec.fs, len(rec.channels)
    win = windows(rec)
    sel, rest_source = selections(rec, win)
    link = link_report(rec)

    rest_rows = _rows(sel["REST"])
    rms = lambda rows, x: (np.sqrt(np.mean(x[rows] ** 2, axis=0)) if len(rows) else np.full(nc, np.nan))
    gesture_rms = pd.DataFrame({k: rms(_rows(v), rec.emg) for k, v in sel.items()},
                               index=rec.channels).T
    psd_rest = spectrum(rec, sel["REST"])
    active_runs = [r for k, v in sel.items() if k != "REST" for r in v]
    psd_active = spectrum(rec, active_runs)
    m = th["mains_hz"]
    if psd_rest is not None:
        mains = _narrowband_uv(*psd_rest, m)
        mains2 = _narrowband_uv(*psd_rest, 2 * m)
        spur_table = spurs(rec, *psd_rest)
    else:
        mains = mains2 = np.full(nc, np.nan)
        spur_table = spurs(rec, np.arange(2.0), np.zeros((2, nc)))
    noise = rms(rest_rows, rec.emg)
    lowfreq = rms(rest_rows, rec.low)
    corr = None
    if len(rest_rows) > fs and nc > 1:
        corr = np.corrcoef(rec.emg[rest_rows].T)

    valid_min = rec.valid.sum() / fs / 60
    jumps_total = np.zeros(nc)
    dc_seg = []
    for a, b in rec.segments:
        x = rec.raw[a:b]
        dc_seg.append(np.median(x, axis=0))
        if b - a > 1:
            jump = np.abs(np.diff(rec.notched[a:b], axis=0)) > th["pop_uv"]
            for c in range(nc):
                idx = np.flatnonzero(jump[:, c])
                if idx.size:   # jumps closer than 0.1 s are one pop
                    jumps_total[c] += 1 + np.count_nonzero(np.diff(idx) > 0.1 * fs)
    dc_seg = np.array(dc_seg)
    dc_max = dc_seg[np.argmax(np.abs(dc_seg), axis=0), np.arange(nc)]
    wander = np.full(nc, np.nan)
    if len(win.info):
        dev = []
        for k, g in win.info.groupby("segment"):
            means = win.metrics["raw_mean"][g.index]
            dev.append(np.abs(means - np.median(means, axis=0)))
        wander = np.percentile(np.concatenate(dev), 95, axis=0)
    valid_rows = rec.raw[rec.valid]
    railed = np.mean(np.abs(valid_rows) >= th["railed_level"] * rec.full_scale_uv, axis=0)
    flat = (win.cls == "flat").mean(axis=0) if len(win.info) else np.zeros(nc)
    gestures = [k for k in gesture_rms.index if k != "REST"]

    rows, problems = [], {}
    for c, name in enumerate(rec.channels):
        p = []
        dc_frac = abs(dc_max[c]) / rec.full_scale_uv if np.isfinite(rec.full_scale_uv) else np.nan
        snr = {g: gesture_rms.loc[g, name] / noise[c] for g in gestures} if noise[c] > 0 else {}
        best = max(snr, key=snr.get) if snr else None

        def check(code, value, warn, bad, text, low_is_bad=False):
            g = _grade(value, warn, bad, low_is_bad)
            if g in ("WARN", "BAD"):
                p.append((g, code, text))
        check("railed", railed[c], th["railed_bad_frac"] / 10, th["railed_bad_frac"],
              f"railed {100 * railed[c]:.2f}% of the samples")
        check("flat", flat[c], th["flat_bad_frac"] / 5, th["flat_bad_frac"],
              f"flat in {100 * flat[c]:.0f}% of the windows")
        check("dc", dc_frac, th["dc_warn_frac"], th["dc_bad_frac"],
              f"electrode offset {dc_max[c] / 1000:.1f} mV ({100 * dc_frac:.0f}% of full scale)")
        check("wander", wander[c], th["wander_warn_uv"], th["wander_bad_uv"],
              f"baseline wander {wander[c]:.0f} µV")
        pops_min = jumps_total[c] / valid_min if valid_min > 0 else np.nan
        check("pops", pops_min, th["pop_warn_per_min"], th["pop_bad_per_min"],
              f"{int(jumps_total[c])} pop(s), {pops_min:.1f}/min")
        check("mains", mains[c], th["mains_warn_uv"], th["mains_bad_uv"],
              f"50 Hz at rest {mains[c]:.1f} µV RMS")
        check("noise", noise[c], th["noise_warn_uv"], th["noise_bad_uv"],
              f"noise floor at rest {noise[c]:.1f} µV RMS")
        check("lowfreq", lowfreq[c], th["lowfreq_warn_uv"], th["lowfreq_bad_uv"],
              f"1-15 Hz at rest {lowfreq[c]:.1f} µV RMS")
        if best is not None:
            check("snr", snr[best], th["snr_warn"], th["snr_bad"],
                  f"weak EMG: best gesture {best} only {snr[best]:.1f}x rest", low_is_bad=True)
        if corr is not None:
            for d, other in enumerate(rec.channels):
                if d != c and abs(corr[c, d]) > th["corr_warn"]:
                    p.append(("WARN", "corr", f"correlated with {other} at rest (r = {corr[c, d]:.2f})"))
        problems[name] = p
        rows.append({
            "channel": name,
            "status": _worst([x[0] for x in p] or ["GOOD"]),
            "dc_mV": round(dc_max[c] / 1000, 2),
            "wander_uV": round(float(wander[c]), 1),
            "railed_pct": round(100 * float(railed[c]), 3),
            "flat_pct": round(100 * float(flat[c]), 1),
            "pops": int(jumps_total[c]),
            "mains_rest_uV": round(float(mains[c]), 1),
            "mains100_rest_uV": round(float(mains2[c]), 1),
            "noise_rest_uV": round(float(noise[c]), 2),
            "lowfreq_rest_uV": round(float(lowfreq[c]), 1),
            "best_gesture": best or "-",
            "snr_best": round(float(snr[best]), 1) if best else float("nan"),
            "problems": "; ".join(x[2] for x in p) or "-",
        })
    channels = pd.DataFrame(rows)

    n_bad = int((channels["status"] == "BAD").sum())
    if link["status"] == "BAD" or n_bad >= max(1, int(np.ceil(nc / 2))):
        verdict = "BAD"
        text = "poor stream: fix the setup before recording more"
    elif link["status"] == "WARN" or (channels["status"] != "GOOD").any():
        verdict = "WARN"
        text = "usable, but check the flagged items"
    else:
        verdict, text = "GOOD", "good stream: usable as it is"

    trials = an.trial_table(rec.session) if rec.mode != "free" else None
    return Report(rec=rec, win=win, link=link, channels=channels, problems=problems,
                  gesture_rms=gesture_rms, rest_source=rest_source,
                  rest_s=round(len(rest_rows) / fs, 1), psd_rest=psd_rest, psd_active=psd_active,
                  spurs=spur_table, corr=corr, trials=trials, verdict=verdict, verdict_text=text)


def advice(rep: Report) -> list[tuple[str, str, str]]:
    """(grade, where, advice) for every problem code found, the most serious first."""
    found: dict[str, tuple[str, list[str]]] = {}
    sources = [("link", rep.link["problems"])] + list(rep.problems.items())
    for where, items in sources:
        for grade, code, _ in items:
            g, wh = found.get(code, ("GOOD", []))
            found[code] = (_worst([g, grade]), wh + ([where] if where not in wh else []))
    ordered = sorted(found.items(), key=lambda kv: -GRADES.index(kv[1][0]))
    return [(g, ", ".join(wh), ADVICE[code]) for code, (g, wh) in ordered]


# ---------------------------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------------------------
def report_text(rep: Report) -> str:
    rec, link, m = rep.rec, rep.link, rep.rec.meta
    q = (rec.session.quality or {}).get("attempts") or []
    lines = [
        f"Session {rec.folder.name}  ({m.get('status')}, {rec.mode})",
        f"VERDICT: {rep.verdict} - {rep.verdict_text}",
        "",
        f"Link: {link['status']}",
        f"  {link['samples']} samples, {link['duration_s']} s, {link['rate_hz']} samples/s "
        f"(nominal {link['nominal_hz']:g}), {link['segments']} segment(s)",
        f"  lost {link['lost']} ({link['loss_pct']:.3f}%) in {link['loss_events']} events, "
        f"{link['duplicates']} duplicates, {link['zero_rows']} all-zero rows, "
        f"{len(link['gaps'])} gaps ({link['gap_total_s']} s), {link['reconnects']} reconnections",
    ]
    lines += [f"  {g}: {t}" for g, _, t in link["problems"]]
    if q:
        lines.append(f"Quality check in the app: {q[-1]['result']} ({len(q)} attempt(s))")
    if rep.trials is not None and len(rep.trials):
        t = rep.trials
        lines.append(f"Trials: {int(t['complete'].sum())} complete, {int(t['bad'].sum())} bad, "
                     f"{int((t['attempts'] > 1).sum())} repeated")
    lines += ["", f"Channels (rest: {rep.rest_source}, {rep.rest_s} s):"]
    cols = ["channel", "status", "dc_mV", "wander_uV", "pops", "mains_rest_uV", "noise_rest_uV",
            "lowfreq_rest_uV", "best_gesture", "snr_best"]
    lines += ["  " + line for line in rep.channels[cols].to_string(index=False).splitlines()]
    for name, items in rep.problems.items():
        for g, _, text in items:
            lines.append(f"  {name} {g}: {text}")
    lines += ["", "RMS per gesture (µV, 20-120 Hz, mains notched):"]
    lines += ["  " + line for line in rep.gesture_rms.round(1).to_string().splitlines()]
    other = rep.spurs[rep.spurs["source"] != "mains"]
    if len(other):
        lines += ["", "Other interference lines at rest (not mains):"]
        lines += ["  " + line for line in other.head(8).to_string(index=False).splitlines()]
    tips = advice(rep)
    if tips:
        lines += ["", "What to do:"]
        lines += [f"  [{g}] {where}: {text}" for g, where, text in tips]
    return "\n".join(lines) + "\n"


def summary_row(folder: str | Path, thresholds: dict | None = None) -> dict:
    """One line per session, for comparing several recordings."""
    try:
        rep = analyze(folder, thresholds)
    except Exception as e:
        return {"session": Path(folder).name, "verdict": "ERROR", "note": f"{type(e).__name__}: {e}"}
    row = {"session": rep.rec.folder.name, "verdict": rep.verdict, "status": rep.rec.meta.get("status"),
           "mode": rep.rec.mode, "duration_s": rep.link["duration_s"], "loss_pct": rep.link["loss_pct"],
           "gaps": len(rep.link["gaps"]), "zero_rows": rep.link["zero_rows"]}
    for _, r in rep.channels.iterrows():
        row[r["channel"]] = f"{r['status']} n{r['noise_rest_uV']:.0f} m{r['mains_rest_uV']:.0f}"
    return row


# ---------------------------------------------------------------------------------------------
# Plots (matplotlib, for the notebook)
# ---------------------------------------------------------------------------------------------
LABEL_COLORS = {"OPEN": "#2F80ED", "CLOSE": "#F2994A", "INDEX": "#BB6BD9", "REST": "#6FA8A0",
                "TRANSITION": "#BDBDBD", "PAUSE": "#8A9BA8"}
# distinct from the protocol colours (blue, orange, violet, teal, grey) drawn in the same plot
CLASS_COLORS = {"ok": "#0ca30c", "mains": "#fab219", "noisy": "#e87ba4", "low-freq": "#8a5a00",
                "pop": "#0b0b0b", "flat": "#c3c2b7", "railed": "#d03b3b"}
INK, MUTED = "#0b0b0b", "#52514e"


def _colors(rec: Recording) -> dict:
    return {**LABEL_COLORS, **rec.meta.get("config", {}).get("display", {}).get("colors", {})}


def _span(rec: Recording, t0: float | None, t1: float | None) -> tuple[int, int]:
    a = 0 if t0 is None else int(np.searchsorted(rec.t, t0))
    b = rec.n if t1 is None else int(np.searchsorted(rec.t, t1))
    return a, max(a + 1, b)


def shade_labels(ax, rec: Recording, a: int = 0, b: int | None = None) -> None:
    """Gesture contractions, cues and pauses as background bands; gaps hatched."""
    b = rec.n if b is None else b
    colors = _colors(rec)
    for i, j in _key_runs([rec.phase[a:b], rec.label[a:b]], b - a):
        ph, lb = rec.phase[a + i], rec.label[a + i]
        x0, x1 = rec.t[a + i], rec.t[a + j - 1]
        if lb in GESTURES:
            ax.axvspan(x0, x1, color=colors[lb], alpha=0.22, lw=0)
        elif ph == "PAUSE" or lb == "TRANSITION":
            ax.axvspan(x0, x1, color=colors["PAUSE" if ph == "PAUSE" else "TRANSITION"], alpha=0.15, lw=0)
    for (_, b0), (a1, _) in zip(rec.segments[:-1], rec.segments[1:]):
        if a <= b0 <= b:
            ax.axvspan(rec.t[b0 - 1], rec.t[a1], fill=False, hatch="xx", edgecolor="#d03b3b", lw=0)


def _legend_labels(fig, rec: Recording) -> None:
    from matplotlib.patches import Patch

    colors = _colors(rec)
    present = set(rec.label)
    handles = [Patch(color=colors[g], alpha=0.4, label=g) for g in GESTURES if g in present]
    handles += [Patch(color=colors["TRANSITION"], alpha=0.3, label="cue"),
                Patch(color=colors["PAUSE"], alpha=0.3, label="pause"),
                Patch(fill=False, hatch="xx", edgecolor="#d03b3b", label="gap")]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8, frameon=False)


def _plot_trace(ax, rec: Recording, y: np.ndarray, a: int, b: int, max_points: int = 6000) -> None:
    """Per segment; long spans drawn as a min/max band so spikes are not lost. The band width
    (samples per point) is set by the whole span, so every segment is drawn at the same density."""
    k = max(1, (b - a) // (max_points // 2))
    for s0, s1 in rec.segments:
        lo, hi = max(a, s0), min(b, s1)
        if hi - lo < 2:
            continue
        t, v = rec.t[lo:hi], y[lo:hi]
        n = (hi - lo) // k * k
        if k > 1 and n >= 2 * k:
            tb = t[:n].reshape(-1, k)[:, 0]
            vb = v[:n].reshape(-1, k)
            ax.fill_between(tb, np.nanmin(vb, axis=1), np.nanmax(vb, axis=1), color=INK, lw=0.3,
                            edgecolor=INK)
        else:
            ax.plot(t, v, lw=0.5, color=INK)


def plot_signals(rec: Recording, kind: str = "emg", t0: float | None = None, t1: float | None = None,
                 scale_uv: float | None = None):
    """kind: 'emg' (20-120 Hz, mains notched), 'raw' (segment median removed) or 'lowfreq'."""
    import matplotlib.pyplot as plt

    y = {"emg": rec.emg, "raw": rec.centered, "lowfreq": rec.low}[kind]
    title = {"emg": "filtered 20-120 Hz, 50/100 Hz notched", "raw": "raw, segment median removed",
             "lowfreq": "1-15 Hz (movement, cables)"}[kind]
    a, b = _span(rec, t0, t1)
    nc = len(rec.channels)
    fig, axes = plt.subplots(nc, 1, sharex=True, figsize=(15, 1.5 * nc + 1.2), squeeze=False)
    for c, ax in enumerate(axes[:, 0]):
        shade_labels(ax, rec, a, b)
        _plot_trace(ax, rec, y[:, c], a, b)
        seg = y[a:b, c]
        lim = scale_uv or (np.nanpercentile(np.abs(seg), 99.5) * 1.3 + 1e-9 if np.isfinite(seg).any() else 1)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(f"{rec.channels[c]}\nµV", rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 0].set_xlim(rec.t[a], rec.t[b - 1])
    _legend_labels(fig, rec)
    fig.suptitle(f"{rec.folder.name}: {title}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_link(rep: Report):
    """Samples received per second (from the timestamps) and lost packets over time."""
    import matplotlib.pyplot as plt

    rec = rep.rec
    fig, ax = plt.subplots(figsize=(15, 2.8))
    edges = np.arange(0, max(1.0, np.floor(rec.t[-1])) + 0.5, 1.0)    # whole seconds only
    counts, _ = np.histogram(rec.t[rec.valid], bins=edges)
    ax.step(edges[:-1], counts, where="post", lw=1, color="#2a78d6", label="samples per second")
    ax.axhline(rec.fs, color=MUTED, lw=1, ls="--", label=f"nominal {rec.fs:g}")
    pk = rec.session.emg["package_num"].to_numpy(np.int64)
    mod = rec.meta.get("source", {}).get("package_modulus", 256)
    loss_t = []
    for a, b in rec.segments:
        d = np.diff(pk[a:b])
        d = d % int(mod) if mod else np.where(d < 0, 1, d)
        loss_t += rec.t[a + 1 + np.flatnonzero(d > 1)].tolist()
    if loss_t:
        ax.plot(loss_t, np.full(len(loss_t), rec.fs * 1.08), "v", ms=8, color="#d03b3b",
                label=f"packet loss ({len(loss_t)} events)")
    for g in rep.link["gaps"]:
        ax.axvspan(g["t_s"], g["t_s"] + g["gap_s"], fill=False, hatch="xx", edgecolor="#d03b3b", lw=0)
    ax.set_ylim(0, rec.fs * 1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("samples / s")
    ax.set_title(f"Link: {rep.link['status']}", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def plot_psd(rep: Report):
    """Spectrum per channel at rest and during the gestures, mains lines marked."""
    import matplotlib.pyplot as plt

    rec = rep.rec
    nc = len(rec.channels)
    cols = min(nc, 4)
    rows = int(np.ceil(nc / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False, sharey=True)
    m = rec.th["mains_hz"]
    for c, name in enumerate(rec.channels):
        ax = axes[c // cols, c % cols]
        if rep.psd_rest is not None:
            ax.semilogy(rep.psd_rest[0], rep.psd_rest[1][:, c], lw=1.2, color="#2a78d6", label="rest")
        if rep.psd_active is not None:
            ax.semilogy(rep.psd_active[0], rep.psd_active[1][:, c], lw=1.2, color="#eb6834",
                        label="gestures" if rec.mode != "free" else "loudest windows")
        for f0 in (m, 2 * m):
            if f0 < rec.fs / 2:
                ax.axvline(f0, color=MUTED, lw=0.8, ls=":")
        ax.axvspan(*EMG_BAND, color="#f0efec", zorder=-1)
        ax.set_title(f"{name}  ({rep.channels.loc[c, 'status']})", fontsize=10, loc="left")
        ax.set_xlabel("Hz")
        ax.grid(alpha=0.15, which="both")
    for k in range(nc, rows * cols):
        axes[k // cols, k % cols].axis("off")
    axes[0, 0].set_ylabel("µV² / Hz")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Spectrum (dotted: mains 50/100 Hz; shaded: EMG band)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_envelope(rec: Recording, window_s: float = 0.25, t0: float | None = None, t1: float | None = None):
    """Moving RMS of the filtered signal, gestures shaded: the EMG should rise in the bands."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    k = max(1, int(round(window_s * rec.fs)))
    env = np.full_like(rec.emg, np.nan)
    for a, b in rec.segments:
        if np.isfinite(rec.emg[a, 0]):
            env[a:b] = np.sqrt(uniform_filter1d(rec.emg[a:b] ** 2, k, axis=0))
    a, b = _span(rec, t0, t1)
    nc = len(rec.channels)
    fig, axes = plt.subplots(nc, 1, sharex=True, figsize=(15, 1.4 * nc + 1.2), squeeze=False)
    for c, ax in enumerate(axes[:, 0]):
        shade_labels(ax, rec, a, b)
        for s0, s1 in rec.segments:
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                ax.plot(rec.t[lo:hi], env[lo:hi, c], lw=0.8, color=INK)
        top = np.nanpercentile(env[a:b, c], 99.5) if np.isfinite(env[a:b, c]).any() else 1.0
        ax.set_ylim(0, top * 1.15 + 1e-9)
        ax.set_ylabel(f"{rec.channels[c]}\nµV", rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 0].set_xlim(rec.t[a], rec.t[b - 1])
    _legend_labels(fig, rec)
    fig.suptitle(f"{rec.folder.name}: RMS envelope ({window_s * 1000:.0f} ms, filtered)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_gesture_rms(rep: Report):
    import matplotlib.pyplot as plt

    g = rep.gesture_rms
    colors = {**_colors(rep.rec), "ACTIVE": "#eb6834"}
    nc = len(rep.rec.channels)
    x = np.arange(nc)
    w = 0.8 / max(1, len(g))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * nc + 3), 4))
    for k, name in enumerate(g.index):
        ax.bar(x + (k - (len(g) - 1) / 2) * w, g.loc[name].to_numpy(), w * 0.92,
               color=colors.get(name, "#999"), label=name)
    ax.set_xticks(x, rep.rec.channels)
    ax.set_ylabel("RMS (µV, 20-120 Hz)")
    ax.set_title("RMS per channel: rest vs gestures (the gestures should stand well above rest)",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def plot_corr(rep: Report):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if rep.corr is None:
        return None
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    ch = rep.rec.channels
    fig, ax = plt.subplots(figsize=(1.1 * len(ch) + 2, 1.0 * len(ch) + 1.2))
    im = ax.imshow(rep.corr, vmin=-1, vmax=1, cmap=cmap)
    for i in range(len(ch)):
        for j in range(len(ch)):
            ax.text(j, i, f"{rep.corr[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(ch)), ch)
    ax.set_yticks(range(len(ch)), ch)
    ax.set_title("Correlation at rest (filtered)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_timeline(rep: Report):
    """Dominant problem of each channel in each 1 s window; top row: the protocol."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rec, win = rep.rec, rep.win
    nc = len(rec.channels)
    fig, ax = plt.subplots(figsize=(15, 0.45 * (nc + 1) + 1.4))
    starts = win.info["t0"].to_numpy()
    widths = (win.info["t1"] - win.info["t0"]).to_numpy()
    for c in range(nc):
        for name, color in CLASS_COLORS.items():
            sel = win.cls[:, c] == name
            if sel.any():
                ax.broken_barh(list(zip(starts[sel], widths[sel])), (nc - c - 0.4, 0.8), color=color, lw=0)
    colors = _colors(rec)
    for i, j in _key_runs([rec.phase, rec.label], rec.n):
        lb, ph = rec.label[i], rec.phase[i]
        key = lb if lb in GESTURES else "PAUSE" if ph == "PAUSE" else "REST" if lb == "REST" else None
        if key:
            ax.broken_barh([(rec.t[i], rec.t[j - 1] - rec.t[i])], (nc + 0.6, 0.8), color=colors[key], lw=0)
    for g in rep.link["gaps"]:
        ax.axvspan(g["t_s"], g["t_s"] + g["gap_s"], fill=False, hatch="xx", edgecolor="#d03b3b", lw=0)
    ax.set_yticks([*range(nc, 0, -1), nc + 1], [*rec.channels, "protocol"])
    ax.set_ylim(0.4, nc + 1.6)
    ax.set_xlim(0, rec.t[-1])
    ax.set_xlabel("time (s)")
    present = [k for k in CLASS_COLORS if (win.cls == k).any()]
    handles = [Patch(color=CLASS_COLORS[k], label=k) for k in present]
    protocol = [k for k in (*GESTURES, "REST") if k in set(rec.label)] + (["PAUSE"] if "PAUSE" in set(rec.phase) else [])
    handles += [Patch(color=colors[k], label=f"protocol: {k}") for k in protocol]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=len(handles),
              fontsize=8, frameon=False)
    ax.set_title("What disturbs each channel, second by second", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path, help="session folder (default: the latest)")
    ap.add_argument("--all", action="store_true", help="one summary line per session in data/")
    args = ap.parse_args(argv)
    if args.all:
        rows = [summary_row(f) for f in list_sessions()]
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(pd.DataFrame(rows).to_string(index=False))
        return 0
    try:
        rep = analyze(args.folder)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report_text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
