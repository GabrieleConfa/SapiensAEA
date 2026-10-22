"""Reading a session folder back: trials, packet loss, gaps, per-gesture RMS, summary.txt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from emgacq.markers import GESTURES
from emgacq.signals import filtfilt


@dataclass
class Session:
    folder: Path
    emg: pd.DataFrame
    events: pd.DataFrame
    meta: dict
    quality: dict | None
    order: dict | None

    @property
    def fs(self) -> float:
        return float(self.meta.get("sampling_rate_hz", 250.0))

    @property
    def channel_columns(self) -> list[str]:
        return [c for c in self.emg.columns if c.startswith("ch") and c[2:].isdigit()]


def load_session(folder: str | Path) -> Session:
    folder = Path(folder)
    emg = pd.read_csv(folder / "emg.csv", keep_default_na=False,
                      dtype={"phase": str, "label": str})
    events = (pd.read_csv(folder / "events.csv", keep_default_na=False)
              if (folder / "events.csv").exists() else pd.DataFrame())
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    qc_path, order_path = folder / "quality_check.json", folder / "trial_order.json"
    quality = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else None
    order = json.loads(order_path.read_text(encoding="utf-8")) if order_path.exists() else None
    s = Session(folder, emg, events, meta, quality, order)
    _apply_logged_bad_flags(s)
    return s


def _apply_logged_bad_flags(s: Session) -> None:
    """If the session crashed before the bad-trial update, apply the flags from events.csv."""
    if s.meta.get("status") != "recording" or s.events.empty:
        return
    bad: set[int] = set()
    for _, row in s.events.iterrows():
        if row["event"] == "TRIAL_BAD":
            bad.add(int(row["trial_id"]))
        elif row["event"] == "TRIAL_BAD_UNDO":
            bad.discard(int(row["trial_id"]))
    if bad:
        s.emg.loc[s.emg["trial_id"].isin(bad), "trial_bad"] = 1


def _last_attempt(g: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """(number of attempts, rows of the last attempt) of one trial's rows.

    Every attempt starts with a CUE marker row; a trial paused during its cue or hold is repeated.
    """
    starts = np.flatnonzero(((g["phase"] == "CUE") & (g["marker"] != 0)).to_numpy())
    if len(starts) == 0:
        return 1, g
    return len(starts), g.iloc[starts[-1]:]


def trial_table(s: Session) -> pd.DataFrame:
    """One row per trial: gesture, block, attempts, hold length, bad, complete.

    A trial paused during its cue or hold is repeated; its interrupted attempts are marked bad
    row by row, and the trial itself is judged on its last attempt.
    """
    df = s.emg
    rows = []
    for tid, g in df[df["trial_id"] > 0].groupby("trial_id", sort=True):
        attempts, last = _last_attempt(g)
        hold = last[last["phase"] == "HOLD"]
        if len(hold):
            gesture = hold["label"].iloc[0]
            bad = bool(hold["trial_bad"].max())
        else:
            gesture = s.order["trials"][tid - 1]["gesture"] if s.order else ""
            bad = bool(g["trial_bad"].max())
        rows.append({"trial_id": int(tid), "block": int(g["block"].iloc[0]), "gesture": gesture,
                     "attempts": attempts, "hold_s": round(len(hold) / s.fs, 2), "bad": bad,
                     "complete": bool(len(hold)) and bool((last["phase"] == "REST").any())})
    # explicit types: with no trial yet (aborted early) the columns would be untyped and a
    # boolean mask like trials[trials["bad"]] would be read as a column selection
    return pd.DataFrame(rows, columns=["trial_id", "block", "gesture", "attempts", "hold_s",
                                       "bad", "complete"]).astype(
        {"trial_id": int, "block": int, "attempts": int, "hold_s": float, "bad": bool, "complete": bool})


def packet_loss(s: Session) -> dict:
    """Lost samples from the package numbers in emg.csv (modulus from session.json)."""
    mod = s.meta.get("source", {}).get("package_modulus", 256)
    pk = s.emg["package_num"].to_numpy().astype(np.int64)
    if len(pk) < 2:
        return {"received": len(pk), "lost": 0, "loss_pct": 0.0, "events": 0}
    steps = np.diff(pk)
    if mod:
        steps = steps % int(mod)
    else:
        steps = np.where(steps < 0, 1, steps)
    lost_each = np.where(steps > 1, steps - 1, 0)
    lost = int(lost_each.sum())
    return {"received": int(len(pk)), "lost": lost,
            "loss_pct": round(100.0 * lost / (len(pk) + lost), 4),
            "events": int(np.count_nonzero(lost_each)),
            "largest": int(lost_each.max()) if lost else 0}


def time_gaps(s: Session, min_gap_s: float = 0.5) -> list[dict]:
    """Holes in the timestamp sequence longer than `min_gap_s`."""
    ts = s.emg["timestamp"].to_numpy()
    if len(ts) < 2:
        return []
    d = np.diff(ts)
    out = []
    for i in np.flatnonzero(d > min_gap_s):
        out.append({"after_sample": int(s.emg["sample_index"].iloc[i]), "gap_s": round(float(d[i]), 3),
                    "phase": s.emg["phase"].iloc[i + 1]})
    return out


def filtered(s: Session) -> np.ndarray:
    """Zero-phase band-pass 20-120 Hz + 50/100 Hz notches of every channel (display/inspection)."""
    x = s.emg[s.channel_columns].to_numpy(float)
    return filtfilt(x, s.fs)


def rms_per_trial(s: Session, skip_s: float = 0.5, include_rest: bool = True) -> pd.DataFrame:
    """RMS per channel of each usable trial's last HOLD (first `skip_s` skipped) and of the
    trial REST periods (first second skipped), on the filtered signal. Skips scale with the
    session's speed factor."""
    y = filtered(s)
    df = s.emg
    speed = float(s.meta.get("speed", 1.0) or 1.0)     # sped-up test runs have shorter steps
    skip = int(skip_s * s.fs / speed)
    rest_skip = int(s.fs / speed)
    rows = []
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for tid, g in df[df["trial_id"] > 0].groupby("trial_id", sort=True):
        _, last = _last_attempt(g)
        hold = last[last["phase"] == "HOLD"]
        seg = pos[hold.index].to_numpy()[skip:]
        if len(seg) > 10 and not hold["trial_bad"].max():
            rows.append({"trial_id": int(tid), "gesture": hold["label"].iloc[0],
                         **dict(zip(s.channel_columns, np.sqrt(np.mean(y[seg] ** 2, axis=0))))})
        if include_rest:
            rest = last[last["phase"] == "REST"]
            seg = pos[rest.index].to_numpy()[rest_skip:]
            if len(seg) > 10:
                rows.append({"trial_id": int(tid), "gesture": "REST",
                             **dict(zip(s.channel_columns, np.sqrt(np.mean(y[seg] ** 2, axis=0))))})
    return pd.DataFrame(rows)


def separability(per_trial: pd.DataFrame, channels: list[str]) -> float | None:
    """Leave-one-out nearest-centroid accuracy on log RMS: a quick 'are gestures distinct?'."""
    d = per_trial[per_trial["gesture"].isin(GESTURES)]
    if d["gesture"].nunique() < 2 or d.groupby("gesture").size().min() < 2:
        return None
    X = np.log(d[channels].to_numpy(float) + 1e-6)
    y = d["gesture"].to_numpy()
    hits = 0
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        cents = {g: X[keep & (y == g)].mean(axis=0) for g in np.unique(y[keep])}
        pred = min(cents, key=lambda g: np.sum((X[i] - cents[g]) ** 2))
        hits += pred == y[i]
    return hits / len(X)


def summary_text(s: Session) -> str:
    m = s.meta
    trials = trial_table(s)
    loss = packet_loss(s)
    gaps = time_gaps(s, max(0.5, float(m.get("config", {}).get("monitor", {}).get("gap_s", 1.0))))
    rows = len(s.emg)
    dur = (s.emg["timestamp"].iloc[-1] - s.emg["timestamp"].iloc[0]) if rows > 1 else 0.0
    lines = [
        f"Session {s.folder.name}  ({m.get('status')})",
        f"Subject {m.get('subject')}, session {m.get('session')}, day {m.get('day')}, "
        f"role {m.get('role')}, arm {m.get('arm')}",
        f"Source: {m.get('source', {}).get('source')} at {s.fs:g} Hz, channels "
        f"{', '.join(s.channel_columns)}" + (f", speed x{m.get('speed')}" if m.get("speed", 1) != 1 else ""),
        f"Started {m.get('started')}, ended {m.get('ended')}",
        f"Duration: {int(dur // 60)} min {dur % 60:04.1f} s of data, {rows} samples "
        f"({rows / dur if dur else 0:.1f} samples/s)",
        f"Packet loss: {loss['lost']} samples lost ({loss['loss_pct']:.3f}%) in {loss['events']} events",
        f"Gaps over {max(0.5, float(m.get('config', {}).get('monitor', {}).get('gap_s', 1.0))):g} s "
        f"in the data: {len(gaps)}" + "".join(f"\n  after sample {g['after_sample']}: {g['gap_s']} s ({g['phase']})" for g in gaps),
        f"Reconnections: {m.get('stats', {}).get('reconnects', 0)}",
        "",
    ]
    if m.get("mode") == "free":
        lines.append("Free recording: no protocol, no trials, no quality check (label NONE).")
        return "\n".join(lines) + "\n"
    lines.append("Trials per gesture (complete / bad / usable):")
    for g in GESTURES:
        t = trials[trials["gesture"] == g]
        usable = t[t["complete"] & ~t["bad"]]
        lines.append(f"  {g:6} {int(t['complete'].sum()):3d} / {int(t['bad'].sum()):2d} / {len(usable):3d}")
    bad = trials[trials["bad"]]["trial_id"].tolist()
    lines.append(f"Bad trials: {', '.join(map(str, bad)) if bad else 'none'}")
    repeated = trials[trials["attempts"] > 1]["trial_id"].tolist()
    lines.append(f"Trials repeated after a pause: {', '.join(map(str, repeated)) if repeated else 'none'}")
    if s.order:
        planned = len(s.order["trials"])
        lines.append(f"Trials recorded: {len(trials)} of {planned} planned")
    lines.append("")
    q = s.quality or {}
    if q.get("attempts"):
        last = q["attempts"][-1]
        lines.append(f"Quality check: {last['result']} (attempt {last.get('attempt')}, "
                     f"{len(q['attempts'])} in total), decision: {last.get('decision', 'none')}")
        for r in last.get("reasons", []):
            lines.append(f"  - {r}")
        if last.get("operator_note"):
            lines.append(f"  operator note: {last['operator_note']}")
    else:
        lines.append("Quality check: not completed")
    return "\n".join(lines) + "\n"


def write_summary(folder: str | Path) -> str:
    s = load_session(folder)
    text = summary_text(s)
    (Path(folder) / "summary.txt").write_text(text, encoding="utf-8")
    return text
