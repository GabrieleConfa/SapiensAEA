"""Check a recorded session right after recording.

    python -m tools.inspect_session data/<folder>
    python -m tools.inspect_session data/<folder> --window 120 180 --show

Prints trials per gesture, bad and repeated trials, packet loss and gaps, the RMS of every
channel per gesture and a quick separability score (leave-one-out nearest centroid on log RMS).
Saves to <folder>/inspect/:
    signals_raw.png       raw signals (DC removed) with the labels shaded
    signals_filtered.png  20-120 Hz band-pass + 50/100 Hz notch, labels shaded
    rms_per_gesture.png   RMS per channel and gesture (mean and SD over trials)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from emgacq import analysis as an
from emgacq.markers import GESTURES

DEFAULT_COLORS = {"OPEN": "#2F80ED", "CLOSE": "#F2994A", "INDEX": "#BB6BD9", "REST": "#6FA8A0",
                  "TRANSITION": "#BDBDBD", "PAUSE": "#8A9BA8"}


def report(s: an.Session) -> str:
    trials = an.trial_table(s)
    loss = an.packet_loss(s)
    gap_s = float(s.meta.get("config", {}).get("monitor", {}).get("gap_s", 1.0))
    gaps = an.time_gaps(s, 0.5)
    lines = [an.summary_text(s).rstrip(), "", "Packet loss (from package_num): "
             f"{loss['lost']} lost ({loss['loss_pct']:.3f}%), {loss['events']} events, "
             f"largest {loss.get('largest', 0)}",
             f"Timestamp gaps > 0.5 s: {len(gaps)}" + "".join(
                 f"\n  after sample {g['after_sample']}: {g['gap_s']} s ({g['phase']})" for g in gaps)]
    if not s.events.empty:
        ev = s.events[s.events["event"].isin(["GAP_START", "GAP_END", "RECONNECT_OK",
                                              "RECONNECT_FAILED", "PAUSE"])]
        if len(ev):
            lines.append(f"Logged interruptions (gap threshold {gap_s:g} s):")
            for _, r in ev.iterrows():
                lines.append(f"  {r['wall_time'][11:19]} {r['event']:16} {r['detail']}")
    per_trial = an.rms_per_trial(s)
    if len(per_trial):
        ch = s.channel_columns
        mean = per_trial.groupby("gesture")[ch].mean()
        order = [g for g in [*GESTURES, "REST"] if g in mean.index]
        lines += ["", "RMS per gesture (uV, filtered, last hold of each usable trial, first 0.5 s skipped):",
                  f"  {'':6}" + "".join(f"{c:>8}" for c in ch)]
        for g in order:
            lines.append(f"  {g:6}" + "".join(f"{v:8.1f}" for v in mean.loc[g, ch]))
        acc = an.separability(per_trial, ch)
        if acc is not None:
            n = int(per_trial["gesture"].isin(GESTURES).sum())
            lines.append(f"\nSeparability: leave-one-out nearest-centroid accuracy {100 * acc:.0f}% on "
                         f"{n} trials (chance {100 / per_trial[per_trial['gesture'].isin(GESTURES)]['gesture'].nunique():.0f}%)")
    lines.append("\nTrials:\n" + trials.to_string(index=False))
    return "\n".join(lines)


def _label_runs(s: an.Session, a: int, b: int):
    phase = s.emg["phase"].to_numpy()[a:b]
    label = s.emg["label"].to_numpy()[a:b]
    bad = s.emg["trial_bad"].to_numpy()[a:b]
    key = np.array([f"{p}|{l}|{x}" for p, l, x in zip(phase, label, bad)])
    change = np.flatnonzero(key[1:] != key[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(key)]])
    for i, j in zip(starts, ends):
        yield a + i, a + j, phase[i], label[i], bool(bad[i])


def plot_signals(s: an.Session, y: np.ndarray, title: str, path: Path, colors: dict,
                 window: tuple[float, float] | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fs = s.fs
    a, b = 0, len(y)
    if window:
        a, b = max(0, int(window[0] * fs)), min(len(y), int(window[1] * fs))
    t = np.arange(a, b) / fs
    ch = s.channel_columns
    fig, axes = plt.subplots(len(ch), 1, sharex=True, figsize=(16, 1.4 * len(ch) + 1.2))
    axes = np.atleast_1d(axes)
    runs = list(_label_runs(s, a, b))
    lim = np.percentile(np.abs(y[a:b]), 99.5, axis=0) * 1.2 + 1e-9
    for c, ax in enumerate(axes):
        for i, j, ph, lb, bad in runs:
            if lb in GESTURES:
                ax.axvspan(i / fs, j / fs, color=colors.get(lb, "#999"), alpha=0.25 if not bad else 0.08, lw=0)
                if bad:
                    ax.axvspan(i / fs, j / fs, fill=False, hatch="///", edgecolor="#D32F2F", lw=0)
            elif lb == "TRANSITION" or ph == "PAUSE":
                ax.axvspan(i / fs, j / fs, color=colors.get("TRANSITION" if ph != "PAUSE" else "PAUSE", "#ccc"),
                           alpha=0.15, lw=0)
        ax.plot(t, y[a:b, c], lw=0.4, color="#222")
        ax.set_ylim(-lim[c], lim[c])
        ax.set_ylabel(f"{ch[c]}\nµV", rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s, sample_index / fs)")
    handles = [Patch(color=colors[g], alpha=0.4, label=g) for g in GESTURES]
    handles += [Patch(color=colors["TRANSITION"], alpha=0.3, label="cue / transition"),
                Patch(color=colors["PAUSE"], alpha=0.3, label="pause"),
                Patch(fill=False, hatch="///", edgecolor="#D32F2F", label="bad trial")]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8)
    fig.suptitle(f"{s.folder.name}: {title}", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)


def plot_rms(s: an.Session, per_trial, path: Path, colors: dict) -> None:
    import matplotlib.pyplot as plt

    ch = s.channel_columns
    groups = [g for g in [*GESTURES, "REST"] if g in set(per_trial["gesture"])]
    mean = per_trial.groupby("gesture")[ch].mean()
    sd = per_trial.groupby("gesture")[ch].std().fillna(0)
    x = np.arange(len(ch))
    w = 0.8 / max(1, len(groups))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(ch) + 3), 5))
    for k, g in enumerate(groups):
        n = int((per_trial["gesture"] == g).sum())
        ax.bar(x + (k - (len(groups) - 1) / 2) * w, mean.loc[g, ch], w, yerr=sd.loc[g, ch],
               color=colors.get(g, "#999"), label=f"{g} (n={n})", capsize=2)
    ax.set_xticks(x, ch)
    ax.set_ylabel("RMS (µV, 20-120 Hz)")
    ax.set_title(f"{s.folder.name}: RMS per channel and gesture (mean ± SD over trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_S", "END_S"),
                    help="time window of the signal plots (default: whole session)")
    ap.add_argument("--show", action="store_true", help="open the plots in windows")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    if not (args.folder / "emg.csv").exists():
        print(f"no emg.csv in {args.folder}")
        return 1
    s = an.load_session(args.folder)
    print(report(s))
    if args.no_plots:
        return 0
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {**DEFAULT_COLORS, **s.meta.get("config", {}).get("display", {}).get("colors", {})}
    out = args.folder / "inspect"
    out.mkdir(exist_ok=True)
    raw = s.emg[s.channel_columns].to_numpy(float)
    raw = raw - np.median(raw, axis=0)
    window = tuple(args.window) if args.window else None
    plot_signals(s, raw, "raw (median removed)", out / "signals_raw.png", colors, window)
    plot_signals(s, an.filtered(s), "filtered 20-120 Hz + 50/100 Hz notch", out / "signals_filtered.png",
                 colors, window)
    per_trial = an.rms_per_trial(s)
    if len(per_trial):
        plot_rms(s, per_trial, out / "rms_per_gesture.png", colors)
    print(f"\nplots in {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
