"""Pool the trials of several sessions, shuffle them and split them into train / val / test.

    python -m tools.split_dataset data/TEST2_S2_2026-11-04_1854 data/TEST3_S2_2026-11-04_1920
    python -m tools.split_dataset <folders...> --fractions 0.7 0.15 0.15 --seed 0 --out data/splits/x

The unit that is shuffled is the trial (cue + hold + rest of one trial), never the single sample:
neighbouring samples are almost identical, so a split by sample would put the same movement in
train and test and the test score would be far too optimistic. The split is stratified by session
and gesture, so every split gets the same gesture mix from every session. Inside a trial the rows
stay in time order (needed for windowing); the order of the trials is shuffled.

Used: trials that are complete and not flagged bad; of a trial repeated after a pause only its
last attempt. Left out: quality check, calibration, pauses and the closing rest.

Writes to the output folder (default <data>/splits/<subjects>_seed<seed>/):
    train.csv, val.csv, test.csv   the emg.csv rows of the trials of each split, with two extra
                                   columns first: session (folder name) and trial_uid
    split.json                     sessions, fractions, seed and the trial list of each split
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from emgacq import analysis as an
from emgacq.markers import GESTURES

SPLITS = ("train", "val", "test")
TRIAL_PHASES = ("CUE", "HOLD", "REST")


def usable_trials(s: an.Session) -> pd.DataFrame:
    trials = an.trial_table(s)
    return trials[trials["complete"] & ~trials["bad"]]


def trial_rows(s: an.Session, trial_id: int) -> pd.DataFrame:
    """Rows of the last attempt of one trial (cue, hold, rest), in time order."""
    _, last = an._last_attempt(s.emg[s.emg["trial_id"] == trial_id])
    return last[last["phase"].isin(TRIAL_PHASES)]


def split_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """(train, val, test) trial counts for a stratum of n trials; train takes the remainder."""
    n_val, n_test = round(n * fractions[1]), round(n * fractions[2])
    if n_val + n_test > n:
        n_test = n - n_val
    return n - n_val - n_test, n_val, n_test


def make_split(sessions: list[an.Session], fractions: tuple[float, float, float],
               seed: int) -> dict[str, list[dict]]:
    """Trials of each split as dicts {session, trial_id, gesture}, in shuffled order."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict]] = {k: [] for k in SPLITS}
    for s in sessions:
        trials = usable_trials(s)
        for gesture in GESTURES:
            ids = trials.loc[trials["gesture"] == gesture, "trial_id"].to_numpy()
            ids = ids[rng.permutation(len(ids))]
            n_train, n_val, _ = split_counts(len(ids), fractions)
            parts = (ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:])
            for name, part in zip(SPLITS, parts):
                out[name] += [{"session": s.folder.name, "trial_id": int(t), "gesture": gesture}
                              for t in part]
    for name in SPLITS:
        out[name] = [out[name][i] for i in rng.permutation(len(out[name]))]
    return out


def check_compatible(sessions: list[an.Session]) -> None:
    names = [s.folder.name for s in sessions]
    if len(set(names)) != len(names):
        sys.exit(f"Two sessions have the same folder name: {names}")
    ref = sessions[0]
    for s in sessions[1:]:
        if s.channel_columns != ref.channel_columns:
            sys.exit(f"Channels differ: {ref.folder.name} {ref.channel_columns}, "
                     f"{s.folder.name} {s.channel_columns}")
        if s.fs != ref.fs:
            sys.exit(f"Sampling rates differ: {ref.folder.name} {ref.fs} Hz, {s.folder.name} {s.fs} Hz")
    for s in sessions:
        if s.meta.get("mode") == "free":
            sys.exit(f"{s.folder.name} is a free recording: it has no trials to split")


def split_frame(sessions: dict[str, an.Session], trials: list[dict]) -> pd.DataFrame:
    parts = []
    for t in trials:
        rows = trial_rows(sessions[t["session"]], t["trial_id"]).copy()
        rows.insert(0, "trial_uid", f"{t['session']}#{t['trial_id']:02d}")
        rows.insert(0, "session", t["session"])
        parts.append(rows)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def default_out(sessions: list[an.Session], seed: int) -> Path:
    tags = [f"{s.meta.get('subject', s.folder.name)}_S{s.meta.get('session', '')}" for s in sessions]
    return sessions[0].folder.parent / "splits" / f"{'+'.join(tags)}_seed{seed}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", help="session folders (protocol recordings)")
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"), help="share of the trials (default 0.7 0.15 0.15)")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the shuffle (default 0)")
    ap.add_argument("--out", help="output folder (default <data>/splits/<subjects>_seed<seed>)")
    ap.add_argument("--force", action="store_true", help="replace the output folder if it exists")
    args = ap.parse_args(argv)

    fractions = tuple(args.fractions)
    if min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-6:
        sys.exit(f"--fractions must be >= 0 and add up to 1 (got {fractions})")
    for f in args.folders:
        if not (Path(f) / "emg.csv").exists():
            sys.exit(f"No emg.csv in {f}")

    sessions = [an.load_session(f) for f in args.folders]
    check_compatible(sessions)
    by_name = {s.folder.name: s for s in sessions}

    out = Path(args.out) if args.out else default_out(sessions, args.seed)
    if out.exists():
        if not args.force:
            sys.exit(f"{out} already exists (use --force to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    split = make_split(sessions, fractions, args.seed)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "unit": "trial (cue + hold + rest of the last attempt)",
        "stratified_by": ["session", "gesture"],
        "fractions": dict(zip(SPLITS, fractions)),
        "seed": args.seed,
        "sampling_rate_hz": sessions[0].fs,
        "channels": sessions[0].channel_columns,
        "sessions": [],
        "splits": {},
    }
    for s in sessions:
        all_trials, used = an.trial_table(s), usable_trials(s)
        manifest["sessions"].append({
            "folder": str(s.folder), "name": s.folder.name, "subject": s.meta.get("subject"),
            "session": s.meta.get("session"), "trials": len(all_trials), "used": len(used),
            "left_out": sorted(set(all_trials["trial_id"]) - set(used["trial_id"]))})

    print(f"{'split':<6} " + " ".join(f"{g:>6}" for g in GESTURES) + f" {'trials':>7} {'rows':>8}")
    for name in SPLITS:
        df = split_frame(by_name, split[name])
        df.to_csv(out / f"{name}.csv", index=False)
        counts = {g: sum(t["gesture"] == g for t in split[name]) for g in GESTURES}
        manifest["splits"][name] = {"trials": len(split[name]), "rows": len(df),
                                    "per_gesture": counts, "trial_list": split[name]}
        print(f"{name:<6} " + " ".join(f"{counts[g]:>6}" for g in GESTURES)
              + f" {len(split[name]):>7} {len(df):>8}")
        if len(split[name]) and min(counts.values()) == 0:
            print(f"  warning: {name} has no trial of "
                  + ", ".join(g for g in GESTURES if counts[g] == 0))

    (out / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for s in manifest["sessions"]:
        if s["left_out"]:
            print(f"{s['name']}: left out trials {s['left_out']} (bad or incomplete)")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
