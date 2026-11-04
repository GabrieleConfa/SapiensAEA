from pathlib import Path

import numpy as np
import pandas as pd

from emgacq import analysis as an
from tools import split_dataset as sd


def fake_session(name: str, per_gesture: int = 6, bad: tuple[int, ...] = ()) -> an.Session:
    """Session with per_gesture trials of every gesture: cue 3 rows, hold 5 rows, rest 4 rows."""
    gestures = [g for _ in range(per_gesture) for g in ("OPEN", "CLOSE", "INDEX")]
    rows = []
    for tid, g in enumerate(gestures, start=1):
        for phase, n, label in (("CUE", 3, "TRANSITION"), ("HOLD", 5, g), ("REST", 4, "REST")):
            for i in range(n):
                rows.append({"marker": float(tid) if i == 0 else 0.0, "phase": phase, "label": label,
                             "block": 1, "trial_id": tid, "trial_bad": int(tid in bad)})
    emg = pd.DataFrame(rows)
    emg.insert(0, "ch1", np.arange(len(emg), dtype=float))
    emg.insert(0, "sample_index", np.arange(len(emg)))
    return an.Session(Path(name), emg, pd.DataFrame(), {"sampling_rate_hz": 250.0}, None, None)


def test_split_by_trial_stratified_and_reproducible():
    sessions = [fake_session("A", bad=(2,)), fake_session("B")]
    split = sd.make_split(sessions, (0.5, 0.25, 0.25), seed=3)
    keys = {k: {(t["session"], t["trial_id"]) for t in v} for k, v in split.items()}
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"])
    assert len(set().union(*keys.values())) == 35          # 36 trials, trial 2 of A is bad
    assert ("A", 2) not in set().union(*keys.values())
    for name in ("val", "test"):                            # 6 per gesture per session -> 2 / 2 / 2
        for g in ("OPEN", "CLOSE", "INDEX"):
            for s in ("A", "B"):
                assert sum(t["gesture"] == g and t["session"] == s for t in split[name]) in (1, 2)
    assert split == sd.make_split(sessions, (0.5, 0.25, 0.25), seed=3)
    assert split != sd.make_split(sessions, (0.5, 0.25, 0.25), seed=4)


def test_split_frame_keeps_trial_rows_in_order():
    s = fake_session("A")
    df = sd.split_frame({"A": s}, [{"session": "A", "trial_id": 5, "gesture": "CLOSE"}])
    assert list(df["phase"]) == ["CUE"] * 3 + ["HOLD"] * 5 + ["REST"] * 4
    assert (np.diff(df["sample_index"]) == 1).all() and set(df["trial_uid"]) == {"A#05"}
