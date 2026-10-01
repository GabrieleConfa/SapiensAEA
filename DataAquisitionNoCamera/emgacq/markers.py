"""Marker codes stamped into the EMG stream at every phase change.

The code alone tells what started at that sample (phase, label, gesture, trial), so the labels of
emg.csv can be rebuilt from the marker column. Table: docs/DATA_FORMAT.md, or

    python -m emgacq.markers
"""

from __future__ import annotations

from dataclasses import dataclass

GESTURES = ("OPEN", "CLOSE", "INDEX")
GESTURE_CODE = {"OPEN": 1, "CLOSE": 2, "INDEX": 3}
CODE_GESTURE = {v: k for k, v in GESTURE_CODE.items()}

PHASES = ("QUALITY", "CALIBRATION", "CUE", "HOLD", "REST", "PAUSE")
LABELS = ("OPEN", "CLOSE", "INDEX", "REST", "TRANSITION")
TRANSITION = "TRANSITION"
REST = "REST"

SESSION_END = 2
SESSION_ABORT = 3
QC_REST = 10
QC_CUE = 11
QC_FIST = 12
QC_REVIEW = 13
CAL_REST = 20
CAL_CUE_BASE = 20          # + gesture code: 21 OPEN, 22 CLOSE, 23 INDEX
CAL_HOLD_BASE = 30         # + gesture code: 31 OPEN, 32 CLOSE, 33 INDEX
BLOCK_PAUSE = 40
OPERATOR_PAUSE = 41
GAP_PAUSE = 42
CLOSING_REST = 50

# Trial markers: 10000 * kind + 100 * trial_id + gesture code, e.g. 20702 = HOLD of trial 7, CLOSE.
TRIAL_CUE, TRIAL_HOLD, TRIAL_REST = 1, 2, 3
MAX_TRIALS = 99

QC_GESTURE = "CLOSE"       # the quality-check fist is a power grip


@dataclass(frozen=True)
class MarkerInfo:
    code: int
    name: str
    phase: str             # phase that starts at this sample; "" for the end markers
    label: str
    gesture: str = ""      # gesture asked for in this step (calibration, trial, QC fist)
    trial_id: int = 0


def trial_marker(kind: int, trial_id: int, gesture: str) -> int:
    if not 1 <= trial_id <= MAX_TRIALS:
        raise ValueError(f"trial_id {trial_id} out of range 1..{MAX_TRIALS}")
    return 10000 * kind + 100 * trial_id + GESTURE_CODE[gesture]


def cal_cue_marker(gesture: str) -> int:
    return CAL_CUE_BASE + GESTURE_CODE[gesture]


def cal_hold_marker(gesture: str) -> int:
    return CAL_HOLD_BASE + GESTURE_CODE[gesture]


_FIXED = {
    SESSION_END: MarkerInfo(SESSION_END, "SESSION_END", "", ""),
    SESSION_ABORT: MarkerInfo(SESSION_ABORT, "SESSION_ABORT", "", ""),
    QC_REST: MarkerInfo(QC_REST, "QC_REST", "QUALITY", REST),
    QC_CUE: MarkerInfo(QC_CUE, "QC_CUE", "QUALITY", TRANSITION, QC_GESTURE),
    QC_FIST: MarkerInfo(QC_FIST, "QC_FIST", "QUALITY", QC_GESTURE, QC_GESTURE),
    QC_REVIEW: MarkerInfo(QC_REVIEW, "QC_REVIEW", "PAUSE", REST),
    CAL_REST: MarkerInfo(CAL_REST, "CAL_REST", "CALIBRATION", REST),
    BLOCK_PAUSE: MarkerInfo(BLOCK_PAUSE, "BLOCK_PAUSE", "PAUSE", REST),
    OPERATOR_PAUSE: MarkerInfo(OPERATOR_PAUSE, "OPERATOR_PAUSE", "PAUSE", TRANSITION),
    GAP_PAUSE: MarkerInfo(GAP_PAUSE, "GAP_PAUSE", "PAUSE", TRANSITION),
    CLOSING_REST: MarkerInfo(CLOSING_REST, "CLOSING_REST", "REST", REST),
}
for _g, _c in GESTURE_CODE.items():
    _FIXED[CAL_CUE_BASE + _c] = MarkerInfo(CAL_CUE_BASE + _c, f"CAL_CUE_{_g}", "CALIBRATION",
                                           TRANSITION, _g)
    _FIXED[CAL_HOLD_BASE + _c] = MarkerInfo(CAL_HOLD_BASE + _c, f"CAL_HOLD_{_g}", "CALIBRATION",
                                            _g, _g)


def decode(code: int | float) -> MarkerInfo:
    code = int(round(code))
    if code in _FIXED:
        return _FIXED[code]
    kind, rest = divmod(code, 10000)
    trial_id, g = divmod(rest, 100)
    if kind in (TRIAL_CUE, TRIAL_HOLD, TRIAL_REST) and 1 <= trial_id <= MAX_TRIALS and g in CODE_GESTURE:
        gesture = CODE_GESTURE[g]
        if kind == TRIAL_CUE:
            return MarkerInfo(code, "TRIAL_CUE", "CUE", TRANSITION, gesture, trial_id)
        if kind == TRIAL_HOLD:
            return MarkerInfo(code, "TRIAL_HOLD", "HOLD", gesture, gesture, trial_id)
        return MarkerInfo(code, "TRIAL_REST", "REST", REST, gesture, trial_id)
    raise ValueError(f"unknown marker code {code}")


def code_table() -> list[tuple[str, str, str, str, str]]:
    """(code, name, phase, label, meaning) rows for the documentation."""
    rows = [(str(c), m.name, m.phase or "-", m.label or "-", "") for c, m in sorted(_FIXED.items())]
    rows += [
        ("1TTgg", "TRIAL_CUE", "CUE", "TRANSITION", "10000 + 100*trial_id + gesture"),
        ("2TTgg", "TRIAL_HOLD", "HOLD", "<gesture>", "20000 + 100*trial_id + gesture"),
        ("3TTgg", "TRIAL_REST", "REST", "REST", "30000 + 100*trial_id + gesture"),
    ]
    return rows


if __name__ == "__main__":
    print("gesture codes: " + ", ".join(f"{g}={c}" for g, c in GESTURE_CODE.items()))
    print(f"{'code':>6}  {'name':16} {'phase':12} {'label':11} meaning")
    for row in code_table():
        print(f"{row[0]:>6}  {row[1]:16} {row[2]:12} {row[3]:11} {row[4]}")
