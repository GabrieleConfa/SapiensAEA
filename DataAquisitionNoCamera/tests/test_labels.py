import numpy as np
import pytest

from emgacq import markers as mk
from emgacq.labels import Labeler


def run(markers_at: dict[int, int], n: int, chunk: int = 7, trials_per_block: int = 18):
    """Label n samples with markers at the given sample indices, fed in chunks."""
    lab = Labeler(trials_per_block)
    col = np.zeros(n)
    for i, code in markers_at.items():
        col[i] = code
    parts = [lab.label_chunk(col[a:a + chunk], a) for a in range(0, n, chunk)]
    cat = lambda f: np.concatenate([getattr(p, f) for p in parts])
    return lab, cat("phase"), cat("label"), cat("block"), cat("trial_id")


def test_marker_codes_roundtrip():
    for kind, phase in ((mk.TRIAL_CUE, "CUE"), (mk.TRIAL_HOLD, "HOLD"), (mk.TRIAL_REST, "REST")):
        for g in mk.GESTURES:
            code = mk.trial_marker(kind, 23, g)
            info = mk.decode(code)
            assert (info.phase, info.trial_id, info.gesture) == (phase, 23, g)
    assert mk.decode(mk.trial_marker(mk.TRIAL_HOLD, 7, "CLOSE")).code == 20702
    assert mk.decode(mk.cal_hold_marker("OPEN")).label == "OPEN"
    assert mk.decode(mk.QC_FIST).label == "CLOSE"
    with pytest.raises(ValueError):
        mk.decode(99)


def test_trial_labels():
    t = 20   # trial 20 = block 2 when 18 trials per block
    m = {0: mk.QC_REST, 10: mk.trial_marker(mk.TRIAL_CUE, t, "INDEX"),
         15: mk.trial_marker(mk.TRIAL_HOLD, t, "INDEX"), 25: mk.trial_marker(mk.TRIAL_REST, t, "INDEX"),
         35: mk.BLOCK_PAUSE, 40: mk.SESSION_END}
    lab, phase, label, block, trial = run(m, 45)
    assert list(phase[:10]) == ["QUALITY"] * 10 and set(label[:10]) == {"REST"}
    assert set(phase[10:15]) == {"CUE"} and set(label[10:15]) == {"TRANSITION"}
    assert set(phase[15:25]) == {"HOLD"} and set(label[15:25]) == {"INDEX"}
    assert set(phase[25:35]) == {"REST"} and set(label[25:35]) == {"REST"}
    assert set(block[10:35]) == {2} and set(trial[10:35]) == {t}
    assert set(trial[35:]) == {0} and set(block[35:]) == {0}
    assert set(phase[35:]) == {"PAUSE"}     # the end marker keeps the labels of the step before
    assert lab.ended and lab.end_sample == 40 and lab.markers_seen == 6


def test_calibration_and_quality_labels():
    m = {0: mk.QC_REST, 3: mk.QC_CUE, 5: mk.QC_FIST, 8: mk.QC_REVIEW, 10: mk.CAL_REST,
         12: mk.cal_cue_marker("CLOSE"), 14: mk.cal_hold_marker("CLOSE"), 18: mk.CLOSING_REST}
    _, phase, label, _, trial = run(m, 20, chunk=3)
    assert list(label[3:8]) == ["TRANSITION"] * 2 + ["CLOSE"] * 3
    assert list(phase[8:10]) == ["PAUSE"] * 2
    assert list(zip(phase[12:18], label[12:18])) == [("CALIBRATION", "TRANSITION")] * 2 + [("CALIBRATION", "CLOSE")] * 4
    assert list(phase[18:]) == ["REST"] * 2
    assert not trial.any()


def test_interrupted_trial_is_recorded():
    cue = mk.trial_marker(mk.TRIAL_CUE, 5, "OPEN")
    hold = mk.trial_marker(mk.TRIAL_HOLD, 5, "OPEN")
    m = {0: cue, 4: hold, 8: mk.OPERATOR_PAUSE, 12: cue, 16: hold,
         20: mk.trial_marker(mk.TRIAL_REST, 5, "OPEN"), 22: mk.OPERATOR_PAUSE}
    lab, phase, label, _, trial = run(m, 26)
    assert [(x.trial_id, x.start_sample, x.end_sample) for x in lab.interruptions] == [(5, 0, 8)]
    assert set(label[8:12]) == {"TRANSITION"} and set(trial[8:12]) == {0}
    # a pause during the rest restarts only the rest: no repeated cue, no interruption
    assert len(lab.interruptions) == 1


def test_markers_arriving_late_after_a_gap():
    """Radio gap during a hold: the queued REST and pause markers land on consecutive samples
    after the gap, then the trial is repeated from its cue; the first attempt is the interruption."""
    cue = mk.trial_marker(mk.TRIAL_CUE, 8, "CLOSE")
    hold = mk.trial_marker(mk.TRIAL_HOLD, 8, "CLOSE")
    rest = mk.trial_marker(mk.TRIAL_REST, 8, "CLOSE")
    m = {0: cue, 5: hold, 9: rest, 10: mk.GAP_PAUSE, 15: cue, 18: hold, 22: rest}
    lab, phase, label, _, trial = run(m, 26)
    assert [(x.trial_id, x.start_sample, x.end_sample) for x in lab.interruptions] == [(8, 0, 10)]
    assert list(phase[9:11]) == ["REST", "PAUSE"]
    assert set(trial[15:]) == {8}


def test_unknown_marker_kept_out():
    lab, phase, *_ = run({0: mk.QC_REST, 3: 77}, 6)
    assert lab.unknown_markers == [(3, 77.0)]
    assert set(phase) == {"QUALITY"}
