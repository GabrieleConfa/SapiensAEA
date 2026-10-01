from collections import Counter

import pytest

from emgacq import markers as mk
from emgacq.protocol import build_timeline, session_seed, trial_order, trial_order_from_config

GESTURES = ["OPEN", "CLOSE", "INDEX"]


@pytest.mark.parametrize("subject,session", [("A", 1), ("B", 2), ("S07", 3), ("X-1", 12)])
def test_counts_and_no_repeats(subject, session):
    o = trial_order(subject, session, GESTURES, 6, 2)
    assert len(o["blocks"]) == 2
    for block in o["blocks"]:
        assert len(block) == 18
        assert Counter(block) == {g: 6 for g in GESTURES}
        assert all(a != b for a, b in zip(block, block[1:]))
    # also no repeat across the block boundary, and the blocks differ
    assert o["blocks"][0][-1] != o["blocks"][1][0]
    assert o["blocks"][0] != o["blocks"][1]
    assert [t["trial_id"] for t in o["trials"]] == list(range(1, 37))
    assert [t["gesture"] for t in o["trials"]] == o["blocks"][0] + o["blocks"][1]


def test_reproducible():
    assert trial_order("A", 1, GESTURES, 6, 2) == trial_order("A", 1, GESTURES, 6, 2)
    # fixed value: the seed must not depend on the Python hash seed or the platform
    assert session_seed("A", 1) == int.from_bytes(
        __import__("hashlib").sha256(b"A|1").digest()[:8], "big")


def test_different_subjects_and_sessions_differ():
    orders = {tuple(trial_order(s, n, GESTURES, 6, 2)["blocks"][0]) for s in "ABC" for n in (1, 2, 3)}
    assert len(orders) == 9


def test_many_seeds_valid():
    for n in range(200):
        o = trial_order("T", n, GESTURES, 6, 2)
        for block in o["blocks"]:
            assert all(a != b for a, b in zip(block, block[1:]))


def test_timeline_from_config(cfg):
    order = trial_order_from_config(cfg, "A", 1)
    steps = build_timeline(cfg, order)
    kinds = Counter(s.kind for s in steps)
    assert kinds["trial_cue"] == kinds["trial_hold"] == kinds["trial_rest"] == 36
    assert kinds["qc_fist"] == cfg["quality.reps"]
    assert kinds["cal_hold"] == 6 and kinds["block_pause"] == 1 and kinds["qc_review"] == 1
    holds = [s for s in steps if s.kind == "trial_hold"]
    assert [s.gesture for s in holds] == [t["gesture"] for t in order["trials"]]
    for s in steps:   # every marker decodes to the step it starts
        info = mk.decode(s.marker)
        assert info.trial_id == s.trial_id
        if s.kind == "trial_hold":
            assert info.label == s.gesture and info.phase == "HOLD"
        if s.kind == "trial_hold" or s.kind == "cal_hold":
            assert steps[s.restart_index].kind in ("trial_cue", "cal_cue")
    assert steps[-1].kind == "closing_rest"
