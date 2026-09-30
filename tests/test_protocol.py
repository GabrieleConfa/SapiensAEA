from collections import Counter

import pytest

from common.config import load_config
from protocol import generate_sequences as gs
from protocol.timeline import build_timeline


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def seqs():
    return gs.build()


def test_sequences_structure(seqs):
    assert [s["id"] for s in seqs["sessions"]] == ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"]
    for k, s in enumerate(seqs["sessions"]):
        assert s["seed"] == 1000 + 17 * k
        assert s["role"] == ("train" if s["day"] == 1 else "test")
        assert Counter(s["trials"]) == {g: 8 for g in gs.GRIPS}
        assert all(a != b for a, b in zip(s["trials"], s["trials"][1:]))
        assert sorted(s["ramps"]) == ["FIST", "PINCH", "POINT"]


def test_sequences_deterministic_and_saved(seqs):
    assert gs.build()["sessions"] == seqs["sessions"]
    saved = gs.load_sequences()
    assert saved["sessions"] == seqs["sessions"], "protocol/sequences.json is stale: rerun the generator"


def test_sequences_match_reference_code(seqs):
    # Independent re-implementation of the reference rule for one session.
    import random
    r = random.Random(1000 + 17 * 4)
    while True:
        s = ["OPEN", "FIST", "PINCH", "POINT"] * 8
        r.shuffle(s)
        if all(s[i] != s[i + 1] for i in range(31)):
            break
    assert seqs["sessions"][4]["trials"] == s


def test_timeline_contiguous_and_complete(cfg, seqs):
    for s in seqs["sessions"]:
        tl = build_timeline(cfg, s)
        segs = tl.segments
        for a, b in zip(segs, segs[1:]):
            assert b.start_s == pytest.approx(a.end_s)
        assert [x.index for x in segs] == list(range(len(segs)))
        holds = [x for x in segs if x.kind == "trial_hold"]
        assert [x.grip for x in holds] == s["trials"]
        assert [x.trial for x in holds] == list(range(1, 33))
        assert [x.phase for x in holds] == ["block1"] * 16 + ["block2"] * 16


def test_trial_timing(cfg, seqs):
    tl = build_timeline(cfg, seqs["sessions"][0])
    t1 = [x for x in tl.segments if x.trial == 1]
    assert [(x.kind, x.duration_s) for x in t1] == [("trial_cue", 1.0), ("trial_hold", 4.0), ("trial_rest", 4.0)]
    block1 = [x for x in tl.segments if x.phase == "block1"]
    assert sum(x.duration_s for x in block1) == pytest.approx(16 * 9)


def test_ramps(cfg, seqs):
    s = seqs["sessions"][0]
    tl = build_timeline(cfg, s)
    ramps = [x for x in tl.segments if x.phase == "ramps"]
    intros = [x.text for x in ramps if x.kind == "ramp_intro"]
    assert intros == [f"Ramps: {g}" for g in s["ramps"]]
    one = ramps[1:9]  # first ramp of the first grip, after its intro
    assert [x.grip for x in one] == [s["ramps"][0]] * 7 + ["REST"]
    assert [x.rep for x in one] == [1] * 8
    assert [x.duration_s for x in one] == [2, 2, 2, 2, 2, 2, 2, 3]
    assert [(x.closure_from, x.closure_to) for x in one[:7]] == [
        (0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 0.5), (0.5, 0.0)]
    assert sum(x.duration_s for x in ramps) == pytest.approx(3 * (3 + 4 * 17))


def test_calibration_and_closing(cfg, seqs):
    tl = build_timeline(cfg, seqs["sessions"][0])
    cal = [x for x in tl.segments if x.kind == "cal_contraction"]
    assert [(x.grip, x.rep, x.duration_s) for x in cal] == [
        ("FIST", 1, 3.0), ("FIST", 2, 3.0), ("FIST", 3, 3.0),
        ("EXTENSION", 1, 3.0), ("EXTENSION", 2, 3.0), ("EXTENSION", 3, 3.0)]
    poses = [(x.phase, x.grip, x.closure_to, x.duration_s) for x in tl.segments if x.kind == "ref_pose"]
    assert poses == [("calibration", "OPEN", 0.0, 5.0), ("calibration", "FIST", 1.0, 5.0),
                     ("closing", "OPEN", 0.0, 5.0), ("closing", "FIST", 1.0, 5.0)]
    assert tl.segments[0].kind == "sync" and tl.segments[-1].kind == "sync"
    assert tl.segments[-1].phase == "sync_end"


def test_flash_schedule(cfg, seqs):
    tl = build_timeline(cfg, seqs["sessions"][0])
    f = tl.flash_times_s
    assert f[0] == pytest.approx(tl.segments[0].start_s + 1.0)
    assert f[-1] == pytest.approx(tl.segments[-1].start_s + 1.0)
    gaps = [b - a for a, b in zip(f, f[1:])]
    assert all(g == pytest.approx(60.0) for g in gaps[:-1])
    assert 30.0 <= gaps[-1] <= 90.0
    assert f[-1] < tl.duration_s


def test_segment_at(cfg, seqs):
    tl = build_timeline(cfg, seqs["sessions"][0])
    for seg in tl.segments:
        assert tl.segment_at(seg.start_s) is seg
        assert tl.segment_at(seg.start_s + seg.duration_s / 2) is seg
    assert tl.segment_at(tl.duration_s + 1) is None
