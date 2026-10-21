"""Full sessions on the BrainFlow synthetic board, sped up, with scripted operator events."""

import json

import pytest

from emgacq import analysis as an
from emgacq.headless import Script, run_headless
from emgacq.labels import Labeler
from emgacq.sources import make_source

pytestmark = pytest.mark.slow

META = {"subject": "T", "session": 1, "day": 1, "role": "train", "arm": "right",
        "band_position_pct": 30.0, "forearm_length_cm": 26.0,
        "channel_map": [{"channel": c, "column": f"ch{c}", "position": ""} for c in range(1, 9)],
        "electrode_type": "test", "notes": ""}


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from emgacq.config import load_config

    cfg = load_config()
    data = tmp_path_factory.mktemp("data")
    script = Script(flag_bad={3}, operator_pause_trial=5, stall_trial=8, stall_s=2.5,
                    reconnect_on_gap=True)
    ctrl = run_headless(cfg, make_source(cfg, "synthetic"), META, speed=25, script=script,
                        data_dir=data, log=lambda *_: None)
    return ctrl, an.load_session(ctrl.folder)


def test_files(session):
    ctrl, s = session
    for f in ("emg.csv", "events.csv", "session.json", "quality_check.json", "trial_order.json",
              "summary.txt"):
        assert (ctrl.folder / f).exists(), f
    assert s.meta["status"] == "completed"
    assert s.meta["source"]["source"] == "synthetic"
    assert ctrl.folder.name.startswith("T_S1_")
    order = json.loads((ctrl.folder / "trial_order.json").read_text())
    assert len(order["trials"]) == 36


def test_emg_csv(session):
    _, s = session
    df = s.emg
    assert df["sample_index"].tolist() == list(range(len(df)))
    assert df["marker"].iloc[0] == 10                       # starts at the first protocol step
    assert df["marker"].iloc[-1] == 2                       # ends with SESSION_END
    assert set(df["phase"]) == {"QUALITY", "CALIBRATION", "CUE", "HOLD", "REST", "PAUSE"}
    assert set(df["label"]) == {"OPEN", "CLOSE", "INDEX", "REST", "TRANSITION"}
    assert (df.loc[df["phase"] == "CUE", "label"] == "TRANSITION").all()
    assert df["timestamp"].is_monotonic_increasing


def test_labels_match_markers(session):
    """Relabelling from the marker column reproduces the label columns."""
    _, s = session
    df = s.emg
    lab = Labeler(18).label_chunk(df["marker"].to_numpy(float), 0)
    assert (lab.phase == df["phase"].to_numpy()).all()
    assert (lab.label == df["label"].to_numpy()).all()
    assert (lab.trial_id == df["trial_id"].to_numpy()).all()
    assert (lab.block == df["block"].to_numpy()).all()


def test_trials(session):
    ctrl, s = session
    t = an.trial_table(s).set_index("trial_id")
    assert len(t) == 36 and t["complete"].all()
    assert t["gesture"].tolist() == [x["gesture"] for x in ctrl.order["trials"]]
    assert (t.groupby("gesture").size() == 12).all()
    assert t.loc[3, "bad"] and t["bad"].sum() == 1
    assert t.loc[5, "attempts"] == 2 and t.loc[8, "attempts"] == 2     # pause and gap repeat the trial
    assert not t.loc[5, "bad"] and not t.loc[8, "bad"]
    df = s.emg
    rows3 = df[df["trial_id"] == 3]
    assert (rows3["trial_bad"] == 1).all()
    # the interrupted attempt of trial 5 is bad row by row, the repeated one is not
    rows5 = df[df["trial_id"] == 5]
    assert rows5["trial_bad"].iloc[0] == 1 and rows5["trial_bad"].iloc[-1] == 0


def test_gap_and_packet_loss(session):
    ctrl, s = session
    ev = set(s.events["event"])
    assert {"GAP_START", "GAP_END", "RECONNECT_OK", "PAUSE", "RESUME", "TRIAL_BAD", "QC_RESULT",
            "SESSION_END", "SESSION_CLOSED"} <= ev
    loss = an.packet_loss(s)
    assert loss["lost"] > 0                     # simulated single losses
    assert ctrl.reconnects == 1
    assert any(g["gap_s"] > 1.0 for g in an.time_gaps(s, 0.5))
    assert (s.emg["phase"] == "PAUSE").any()
    assert s.meta["stats"]["packets_lost"] >= 1


def test_quality_and_summary(session):
    ctrl, s = session
    q = s.quality
    assert q["attempts"] and q["attempts"][-1]["decision"] == "proceed"
    assert len(q["attempts"][-1]["channels"]) == len(s.channel_columns)
    text = (ctrl.folder / "summary.txt").read_text()
    assert "36 of 36" in text and "Bad trials: 3" in text


def test_gestures_distinguishable(session):
    _, s = session
    per = an.rms_per_trial(s)
    acc = an.separability(per, s.channel_columns)
    assert acc is not None and acc > 0.8
