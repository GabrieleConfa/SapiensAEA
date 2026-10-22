"""Free recording on the BrainFlow synthetic board: no protocol, raw stream for a set duration."""

import pytest

from emgacq import analysis as an
from emgacq import markers as mk
from emgacq.headless import run_headless
from emgacq.labels import Labeler
from emgacq.sources import make_source

pytestmark = pytest.mark.slow

META = {"subject": "T", "session": 1, "day": 1, "role": "train", "arm": "right",
        "band_position_pct": None, "forearm_length_cm": None,
        "channel_map": [{"channel": c, "column": f"ch{c}", "position": ""} for c in range(1, 9)],
        "electrode_type": "", "notes": ""}


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from emgacq.config import load_config

    cfg = load_config()
    data = tmp_path_factory.mktemp("data")
    ctrl = run_headless(cfg, make_source(cfg, "synthetic"), META, free=True, duration_s=3.0,
                        data_dir=data, log=lambda *_: None)
    return ctrl, an.load_session(ctrl.folder)


def test_files(session):
    ctrl, s = session
    assert ctrl.folder.name.startswith("T_S1_FREE_")
    for f in ("emg.csv", "events.csv", "session.json", "summary.txt"):
        assert (ctrl.folder / f).exists(), f
    for f in ("trial_order.json", "quality_check.json"):
        assert not (ctrl.folder / f).exists(), f
    assert s.meta["status"] == "completed"
    assert s.meta["mode"] == "free"
    assert s.meta["protocol"] is None and s.meta["seed"] is None
    assert "Free recording" in (ctrl.folder / "summary.txt").read_text(encoding="utf-8")


def test_emg_csv(session):
    _, s = session
    df = s.emg
    assert df["sample_index"].tolist() == list(range(len(df)))
    assert df["marker"].iloc[0] == mk.FREE_RECORDING
    assert df["marker"].iloc[-1] == mk.SESSION_END
    assert set(df["marker"].unique()) == {0, mk.FREE_RECORDING, mk.SESSION_END}
    assert set(df["phase"]) == {"FREE"} and set(df["label"]) == {"NONE"}
    assert (df["trial_id"] == 0).all() and (df["trial_bad"] == 0).all()
    assert len(df) / s.fs == pytest.approx(3.0, abs=0.6)
    lab = Labeler(1).label_chunk(df["marker"].to_numpy(float), 0)
    assert (lab.phase == df["phase"].to_numpy()).all()


def test_operator_stop(cfg, tmp_path):
    """Without a duration the recording runs until stop(); it then closes as completed."""
    import time

    from emgacq.recorder import Recorder
    from emgacq.session import SessionController, State

    source = make_source(cfg, "synthetic")
    recorder = Recorder(source, 0.02)
    ctrl = SessionController(cfg, source, recorder, META, data_dir=tmp_path, free=True)
    try:
        source.connect()
        source.start()
        recorder.start()
        time.sleep(0.3)
        ctrl.start()
        t_stop = time.monotonic() + 1.5
        while ctrl.state != State.DONE:
            now = time.monotonic()
            ctrl.tick(now)
            assert not ctrl.can_pause()
            if now > t_stop and ctrl.state == State.RUNNING:
                assert ctrl.stop("test", now)
            assert now < t_stop + 5
            time.sleep(0.01)
    finally:
        ctrl.shutdown("test exit")
    s = an.load_session(ctrl.folder)
    assert s.meta["status"] == "completed"
    assert s.emg["marker"].iloc[-1] == mk.SESSION_END
    assert "STOP" in set(s.events["event"])


def test_headless_free_needs_duration(cfg):
    with pytest.raises(ValueError):
        run_headless(cfg, make_source(cfg, "synthetic"), META, free=True)
