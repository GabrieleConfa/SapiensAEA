"""Reconnections: markers lost with the old board session, invalid first samples, counter restarts."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emgacq import analysis as an
from emgacq.recorder import Recorder
from emgacq.session import SessionController, State
from emgacq.sources import make_source
from emgacq.sources.brainflow_source import BrainFlowSource

META = {"subject": "T", "session": 1, "day": 1, "role": "train", "arm": "right",
        "band_position_pct": None, "forearm_length_cm": None,
        "channel_map": [{"channel": c, "column": f"ch{c}", "position": ""} for c in range(1, 9)],
        "electrode_type": "", "notes": ""}


@pytest.mark.slow
def test_quality_check_completes_after_a_reconnection_during_it(cfg, tmp_path):
    """The gap-pause marker queued during the stall is lost with the old board session; the
    repeated quality check must still be evaluated (it used to wait for that marker forever)."""
    source = make_source(cfg, "synthetic")
    recorder = Recorder(source, float(cfg["board.read_interval_s"]))
    ctrl = SessionController(cfg, source, recorder, META, speed=10, data_dir=tmp_path)
    source.connect()
    source.start()
    recorder.start()
    time.sleep(0.5)
    try:
        ctrl.start()
        stalled = reconnected = False
        deadline = time.monotonic() + 60
        while ctrl.qc_report is None and time.monotonic() < deadline:
            now = time.monotonic()
            ctrl.tick(now)
            if not stalled and ctrl.step.kind == "qc_fist":
                source.simulate_stall(2.0)
                stalled = True
            if ctrl.state == State.PAUSED and not reconnected:
                ctrl.reconnect()
                reconnected = True
            if ctrl.can_resume():
                ctrl.resume(now)
            time.sleep(0.005)
        assert reconnected
        assert ctrl.qc_report is not None, "quality check never evaluated after the reconnection"
        assert any("queued marker(s) lost" in r["detail"] for r in ctrl.events.rows)
    finally:
        ctrl.shutdown("test")


class FakeBoard:
    def __init__(self, data):
        self.data = data

    def get_board_data(self):
        data, self.data = self.data, np.zeros((self.data.shape[0], 0))
        return data


def test_cyton_all_zero_sample_is_dropped():
    src = BrainFlowSource("cyton", "COM3", [1, 2], 250.0, "x{ch}060100X", "{ch}", 187500.0)
    # rows: 0 timestamp, 1 package, 2-3 channels, 4 marker
    data = np.array([[0.0, 0.004, 0.008, 0.012],
                     [0, 1, 2, 3],
                     [0.0, 19000.0, 0.0, 19010.0],
                     [0.0, -3000.0, 0.0, -3005.0],
                     [0, 0, 10, 0]])
    src._board, src._streaming = FakeBoard(data), True
    src._rows_uv, src._row_ts, src._row_pkg, src._row_marker = [2, 3], 0, 1, 4
    chunk = src.read()
    # the first all-zero row goes; the third is all zero too but carries a marker, so it stays
    assert chunk.package_num.tolist() == [1, 2, 3]
    assert chunk.marker.tolist() == [0, 10, 0]
    assert src.zero_rows_dropped == 1 and src.info["zero_rows_dropped"] == 1


def test_counter_restart_after_a_reconnection_is_not_loss():
    n = 600
    ts = np.arange(n) / 250.0
    ts[300:] += 11.0                               # reconnection: 11 s without data
    pk = np.r_[np.arange(300) % 256, np.arange(n - 300) % 256]   # counter restarts at 300
    emg = pd.DataFrame({"sample_index": np.arange(n), "timestamp": ts, "package_num": pk,
                        "ch1": 1.0, "marker": 0, "phase": "FREE", "label": "NONE", "block": 0,
                        "trial_id": 0, "trial_bad": 0})
    emg = emg.drop(index=[100, 101]).reset_index(drop=True)      # 2 samples really lost
    s = an.Session(Path("."), emg, pd.DataFrame(),
                   {"sampling_rate_hz": 250.0, "source": {"package_modulus": 256}}, None, None)
    assert an.packet_loss(s)["lost"] == 2
