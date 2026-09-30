"""Full recording with the simulator and the synthetic camera, headless, 100x speed."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from acquisition.__main__ import main as acquisition_main
from common.camera import SYNTHETIC_ROI
from common.config import load_config
from common.recording import load_emg
from protocol.generate_sequences import get_session
from protocol.timeline import build_timeline
from simulator.giga_sim import GigaSimulator


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    cfg = load_config()
    data = tmp_path_factory.mktemp("data")
    sim = GigaSimulator(cfg, port=0)
    port = sim.start()
    try:
        rc = acquisition_main(["--subject", "B", "--session", "2", "--source", "sim",
                               "--port", f"socket://127.0.0.1:{port}", "--camera", "synthetic",
                               "--headless", "--speed", "100", "--forearm-cm", "26",
                               "--band-pct", "30", "--data-dir", str(data)])
    finally:
        sim.stop()
    folders = list(data.iterdir())
    assert len(folders) == 1
    return rc, folders[0], sim, cfg


@pytest.mark.slow
def test_files_and_metadata(recorded):
    rc, out, sim, cfg = recorded
    assert rc == 0
    assert out.name.startswith("B_S2_")
    for f in ["emg_raw.bin", "emg_host_times.csv", "video.mp4", "frame_times.csv", "cues.csv",
              "sync_events.csv", "session.json"]:
        assert (out / f).exists(), f
    meta = json.loads((out / "session.json").read_text())
    assert meta["status"] == "completed"
    assert meta["role"] == "train" and meta["day"] == 1
    assert meta["trial_order"] == get_session("B2")["trials"]
    assert meta["forearm_length_cm"] == 26 and meta["band_position_pct_forearm"] == 30
    assert meta["config_sha256"] == cfg.sha256


@pytest.mark.slow
def test_cues_follow_timeline(recorded):
    _, out, _, cfg = recorded
    tl = build_timeline(cfg, get_session("B2"))
    cues = pd.read_csv(out / "cues.csv")
    assert cues["segment_index"].tolist() == list(range(len(tl.segments)))
    holds = cues[cues["kind"] == "trial_hold"]
    assert holds["grip"].tolist() == get_session("B2")["trials"]
    assert (cues["host_time_ns"].diff().dropna() >= 0).all()


@pytest.mark.slow
def test_emg_complete_and_led_events_logged(recorded):
    _, out, sim, cfg = recorded
    rec = load_emg(out)
    assert rec.stats.gaps == 0 and rec.stats.crc_errors == 0
    assert np.all(np.diff(rec.counters) == 1)
    sync = pd.read_csv(out / "sync_events.csv")
    n_flash = len(build_timeline(cfg, get_session("B2")).flash_times_s)
    assert (sync["event"] == "flash_request").sum() == n_flash
    rises = sync[sync["event"] == "led_rise"]["emg_counter"].astype(int).tolist()
    assert rises == rec.led_rise_counters.tolist() == sim.flash_counters


@pytest.mark.slow
def test_led_visible_in_video_matches_emg(recorded):
    _, out, _, cfg = recorded
    frames = pd.read_csv(out / "frame_times.csv")
    cap = cv2.VideoCapture(str(out / "video.mp4"))
    bright = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        bright.append(SYNTHETIC_ROI.brightness(f))
    cap.release()
    assert len(bright) == len(frames)
    on = np.asarray(bright) > 150
    video_rises = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    rec = load_emg(out)
    assert len(video_rises) == len(rec.led_rise_counters)
    # host time of each EMG LED edge vs capture time of the first bright frame: within ~2 frames
    emg_rise_host = rec.host_time_ns[(rec.flags & 2) != 0]
    frame_rise_host = frames["host_time_ns"].to_numpy()[video_rises]
    assert np.all(np.abs(frame_rise_host - emg_rise_host) < 0.1e9)
