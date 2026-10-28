import json

import numpy as np
import pandas as pd

from tools import stream_quality as sq

FS = 250.0


def _session(folder, n_s=80, seed=0):
    """Labelled session with known defects: ch1 clean, ch2 mains, ch3 flat, ch4 pops; a lost run
    of 3 samples, an 11 s gap after a reconnection that restarts the counter on an all-zero row."""
    rng = np.random.default_rng(seed)
    n = int(n_s * FS)
    t = np.arange(n) / FS
    label = np.where((t % 10) >= 6, "CLOSE", "REST")
    phase = np.where(label == "CLOSE", "HOLD", "REST")
    noise = rng.normal(0, 2.0, (n, 4))
    burst = rng.normal(0, 40.0, (n, 4)) * (label == "CLOSE")[:, None]
    x = noise + burst + np.array([20000.0, -3000.0, 5000.0, 1000.0])
    x[:, 1] += 80 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
    x[:, 2] = 5000.0 + rng.normal(0, 0.05, n)
    for k in (5, 25, 45, 65):
        x[int(k * FS):, 3] += 3000.0
    ts = 1.79e9 + t
    pk = np.arange(n) % 256
    cut = n // 2
    ts[cut:] += 11.0                       # reconnection: 11 s without data
    pk[cut:] = np.arange(n - cut) % 256    # counter restarts
    x[cut] = 0.0                           # first sample after the restart is invalid
    keep = np.ones(n, bool)
    keep[1000:1003] = False                # 3 samples lost on the radio
    df = pd.DataFrame({"sample_index": 0, "timestamp": ts, "package_num": pk,
                       **{f"ch{c + 1}": x[:, c] for c in range(4)}, "marker": 0,
                       "phase": phase, "label": label, "block": 1, "trial_id": 0, "trial_bad": 0})[keep]
    df["sample_index"] = np.arange(len(df))
    folder.mkdir()
    df.to_csv(folder / "emg.csv", index=False)
    meta = {"status": "completed", "mode": "protocol", "sampling_rate_hz": FS,
            "source": {"full_scale_uv": 187500.0, "package_modulus": 256}}
    (folder / "session.json").write_text(json.dumps(meta))
    pd.DataFrame({"wall_time": ["2026-10-28T10:00:00.000"], "event": ["RECONNECT_OK"],
                  "detail": [""]}).to_csv(folder / "events.csv", index=False)
    return folder


def test_defects_are_found(tmp_path):
    rep = sq.analyze(_session(tmp_path / "S"))
    link = rep.link
    assert link["lost"] == 3                   # the restart of the counter is not counted as loss
    assert link["zero_rows"] == 1 and len(link["gaps"]) == 1 and link["reconnects"] == 1
    assert link["segments"] == 2
    ch = rep.channels.set_index("channel")
    assert ch.loc["ch1", "status"] == "GOOD", rep.problems["ch1"]
    assert abs(ch.loc["ch2", "mains_rest_uV"] - 80) < 8
    assert ch.loc["ch2", "status"] == "BAD"
    assert "flat" in {code for _, code, _ in rep.problems["ch3"]}
    assert ch.loc["ch4", "pops"] == 4
    assert ch.loc["ch1", "snr_best"] > 10
    assert rep.verdict == "BAD"
    assert "What to do" in sq.report_text(rep)


def test_free_recording_uses_quietest_windows(tmp_path):
    folder = _session(tmp_path / "F")
    df = pd.read_csv(folder / "emg.csv")
    df["phase"], df["label"] = "FREE", "NONE"
    df.to_csv(folder / "emg.csv", index=False)
    meta = json.loads((folder / "session.json").read_text())
    (folder / "session.json").write_text(json.dumps({**meta, "mode": "free"}))
    rep = sq.analyze(folder)
    assert rep.rest_source.startswith("quietest")
    assert "ACTIVE" in rep.gesture_rms.index
    assert rep.channels.set_index("channel").loc["ch1", "snr_best"] > 5
