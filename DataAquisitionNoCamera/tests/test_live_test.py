import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from tools import live_test as lt

FS, FRAME_LEN, HOP, THR = 250.0, 32, 16, 5.0
ALL = ["MAV", "RMS", "WL", "ZC", "SSC"]


def notebook_frame_feature_array(frames, threshold):
    # copy of frame_feature_array in models/emg_cnn1d.ipynb: (n, channels, L) -> (n, channels, 5)
    L = frames.shape[-1]
    d = np.diff(frames, axis=-1)
    mav = np.abs(frames).mean(-1)
    rms = np.sqrt((frames ** 2).mean(-1))
    wl = np.abs(d).sum(-1)
    zc = ((frames[..., :-1] * frames[..., 1:] < 0) & (np.abs(d) >= threshold)).sum(-1)
    ssc = ((d[..., :-1] * -d[..., 1:]) >= threshold ** 2).sum(-1)
    return np.stack([np.log1p(mav), np.log1p(rms), np.log1p(wl), zc / L, ssc / L], axis=-1)


def make_sos():
    sos = [signal.butter(4, (20.0, 110.0), btype="bandpass", fs=FS, output="sos"),
           signal.butter(2, (45.0, 55.0), btype="bandstop", fs=FS, output="sos")]
    b, a = signal.iirnotch(100.0, Q=10.0, fs=FS)
    return np.vstack(sos + [signal.tf2sos(b, a)])


def test_streaming_features_match_the_notebook_in_any_chunking():
    rng = np.random.default_rng(1)
    x = 2000 + 30 * rng.standard_normal((1000, 4))          # raw µV with an electrode offset
    sos = make_sos()
    # notebook: whole recording filtered at once, frames every HOP
    zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    y = signal.sosfilt(sos, x, axis=0, zi=zi)[0]
    feats = ["MAV", "WL", "ZC", "SSC"]
    idx = [ALL.index(f) for f in feats]
    ref = notebook_frame_feature_array(sliding_window_view(y, FRAME_LEN, axis=0)[::HOP], THR)[:, :, idx]
    ref = ref.reshape(len(ref), -1)
    for sizes in ([1000], [1] * 1000, list(rng.integers(1, 40, 200))):
        sf = lt.StreamingFeatures(sos, 4, FRAME_LEN, HOP, feats, THR)
        out, pos = [], 0
        for n in sizes:
            if pos >= len(x):
                break
            out += sf.push(x[pos:pos + n])
            pos += n
        ends = [e for e, _ in out]
        got = np.array([v for _, v in out])
        n = min(len(got), len(ref))
        assert n >= len(ref) - 1
        assert ends[:n] == [FRAME_LEN + HOP * i for i in range(n)]
        np.testing.assert_allclose(got[:n], ref[:n], rtol=1e-9, atol=1e-9)


class FakeModel:
    classes = ["OPEN", "CLOSE", "INDEX"]
    target = lt.Model.target


def test_score_uses_only_settled_windows_and_maps_rest_to_open():
    fs, win = 100.0, 20
    labels = np.array(["REST"] * 100 + ["CLOSE"] * 100 + ["REST"] * 100 + ["INDEX"] * 100, dtype=object)
    trial = np.r_[np.zeros(100), np.ones(200), 2 * np.ones(100)].astype(int)
    rows = []
    for end in range(win, len(labels) + 1, 10):
        lab = labels[end - 1]
        pred = {"REST": "OPEN", "CLOSE": "CLOSE", "INDEX": "OPEN"}[lab]     # INDEX always missed
        rows.append({"end": end, "pred": pred, "trial_id": trial[end - 1]})
    res = lt.score(labels.astype(str), pd.DataFrame(rows), FakeModel(), win, fs)
    assert res["windows"] > 0
    assert res["recall"]["OPEN"] == 1.0 and res["recall"]["CLOSE"] == 1.0 and res["recall"]["INDEX"] == 0.0
    assert abs(res["balanced_acc"] - 2 / 3) < 1e-9
    assert res["share"].loc["INDEX", "OPEN"] == 1.0
    t = res["trials"].set_index("posture")
    assert bool(t.loc["CLOSE", "ok"]) and not bool(t.loc["INDEX", "ok"])


def test_trial_order_has_every_gesture_and_no_repeats():
    order = lt.make_trials(["OPEN", "CLOSE", "INDEX"], 5, seed=3)
    assert sorted(order) == sorted(["OPEN", "CLOSE", "INDEX"] * 5)
    assert all(a != b for a, b in zip(order, order[1:]))
