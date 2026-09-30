import numpy as np
import pytest

from common import quality as ql
from common.config import load_config
from simulator.giga_sim import SyntheticEmg

FS = 1000.0
LABELS = [f"CH{i + 1}" for i in range(8)]


@pytest.fixture
def q():
    return load_config().section("quality")


def synth(grip, level=1.0, n=5000, line_uv=5.0, seed=0):
    e = SyntheticEmg(FS, 8, uv_per_lsb=1.0, seed=seed, line_uv=line_uv)
    e.set_grip(grip, level)
    e.generate(1000)
    return e.generate(n).astype(float)


def test_line_share_detects_mains():
    t = np.arange(5000) / FS
    noise = np.random.default_rng(0).standard_normal((5000, 2))
    x = noise.copy()
    x[:, 1] += 20 * np.sin(2 * np.pi * 50 * t)
    s = ql.line_share(x, FS)
    assert s[0] < 0.05 and s[1] > 0.8


def test_flat_and_saturation():
    x = np.random.default_rng(1).standard_normal((1000, 3)) * 10
    x[:, 1] = 123.0
    x[:500, 2] = 1000.0
    assert ql.flat(x, 0.5).tolist() == [False, True, False]
    assert ql.saturation_fraction(x, 1000.0)[2] == pytest.approx(0.5)


def test_good_recording_passes(q):
    rep = ql.evaluate(synth("REST", seed=1), synth("FIST", seed=2), FS, q, saturation_uv=187500,
                      led={"ok": True}, channel_labels=LABELS)
    assert rep["result"] == "PASS", rep["reasons"]
    assert all(ch["rms_ratio_fist_rest"] > 3 for ch in rep["channels"])


def test_bad_recording_fails_with_reasons(q):
    rest = synth("REST", line_uv=200.0, seed=3)
    fist = synth("FIST", line_uv=200.0, seed=4)
    rest[:, 4] = 7.0
    fist[:, 4] = 7.0                      # channel 5 flat
    rep = ql.evaluate(rest, fist, FS, q, saturation_uv=187500, led={"ok": False, "message": "no edge"},
                      channel_labels=LABELS)
    assert rep["result"] == "FAIL"
    text = " ".join(rep["reasons"])
    assert "CH5: flat" in text and "50 Hz share" in text and "LED: no edge" in text


def test_cyton_rate_band_is_clipped(q):
    lo, hi = ql.usable_band((20.0, 450.0), 250.0)
    assert hi == pytest.approx(112.5)
    x = np.random.default_rng(0).standard_normal((1250, 8))
    rep = ql.evaluate(x, 5 * x, 250.0, q, saturation_uv=187500, led=None, channel_labels=LABELS)
    assert rep["led"] is None and len(rep["channels"]) == 8
