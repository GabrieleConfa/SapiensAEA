import numpy as np

from emgacq.quality import evaluate
from emgacq.signals import ChannelMonitor

FS = 250.0
NAMES = [f"ch{i}" for i in range(1, 5)]


def noise(n, uv, rng):
    from scipy import signal
    sos = signal.butter(4, [20, 110], btype="bandpass", fs=FS, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal((n, len(uv))), axis=0)
    return x / x.std(axis=0) * np.asarray(uv)


def captures(rng, rest_uv=(2, 2, 2, 2), fist_uv=(40, 40, 40, 40), line_uv=0.0):
    n_rest, n_fist = int(15 * FS), int(5 * FS)
    t = np.arange(n_rest) / FS
    rest = [noise(n_rest, rest_uv, rng) + line_uv * np.sin(2 * np.pi * 50 * t)[:, None] + 100.0
            for _ in range(2)]
    fist = [noise(n_fist, fist_uv, rng) + 100.0 for _ in range(2)]
    return rest, fist


def judged(cfg, **over):
    """The [quality] section with PASS/FAIL on, whatever the local config says."""
    return {**cfg.section("quality"), "judge": True, **over}


def test_pass(cfg):
    rng = np.random.default_rng(1)
    rest, fist = captures(rng)
    r = evaluate(rest, fist, FS, judged(cfg), 187500.0, NAMES, 0.1)
    assert r["result"] == "PASS", r["reasons"]
    assert all(ch["rms_ratio_fist_rest"] > 10 for ch in r["channels"])


def test_fail_flat_railed_mains_and_loss(cfg):
    rng = np.random.default_rng(2)
    rest, fist = captures(rng, line_uv=20.0)
    for x in rest + fist:
        x[:, 1] = 5.0                 # ch2 flat
        x[::50, 2] = 187500.0         # ch3 railed 2% of the time
    r = evaluate(rest, fist, FS, judged(cfg), 187500.0, NAMES, 5.0)
    assert r["result"] == "FAIL"
    by = {c["channel"]: c for c in r["channels"]}
    assert by["ch2"]["flat"] and not by["ch2"]["ok"]
    assert by["ch3"]["railed_fraction"] > 0.01
    assert by["ch1"]["line_share_48_52_rest"] > cfg["quality.line_share_max"]
    assert any("packet loss" in x for x in r["reasons"])


def test_mains_share_judged_or_not(cfg):
    rng = np.random.default_rng(5)
    rest, fist = captures(rng, line_uv=20.0)
    q = judged(cfg, line_share_judged_on="rest")
    r = evaluate(rest, fist, FS, q, 187500.0, NAMES, 0.0)
    assert r["result"] == "FAIL" and all("50 Hz share" in x for x in r["reasons"])
    q["line_share_judged_on"] = "none"
    r = evaluate(rest, fist, FS, q, 187500.0, NAMES, 0.0)
    assert r["result"] == "PASS", r["reasons"]
    assert all(ch["line_share_48_52_rest"] > q["line_share_max"] for ch in r["channels"])


def test_fail_no_contraction(cfg):
    rng = np.random.default_rng(3)
    rest, fist = captures(rng, fist_uv=(3, 3, 3, 3))
    r = evaluate(rest, fist, FS, judged(cfg), 187500.0, NAMES, 0.0)
    assert r["result"] == "FAIL"
    assert any("fist/rest" in x for x in r["reasons"])


def test_not_judged_reports_but_never_fails(cfg):
    rng = np.random.default_rng(6)
    rest, fist = captures(rng, fist_uv=(3, 3, 3, 3), line_uv=20.0)
    for x in rest + fist:
        x[:, 1] = 5.0                 # ch2 flat
    r = evaluate(rest, fist, FS, judged(cfg, judge=False), 187500.0, NAMES, 5.0)
    assert r["result"] == "NOT_JUDGED"
    assert any("flat" in x for x in r["reasons"]) and any("packet loss" in x for x in r["reasons"])
    assert not {c["channel"]: c for c in r["channels"]}["ch2"]["ok"]


def test_live_monitor_flags(cfg):
    m = ChannelMonitor(FS, 3, cfg.section("monitor"), cfg.section("quality"), 187500.0)
    rng = np.random.default_rng(4)
    x = noise(int(2 * FS), (20, 20, 20), rng)
    x[:, 1] = 3.0
    x[:, 2] = 187000.0
    m.update(x)
    flat, railed = m.flags()
    assert list(flat) == [False, True, True] and list(railed) == [False, False, True]
    assert 10 < m.rms()[0] < 40
