import json

import pytest

from common.config import load_config
from simulator.giga_sim import GigaSimulator
from tools.quality_check import main as qc_main


@pytest.fixture
def sim_port():
    sim = GigaSimulator(load_config(), port=0)
    port = sim.start()
    yield port
    sim.stop()


@pytest.mark.slow
def test_guided_check_passes_with_simulator(sim_port, tmp_path):
    rc = qc_main(["--source", "sim", "--port", f"socket://127.0.0.1:{sim_port}",
                  "--camera", "synthetic", "--headless", "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "quality_check.json").read_text())
    assert rc == 0, rep["reasons"]
    assert rep["result"] == "PASS"
    assert rep["led"]["stream_edge"] and rep["led"]["camera_checked"]
    assert rep["led"]["camera_step"] > 100
    assert len(rep["channels"]) == 8


@pytest.mark.slow
def test_led_not_seen_by_camera_fails(sim_port, tmp_path, monkeypatch):
    # synthetic camera whose ROI is somewhere else than the drawn LED
    import tools.quality_check as qcmod
    from common.camera import LedRoi

    real = qcmod.CameraMonitor

    def wrong_roi(capture, roi):
        return real(capture, LedRoi(400, 250, 40, 40))

    monkeypatch.setattr(qcmod, "CameraMonitor", wrong_roi)
    rc = qc_main(["--source", "sim", "--port", f"socket://127.0.0.1:{sim_port}",
                  "--camera", "synthetic", "--headless", "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "quality_check.json").read_text())
    assert rc == 1
    assert rep["result"] == "FAIL"
    assert any("camera brightness step" in r for r in rep["reasons"])
