"""The GIGA source behind the same interface, against the main project's simulator."""

import sys
import time

import numpy as np
import pytest

from emgacq import markers as mk
from emgacq.config import APP_ROOT
from emgacq.sources.giga_source import GigaSource

pytestmark = pytest.mark.slow

REPO = APP_ROOT.parent


@pytest.fixture
def simulator():
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    sim_mod = pytest.importorskip("simulator.giga_sim")
    from common.config import load_config as load_main_config

    sim = sim_mod.GigaSimulator(load_main_config(REPO / "config" / "config.toml"), port=0)
    port = sim.start()
    yield port
    sim.stop()


def test_giga_simulator_stream(simulator):
    src = GigaSource(f"socket://127.0.0.1:{simulator}", [1, 2, 5], 1000.0, 8, str(REPO), simulator=True)
    src.connect()
    src.start()
    try:
        chunks = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            time.sleep(0.05)
            if time.monotonic() - t0 > 0.5 and not any(c.marker.any() for c in chunks):
                src.insert_marker(mk.trial_marker(mk.TRIAL_HOLD, 1, "CLOSE"))
            chunks.append(src.read())
        n = sum(len(c) for c in chunks)
        assert 800 < n < 1200
        pk = np.concatenate([c.package_num for c in chunks])
        assert (np.diff(pk) == 1).all()
        markers = np.concatenate([c.marker for c in chunks])
        assert list(markers[markers != 0]) == [20102]
        assert chunks[-1].data_uv.shape[1] == 3
        assert src.package_modulus is None
    finally:
        src.release()
