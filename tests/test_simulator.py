"""End-to-end: simulator -> TCP -> GigaSerialSource -> parser -> buffer."""

import time

import numpy as np
import pytest

from common.config import load_config
from common.packet import FLAG_LED_ON, FLAG_LED_RISE, parse_bytes
from common.sources.base import RawSink
from common.sources.giga import GigaSerialSource
from simulator.giga_sim import GigaSimulator, SyntheticEmg


class MemorySink(RawSink):
    def __init__(self):
        self.chunks = []

    def write_bytes(self, data, host_time_ns):
        self.chunks.append((data, host_time_ns))


@pytest.fixture
def cfg():
    return load_config()


def run_sim(cfg, seconds, flashes_at=(), **sim_kwargs):
    sim = GigaSimulator(cfg, port=0, **sim_kwargs)
    port = sim.start()
    src = GigaSerialSource(f"socket://127.0.0.1:{port}", 8, 1000.0)
    try:
        info = src.open()
        sink = MemorySink()
        src.start(sink)
        t0 = time.perf_counter()
        pending = sorted(flashes_at)
        while time.perf_counter() - t0 < seconds:
            if pending and time.perf_counter() - t0 >= pending[0]:
                src.flash()
                pending.pop(0)
            time.sleep(0.01)
        src.stop()
    finally:
        sim.stop()
    return sim, src, info, sink


@pytest.mark.slow
def test_stream_rate_counters_and_led(cfg):
    sim, src, info, sink = run_sim(cfg, 2.0, flashes_at=(0.5, 1.2))
    assert info["n_ch"] == 8 and info["fs"] == 1000.0
    st = src.stats()
    assert 1800 <= st["packets"] <= 2200
    assert st["crc_errors"] == 0 and st["gaps"] == 0
    # LED edges are reported at the exact counters where the simulator switched the LED on
    assert [e.counter for e in src.buffer.led_edges] == sim.flash_counters
    assert len(sim.flash_counters) == 2
    # the raw bytes saved by the sink parse back to the same packets
    raw = b"".join(d for d, _ in sink.chunks)
    block, stats = parse_bytes(raw, 8)
    assert stats.packets == st["packets"]
    flash_len = round(cfg["sync.flash_ms"])
    for c in sim.flash_counters:
        i = int(np.searchsorted(block.counters, c))
        assert block.flags[i] & FLAG_LED_RISE
        assert np.all(block.flags[i:i + flash_len] & FLAG_LED_ON)
        assert not block.flags[i + flash_len] & FLAG_LED_ON
    assert "#OK STOP" in [t for _, t in src.messages]


@pytest.mark.slow
def test_parser_survives_drops_and_corruption(cfg):
    sim, src, info, sink = run_sim(cfg, 1.5, drop_rate=0.01, corrupt_rate=0.01, seed=3)
    st = src.stats()
    assert st["packets"] > 1200
    assert st["gaps"] > 0 and st["crc_errors"] > 0
    assert np.all(np.diff(src.buffer.latest(st["packets"])[1]) > 0)


def test_synthetic_emg_follows_grip(cfg):
    emg = SyntheticEmg(1000.0, 8, uv_per_lsb=0.0223, seed=1, line_uv=0.0)
    rest = emg.generate(2000) * 0.0223
    emg.set_grip("FIST", 1.0)
    emg.generate(500)  # settle
    fist = emg.generate(2000) * 0.0223
    rest_rms = (rest - rest.mean(0)).std(0)
    fist_rms = (fist - fist.mean(0)).std(0)
    assert np.all(fist_rms > 3 * rest_rms)
    assert np.argmax(fist_rms) == 1  # FIST pattern peaks on channel 2
