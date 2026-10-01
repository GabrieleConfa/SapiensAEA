import numpy as np

from emgacq.packets import PacketTracker


def test_no_loss_with_wrap():
    t = PacketTracker(256)
    seq = np.arange(1000) % 256
    for a in range(0, 1000, 37):
        assert t.update(seq[a:a + 37], a) == 0
    assert t.lost == 0 and t.received == 1000


def test_loss_inside_and_between_chunks():
    t = PacketTracker(256)
    assert t.update(np.array([250, 251, 253, 254]), 0) == 1       # 252 missing
    assert t.update(np.array([255, 3, 4]), 4) == 3                # 0, 1, 2 missing across the wrap
    assert t.update(np.array([7]), 7) == 2                        # 5, 6 missing between chunks
    assert t.lost == 6
    assert t.loss_events == [(2, 1), (5, 3), (7, 2)]
    assert abs(t.loss_pct - 100 * 6 / (8 + 6)) < 1e-9


def test_duplicate_and_reset():
    t = PacketTracker(256)
    t.update(np.array([1, 2, 2, 3]))
    assert t.duplicates == 1 and t.lost == 0
    t.reset()                                # e.g. after a reconnection the counter restarts
    assert t.update(np.array([0, 1])) == 0


def test_non_wrapping_counter():
    t = PacketTracker(None)
    assert t.update(np.array([10, 11, 15])) == 3
    assert t.update(np.array([2, 3])) == 0      # counter went back: resync, not loss
    assert t.resyncs == 1
