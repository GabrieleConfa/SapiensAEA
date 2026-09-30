import numpy as np
import pytest

from common import packet as pk

N = 8


def make_stream(n=50, start=0, flags=None, seed=0):
    rng = np.random.default_rng(seed)
    counters = np.arange(start, start + n)
    fl = np.zeros(n, np.uint8) if flags is None else np.asarray(flags, np.uint8)
    samples = rng.integers(-(1 << 23), (1 << 23), size=(n, N))
    return counters, fl, samples, pk.encode_packets(counters, fl, samples)


def test_packet_length_and_crc_vector():
    assert pk.packet_length(8) == 33
    # CRC-16/CCITT-FALSE check value
    assert pk.crc16(b"123456789") == 0x29B1


def test_scalar_and_vector_encoders_agree():
    c, f, s, data = make_stream(5)
    scalar = b"".join(pk.encode_packet(int(ci), int(fi), si) for ci, fi, si in zip(c, f, s))
    assert scalar == data


def test_roundtrip_including_extremes():
    samples = np.array([[-(1 << 23), (1 << 23) - 1, 0, -1, 1, 123456, -654321, 42]])
    data = pk.encode_packets([7], [pk.FLAG_LED_ON | pk.FLAG_LED_RISE], samples)
    block, stats = pk.parse_bytes(data, N)
    assert block.counters.tolist() == [7]
    assert block.flags.tolist() == [3]
    assert block.samples.tolist() == samples.tolist()
    assert block.end_offsets.tolist() == [33]
    assert stats.crc_errors == 0 and stats.bytes_skipped == 0


@pytest.mark.parametrize("chunk", [1, 2, 7, 33, 34, 100])
def test_arbitrary_chunking(chunk):
    c, f, s, data = make_stream(40)
    p = pk.PacketParser(N)
    out = [p.feed(data[i:i + chunk]) for i in range(0, len(data), chunk)]
    counters = np.concatenate([b.counters for b in out])
    samples = np.concatenate([b.samples for b in out])
    assert counters.tolist() == c.tolist()
    assert np.array_equal(samples, s)
    assert p.stats.packets == 40 and p.stats.gaps == 0


def test_resync_after_garbage_and_corruption():
    c, f, s, data = make_stream(20)
    corrupted = bytearray(data)
    corrupted[5 * 33 + 10] ^= 0xFF          # damage packet 5
    stream = b"\x00\xA5\x13garbage#\x01\x02" + bytes(corrupted)
    block, stats = pk.parse_bytes(stream, N)
    assert block.counters.tolist() == [i for i in range(20) if i != 5]
    assert stats.crc_errors >= 1
    assert stats.gaps == 1 and stats.lost_samples == 1


def test_text_lines_between_packets():
    c, f, s, data = make_stream(4)
    stream = b"#INFO {\"fs\":1000}\n" + data[:66] + b"#OK FLASH\r\n" + data[66:]
    p = pk.PacketParser(N)
    b1 = p.feed(stream[:10])
    assert b1.texts == [] and len(b1) == 0   # line not complete yet
    b2 = p.feed(stream[10:])
    assert b2.texts == ['#INFO {"fs":1000}', "#OK FLASH"]
    assert b2.counters.tolist() == [0, 1, 2, 3]


def test_hash_inside_binary_is_not_a_text_line():
    samples = np.full((3, N), 0x0A2323)       # bytes '#', '#', '\n' inside samples
    data = pk.encode_packets([0, 1, 2], [0, 0, 0], samples)
    # start mid-packet so the parser sees the binary '#' bytes first
    block, stats = pk.parse_bytes(data[10:], N)
    assert block.counters.tolist() == [1, 2]
    assert block.texts == []


def test_counter_gap_and_restart():
    _, _, _, a = make_stream(10, start=0)
    _, _, _, b = make_stream(10, start=15)
    _, _, _, restart = make_stream(3, start=0)
    block, stats = pk.parse_bytes(a + b + restart, N)
    assert stats.gaps == 2
    assert stats.lost_samples == 5            # the restart is not counted as lost
