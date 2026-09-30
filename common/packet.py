"""Binary packet format of the GIGA stream (see docs/SERIAL_PROTOCOL.md).

One packet per sample instant, little endian:

    offset  size  field
    0       2     sync bytes 0xA5 0x5A
    2       4     sample counter, uint32 (0 at START)
    6       1     flags (FLAG_*)
    7       3*N   N samples, signed 24-bit
    7+3N    2     CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) of bytes 2 .. 7+3N-1

Text lines starting with '#' and ending with '\\n' may appear between packets (command replies).
They never contain 0xA5, so they cannot be confused with a sync word.
"""

from __future__ import annotations

import binascii
import re
from dataclasses import dataclass, field

import numpy as np

SYNC = b"\xA5\x5A"
HEADER_LEN = 7
CRC_LEN = 2

FLAG_LED_ON = 0x01     # sync LED is on during this sample
FLAG_LED_RISE = 0x02   # the LED was switched on at this sample (first sample with LED on)
FLAG_OVERRUN = 0x04    # the firmware lost at least one sample before this one
FLAG_BUTTON = 0x08     # calibration button pressed

MAX_TEXT_LINE = 512
INT24_SIGN = 0x800000
# Firmware text lines: '#' + uppercase keyword + printable ASCII, e.g. "#INFO {...}", "#OK START".
# A line can never span a packet because 0xA5 is not printable.
_TEXT_RUN = re.compile(rb"#[A-Z]{2}[\x20-\x7e\t\r]*")


def packet_length(n_channels: int) -> int:
    return HEADER_LEN + 3 * n_channels + CRC_LEN


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE, same as the firmware implementation."""
    return binascii.crc_hqx(data, 0xFFFF)


def encode_packet(counter: int, flags: int, samples) -> bytes:
    body = bytearray()
    body += (counter & 0xFFFFFFFF).to_bytes(4, "little")
    body.append(flags & 0xFF)
    for s in samples:
        s = int(s)
        if not -INT24_SIGN <= s < INT24_SIGN:
            raise ValueError(f"sample {s} does not fit in 24 bits")
        body += (s & 0xFFFFFF).to_bytes(3, "little")
    return SYNC + bytes(body) + crc16(bytes(body)).to_bytes(2, "little")


def encode_packets(counters, flags, samples) -> bytes:
    """Vectorised encoder: counters (n,), flags (n,), samples (n, N) int."""
    samples = np.asarray(samples, dtype=np.int64)
    n, n_ch = samples.shape
    plen = packet_length(n_ch)
    out = np.zeros((n, plen), dtype=np.uint8)
    out[:, 0], out[:, 1] = 0xA5, 0x5A
    out[:, 2:6] = np.asarray(counters, dtype="<u4").view(np.uint8).reshape(n, 4)
    out[:, 6] = np.asarray(flags, dtype=np.uint8)
    u = (samples & 0xFFFFFF).astype("<u4").view(np.uint8).reshape(n, n_ch, 4)[:, :, :3]
    out[:, 7:7 + 3 * n_ch] = u.reshape(n, 3 * n_ch)
    for i in range(n):
        out[i, plen - 2:] = np.frombuffer(crc16(out[i, 2:plen - 2].tobytes()).to_bytes(2, "little"), np.uint8)
    return out.tobytes()


@dataclass
class ParsedBlock:
    counters: np.ndarray      # (n,) uint32
    flags: np.ndarray         # (n,) uint8
    samples: np.ndarray       # (n, N) int32, raw ADC codes
    end_offsets: np.ndarray   # (n,) int64, stream offset just after each packet
    texts: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.counters)


@dataclass
class ParserStats:
    packets: int = 0
    crc_errors: int = 0
    bytes_skipped: int = 0
    gaps: int = 0             # number of counter discontinuities
    lost_samples: int = 0     # samples missing according to the counter
    overrun_flags: int = 0


def decode_packets(raw: np.ndarray, n_channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode an (n, packet_length) uint8 array of already validated packets."""
    n = raw.shape[0]
    counters = raw[:, 2:6].copy().view("<u4").reshape(n)
    flags = raw[:, 6].copy()
    b = raw[:, 7:7 + 3 * n_channels].reshape(n, n_channels, 3).astype(np.int32)
    v = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16)
    samples = (v ^ INT24_SIGN) - INT24_SIGN
    return counters, flags, samples


class PacketParser:
    """Incremental parser: feed arbitrary byte chunks, get whole packets back."""

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.plen = packet_length(n_channels)
        self.stats = ParserStats()
        self._buf = bytearray()
        self._buf_start = 0          # stream offset of _buf[0]
        self._last_counter: int | None = None

    def feed(self, data: bytes) -> ParsedBlock:
        self._buf += data
        buf, plen = self._buf, self.plen
        good: list[bytes] = []
        ends: list[int] = []
        texts: list[str] = []
        i = 0
        n = len(buf)
        while i < n:
            if buf[i] == 0x23:  # '#': text line, printable ASCII up to '\n'
                m = _TEXT_RUN.match(buf, i, min(n, i + MAX_TEXT_LINE))
                if m is None:
                    if n - i < 3:
                        break  # not enough bytes to decide yet
                elif m.end() < n and buf[m.end()] == 0x0A:
                    texts.append(buf[i:m.end()].decode("ascii").strip())
                    i = m.end() + 1
                    continue
                elif m.end() == n and n - i < MAX_TEXT_LINE:
                    break  # wait for the rest of the line
                i += 1  # a '#' inside binary garbage
                self.stats.bytes_skipped += 1
                continue
            if buf[i] != 0xA5 or (i + 1 < n and buf[i + 1] != 0x5A):
                # skip to the next candidate (sync or text)
                k = buf.find(SYNC, i + 1)
                h = buf.find(b"#", i + 1)
                cands = [c for c in (k, h) if c >= 0]
                nxt = min(cands) if cands else (n - 1 if buf[n - 1] == 0xA5 else n)
                nxt = max(nxt, i + 1)
                self.stats.bytes_skipped += nxt - i
                i = nxt
                continue
            if i + plen > n:
                break  # incomplete packet
            body = bytes(buf[i + 2:i + plen - 2])
            crc = buf[i + plen - 2] | (buf[i + plen - 1] << 8)
            if crc16(body) != crc:
                self.stats.crc_errors += 1
                self.stats.bytes_skipped += 1
                i += 1
                continue
            good.append(bytes(buf[i:i + plen]))
            ends.append(self._buf_start + i + plen)
            i += plen
        del buf[:i]
        self._buf_start += i
        return self._make_block(good, ends, texts)

    def _make_block(self, good: list[bytes], ends: list[int], texts: list[str]) -> ParsedBlock:
        if not good:
            empty = np.zeros(0, np.uint32)
            return ParsedBlock(empty, np.zeros(0, np.uint8), np.zeros((0, self.n_channels), np.int32),
                               np.zeros(0, np.int64), texts)
        raw = np.frombuffer(b"".join(good), dtype=np.uint8).reshape(len(good), self.plen)
        counters, flags, samples = decode_packets(raw, self.n_channels)
        self._update_stats(counters, flags)
        return ParsedBlock(counters, flags, samples, np.asarray(ends, np.int64), texts)

    def _update_stats(self, counters: np.ndarray, flags: np.ndarray) -> None:
        s = self.stats
        s.packets += len(counters)
        s.overrun_flags += int(np.count_nonzero(flags & FLAG_OVERRUN))
        c = counters.astype(np.int64)
        if self._last_counter is not None:
            c = np.concatenate([[self._last_counter], c])
        d = (np.diff(c) % (1 << 32)) if len(c) > 1 else np.zeros(0, np.int64)
        jumps = d[d != 1]
        s.gaps += len(jumps)
        # a counter reset (START again) shows up as a huge jump; do not count it as lost samples
        s.lost_samples += int(np.sum(jumps[(jumps > 1) & (jumps < 1 << 31)] - 1))
        self._last_counter = int(counters[-1])

    def reset_counter_tracking(self) -> None:
        self._last_counter = None


def parse_bytes(data: bytes, n_channels: int) -> tuple[ParsedBlock, ParserStats]:
    """Parse a complete recording (e.g. emg_raw.bin) in one go."""
    p = PacketParser(n_channels)
    block = p.feed(data)
    return block, p.stats
