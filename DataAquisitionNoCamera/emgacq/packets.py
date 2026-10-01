"""Lost-packet detection from the board's package-number channel."""

from __future__ import annotations

import numpy as np


class PacketTracker:
    """Counts samples missing from a package-number sequence.

    `modulus` is the wrap of the counter (256 for the Cyton and the BrainFlow synthetic board),
    or None for a counter that never wraps. A step of 0 (repeated number) counts as a duplicate,
    a backwards step of a non-wrapping counter as a resync; neither counts as loss.
    """

    def __init__(self, modulus: int | None):
        self.modulus = modulus
        self.received = 0
        self.lost = 0
        self.duplicates = 0
        self.resyncs = 0
        self.loss_events: list[tuple[int, int]] = []    # (sample_index after the gap, lost count)
        self._last: int | None = None

    def update(self, package_nums: np.ndarray, first_sample_index: int = 0) -> int:
        """Feed the next chunk; returns the number of samples lost before/inside it."""
        pk = np.asarray(package_nums).astype(np.int64)
        if pk.size == 0:
            return 0
        self.received += pk.size
        if self._last is None:
            seq, offset = pk, 1          # step i lands on sample i+1 of the chunk
        else:
            seq, offset = np.concatenate([[self._last], pk]), 0
        self._last = int(pk[-1])
        steps = np.diff(seq)
        if self.modulus:
            steps = steps % self.modulus
        else:
            self.resyncs += int(np.count_nonzero(steps < 0))
            steps = np.where(steps < 0, 1, steps)
        self.duplicates += int(np.count_nonzero(steps == 0))
        lost_each = np.where(steps > 1, steps - 1, 0)
        total = int(lost_each.sum())
        if total:
            self.lost += total
            for i in np.flatnonzero(lost_each):
                self.loss_events.append((first_sample_index + int(i) + offset, int(lost_each[i])))
        return total

    def reset(self) -> None:
        """Forget the last number (after a reconnection the counter restarts)."""
        self._last = None

    @property
    def loss_pct(self) -> float:
        expected = self.received + self.lost
        return 100.0 * self.lost / expected if expected else 0.0
