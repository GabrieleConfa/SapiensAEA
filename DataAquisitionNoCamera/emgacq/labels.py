"""Per-sample labels (phase, label, block, trial) from the marker column of the stream.

A marker marks the first sample of the step it starts; every following sample keeps that step's
labels until the next marker. The end markers (SESSION_END / SESSION_ABORT) keep the labels of
the step before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emgacq import markers as mk


@dataclass(frozen=True)
class Interruption:
    """A trial attempt that was cut (pause, data gap) and repeated from its cue."""
    trial_id: int
    start_sample: int      # sample of the attempt's CUE marker
    end_sample: int        # first pause marker after it, or the repeated CUE (exclusive)


@dataclass
class LabelArrays:
    phase: np.ndarray
    label: np.ndarray
    block: np.ndarray
    trial_id: np.ndarray
    step: np.ndarray       # running number of the step (increments at every marker)


@dataclass
class Labeler:
    trials_per_block: int
    phase: str = ""
    label: str = ""
    block: int = 0
    trial_id: int = 0
    gesture: str = ""
    step: int = 0
    markers_seen: int = 0
    ended: bool = False
    end_sample: int | None = None
    interruptions: list[Interruption] = field(default_factory=list)
    unknown_markers: list[tuple[int, float]] = field(default_factory=list)
    _attempt_start: int | None = None
    _attempt_trial: int = 0
    _attempt_pause: int | None = None

    def block_of(self, trial_id: int) -> int:
        return (trial_id - 1) // self.trials_per_block + 1 if trial_id > 0 else 0

    def apply(self, code: float, sample_index: int) -> None:
        self.markers_seen += 1
        try:
            info = mk.decode(code)
        except ValueError:
            self.unknown_markers.append((sample_index, float(code)))
            return
        if info.code in (mk.SESSION_END, mk.SESSION_ABORT):
            self.ended = True
            self.end_sample = sample_index
            return
        if info.phase == "PAUSE" and self._attempt_pause is None and self._attempt_start is not None:
            self._attempt_pause = sample_index
        if info.phase == "CUE":
            # a second CUE of the same trial means the previous attempt was cut and is repeated
            if info.trial_id == self._attempt_trial and self._attempt_start is not None:
                end = self._attempt_pause if self._attempt_pause is not None else sample_index
                self.interruptions.append(Interruption(info.trial_id, self._attempt_start, end))
            self._attempt_start, self._attempt_trial, self._attempt_pause = sample_index, info.trial_id, None
        self.phase, self.label, self.gesture = info.phase, info.label, info.gesture
        self.trial_id = info.trial_id
        self.block = self.block_of(info.trial_id)
        self.step += 1

    def label_chunk(self, markers: np.ndarray, first_index: int) -> LabelArrays:
        """Labels for a chunk of samples; `first_index` is the sample_index of its first row."""
        n = len(markers)
        out = LabelArrays(np.empty(n, object), np.empty(n, object), np.zeros(n, np.int32),
                          np.zeros(n, np.int32), np.zeros(n, np.int64))
        start = 0
        for i in np.flatnonzero(markers):
            self._fill(out, start, i)
            self.apply(markers[i], first_index + int(i))
            start = int(i)
        self._fill(out, start, n)
        return out

    def _fill(self, out: LabelArrays, a: int, b: int) -> None:
        out.phase[a:b] = self.phase
        out.label[a:b] = self.label
        out.block[a:b] = self.block
        out.trial_id[a:b] = self.trial_id
        out.step[a:b] = self.step
