"""Drives a session timeline on the laptop clock: logs cues, requests LED flashes, records LED
edges. UI-independent, so it runs the same with the Tk cue screen and headless (tests)."""

from __future__ import annotations

from dataclasses import dataclass

from acquisition.writers import CsvLog
from common.sources.base import EmgSource
from protocol.timeline import Segment, Timeline

# planned_host_time_ns: session start + planned_start_s (the schedule, on the laptop clock)
# host_time_ns: when the cue was actually put on screen (one UI tick later at most)
CUE_COLUMNS = ["segment_index", "phase", "kind", "grip", "trial", "rep", "text",
               "planned_start_s", "planned_duration_s", "closure_from", "closure_to",
               "planned_host_time_ns", "host_time_ns", "emg_last_counter"]
SYNC_COLUMNS = ["event", "host_time_ns", "emg_counter", "planned_t_s", "note"]

# Simulator only: activity level of the synthetic EMG per segment kind.
_SIM_LEVEL = {"trial_hold": 0.6, "cal_contraction": 0.8, "ref_pose": 0.5}


@dataclass
class RunnerState:
    segment: Segment | None
    t_s: float                 # session time (s, already scaled by speed)
    closure_target: float | None
    finished: bool


class SessionRunner:
    def __init__(self, timeline: Timeline, source: EmgSource, cues: CsvLog, sync: CsvLog,
                 speed: float = 1.0, simulate: bool = False):
        self.timeline = timeline
        self.source = source
        self.cues = cues
        self.sync = sync
        self.speed = speed
        self.simulate = simulate
        self.t0_ns: int | None = None
        self.end_ns: int | None = None
        self._seg_index = -1
        self._next_flash = 0
        self._edges_logged = 0
        self._last_sim_ns = 0
        self.aborted = False

    def start(self, now_ns: int) -> None:
        self.t0_ns = now_ns
        self.sync.write(event="session_start", host_time_ns=now_ns, planned_t_s=0.0)

    def session_time(self, now_ns: int) -> float:
        return (now_ns - self.t0_ns) / 1e9 * self.speed

    def tick(self, now_ns: int) -> RunnerState:
        """Advance to now. Call often (every ~10-20 ms)."""
        self.poll_led_edges()
        if self.t0_ns is None:
            return RunnerState(None, 0.0, None, False)
        t = self.session_time(now_ns)
        seg = self.timeline.segment_at(t)

        # Log every segment reached, in order, even if a tick came late and skipped one.
        upto = seg.index if seg is not None else len(self.timeline.segments) - 1
        while self._seg_index < upto:
            self._seg_index += 1
            s = self.timeline.segments[self._seg_index]
            self.cues.write(segment_index=s.index, phase=s.phase, kind=s.kind, grip=s.grip,
                            trial=s.trial, rep=s.rep, text=s.text,
                            planned_start_s=s.start_s, planned_duration_s=s.duration_s,
                            closure_from=s.closure_from, closure_to=s.closure_to,
                            planned_host_time_ns=self.t0_ns + round(s.start_s / self.speed * 1e9),
                            host_time_ns=now_ns, emg_last_counter=self.source.buffer.last_counter())
            if self.simulate and s is seg:
                self._sim_segment(seg)

        flashes = self.timeline.flash_times_s
        while self._next_flash < len(flashes) and t >= flashes[self._next_flash]:
            ok = self.source.flash()
            self.sync.write(event="flash_request", host_time_ns=now_ns,
                            planned_t_s=flashes[self._next_flash],
                            note="" if ok else "source cannot drive the LED")
            self._next_flash += 1

        closure = None
        if seg is not None and seg.closure_from is not None:
            frac = min(1.0, max(0.0, (t - seg.start_s) / seg.duration_s))
            closure = seg.closure_from + (seg.closure_to - seg.closure_from) * frac
            if self.simulate and seg.kind.startswith("ramp") and now_ns - self._last_sim_ns > 50e6:
                self.source.send(f"SIM GRIP {seg.grip} {0.8 * closure:.3f}")
                self._last_sim_ns = now_ns

        finished = seg is None
        if finished and self.end_ns is None:
            self.end_ns = now_ns
            self.sync.write(event="session_end", host_time_ns=now_ns, planned_t_s=round(t, 3))
        return RunnerState(seg, t, closure, finished)

    def abort(self, now_ns: int) -> None:
        self.aborted = True
        self.end_ns = now_ns
        t = self.session_time(now_ns) if self.t0_ns is not None else None
        self.sync.write(event="session_aborted", host_time_ns=now_ns, planned_t_s=t)

    def poll_led_edges(self) -> None:
        edges = self.source.buffer.led_edges
        while self._edges_logged < len(edges):
            e = edges[self._edges_logged]
            self.sync.write(event="led_rise", host_time_ns=e.host_time_ns, emg_counter=e.counter)
            self._edges_logged += 1

    def _sim_segment(self, seg: Segment) -> None:
        if seg.kind in _SIM_LEVEL and seg.grip not in (None, "REST"):
            self.source.send(f"SIM GRIP {seg.grip} {_SIM_LEVEL[seg.kind]}")
        elif not seg.kind.startswith("ramp_") or seg.kind in ("ramp_rest", "ramp_intro"):
            self.source.send("SIM GRIP REST 0")
