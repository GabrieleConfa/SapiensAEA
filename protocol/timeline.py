"""Session script -> ordered list of timed segments.

This is the single source of truth of the session timing. The acquisition program displays it,
and the Stage 2 labelling reads it back from cues.csv.

Script (times in config [protocol]):
  1. quality check           - done before, by tools/quality_check.py (not in this timeline)
  2. LED sync
  3. calibration: rest; reference contractions (FIST, EXTENSION) 3 x 3 s; reference poses
     OPEN (closure 0) and FIST (closure 1), 5 s each
  4. block 1: trials 1-16    (cue 1 s, hold 4 s, rest 4 s)
  5. pause 2 min
  6. ramps: per closing grip in the session's ramp order, 4 x
     (open->half 2 s, hold 2 s, ->closed 2 s, hold 2 s, ->half 2 s, hold 2 s, ->open 2 s, rest 3 s)
  7. pause 2 min
  8. block 2: trials 17-32
  9. closing: rest 10 s, reference poses again, LED sync
LED flashes: in both sync segments and every [sync] interval_s in between.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from common.config import Config

REST = "REST"


@dataclass(frozen=True)
class Segment:
    index: int
    phase: str             # sync_start, calibration, block1, pause1, ramps, pause2, block2, closing, sync_end
    kind: str              # see module docstring and _Builder methods
    start_s: float
    duration_s: float
    grip: str | None       # OPEN/FIST/PINCH/POINT/EXTENSION/REST, None for sync/pauses/cues
    text: str              # large instruction shown to the subject
    subtext: str = ""
    trial: int | None = None   # 1..32 for trial segments
    rep: int | None = None     # repetition inside calibration or ramps (1-based)
    closure_from: float | None = None
    closure_to: float | None = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def as_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Timeline:
    session_id: str
    segments: tuple[Segment, ...]
    flash_times_s: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s

    def segment_at(self, t: float) -> Segment | None:
        """Segment active at time t (seconds from session start); None after the end."""
        if t < 0:
            return self.segments[0]
        lo, hi = 0, len(self.segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            seg = self.segments[mid]
            if t < seg.start_s:
                hi = mid - 1
            elif t >= seg.end_s:
                lo = mid + 1
            else:
                return seg
        return None


class _Builder:
    def __init__(self) -> None:
        self.t = 0.0
        self.segments: list[Segment] = []
        self.phase = ""

    def add(self, kind: str, duration: float, grip: str | None, text: str, subtext: str = "",
            **extra) -> Segment:
        if duration <= 0:
            return self.segments[-1] if self.segments else None  # zero-length segments are skipped
        seg = Segment(index=len(self.segments), phase=self.phase, kind=kind, start_s=round(self.t, 6),
                      duration_s=duration, grip=grip, text=text, subtext=subtext, **extra)
        self.segments.append(seg)
        self.t += duration
        return seg


GRIP_TEXT = {
    "OPEN": "OPEN - fingers extended and spread",
    "FIST": "FIST - power grip",
    "PINCH": "PINCH - thumb tip to index tip",
    "POINT": "POINT - index extended",
    "EXTENSION": "EXTEND - fingers and wrist back",
}


def build_timeline(cfg: Config, session: dict) -> Timeline:
    """Build the timeline of one session (an entry of protocol/sequences.json)."""
    p = cfg.section("protocol")
    trial_order: list[str] = session["trials"]
    ramp_order: list[str] = session["ramps"]
    n_block = int(p["trials_per_block"])
    if len(trial_order) != 2 * n_block:
        raise ValueError(f"expected {2 * n_block} trials, got {len(trial_order)}")

    b = _Builder()
    flashes: list[float] = []

    def sync(phase: str) -> None:
        b.phase = phase
        seg = b.add("sync", p["sync_duration_s"], None, "Keep still", "LED sync")
        flashes.append(round(seg.start_s + p["sync_flash_offset_s"], 6))

    def reference_poses() -> None:
        for grip, closure in (("OPEN", 0.0), ("FIST", 1.0)):
            b.add("ref_cue", p["ref_pose_cue_s"], None, f"Next: {grip}", "Reference pose",
                  closure_from=closure, closure_to=closure)
            b.add("ref_pose", p["ref_pose_s"], grip, GRIP_TEXT[grip], "Reference pose - hold",
                  closure_from=closure, closure_to=closure)
            b.add("ref_rest", p["ref_pose_rest_s"], REST, "Relax")

    def trial_block(first: int) -> None:
        for i in range(first, first + n_block):
            grip = trial_order[i - 1]
            sub = f"Trial {i}/{2 * n_block}"
            b.add("trial_cue", p["trial_cue_s"], None, f"Next: {grip}", sub, trial=i)
            b.add("trial_hold", p["trial_hold_s"], grip, GRIP_TEXT[grip], sub + " - hold", trial=i)
            b.add("trial_rest", p["trial_rest_s"], REST, "Relax", sub, trial=i)

    # 2. LED sync
    sync("sync_start")

    # 3. calibration
    b.phase = "calibration"
    b.add("cal_rest", p["cal_rest_s"], REST, "Relax", "Calibration - rest")
    for grip in p["cal_contraction_grips"]:
        for rep in range(1, int(p["cal_contraction_reps"]) + 1):
            sub = f"Reference contraction {rep}/{p['cal_contraction_reps']} - strong, NOT maximal"
            b.add("cal_cue", p["cal_contraction_cue_s"], None, f"Next: {grip}", sub, rep=rep)
            b.add("cal_contraction", p["cal_contraction_s"], grip, GRIP_TEXT[grip], sub, rep=rep)
            b.add("cal_contraction_rest", p["cal_contraction_rest_s"], REST, "Relax", rep=rep)
    reference_poses()

    # 4. block 1
    b.phase = "block1"
    trial_block(1)

    # 5. pause
    b.phase = "pause1"
    b.add("pause", p["pause_s"], REST, "Pause", "Relax, the session resumes automatically")

    # 6. ramps
    b.phase = "ramps"
    move, hold = p["ramp_move_s"], p["ramp_hold_s"]
    for grip in ramp_order:
        b.add("ramp_intro", p["ramp_intro_s"], None, f"Ramps: {grip}", "Start from the open hand")
        for rep in range(1, int(p["ramp_reps"]) + 1):
            sub = f"{grip} ramp {rep}/{p['ramp_reps']}"
            for kind, dur, c0, c1, text in (
                ("ramp_move", move, 0.0, 0.5, "Close to HALF"),
                ("ramp_hold", hold, 0.5, 0.5, "Hold HALF"),
                ("ramp_move", move, 0.5, 1.0, "Close fully"),
                ("ramp_hold", hold, 1.0, 1.0, "Hold CLOSED"),
                ("ramp_move", move, 1.0, 0.5, "Open to HALF"),
                ("ramp_hold", hold, 0.5, 0.5, "Hold HALF"),
                ("ramp_move", move, 0.5, 0.0, "Open fully"),
            ):
                b.add(kind, dur, grip, text, sub, rep=rep, closure_from=c0, closure_to=c1)
            b.add("ramp_rest", p["ramp_rest_s"], REST, "Relax", sub, rep=rep)

    # 7. pause
    b.phase = "pause2"
    b.add("pause", p["pause_s"], REST, "Pause", "Relax, the session resumes automatically")

    # 8. block 2
    b.phase = "block2"
    trial_block(n_block + 1)

    # 9. closing
    b.phase = "closing"
    b.add("closing_rest", p["closing_rest_s"], REST, "Relax", "Closing - rest")
    reference_poses()
    sync("sync_end")

    first, last = flashes[0], flashes[-1]
    interval = float(cfg["sync.interval_s"])
    periodic = []
    t = first + interval
    while t < last - interval / 2:
        periodic.append(round(t, 6))
        t += interval
    all_flashes = tuple([first, *periodic, last])

    return Timeline(session_id=session["id"], segments=tuple(b.segments), flash_times_s=all_flashes)
