"""Run a whole session without GUI, with the operator's decisions scripted.

Used by the tests and for checking a setup end to end:

    python -m emgacq --synthetic --headless --speed 20

The script: proceed after the quality check (with a note if it failed), continue after the block
pause as soon as allowed, resume after a pause as soon as data flow again. Optional events:
flag trials bad, pause the operator during a trial, simulate a radio stall (synthetic board),
reconnect on a gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from emgacq.config import Config
from emgacq.recorder import Recorder
from emgacq.session import SessionController, State
from emgacq.sources.base import SampleSource


@dataclass
class Script:
    qc_note: str = "headless run"
    flag_bad: set[int] = field(default_factory=set)       # trials flagged after their hold starts
    operator_pause_trial: int | None = None               # pause during this trial's hold
    stall_trial: int | None = None                        # radio stall during this trial's hold
    stall_s: float = 2.5
    reconnect_on_gap: bool = False
    resume_after_s: float = 0.3
    timeout_s: float = 3600.0


def run_headless(cfg: Config, source: SampleSource, metadata: dict, speed: float = 1.0,
                 script: Script | None = None, data_dir=None, log=print) -> SessionController:
    script = script or Script()
    recorder = Recorder(source, float(cfg["board.read_interval_s"]))
    ctrl = SessionController(cfg, source, recorder, metadata, speed=speed, data_dir=data_dir)
    done_events: set[str] = set()
    try:
        log(f"connecting to {source.kind} ...")
        source.connect()
        for c in source.command_log:
            log(f"  {c['command']!r} -> {c['reply']!r}")
        source.start()
        recorder.start()
        time.sleep(0.5)
        folder = ctrl.start()
        log(f"session folder {folder}")
        t_end = time.monotonic() + script.timeout_s
        last_step = -1
        while ctrl.state != State.DONE:
            now = time.monotonic()
            if now > t_end:
                ctrl.abort("headless timeout")
            ctrl.tick(now)
            step = ctrl.step
            if step is not None and ctrl.i != last_step and ctrl.state == State.RUNNING:
                last_step = ctrl.i
                log(f"  {ctrl.elapsed_s(now):7.1f} s  {step.stage:13} {step.kind:12} {step.gesture:5} "
                    f"trial {step.trial_id}")
            if ctrl.state == State.RUNNING and step.kind == "qc_review" and ctrl.qc_report:
                rep = ctrl.qc_report
                log(f"  quality check {rep['result']}: {'; '.join(rep['reasons']) or 'ok'}")
                ctrl.proceed_qc(script.qc_note if rep["result"] == "FAIL" else "", now)
            elif ctrl.state == State.RUNNING and step.kind == "block_pause":
                ctrl.continue_block_pause(now)
            elif ctrl.state == State.RUNNING and step.kind == "trial_hold":
                t = step.trial_id
                if t in script.flag_bad and f"bad{t}" not in done_events:
                    done_events.add(f"bad{t}")
                    ctrl.flag_last_trial_bad()
                    log(f"  flagged trial {t} bad")
                if t == script.operator_pause_trial and "pause" not in done_events \
                        and ctrl.step_elapsed(now) > 0.5:
                    done_events.add("pause")
                    ctrl.pause("operator", now)
                    log(f"  operator pause during trial {t}")
                if t == script.stall_trial and "stall" not in done_events:
                    done_events.add("stall")
                    source.simulate_stall(script.stall_s)
                    log(f"  simulated radio stall of {script.stall_s} s during trial {t}")
            elif ctrl.state == State.PAUSED:
                if ctrl.pause_reason == "gap" and script.reconnect_on_gap and "reconnect" not in done_events:
                    done_events.add("reconnect")
                    log("  gap: reconnecting")
                    ctrl.reconnect()
                if ctrl.can_resume() and now - ctrl.pause_start >= script.resume_after_s:
                    ctrl.resume(now)
                    log("  resumed")
            time.sleep(0.005)
        log(f"done: {'aborted' if ctrl.aborted else 'completed'}")
    finally:
        ctrl.shutdown("headless exit")
    return ctrl
