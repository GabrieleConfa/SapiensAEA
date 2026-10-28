"""Session controller: walks the timeline, stamps markers, handles operator commands and gaps.

Independent of the GUI: the GUI and the headless runner call `tick()` periodically and the
command methods (start, pause, resume, proceed_qc, ...) on operator input.

Free recording (`free=True`): no quality check, no calibration, no trials. A single FREE step is
recorded until the operator stops it (`stop`) or `duration_s` has passed; gaps are logged but do
not pause anything.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

from emgacq import __version__
from emgacq import markers as mk
from emgacq.analysis import write_summary
from emgacq.config import APP_ROOT, Config
from emgacq.labels import Labeler
from emgacq.protocol import (Step, build_free_timeline, build_timeline, planned_duration_s,
                             trial_order_from_config)
from emgacq.quality import evaluate
from emgacq.recorder import Recorder
from emgacq.sources.base import SampleSource
from emgacq.writer import EmgCsvWriter, EventLog, mark_bad_trials

SCHEMA = "emgacq-session/1"
END_MARKER_TIMEOUT_S = 2.0


class State(str, Enum):
    IDLE = "IDLE"            # connected, live view only, nothing written
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHING = "FINISHING"  # end marker sent, waiting for it in the stream
    DONE = "DONE"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=APP_ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=APP_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def session_folder(data_dir: Path, subject: str, session: int, when: datetime,
                   tag: str = "") -> Path:
    name = f"{subject}_S{session}_{tag + '_' if tag else ''}{when:%Y-%m-%d_%H%M}"
    folder, k = data_dir / name, 2
    while folder.exists():
        folder, k = data_dir / f"{name}_{k}", k + 1
    return folder


class SessionController:
    def __init__(self, cfg: Config, source: SampleSource, recorder: Recorder, metadata: dict,
                 speed: float = 1.0, data_dir: Path | None = None, free: bool = False,
                 duration_s: float = 0.0):
        self.cfg = cfg
        self.source = source
        self.recorder = recorder
        self.metadata = dict(metadata)
        self.speed = float(speed)
        self.data_dir = Path(data_dir) if data_dir else cfg.path_of("paths.data_dir")
        self.free = free
        if free:
            self.order = None
            self.steps: list[Step] = build_free_timeline(duration_s)
            self.n_trials = 0
            self.trials_per_block = 1
        else:
            self.order = trial_order_from_config(cfg, metadata["subject"], int(metadata["session"]))
            self.steps = build_timeline(cfg, self.order)
            self.n_trials = len(self.order["trials"])
            self.trials_per_block = len(self.order["blocks"][0])
        self.gap_s = float(cfg["monitor.gap_s"])
        self.advance_hold_s = float(cfg["monitor.advance_hold_s"])
        self.channel_names = [f"ch{c}" for c in source.channels]

        self.state = State.IDLE
        self.aborted = False
        self.abort_reason = ""
        self.i = -1
        self.step_start = 0.0
        self.session_start = 0.0
        self.started_wall: datetime | None = None
        self.pause_reason = ""
        self.pause_start = 0.0
        self.resume_index = 0
        self._history: list[tuple[float, int]] = []   # (time, step index) of recent steps
        self.markers_inserted = 0
        self.last_trial = 0
        self.bad_trials: set[int] = set()
        self.qc_attempts: list[dict] = []
        self.qc_report: dict | None = None
        self._qc_marker_target = 0
        self._qc_loss0 = (0, 0)
        self.gap_active = False
        self.gap_start = 0.0
        self.gap_start_sample = 0
        self.gaps: list[dict] = []
        self.reconnects = 0
        self.finish_deadline = 0.0
        self.folder: Path | None = None
        self.events = EventLog(None)       # memory only until the session starts
        self.writer: EmgCsvWriter | None = None
        self.labeler: Labeler | None = None
        self.summary_text = ""

    # -- properties -------------------------------------------------------------------------
    @property
    def step(self) -> Step | None:
        return self.steps[self.i] if 0 <= self.i < len(self.steps) else None

    def step_elapsed(self, now: float) -> float:
        return (now - self.step_start) * self.speed

    def qc_start_index(self) -> int:
        return next((s.index for s in self.steps if s.kind == "qc_rest"), -1)

    @property
    def duration_s(self) -> float:
        """Free recording: planned length in seconds (0 = until the operator stops)."""
        return self.steps[0].duration_s if self.free else 0.0

    # -- start --------------------------------------------------------------------------------
    def start(self, now: float | None = None) -> Path:
        if self.state != State.IDLE:
            raise RuntimeError("session already started")
        now = time.monotonic() if now is None else now
        self.started_wall = datetime.now()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.folder = session_folder(self.data_dir, self.metadata["subject"],
                                     int(self.metadata["session"]), self.started_wall,
                                     tag="FREE" if self.free else "")
        self.folder.mkdir(parents=True)
        if self.order is not None:
            (self.folder / "trial_order.json").write_text(json.dumps(self.order, indent=2) + "\n",
                                                          encoding="utf-8")
        pre_start_events = self.events.rows
        self.events = EventLog(self.folder / "events.csv")
        for row in pre_start_events:
            self.events.log(row["event"], detail=f"[before start {row['wall_time']}] {row['detail']}")
        self.writer = EmgCsvWriter(self.folder / "emg.csv", self.channel_names,
                                   float(self.cfg["monitor.csv_flush_s"]))
        self.labeler = Labeler(self.trials_per_block)
        self.write_session_json("recording")
        self.recorder.start_recording(self.writer, self.labeler, self.events)
        if self.free:
            mode = "free recording, " + (f"{self.duration_s:g} s" if self.duration_s else "until stopped")
        else:
            mode = "protocol"
        self.events.log("SESSION_START", detail=f"folder {self.folder.name}, {mode}, speed {self.speed}")
        self.state = State.RUNNING
        self.session_start = now
        self._enter(0, now)
        return self.folder

    # -- main loop ------------------------------------------------------------------------------
    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._check_gap(now)
        if self.state == State.RUNNING:
            step = self.step
            if step.kind == "qc_review":
                if self.qc_report is None and self.recorder.markers_seen >= self._qc_marker_target:
                    self._evaluate_qc()
            elif step.kind == "free":
                if step.duration_s and self.step_elapsed(now) >= step.duration_s:
                    self._finish(now, aborted=False)
            elif (not step.wait_operator and self.step_elapsed(now) >= step.duration_s
                  and self.recorder.seconds_since_last_sample() <= self.advance_hold_s):
                self._advance(now)     # held while the link is silent: no step starts without data
        if self.state == State.FINISHING:
            if self.recorder.finished or now >= self.finish_deadline:
                self._finalize()

    def _check_gap(self, now: float) -> None:
        if self.state == State.DONE:
            return
        silent = self.recorder.seconds_since_last_sample()
        if silent > self.gap_s and not self.gap_active:
            self.gap_active = True
            self.gap_start = now - silent
            self.gap_start_sample = self.recorder.rows_written
            self.events.log("GAP_START", sample_index=self.gap_start_sample,
                            detail=f"no samples for {silent:.1f} s")
            if self.state == State.RUNNING and not self.step.wait_operator:
                self.pause("gap", now, data_stop=now - silent)
        elif silent <= self.gap_s and self.gap_active:
            self.gap_active = False
            duration = now - self.gap_start
            self.gaps.append({"start_sample": self.gap_start_sample,
                              "end_sample": self.recorder.rows_written,
                              "duration_s": round(duration, 3)})
            self.events.log("GAP_END", sample_index=self.recorder.rows_written,
                            detail=f"samples again after {duration:.1f} s")

    def _insert(self, code: int) -> None:
        self.source.insert_marker(code)
        self.markers_inserted += 1

    def _enter(self, index: int, now: float) -> None:
        self.i = index
        self.step_start = now
        self._history = self._history[-30:] + [(now, index)]
        step = self.steps[index]
        if index == self.qc_start_index():
            self.recorder.reset_qc()
            self.qc_report = None
            self._qc_loss0 = (self.recorder.packets.received, self.recorder.packets.lost)
        self._insert(step.marker)
        info = step.info
        self.events.log(info.name, marker=step.marker, phase=info.phase, gesture=step.gesture,
                        block=step.block, trial_id=step.trial_id,
                        sample_index=self.recorder.rows_written, detail=step.stage)
        if step.kind == "qc_review":
            self._qc_marker_target = self.markers_inserted
        if step.kind == "trial_hold":
            self.last_trial = step.trial_id

    def _advance(self, now: float) -> None:
        if self.i + 1 < len(self.steps):
            self._enter(self.i + 1, now)
        else:
            self._finish(now, aborted=False)

    # -- operator commands ------------------------------------------------------------------
    def can_pause(self) -> bool:
        return self.state == State.RUNNING and not self.step.wait_operator

    def pause(self, reason: str = "operator", now: float | None = None,
              data_stop: float | None = None) -> bool:
        """Pause; resume restarts the interrupted step (a trial from its cue).

        For a data gap, `data_stop` is when the last sample arrived: the restart point is taken
        from the step running at that moment, which may precede the current one.
        """
        now = time.monotonic() if now is None else now
        if not self.can_pause():
            return False
        step = self.step
        self.state = State.PAUSED
        self.pause_reason = reason
        self.pause_start = now
        self.resume_index = step.restart_index
        if data_stop is not None:
            origin = next((self.steps[i] for t, i in reversed(self._history) if t <= data_stop), step)
            if not origin.wait_operator:
                self.resume_index = min(self.resume_index, origin.restart_index)
        code = mk.GAP_PAUSE if reason == "gap" else mk.OPERATOR_PAUSE
        self._insert(code)
        repeats = ""
        if step.restart_index != step.index or step.kind in ("trial_cue", "cal_cue"):
            repeats = f"; {self.steps[step.restart_index].kind} of this step will be repeated"
        self.events.log("PAUSE", marker=code, phase="PAUSE", gesture=step.gesture,
                        block=step.block, trial_id=step.trial_id,
                        sample_index=self.recorder.rows_written,
                        detail=f"{reason} during {step.kind}{repeats}")
        return True

    def can_resume(self) -> bool:
        return self.state == State.PAUSED and not self.gap_active

    def resume(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not self.can_resume():
            return False
        self.events.log("RESUME", sample_index=self.recorder.rows_written,
                        detail=f"after {now - self.pause_start:.1f} s ({self.pause_reason})")
        self.state = State.RUNNING
        self.pause_reason = ""
        self._enter(self.resume_index, now)
        return True

    def proceed_qc(self, note: str = "", now: float | None = None) -> None:
        """Accept the quality check and go on. After a FAIL a note is required."""
        now = time.monotonic() if now is None else now
        if self.state != State.RUNNING or self.step.kind != "qc_review" or self.qc_report is None:
            raise RuntimeError("no quality-check result to accept")
        if self.gap_active:
            raise RuntimeError("no data from the board")
        if self.qc_report["result"] == "FAIL" and not note.strip():
            raise ValueError("the quality check FAILED: type a note to proceed anyway")
        self._qc_decide("proceed", note)
        self._advance(now)

    def repeat_qc(self, note: str = "", now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state != State.RUNNING or self.step.kind != "qc_review":
            raise RuntimeError("not reviewing a quality check")
        if self.qc_report is not None:
            self._qc_decide("repeat", note)
        self._enter(self.qc_start_index(), now)

    def block_pause_left(self, now: float | None = None) -> float:
        """Seconds (real time) until the operator may continue after the block pause."""
        now = time.monotonic() if now is None else now
        step = self.step
        if step is None or step.kind != "block_pause":
            return 0.0
        return max(0.0, (step.min_s - self.step_elapsed(now)) / self.speed)

    def can_continue_block_pause(self, now: float | None = None) -> bool:
        return (self.state == State.RUNNING and self.step.kind == "block_pause"
                and self.block_pause_left(now) <= 0 and not self.gap_active)

    def continue_block_pause(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not self.can_continue_block_pause(now):
            return False
        self.events.log("BLOCK_PAUSE_END", sample_index=self.recorder.rows_written,
                        detail=f"after {self.step_elapsed(now) / self.speed:.1f} s")
        self._advance(now)
        return True

    def flag_last_trial_bad(self) -> tuple[int, bool] | None:
        """Toggle the bad flag of the last trial whose HOLD has started. Returns (trial, bad)."""
        if self.last_trial == 0 or self.state in (State.IDLE, State.DONE):
            return None
        t = self.last_trial
        bad = t not in self.bad_trials
        (self.bad_trials.add if bad else self.bad_trials.discard)(t)
        g = self.order["trials"][t - 1]["gesture"]
        self.events.log("TRIAL_BAD" if bad else "TRIAL_BAD_UNDO", gesture=g,
                        block=self.order["trials"][t - 1]["block"], trial_id=t,
                        sample_index=self.recorder.rows_written,
                        detail="flagged by the operator" if bad else "flag removed by the operator")
        return t, bad

    def stop(self, reason: str = "operator", now: float | None = None) -> bool:
        """Free recording: end it normally (status `completed`)."""
        now = time.monotonic() if now is None else now
        if not self.free or self.state != State.RUNNING:
            return False
        self.events.log("STOP", sample_index=self.recorder.rows_written, detail=f"stopped: {reason}")
        self._finish(now, aborted=False)
        return True

    def abort(self, reason: str = "operator", now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state in (State.RUNNING, State.PAUSED):
            self.abort_reason = reason
            self._finish(now, aborted=True)
        elif self.state == State.IDLE:
            self.state = State.DONE

    def _finish(self, now: float, aborted: bool) -> None:
        self.aborted = aborted
        code = mk.SESSION_ABORT if aborted else mk.SESSION_END
        self._insert(code)
        self.events.log(mk.decode(code).name, marker=code, sample_index=self.recorder.rows_written,
                        detail=self.abort_reason)
        self.state = State.FINISHING
        self.finish_deadline = now + END_MARKER_TIMEOUT_S

    # -- reconnection -------------------------------------------------------------------------
    def reconnect(self) -> dict:
        """Reopen the source (blocking; the GUI calls it from a worker thread)."""
        self.events.log("RECONNECT_START", sample_index=self.recorder.rows_written)
        try:
            info = self.recorder.reconnect()
        except Exception as e:
            self.events.log("RECONNECT_FAILED", sample_index=self.recorder.rows_written, detail=str(e))
            raise
        self.reconnects += 1
        # markers still queued in the old board session never reach the stream: count only the
        # ones seen, or the quality check (which waits for all its markers) would wait forever
        lost_markers = self.markers_inserted - self.recorder.markers_seen
        self.markers_inserted = self.recorder.markers_seen
        self.events.log("RECONNECT_OK", sample_index=self.recorder.rows_written,
                        detail=f"package counter reset; reconnection {self.reconnects}"
                               + (f"; {lost_markers} queued marker(s) lost" if lost_markers > 0 else ""))
        if self.state == State.PAUSED:
            # markers still queued in the old board session are lost with it: stamp the pause again
            code = mk.GAP_PAUSE if self.pause_reason == "gap" else mk.OPERATOR_PAUSE
            self._insert(code)
            self.events.log("PAUSE", marker=code, phase="PAUSE", sample_index=self.recorder.rows_written,
                            detail="pause marker stamped again after the reconnection")
        return info

    # -- quality check ------------------------------------------------------------------------
    def _evaluate_qc(self) -> None:
        cap = self.recorder.close_qc_capture()
        rec0, lost0 = self._qc_loss0
        p = self.recorder.packets
        lost = p.lost - lost0
        expected = (p.received - rec0) + lost
        loss_pct = 100.0 * lost / expected if expected else 0.0
        q = self.cfg.section("quality")
        q["discard_s"] = q["discard_s"] / self.speed      # captures are shorter when sped up
        try:
            report = evaluate(cap.rest, cap.fist, self.source.fs, q,
                              self.source.full_scale_uv, self.channel_names, loss_pct)
        except ValueError as e:
            report = {"result": "FAIL", "reasons": [str(e)], "channels": [],
                      "packet_loss_pct": round(loss_pct, 3)}
        report["attempt"] = len(self.qc_attempts) + 1
        report["time"] = datetime.now().isoformat(timespec="seconds")
        self.qc_report = report
        self.events.log("QC_RESULT", sample_index=self.recorder.rows_written,
                        detail=f"{report['result']}: " + ("; ".join(report["reasons"]) or "all checks passed"))
        self._save_qc()

    def _qc_decide(self, decision: str, note: str) -> None:
        self.qc_report["decision"] = decision
        self.qc_report["operator_note"] = note.strip()
        self.qc_attempts.append(self.qc_report)
        self.events.log("QC_" + decision.upper(), sample_index=self.recorder.rows_written,
                        detail=f"{self.qc_report['result']}" + (f"; note: {note.strip()}" if note.strip() else ""))
        self._save_qc()

    def _save_qc(self) -> None:
        if self.folder is None:
            return
        attempts = list(self.qc_attempts)
        if self.qc_report is not None and self.qc_report not in attempts:
            attempts.append(self.qc_report)
        final = next((a for a in reversed(attempts) if a.get("decision") == "proceed"), None)
        data = {"final_result": final["result"] if final else None,
                "final_note": final.get("operator_note", "") if final else None,
                "attempts": attempts}
        (self.folder / "quality_check.json").write_text(json.dumps(data, indent=2) + "\n",
                                                        encoding="utf-8")

    # -- close --------------------------------------------------------------------------------
    def _finalize(self) -> None:
        self.recorder.stop_recording()
        ranges = [(x.start_sample, x.end_sample) for x in self.labeler.interruptions]
        n_bad = mark_bad_trials(self.folder / "emg.csv", self.bad_trials, ranges)
        self.events.log("BAD_ROWS_UPDATED", detail=f"{n_bad} rows: trials {sorted(self.bad_trials)}, "
                                                  f"{len(ranges)} interrupted attempts")
        if self.gap_active:
            self.gaps.append({"start_sample": self.gap_start_sample, "end_sample": None,
                              "duration_s": None})
        self.events.log("SESSION_CLOSED", detail="aborted" if self.aborted else "completed")
        self.events.close()
        self.write_session_json("aborted" if self.aborted else "completed")
        self.summary_text = write_summary(self.folder)
        self.state = State.DONE

    def shutdown(self, reason: str = "exit") -> None:
        """Called on any exit path: close the session if needed and free the board."""
        try:
            if self.state in (State.RUNNING, State.PAUSED):
                self.abort(reason)
            if self.state == State.FINISHING:
                deadline = time.monotonic() + END_MARKER_TIMEOUT_S
                while not self.recorder.finished and time.monotonic() < deadline:
                    time.sleep(0.02)
                self._finalize()
        finally:
            self.recorder.stop()
            self.source.release()

    # -- session.json -------------------------------------------------------------------------
    def write_session_json(self, status: str) -> None:
        p = self.recorder.packets
        data = {
            "schema": SCHEMA,
            "app": {"name": "emgacq", "version": __version__, "git_commit": git_commit(),
                    "python": platform.python_version(), "platform": platform.platform()},
            "status": status,
            "abort_reason": self.abort_reason or None,
            "started": self.started_wall.isoformat(timespec="seconds") if self.started_wall else None,
            "ended": datetime.now().isoformat(timespec="seconds") if status != "recording" else None,
            **self.metadata,
            "source": {k: v for k, v in self.source.info.items()},
            "sampling_rate_hz": self.source.fs,
            "channels": self.source.channels,
            "csv_channel_columns": self.channel_names,
            "units": "microvolt, raw (unfiltered)",
            "mode": "free" if self.free else "protocol",
            "speed": self.speed,
            "seed": self.order["seed"] if self.order else None,
            "protocol": None if self.free else self.cfg.section("protocol"),
            "quality_thresholds": None if self.free else self.cfg.section("quality"),
            "planned_duration_s": round(planned_duration_s(self.steps), 1) or None,
            "config_sha256": self.cfg.sha256,
            "config": self.cfg.as_dict(),
        }
        if status != "recording":
            data["stats"] = {
                "rows": self.writer.rows if self.writer else 0,
                "packets_received": p.received, "packets_lost": p.lost,
                "packet_loss_pct": round(p.loss_pct, 4), "duplicates": p.duplicates,
                "gaps": self.gaps, "reconnects": self.reconnects,
                "bad_trials": sorted(self.bad_trials),
                "interrupted_attempts": [vars(x) for x in self.labeler.interruptions] if self.labeler else [],
                "quality_check": self.qc_attempts[-1]["result"] if self.qc_attempts else None,
            }
        (self.folder / "session.json").write_text(json.dumps(data, indent=2, default=str) + "\n",
                                                  encoding="utf-8")

    # -- status for the views ---------------------------------------------------------------
    def remaining_s(self, now: float) -> float:
        """Planned time left (real seconds), operator waits excluded except the minimum pause.

        A free recording without a set duration has no planned end: 0."""
        if self.step is None or self.state in (State.IDLE,):
            return planned_duration_s(self.steps) / self.speed
        if self.state in (State.FINISHING, State.DONE):
            return 0.0
        step = self.step
        own = max(0.0, (step.duration_s or step.min_s) - self.step_elapsed(now))
        later = planned_duration_s(self.steps[self.i + 1:])
        return (own + later) / self.speed

    def elapsed_s(self, now: float) -> float:
        return now - self.session_start if self.state != State.IDLE else 0.0
