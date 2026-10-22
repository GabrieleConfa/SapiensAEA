"""What the subject screen shows, derived from the controller state (no Qt here).

The subject screen shows only the task: a gesture name or an instruction, a colour and a
countdown bar. Never signals, numbers or predictions: seeing their own EMG changes how the
subject contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from emgacq.session import SessionController, State

GESTURE_HINTS = {
    "OPEN": "Fingers extended and spread",
    "CLOSE": "Fist (power grip)",
    "INDEX": "Point with the index finger",
}


@dataclass(frozen=True)
class SubjectScreen:
    title: str
    subtitle: str
    color_key: str          # key of [display.colors]
    progress: float | None  # fraction of the step still to go (1 -> 0), None = no bar
    image_key: str | None   # assets/<image_key>.png if present


def subject_screen(ctrl: SessionController, now: float) -> SubjectScreen:
    if ctrl.state == State.IDLE:
        return SubjectScreen("Please wait", "The session will start shortly", "PAUSE", None, None)
    if ctrl.state in (State.FINISHING, State.DONE):
        if ctrl.aborted:
            return SubjectScreen("Session stopped", "Thank you", "PAUSE", None, None)
        return SubjectScreen("Session complete", "Thank you!", "REST", None, None)
    if ctrl.state == State.PAUSED:
        return SubjectScreen("Pause", "Relax your hand and wait", "PAUSE", None, None)

    step = ctrl.step
    left = None
    if step.duration_s > 0:
        left = max(0.0, 1.0 - ctrl.step_elapsed(now) / step.duration_s)
    g = step.gesture
    k = step.kind
    if k in ("qc_rest", "cal_rest", "trial_rest", "closing_rest"):
        sub = {"qc_rest": "Relax your hand and arm completely",
               "cal_rest": "Relax your hand and arm completely",
               "trial_rest": "Relax",
               "closing_rest": "Relax - last few seconds"}[k]
        return SubjectScreen("REST", sub, "REST", left, "REST")
    if k in ("qc_cue", "cal_cue", "trial_cue"):
        return SubjectScreen(f"Get ready: {g}", GESTURE_HINTS.get(g, ""), g, left, g)
    if k == "qc_fist":
        return SubjectScreen(g, "Firm, comfortable fist - hold", g, left, g)
    if k == "cal_hold":
        return SubjectScreen(g, "Strong but comfortable, never maximal - hold", g, left, g)
    if k == "trial_hold":
        return SubjectScreen(g, "Perform and hold", g, left, g)
    if k == "qc_review":
        return SubjectScreen("Relax", "The operator is checking the electrodes", "PAUSE", None, None)
    if k == "block_pause":
        return SubjectScreen("Pause", "Relax. The next block starts when you are ready", "PAUSE", None, None)
    if k == "free":
        return SubjectScreen("Recording", "Follow the operator's instructions", "PAUSE", left, None)
    return SubjectScreen("", "", "PAUSE", None, None)


def status_line(ctrl: SessionController, now: float) -> str:
    """One line for the operator (status strip in single-screen mode, window title)."""
    step = ctrl.step
    p = ctrl.recorder.packets
    parts = [ctrl.state.value]
    if step is not None and ctrl.state not in (State.IDLE, State.DONE):
        parts.append(step.stage)
        if step.trial_id:
            in_block = (step.trial_id - 1) % ctrl.trials_per_block + 1
            parts.append(f"trial {in_block}/{ctrl.trials_per_block} ({step.gesture})")
    parts.append(f"lost {p.lost} ({p.loss_pct:.2f}%)")
    if ctrl.gap_active:
        parts.append("NO DATA")
    if ctrl.bad_trials:
        parts.append(f"bad: {','.join(map(str, sorted(ctrl.bad_trials)))}")
    if ctrl.free and not ctrl.duration_s:
        parts.append("until stopped")
    else:
        rem = ctrl.remaining_s(now)
        parts.append(f"left ~{int(rem // 60)}:{int(rem % 60):02d}")
    return "   |   ".join(parts)
