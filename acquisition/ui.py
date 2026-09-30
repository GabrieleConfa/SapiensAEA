"""Full-screen cue display (tkinter) for the operator and the subject.

Keys: SPACE start the session, ESC twice within 2 s abort, F11 toggle full screen.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Callable

from acquisition.runner import RunnerState, SessionRunner

BG = "#101214"
COLORS = {
    "cue": "#FFC107",      # get ready
    "active": "#4CAF50",   # hold / contract / ramp
    "rest": "#9E9E9E",
    "sync": "#FFFFFF",
    "warn": "#FF5252",
    "status": "#80868B",
}
CUE_KINDS = {"trial_cue", "cal_cue", "ref_cue", "ramp_intro"}
ACTIVE_KINDS = {"trial_hold", "cal_contraction", "ref_pose", "ramp_move", "ramp_hold"}
PHASE_NAMES = {"sync_start": "LED sync", "calibration": "Calibration", "block1": "Block 1",
               "pause1": "Pause", "ramps": "Ramps", "pause2": "Pause", "block2": "Block 2",
               "closing": "Closing", "sync_end": "LED sync"}


def color_for(kind: str) -> str:
    if kind == "sync":
        return COLORS["sync"]
    if kind in CUE_KINDS:
        return COLORS["cue"]
    if kind in ACTIVE_KINDS:
        return COLORS["active"]
    return COLORS["rest"]


class CueScreen:
    def __init__(self, runner: SessionRunner, title: str, status: Callable[[], tuple[str, bool]],
                 on_finish: Callable[[bool], None], fullscreen: bool = True):
        """status() returns (status line, ok); on_finish(aborted) is called once at the end."""
        self.runner = runner
        self.status = status
        self.on_finish = on_finish
        self.title = title
        self.root = tk.Tk()
        self.root.title(f"EMG session {title}")
        self.root.configure(bg=BG)
        self.root.attributes("-fullscreen", fullscreen)
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<F11>", lambda e: self.root.attributes(
            "-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._on_escape(None, force=True))
        self._started = False
        self._done = False
        self._esc_ns = 0
        self._end_shown_ns = 0

    def run(self) -> None:
        self.root.after(15, self._tick)
        self.root.mainloop()

    # -- events -----------------------------------------------------------------------------
    def _on_space(self, _event) -> None:
        if not self._started:
            _, ok = self.status()
            self._started = True
            self.runner.start(time.perf_counter_ns())
            if not ok:
                print("WARNING: session started while the status line reported a problem")

    def _on_escape(self, _event, force: bool = False) -> None:
        now = time.perf_counter_ns()
        if self._done:
            self._close()
            return
        if force or now - self._esc_ns < 2e9:
            self.runner.abort(now)
            self._finish(aborted=True)
        else:
            self._esc_ns = now

    def _finish(self, aborted: bool) -> None:
        if not self._done:
            self._done = True
            self._end_shown_ns = time.perf_counter_ns()
            self.on_finish(aborted)

    def _close(self) -> None:
        self.root.quit()
        self.root.destroy()

    # -- drawing ----------------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.perf_counter_ns()
        state = self.runner.tick(now)
        if state.finished and not self._done:
            self._finish(aborted=False)
        self._draw(state, now)
        if self._done and now - self._end_shown_ns > 5e9:
            self._close()
            return
        self.root.after(15, self._tick)

    def _draw(self, state: RunnerState, now: int) -> None:
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 800), max(c.winfo_height(), 600)
        big = int(h * 0.11)
        mid = int(h * 0.045)
        small = int(h * 0.022)
        status_text, ok = self.status()

        if self._done:
            msg = "Session aborted" if self.runner.aborted else "Session complete"
            c.create_text(w / 2, h / 2, text=msg, fill=COLORS["warn"] if self.runner.aborted
                          else COLORS["active"], font=("Segoe UI", big, "bold"))
            c.create_text(w / 2, h / 2 + big, text="Saving files... the window closes by itself",
                          fill=COLORS["rest"], font=("Segoe UI", small))
        elif not self._started:
            c.create_text(w / 2, h * 0.35, text=f"Session {self.title}", fill="white",
                          font=("Segoe UI", big, "bold"))
            c.create_text(w / 2, h * 0.52, text="Press SPACE to start", fill=COLORS["cue"],
                          font=("Segoe UI", mid))
            if not ok:
                c.create_text(w / 2, h * 0.62, text="Check the status line before starting",
                              fill=COLORS["warn"], font=("Segoe UI", small, "bold"))
        elif state.segment is not None:
            seg = state.segment
            col = color_for(seg.kind)
            remaining = max(0.0, seg.end_s - state.t_s)
            total = self.runner.timeline.duration_s
            c.create_text(30, 30, anchor="nw", fill=COLORS["status"], font=("Segoe UI", small),
                          text=f"{PHASE_NAMES.get(seg.phase, seg.phase)}   "
                               f"{_mmss(state.t_s)} / {_mmss(total)}")
            c.create_text(w / 2, h * 0.36, text=seg.text, fill=col, font=("Segoe UI", big, "bold"),
                          width=w * 0.9)
            if seg.subtext:
                c.create_text(w / 2, h * 0.50, text=seg.subtext, fill=COLORS["rest"],
                              font=("Segoe UI", mid), width=w * 0.9)
            c.create_text(w / 2, h * 0.64, text=str(math.ceil(remaining)), fill=col,
                          font=("Segoe UI", mid, "bold"))
            frac = 1.0 - remaining / seg.duration_s if seg.duration_s else 1.0
            x0, x1, y = w * 0.25, w * 0.75, h * 0.72
            c.create_rectangle(x0, y, x1, y + 8, outline="", fill="#2A2E33")
            c.create_rectangle(x0, y, x0 + (x1 - x0) * frac, y + 8, outline="", fill=col)
            if state.closure_target is not None and seg.phase in ("ramps",):
                self._draw_closure(state.closure_target, w, h, small)
        c.create_text(30, h - 30, anchor="sw", text=status_text,
                      fill=COLORS["status"] if ok else COLORS["warn"], font=("Consolas", small))
        c.create_text(w - 30, h - 30, anchor="se", fill=COLORS["status"], font=("Segoe UI", small),
                      text="ESC twice = abort")

    def _draw_closure(self, closure: float, w: int, h: int, small: int) -> None:
        c = self.canvas
        x0, x1, y = w * 0.2, w * 0.8, h * 0.82
        c.create_rectangle(x0, y, x1, y + 24, outline="#555", width=2)
        for frac, label in ((0.0, "open"), (0.5, "half"), (1.0, "closed")):
            x = x0 + (x1 - x0) * frac
            c.create_line(x, y - 6, x, y + 30, fill="#555", width=2)
            c.create_text(x, y + 44, text=label, fill=COLORS["rest"], font=("Segoe UI", small))
        x = x0 + (x1 - x0) * closure
        c.create_rectangle(x - 10, y - 10, x + 10, y + 34, outline="", fill=COLORS["active"])


def _mmss(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"
