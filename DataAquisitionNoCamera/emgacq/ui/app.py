"""GUI entry point: metadata form -> connect -> operator view + subject view.

Two screens: subject view full screen on the second monitor, operator view on the laptop.
One screen: the operator view is shown before the start, during the quality-check review, on a
data gap and at the end; during the protocol the subject view fills the screen, with a thin text
status strip for the operator (no signals). The operator view can be brought back with O while
the protocol waits (paused, between blocks, quality-check review).
Free recording (--free): no cues; the operator view stays visible, Esc stops the recording.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMessageBox, QWidget

from emgacq.config import Config
from emgacq.presentation import status_line, subject_screen
from emgacq.recorder import Recorder
from emgacq.session import SessionController, State
from emgacq.signals import ChannelMonitor
from emgacq.sources.base import GUI_HINT, SampleSource, SourceError
from emgacq.ui.metadata_dialog import MetadataDialog
from emgacq.ui.operator_view import BAD_COLOR, OK_COLOR, WARN_COLOR, OperatorWindow
from emgacq.ui.subject_view import SubjectView


class Actions:
    """Operator commands, shared by the buttons, the keys of both windows and the auto mode."""

    def __init__(self, app: QApplication, ctrl: SessionController, single: bool, auto: bool):
        self.app = app
        self.ctrl = ctrl
        self.single = single
        self.auto = auto
        self.operator: OperatorWindow | None = None
        self.subject: SubjectView | None = None
        self.message = ""
        self.message_color: str | None = None
        self._message_until = 0.0
        self.reconnecting = False
        self._reconnect_result: tuple[bool, str] | None = None
        self.show_operator = False      # single screen: operator view brought back with O
        self.quit_reason: str | None = None
        self._done_at: float | None = None
        self._operator_visible: bool | None = None

    def say(self, text: str, color: str | None = WARN_COLOR, seconds: float = 6.0) -> None:
        self.message, self.message_color = text, color
        self._message_until = time.monotonic() + seconds

    # -- commands -----------------------------------------------------------------------------
    def start(self) -> None:
        if self.ctrl.state != State.IDLE:
            return
        if self.ctrl.gap_active:
            self.say("No data from the board: cannot start.", BAD_COLOR)
            return
        folder = self.ctrl.start()
        self.say(f"Recording to {folder}", OK_COLOR)
        if self.subject is not None:
            self.subject.activateWindow()

    def toggle_pause(self) -> None:
        if self.ctrl.can_resume():
            self.ctrl.resume()
        elif self.ctrl.can_pause():
            self.ctrl.pause("operator")
        elif self.ctrl.state == State.PAUSED and self.ctrl.gap_active:
            self.say("Cannot resume: no data from the board.", BAD_COLOR)

    def continue_(self) -> None:
        if not self.ctrl.continue_block_pause():
            left = self.ctrl.block_pause_left()
            if left > 0:
                self.say(f"Minimum pause: {left:.0f} s left.")

    def flag_bad(self) -> None:
        r = self.ctrl.flag_last_trial_bad()
        if r is None:
            self.say("No trial to flag yet.")
        else:
            t, bad = r
            g = self.ctrl.order["trials"][t - 1]["gesture"]
            self.say(f"Trial {t} ({g}) flagged BAD" if bad else f"Trial {t} ({g}): BAD flag removed",
                     BAD_COLOR if bad else OK_COLOR)

    def reconnect(self) -> None:
        if self.reconnecting:
            return
        self.reconnecting = True
        self.say("Reconnecting...", WARN_COLOR, 30)

        def work():
            try:
                self.ctrl.reconnect()
                self._reconnect_result = (True, "Reconnected. Resume with P when the signals look right.")
            except Exception as e:
                self._reconnect_result = (False, f"Reconnection failed: {e}")
        threading.Thread(target=work, name="reconnect", daemon=True).start()

    def abort(self, confirm: bool = True) -> None:
        if self.ctrl.state == State.IDLE:
            self.quit_reason = "closed before start"
            return
        if self.ctrl.state not in (State.RUNNING, State.PAUSED):
            return
        if self.ctrl.free:
            if not confirm or self._confirm("Stop the recording?", "The files are closed and kept."):
                self.ctrl.stop("operator")
            return
        if confirm and not self._confirm("Abort the session?",
                                         "The data recorded so far are kept and the session is marked aborted."):
            return
        self.ctrl.abort("operator")

    def proceed_qc(self, note: str) -> None:
        try:
            self.ctrl.proceed_qc(note)
        except (ValueError, RuntimeError) as e:
            self.say(str(e), BAD_COLOR)

    def repeat_qc(self, note: str) -> None:
        try:
            self.ctrl.repeat_qc(note)
        except RuntimeError as e:
            self.say(str(e), BAD_COLOR)

    def reviewing_qc(self) -> bool:
        step = self.ctrl.step
        return self.ctrl.state == State.RUNNING and step is not None and step.kind == "qc_review"

    def _qc_note(self) -> str:
        return self.operator.qc_note.text() if self.operator is not None else ""

    def handle_key(self, event) -> bool:
        key = event.key()
        if key == Qt.Key_P:
            self.toggle_pause()
        elif key == Qt.Key_B:
            self.flag_bad()
        elif key == Qt.Key_Space:
            if self.reviewing_qc():
                self.proceed_qc(self._qc_note())
            else:
                self.continue_()
        elif key == Qt.Key_R and self.reviewing_qc():
            self.repeat_qc(self._qc_note())
        elif key == Qt.Key_Escape:
            self.abort()
        elif key == Qt.Key_O and self.single:
            waiting = (self.ctrl.state == State.PAUSED or
                       (self.ctrl.step is not None and self.ctrl.step.wait_operator))
            if waiting or self.show_operator:
                self.show_operator = not self.show_operator
        else:
            return False
        return True

    def request_close(self) -> bool:
        """A window is being closed: abort a running session (after confirmation) and quit."""
        if self.ctrl.state in (State.RUNNING, State.PAUSED):
            if not self._confirm("Stop the recording and quit?" if self.ctrl.free else
                                 "Abort the session and quit?", "The data recorded so far are kept."):
                return False
            if not self.ctrl.stop("window closed"):
                self.ctrl.abort("window closed")
        self.quit_reason = "window closed"
        return True

    def _confirm(self, title: str, text: str) -> bool:
        parent = self.operator if self.operator is not None and self.operator.isVisible() else self.subject
        return QMessageBox.question(parent, title, text) == QMessageBox.Yes

    # -- periodic -------------------------------------------------------------------------------
    def tick(self) -> None:
        now = time.monotonic()
        ctrl = self.ctrl
        try:
            ctrl.tick(now)
        except Exception:
            traceback.print_exc()
            ctrl.events.log("ERROR", detail=traceback.format_exc(limit=1).strip().splitlines()[-1])
            self.say("Internal error, see the console. The session is being aborted.", BAD_COLOR, 60)
            ctrl.abort("internal error")
        if self._reconnect_result is not None:
            ok, text = self._reconnect_result
            self._reconnect_result = None
            self.reconnecting = False
            self.say(text, OK_COLOR if ok else BAD_COLOR, 10)
        if self.message and now > self._message_until:
            self.message = ""
        if self.auto:
            self._auto_step(now)
        if self.subject is not None:
            self.subject.set_screen(subject_screen(ctrl, now),
                                    status_line(ctrl, now) if self.single else "")
        if self.single:
            self._single_screen_layout()
        if ctrl.state == State.DONE:
            if self._done_at is None:
                self._done_at = now
                print(ctrl.summary_text)
            if self.auto and now - self._done_at > 3.0:
                self.quit_reason = "auto mode finished"
        if self.quit_reason and ctrl.state in (State.RUNNING, State.PAUSED):
            ctrl.abort(self.quit_reason)    # the app quits once the files are closed (finish_check)

    def _single_screen_layout(self) -> None:
        ctrl = self.ctrl
        need = (ctrl.free or ctrl.state in (State.IDLE, State.DONE) or ctrl.gap_active or self.show_operator
                or (ctrl.step is not None and ctrl.step.kind == "qc_review" and ctrl.state == State.RUNNING))
        if ctrl.state == State.RUNNING and ctrl.step is not None and not ctrl.step.wait_operator:
            self.show_operator = False
        if need != self._operator_visible:
            self._operator_visible = need
            if need:
                self.operator.show()
                self.operator.raise_()
                self.operator.activateWindow()
            else:
                self.operator.hide()
                self.subject.raise_()
                self.subject.activateWindow()

    def _auto_step(self, now: float) -> None:
        ctrl = self.ctrl
        if ctrl.state == State.IDLE and now - getattr(self, "_t_open", now) > 2.0 and not ctrl.gap_active:
            self.start()
        elif ctrl.state == State.RUNNING and ctrl.step.kind == "qc_review" and ctrl.qc_report:
            note = "auto demo run" if ctrl.qc_report["result"] == "FAIL" else ""
            ctrl.proceed_qc(note)
        elif ctrl.state == State.RUNNING and ctrl.step.kind == "block_pause":
            ctrl.continue_block_pause()
        elif ctrl.state == State.PAUSED and ctrl.can_resume():
            ctrl.resume()


class OperatorKeys(QObject):
    """Application-wide key filter for the operator and subject windows.

    The operator keys work whatever widget has the focus (tables, lists and the scale box would
    otherwise take Space and the letters). A text box keeps its keys while it is being typed in:
    Enter there = Proceed, Esc leaves it. Dialogs (e.g. the abort confirmation) are not touched.
    """

    def __init__(self, actions: Actions, windows: tuple[QWidget, ...]):
        super().__init__()
        self.actions = actions
        self.windows = windows

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.KeyPress or not isinstance(obj, QWidget):
            return False
        if obj.window() not in self.windows or event.isAutoRepeat():
            return False
        if isinstance(obj, QLineEdit):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.actions.reviewing_qc():
                self.actions.proceed_qc(obj.text())
                obj.clearFocus()
                return True
            if event.key() == Qt.Key_Escape:
                obj.clearFocus()
                return True
            return False
        return self.actions.handle_key(event)


def _pick_subject_screen(app: QApplication, setting, force_single: bool):
    """(screen for the subject view or None for single-screen mode, description)."""
    screens = app.screens()
    primary = app.primaryScreen()
    if force_single or setting == "single":
        return None, "single-screen mode (forced)"
    if isinstance(setting, int):
        if 0 <= setting < len(screens):
            return screens[setting], f"subject view on screen {setting}"
        return None, f"screen {setting} not present: single-screen mode"
    others = [s for s in screens if s is not primary]
    if others:
        return others[0], f"subject view on the second screen ({others[0].name()})"
    return None, "one screen found: single-screen mode"


def run_gui(cfg: Config, args, source: SampleSource, kind: str) -> int:
    from emgacq.__main__ import metadata_from_args

    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=False, background="#111417", foreground="#C9D1D9")

    meta = metadata_from_args(args, cfg, source.channels)
    where = getattr(source, "serial_port", None) or getattr(source, "url", None) or ""
    source_text = f"{kind}" + (f" on {where}" if where else "")
    if kind == "cyton":
        source_text += f". {GUI_HINT}"
    if args.free:
        source_text += ("<br><b>Free recording</b> (no protocol), "
                        + (f"{args.duration:g} s." if args.duration else "until stopped."))
    if not args.auto:
        dlg = MetadataDialog(meta, source_text)
        if dlg.exec() != QDialog.Accepted:
            return 1
        meta = dlg.result_data

    while True:
        try:
            source.connect()
            source.start()
            break
        except SourceError as e:
            source.release()
            box = QMessageBox(QMessageBox.Critical, "Cannot connect", str(e),
                              QMessageBox.Retry | QMessageBox.Cancel)
            if box.exec() != QMessageBox.Retry:
                return 1

    recorder = Recorder(source, float(cfg["board.read_interval_s"]))
    recorder.start()
    ctrl = SessionController(cfg, source, recorder, meta, speed=args.speed, free=args.free,
                             duration_s=args.duration)
    for c in source.command_log:
        ctrl.events.log("BOARD_CONFIG", detail=f"{c['command']!r} -> {c['reply']!r}")
    try:
        screen, mode_text = _pick_subject_screen(app, cfg["display.subject_screen"], args.single_screen)
        single = screen is None and not args.windowed
        ctrl.events.log("DISPLAY", detail=mode_text if not args.windowed else "windowed (testing)")
        actions = Actions(app, ctrl, single=single, auto=args.auto)
        actions._t_open = time.monotonic()
        mon = ChannelMonitor(source.fs, source.n_channels, cfg.section("monitor"),
                             cfg.section("quality"), source.full_scale_uv)
        positions = [c.get("position", "") for c in meta["channel_map"]]
        operator = OperatorWindow(ctrl, mon, positions, actions)
        subject = SubjectView(dict(cfg["display.colors"]), cfg["display.font_family"],
                              cfg.path_of("paths.assets_dir"), status_strip=single,
                              on_key=actions.handle_key)
        subject.closeEvent = lambda e: e.accept() if actions.request_close() else e.ignore()
        actions.operator, actions.subject = operator, subject
        keys = OperatorKeys(actions, (operator, subject))
        app.installEventFilter(keys)

        if args.windowed:
            subject.resize(960, 600)
            subject.setCursor(Qt.ArrowCursor)
            subject.show()
            operator.show()
        elif screen is not None:
            subject.setGeometry(screen.geometry())
            subject.showFullScreen()
            operator.showMaximized()
        else:
            subject.setGeometry(app.primaryScreen().geometry())
            subject.showFullScreen()
            operator.showMaximized()
            operator.raise_()

        signal.signal(signal.SIGINT, lambda *_: setattr(actions, "quit_reason", "Ctrl+C"))
        tick = QTimer()
        tick.timeout.connect(actions.tick)
        tick.start(15)
        draw = QTimer()
        draw.timeout.connect(lambda: operator.refresh(time.monotonic()) if operator.isVisible() else
                             recorder.take_live())
        draw.start(50)

        def finish_check():
            if actions.quit_reason and ctrl.state in (State.IDLE, State.DONE):
                app.quit()
        closer = QTimer()
        closer.timeout.connect(finish_check)
        closer.start(100)
        app.exec()
    finally:
        ctrl.shutdown("application exit")
    if ctrl.folder is not None:
        print(f"files in {ctrl.folder}")
    return 0
