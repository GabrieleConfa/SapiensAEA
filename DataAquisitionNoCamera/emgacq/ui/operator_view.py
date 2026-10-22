"""Operator view: live filtered signals, RMS bars, channel flags, link status, session controls,
quality-check review and the event log. Never shown to the subject."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QProgressBar, QPushButton,
                               QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from emgacq.presentation import status_line
from emgacq.session import SessionController, State
from emgacq.signals import ChannelMonitor

SCALES_UV = [("±50 µV", 50), ("±100 µV", 100), ("±200 µV", 200), ("±500 µV", 500),
             ("±1000 µV", 1000), ("auto", None)]
OK_COLOR, BAD_COLOR, WARN_COLOR = "#3FB950", "#F85149", "#D29922"


def _mmss(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


class OperatorWindow(QMainWindow):
    def __init__(self, ctrl: SessionController, monitor: ChannelMonitor, positions: list[str],
                 actions):
        super().__init__()
        self.ctrl = ctrl
        self.recorder = ctrl.recorder
        self.monitor = monitor
        self.actions = actions
        self.channels = ctrl.channel_names
        self.positions = positions
        self.fs = ctrl.source.fs
        self._rate = (time.monotonic(), 0, 0.0)
        self._n_events_shown = 0
        self._qc_shown = None
        self.setWindowTitle("EMG acquisition - operator")
        self.resize(1500, 900)

        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)

        # left: banner + signals
        left = QVBoxLayout()
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setMinimumHeight(34)
        f = self.banner.font()
        f.setPointSize(12)
        f.setBold(True)
        self.banner.setFont(f)
        left.addWidget(self.banner)
        top = QHBoxLayout()
        top.addWidget(QLabel("Signals, 20-120 Hz band-pass + 50/100 Hz notch (display only; "
                             "the saved data are raw)"))
        top.addStretch(1)
        top.addWidget(QLabel("Scale"))
        self.scale = QComboBox()
        for text, _ in SCALES_UV:
            self.scale.addItem(text)
        self.scale.setCurrentIndex(2)
        self.scale.currentIndexChanged.connect(self._apply_scale)
        top.addWidget(self.scale)
        left.addLayout(top)
        self.plots_widget = pg.GraphicsLayoutWidget()
        self.curves, self.plots = [], []
        n = len(self.channels)
        self.t_axis = None
        for i, name in enumerate(self.channels):
            pl = self.plots_widget.addPlot(row=i, col=0)
            pl.setMenuEnabled(False)
            pl.setMouseEnabled(x=False, y=False)
            pl.hideButtons()
            pl.showGrid(x=True, y=False, alpha=0.2)
            label = name + (f"  {positions[i]}" if i < len(positions) and positions[i] else "")
            pl.setLabel("left", label)
            pl.getAxis("left").setWidth(70)
            if i < n - 1:
                pl.getAxis("bottom").setStyle(showValues=False)
            else:
                pl.setLabel("bottom", "time (s)")
            if self.plots:
                pl.setXLink(self.plots[0])
            self.curves.append(pl.plot(pen=pg.mkPen(pg.intColor(i, hues=max(n, 8)), width=1)))
            self.plots.append(pl)
        left.addWidget(self.plots_widget, 1)
        lay.addLayout(left, 1)

        # right: status and controls
        right = QVBoxLayout()
        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(500)
        lay.addWidget(right_w)

        link = QGroupBox("Link")
        g = QGridLayout(link)
        self.lbl_rate = QLabel("-")
        self.lbl_lost = QLabel("-")
        self.lbl_gaps = QLabel("-")
        for r, (k, w) in enumerate((("Samples/s", self.lbl_rate), ("Lost packets", self.lbl_lost),
                                    ("Gaps / reconnections", self.lbl_gaps))):
            g.addWidget(QLabel(k), r, 0)
            g.addWidget(w, r, 1)
        right.addWidget(link)

        chbox = QGroupBox("Channels (RMS of the last 250 ms, filtered)")
        v = QVBoxLayout(chbox)
        self.ch_table = QTableWidget(n, 3)
        self.ch_table.setHorizontalHeaderLabels(["Channel", "RMS", "Flag"])
        self.ch_table.verticalHeader().setVisible(False)
        self.ch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.ch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ch_table.setSelectionMode(QTableWidget.NoSelection)
        self.bars = []
        for i, name in enumerate(self.channels):
            pos = positions[i] if i < len(positions) else ""
            self.ch_table.setItem(i, 0, QTableWidgetItem(name + (f" ({pos})" if pos else "")))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(True)
            bar.setFormat("-")
            self.ch_table.setCellWidget(i, 1, bar)
            self.bars.append(bar)
            self.ch_table.setItem(i, 2, QTableWidgetItem("-"))
        self.ch_table.setFixedHeight(28 + 30 * n)
        v.addWidget(self.ch_table)
        right.addWidget(chbox)

        sess = QGroupBox("Session")
        g = QGridLayout(sess)
        self.lbl_state = QLabel("-")
        self.lbl_stage = QLabel("-")
        self.lbl_trial = QLabel("-")
        self.lbl_time = QLabel("-")
        self.lbl_bad = QLabel("-")
        for r, (k, w) in enumerate((("State", self.lbl_state), ("Stage / step", self.lbl_stage),
                                    ("Trial", self.lbl_trial), ("Elapsed / left", self.lbl_time),
                                    ("Bad trials", self.lbl_bad))):
            g.addWidget(QLabel(k), r, 0)
            g.addWidget(w, r, 1)
        self.btn_start = QPushButton("Start session")
        self.btn_pause = QPushButton("Pause  [P]")
        self.btn_continue = QPushButton("Continue  [Space]")
        self.btn_bad = QPushButton("Flag last trial BAD  [B]")
        self.btn_reconnect = QPushButton("Reconnect")
        self.btn_abort = QPushButton("Abort  [Esc]")
        self.btn_start.clicked.connect(actions.start)
        self.btn_pause.clicked.connect(actions.toggle_pause)
        self.btn_continue.clicked.connect(actions.continue_)
        self.btn_bad.clicked.connect(actions.flag_bad)
        self.btn_reconnect.clicked.connect(actions.reconnect)
        self.btn_abort.clicked.connect(actions.abort)
        buttons = [self.btn_start, self.btn_pause, self.btn_continue, self.btn_bad,
                   self.btn_reconnect, self.btn_abort]
        for i, b in enumerate(buttons):
            b.setMinimumHeight(34)
            b.setFocusPolicy(Qt.NoFocus)
            g.addWidget(b, 5 + i // 2, i % 2)
        self.btn_bad.setStyleSheet("font-weight: bold;")
        if ctrl.free:
            self.btn_start.setText("Start recording")
            self.btn_abort.setText("Stop recording  [Esc]")
            for b in (self.btn_pause, self.btn_continue, self.btn_bad):
                b.setVisible(False)
        right.addWidget(sess)

        self.qc_box = QGroupBox("Quality check")
        v = QVBoxLayout(self.qc_box)
        self.qc_result = QLabel("")
        self.qc_result.setWordWrap(True)
        v.addWidget(self.qc_result)
        self.qc_table = QTableWidget(n, 6)
        self.qc_table.setHorizontalHeaderLabels(["Ch", "50 Hz rest", "rest µV", "fist/rest",
                                                 "flat/railed", "ok"])
        self.qc_table.verticalHeader().setVisible(False)
        self.qc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.qc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.qc_table.setFixedHeight(28 + 30 * n)
        v.addWidget(self.qc_table)
        self.qc_note = QLineEdit()
        self.qc_note.setPlaceholderText("Note: click to type (required to proceed after a FAIL); "
                                        "Enter = Proceed, Esc = stop typing")
        v.addWidget(self.qc_note)
        h = QHBoxLayout()
        self.btn_qc_proceed = QPushButton("Proceed  [Space]")
        self.btn_qc_repeat = QPushButton("Repeat the check  [R]")
        self.btn_qc_proceed.clicked.connect(lambda: actions.proceed_qc(self.qc_note.text()))
        self.btn_qc_repeat.clicked.connect(lambda: actions.repeat_qc(self.qc_note.text()))
        for b in (self.btn_qc_proceed, self.btn_qc_repeat):
            b.setFocusPolicy(Qt.NoFocus)
            h.addWidget(b)
        v.addLayout(h)
        self.qc_box.setVisible(False)
        right.addWidget(self.qc_box)

        right.addWidget(QLabel("Events"))
        self.log = QListWidget()
        self.log.setFont(QFont("Consolas", 9))
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.log, 1)

        self._apply_scale()

    # ---------------------------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        if self.qc_note.hasFocus() and event.key() not in (Qt.Key_Escape,):
            return super().keyPressEvent(event)
        if not self.actions.handle_key(event):
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.actions.request_close():
            event.accept()
        else:
            event.ignore()

    def _apply_scale(self) -> None:
        val = SCALES_UV[self.scale.currentIndex()][1]
        for pl in self.plots:
            if val is None:
                pl.enableAutoRange(axis="y")
            else:
                pl.disableAutoRange(axis="y")
                pl.setYRange(-val, val, padding=0)

    def set_banner(self, text: str, color: str | None) -> None:
        self.banner.setText(text)
        self.banner.setStyleSheet(f"background: {color}; color: white; padding: 6px;" if color else "")

    # ---------------------------------------------------------------------------------------
    def refresh(self, now: float) -> None:
        ctrl = self.ctrl
        live = self.recorder.take_live()
        self.monitor.update(live.data_uv)
        disp = self.monitor.display.last(self.monitor.display.capacity)
        if len(disp):
            t = (np.arange(len(disp)) - len(disp)) / self.fs
            for c, curve in enumerate(self.curves):
                curve.setData(t, disp[:, c])
            self.plots[0].setXRange(-self.monitor.display.capacity / self.fs, 0, padding=0)

        # channels
        rms = self.monitor.rms()
        flat, railed = self.monitor.flags()
        scale = SCALES_UV[self.scale.currentIndex()][1] or 200
        for c, bar in enumerate(self.bars):
            v = float(rms[c]) if c < len(rms) else 0.0
            bar.setValue(int(min(1000, 1000 * v / scale)))
            bar.setFormat(f"{v:.1f} µV")
            item = self.ch_table.item(c, 2)
            if railed[c]:
                item.setText("RAILED")
                item.setForeground(QColor(BAD_COLOR))
            elif flat[c]:
                item.setText("FLAT")
                item.setForeground(QColor(BAD_COLOR))
            else:
                item.setText("ok")
                item.setForeground(QColor(OK_COLOR))

        # link
        t0, n0, rate = self._rate
        if now - t0 >= 1.0:
            rate = (self.recorder.received - n0) / (now - t0)
            self._rate = (now, self.recorder.received, rate)
        self.lbl_rate.setText(f"{rate:.0f}  (nominal {self.fs:g})")
        p = self.recorder.packets
        self.lbl_lost.setText(f"{p.lost}  ({p.loss_pct:.2f}%)")
        self.lbl_lost.setStyleSheet(f"color: {BAD_COLOR};" if p.loss_pct > 1 else "")
        self.lbl_gaps.setText(f"{len(ctrl.gaps) + int(ctrl.gap_active)} / {ctrl.reconnects}")

        # session
        step = ctrl.step
        self.lbl_state.setText(ctrl.state.value + (f" ({ctrl.pause_reason})" if ctrl.state == State.PAUSED else ""))
        if step is not None and ctrl.state not in (State.IDLE,):
            self.lbl_stage.setText(f"{step.stage} / {step.kind}" + (f" {step.gesture}" if step.gesture else ""))
            if ctrl.free:
                self.lbl_trial.setText("- (free recording)")
            elif step.trial_id:
                in_block = (step.trial_id - 1) % ctrl.trials_per_block + 1
                self.lbl_trial.setText(f"block {step.block}, trial {in_block}/{ctrl.trials_per_block} "
                                       f"(id {step.trial_id}), {step.gesture}")
            else:
                self.lbl_trial.setText(f"last trial {ctrl.last_trial or '-'} of {ctrl.n_trials}")
        else:
            self.lbl_stage.setText("not started")
            self.lbl_trial.setText(f"0 of {ctrl.n_trials}")
        left = "until stopped" if ctrl.free and not ctrl.duration_s else f"~{_mmss(ctrl.remaining_s(now))}"
        self.lbl_time.setText(f"{_mmss(ctrl.elapsed_s(now))} / {left}")
        self.lbl_bad.setText(", ".join(map(str, sorted(ctrl.bad_trials))) or "none")

        running = ctrl.state in (State.RUNNING, State.PAUSED)
        self.btn_start.setEnabled(ctrl.state == State.IDLE)
        self.btn_pause.setEnabled(ctrl.can_pause() or ctrl.can_resume())
        self.btn_pause.setText("Resume  [P]" if ctrl.state == State.PAUSED else "Pause  [P]")
        if step is not None and step.kind == "block_pause" and ctrl.state == State.RUNNING:
            left = ctrl.block_pause_left(now)
            self.btn_continue.setText("Start next block  [Space]" + (f" ({left:.0f} s)" if left > 0 else ""))
            self.btn_continue.setEnabled(ctrl.can_continue_block_pause(now))
        else:
            self.btn_continue.setText("Continue  [Space]")
            self.btn_continue.setEnabled(False)
        self.btn_bad.setEnabled(running and ctrl.last_trial > 0)
        self.btn_reconnect.setEnabled(ctrl.state != State.DONE and not self.actions.reconnecting)
        self.btn_abort.setEnabled(ctrl.state in (State.IDLE, State.RUNNING, State.PAUSED))

        self._refresh_qc()
        self._refresh_banner(now)
        rows = ctrl.events.rows
        for row in rows[self._n_events_shown:]:
            self.log.addItem(f"{row['wall_time'][11:19]} {row['event']:16} "
                             f"{('t' + str(row['trial_id'])) if row['trial_id'] else '':4} {row['detail']}")
        if len(rows) > self._n_events_shown:
            self._n_events_shown = len(rows)
            self.log.scrollToBottom()
        self.setWindowTitle(f"EMG acquisition - operator   |   {status_line(ctrl, now)}")

    def _refresh_banner(self, now: float) -> None:
        ctrl = self.ctrl
        if ctrl.gap_active:
            self.set_banner(f"NO DATA for {self.recorder.seconds_since_last_sample():.0f} s - protocol "
                            "paused. Check the dongle, the board switch and the battery; then Reconnect.",
                            BAD_COLOR)
        elif self.actions.message:
            self.set_banner(self.actions.message, self.actions.message_color)
        elif ctrl.state == State.IDLE and ctrl.free:
            self.set_banner("Free recording: check the signals, then Start recording.", None)
        elif ctrl.state == State.IDLE:
            self.set_banner("Check the signals, then Start session. The subject view shows 'Please wait'.",
                            None)
        elif ctrl.free and ctrl.state == State.RUNNING:
            self.set_banner("Free recording: the raw stream is being saved, no labels. "
                            "Stop recording [Esc] ends it.", OK_COLOR)
        elif ctrl.state == State.PAUSED:
            self.set_banner("Paused. Resume [P] restarts the interrupted step.", WARN_COLOR)
        elif ctrl.step is not None and ctrl.step.kind == "qc_review":
            self.set_banner("Quality check finished: review the table below, then Proceed [Space] "
                            "or Repeat the check [R].", WARN_COLOR)
        elif ctrl.step is not None and ctrl.step.kind == "block_pause":
            left = ctrl.block_pause_left(now)
            self.set_banner("Pause between blocks. " + (f"Minimum pause: {left:.0f} s left." if left > 0
                            else "Press Space when the subject is ready."), WARN_COLOR)
        elif ctrl.state == State.DONE:
            self.set_banner(f"Session {'aborted' if ctrl.aborted else 'complete'}. Files in "
                            f"{ctrl.folder}", OK_COLOR if not ctrl.aborted else WARN_COLOR)
        else:
            self.set_banner("", None)

    def _refresh_qc(self) -> None:
        ctrl = self.ctrl
        reviewing = ctrl.state == State.RUNNING and ctrl.step is not None and ctrl.step.kind == "qc_review"
        self.qc_box.setVisible(reviewing or (ctrl.qc_report is not None and ctrl.state == State.RUNNING
                                             and ctrl.step.kind.startswith("qc")))
        rep = ctrl.qc_report
        self.btn_qc_proceed.setEnabled(reviewing and rep is not None)
        self.btn_qc_repeat.setEnabled(reviewing)
        if rep is None:
            self.qc_result.setText("Running..." if ctrl.step is not None and ctrl.step.kind.startswith("qc")
                                   else "")
            return
        if self._qc_shown is rep:
            return
        self._qc_shown = rep
        color = {"PASS": OK_COLOR, "NOT_JUDGED": WARN_COLOR}.get(rep["result"], BAD_COLOR)
        self.qc_result.setText(f"<b style='color:{color}'>{rep['result']}</b>"
                               f" &nbsp; packet loss {rep.get('packet_loss_pct', 0):.2f}%<br>"
                               + "<br>".join(rep["reasons"]))
        for r, ch in enumerate(rep.get("channels", [])):
            vals = [ch["channel"], f"{ch['line_share_48_52_rest']:.2f}", f"{ch['rest_rms_uv']:.1f}",
                    f"{ch['rms_ratio_fist_rest']:.1f}",
                    ("flat " if ch["flat"] else "") + (f"{100 * ch['railed_fraction']:.1f}%" if ch["railed_fraction"] else "") or "-",
                    "ok" if ch["ok"] else ", ".join(ch["issues"])]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 5:
                    item.setForeground(QColor(OK_COLOR if ch["ok"] else BAD_COLOR))
                self.qc_table.setItem(r, c, item)
