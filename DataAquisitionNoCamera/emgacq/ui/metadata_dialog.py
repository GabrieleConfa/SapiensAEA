"""Session metadata form, shown before connecting. The result goes into session.json."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

SUBJECT_RE = re.compile(r"^[A-Za-z0-9-]+$")


class MetadataDialog(QDialog):
    def __init__(self, initial: dict, source_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New session")
        self.setMinimumWidth(560)
        form = QFormLayout()

        self.subject = QLineEdit(initial.get("subject") or "")
        self.subject.setPlaceholderText("e.g. A or S01 (letters, digits, '-')")
        self.session = QSpinBox(minimum=1, maximum=99, value=int(initial.get("session") or 1))
        self.day = QSpinBox(minimum=1, maximum=365, value=int(initial.get("day") or 1))
        self.role = QComboBox()
        self.role.addItems(["train", "test"])
        self.role.setCurrentText(initial.get("role") or "train")
        self.arm = QComboBox()
        self.arm.addItems(["right", "left"])
        self.arm.setCurrentText(initial.get("arm") or "right")
        self.band = QDoubleSpinBox(minimum=0.0, maximum=100.0, decimals=1, suffix=" %")
        self.band.setValue(initial.get("band_position_pct") or 0.0)
        self.forearm = QDoubleSpinBox(minimum=0.0, maximum=60.0, decimals=1, suffix=" cm")
        self.forearm.setValue(initial.get("forearm_length_cm") or 0.0)
        self.electrode = QLineEdit(initial.get("electrode_type") or "")
        self.electrode.setPlaceholderText("e.g. Ag/AgCl gel, 10 mm, 20 mm spacing")

        cmap = initial["channel_map"]
        self.table = QTableWidget(len(cmap), 2)
        self.table.setHorizontalHeaderLabels(["Channel", "Position (muscle / location)"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for r, ch in enumerate(cmap):
            item = QTableWidgetItem(f"ch{ch['channel']}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, item)
            self.table.setItem(r, 1, QTableWidgetItem(ch.get("position", "")))
        self.table.setMinimumHeight(min(300, 30 + 28 * len(cmap)))
        self._cmap = cmap
        self.notes = QPlainTextEdit(initial.get("notes") or "")
        self.notes.setFixedHeight(70)

        form.addRow("Subject ID", self.subject)
        form.addRow("Session number", self.session)
        form.addRow("Day", self.day)
        form.addRow("Role", self.role)
        form.addRow("Arm", self.arm)
        form.addRow("Band position (from the elbow)", self.band)
        form.addRow("Forearm length", self.forearm)
        form.addRow("Electrode type", self.electrode)
        form.addRow("Channel map", self.table)
        form.addRow("Notes", self.notes)

        info = QLabel(f"<b>Source:</b> {source_text}")
        info.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Connect")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(info)
        lay.addWidget(buttons)
        self.result_data: dict | None = None

    def _accept(self) -> None:
        subject = self.subject.text().strip()
        if not SUBJECT_RE.match(subject):
            QMessageBox.warning(self, "Subject ID", "Enter a subject ID: letters, digits and '-' only.")
            return
        cmap = [{"channel": ch["channel"], "column": ch["column"],
                 "position": (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()}
                for r, ch in enumerate(self._cmap)]
        self.result_data = {
            "subject": subject,
            "session": self.session.value(),
            "day": self.day.value(),
            "role": self.role.currentText(),
            "arm": self.arm.currentText(),
            "band_position_pct": self.band.value() or None,
            "forearm_length_cm": self.forearm.value() or None,
            "channel_map": cmap,
            "electrode_type": self.electrode.text().strip(),
            "notes": self.notes.toPlainText().strip(),
        }
        self.accept()
