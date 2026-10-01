"""Subject view: only the task - gesture name, colour, optional image, countdown bar.

In single-screen mode a thin status strip for the operator is drawn at the bottom (text only,
no signals).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from emgacq.presentation import SubjectScreen


class SubjectView(QWidget):
    def __init__(self, colors: dict, font_family: str, assets_dir: Path, status_strip: bool,
                 on_key=None):
        super().__init__()
        self.setWindowTitle("Task")
        self.colors = colors
        self.font_family = font_family
        self.status_strip = status_strip
        self.on_key = on_key
        self.screen_state = SubjectScreen("Please wait", "", "PAUSE", None, None)
        self.status_text = ""
        self._images: dict[str, QPixmap | None] = {}
        self._assets = assets_dir
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def _image(self, key: str | None) -> QPixmap | None:
        if key is None:
            return None
        if key not in self._images:
            path = self._assets / f"{key}.png"
            pm = QPixmap(str(path)) if path.exists() else None
            self._images[key] = pm if pm is not None and not pm.isNull() else None
        return self._images[key]

    def set_screen(self, screen: SubjectScreen, status_text: str = "") -> None:
        if screen != self.screen_state or status_text != self.status_text:
            self.screen_state = screen
            self.status_text = status_text
            self.update()

    def keyPressEvent(self, event) -> None:
        if self.on_key is not None and self.on_key(event):
            return
        super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:
        s = self.screen_state
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(self.colors["background"]))
        col = QColor(self.colors.get(s.color_key, "#E0E0E0"))

        strip_h = int(h * 0.05) if self.status_strip else 0
        area_h = h - strip_h
        img = self._image(s.image_key)
        title_y = area_h * 0.42
        if img is not None:
            box = min(w * 0.5, area_h * 0.38)
            scaled = img.scaled(int(box), int(box), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(int((w - scaled.width()) / 2), int(area_h * 0.06), scaled)
            title_y = area_h * 0.06 + box + area_h * 0.12

        big = QFont(self.font_family)
        big.setPixelSize(max(24, int(min(area_h * 0.16, w * 0.09))))
        big.setBold(True)
        p.setFont(big)
        p.setPen(col)
        p.drawText(QRectF(0, title_y - area_h * 0.12, w, area_h * 0.24), Qt.AlignCenter, s.title)

        if s.subtitle:
            sub = QFont(self.font_family)
            sub.setPixelSize(max(14, int(area_h * 0.045)))
            p.setFont(sub)
            p.setPen(QColor("#B8C0C8"))
            p.drawText(QRectF(w * 0.05, title_y + area_h * 0.11, w * 0.9, area_h * 0.1),
                       Qt.AlignCenter | Qt.TextWordWrap, s.subtitle)

        if s.progress is not None:
            x0, x1 = w * 0.2, w * 0.8
            y, bh = area_h * 0.82, max(8, area_h * 0.022)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#2A3036"))
            p.drawRoundedRect(QRectF(x0, y, x1 - x0, bh), bh / 2, bh / 2)
            p.setBrush(col)
            p.drawRoundedRect(QRectF(x0, y, (x1 - x0) * s.progress, bh), bh / 2, bh / 2)

        if self.status_strip:
            p.fillRect(QRectF(0, area_h, w, strip_h), QColor("#0B0D10"))
            f = QFont("Consolas")
            f.setPixelSize(max(10, int(strip_h * 0.38)))
            p.setFont(f)
            p.setPen(QColor("#7D8790"))
            p.drawText(QRectF(12, area_h, w - 24, strip_h), Qt.AlignVCenter | Qt.AlignLeft,
                       self.status_text)
            p.drawText(QRectF(12, area_h, w - 24, strip_h), Qt.AlignVCenter | Qt.AlignRight,
                       "B bad trial   P pause   SPACE continue   O operator view   ESC abort")
        p.end()
