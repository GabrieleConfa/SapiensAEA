"""Camera opening with the settings from config, LED ROI storage, and a synthetic camera."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from common.config import REPO_ROOT, Config

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
LED_ROI_PATH = REPO_ROOT / "config" / "led_roi.json"


class CameraError(RuntimeError):
    pass


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def _decode_fourcc(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def open_camera(cfg: Config, index: int | None = None):
    """Open the webcam with the configured resolution, fps, exposure and white balance.

    Returns (capture, settings) where settings are the values read back from the driver:
    drivers silently ignore what they do not support, so the actual values are what is recorded.
    """
    c = cfg.section("camera")
    idx = int(index if index is not None else cfg.hw("camera.index"))
    cap = cv2.VideoCapture(idx, BACKENDS[c["backend"]])
    if not cap.isOpened():
        raise CameraError(f"cannot open camera {idx} with backend {c['backend']}; "
                          "list cameras with: python -m tools.camera_preview --list")
    cap.set(cv2.CAP_PROP_FOURCC, _fourcc(c["capture_fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
    cap.set(cv2.CAP_PROP_FPS, c["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not c["auto_exposure"]:
        # DirectShow: 0.25 = manual (OpenCV convention); then the exposure value
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, c["exposure"])
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    if not c["auto_white_balance"]:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, c["white_balance_k"])
    else:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    return cap, read_settings(cap, idx, c["backend"])


def read_settings(cap, index, backend) -> dict:
    return {
        "index": index,
        "backend": backend,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
        "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
    }


def list_cameras(backend: str = "dshow", max_index: int = 6) -> list[dict]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, BACKENDS[backend])
        if cap.isOpened():
            ok, frame = cap.read()
            found.append({"index": i, "ok": bool(ok),
                          "shape": None if frame is None else list(frame.shape)})
        cap.release()
    return found


# -- LED region of interest ---------------------------------------------------------------------
@dataclass
class LedRoi:
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def brightness(self, frame: np.ndarray) -> float:
        """Mean of the brightest channel inside the ROI (LEDs saturate one colour first)."""
        patch = self.crop(frame)
        if patch.size == 0:
            return float("nan")
        return float(patch.reshape(-1, patch.shape[-1]).mean(axis=0).max())


def save_led_roi(roi: LedRoi, camera_index) -> None:
    LED_ROI_PATH.write_text(json.dumps({"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h,
                                        "camera_index": camera_index,
                                        "set_at": datetime.now().isoformat(timespec="seconds")},
                                       indent=2), encoding="utf-8")


def load_led_roi() -> LedRoi | None:
    if not LED_ROI_PATH.exists():
        return None
    d = json.loads(LED_ROI_PATH.read_text(encoding="utf-8"))
    return LedRoi(d["x"], d["y"], d["w"], d["h"])


# -- synthetic camera for simulator runs ---------------------------------------------------------
class SyntheticCamera:
    """Minimal cv2.VideoCapture stand-in: grey frames paced at `fps`, with a white LED square drawn
    inside `roi` whenever `led_state()` is True. Lets the whole recording chain, including LED
    detection in the video, run with the GIGA simulator."""

    def __init__(self, width: int, height: int, fps: float, roi: LedRoi,
                 led_state: Callable[[], bool]):
        self.width, self.height, self.fps = width, height, fps
        self.roi, self.led_state = roi, led_state
        self._next = time.perf_counter()
        self._n = 0
        self._frame: np.ndarray | None = None

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        self._next += 1.0 / self.fps
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.perf_counter()
        f = np.full((self.height, self.width, 3), 60, np.uint8)
        cv2.putText(f, f"SIMULATED {self._n}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        if self.led_state():
            r = self.roi
            f[r.y:r.y + r.h, r.x:r.x + r.w] = 255
        self._frame = f
        self._n += 1
        return True

    def retrieve(self):
        return self._frame is not None, self._frame

    def read(self):
        self.grab()
        return self.retrieve()

    def get(self, prop) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0.0)

    def release(self) -> None:
        pass


SYNTHETIC_ROI = LedRoi(40, 80, 60, 60)


def synthetic_settings(cam: SyntheticCamera) -> dict:
    return {"index": "synthetic", "backend": "synthetic", "width": cam.width, "height": cam.height,
            "fps_reported": cam.fps, "fourcc": "none"}
