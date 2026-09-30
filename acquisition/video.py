"""Video recording with one host timestamp per frame. MediaPipe never runs here."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from acquisition.writers import CsvLog
from common.camera import LedRoi


class VideoRecorder:
    """Capture thread timestamps each frame right after grab(); a writer thread encodes it.

    frame_times.csv: frame_index (position in video.mp4), capture_index (frames grabbed so far,
    gaps = frames dropped because the writer fell behind), host_time_ns (perf_counter_ns).
    The container fps is nominal; the real timing is in frame_times.csv.
    """

    def __init__(self, cap, video_path: Path, times_path: Path, fps: float, fourcc: str,
                 led_roi: LedRoi | None = None, queue_frames: int = 300):
        self.cap = cap
        self.video_path = video_path
        self.fps = fps
        self.fourcc = fourcc
        self.led_roi = led_roi
        self.times = CsvLog(times_path, ["frame_index", "capture_index", "host_time_ns"])
        self._q: queue.Queue = queue.Queue(maxsize=queue_frames)
        self._running = False
        self.captured = 0
        self.written = 0
        self.dropped = 0
        self.read_failures = 0
        self.led_brightness = float("nan")   # live value inside the LED ROI (quality check / UI)
        self._recent = deque(maxlen=60)
        self.error: str | None = None

    def start(self) -> None:
        self._running = True
        self._cap_thread = threading.Thread(target=self._capture, name="video-capture", daemon=True)
        self._wr_thread = threading.Thread(target=self._write, name="video-writer", daemon=True)
        self._wr_thread.start()
        self._cap_thread.start()

    def _capture(self) -> None:
        while self._running:
            if not self.cap.grab():
                self.read_failures += 1
                if self.read_failures > 50:
                    self.error = "camera stopped delivering frames"
                    break
                time.sleep(0.01)
                continue
            t = time.perf_counter_ns()
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                self.read_failures += 1
                continue
            idx = self.captured
            self.captured += 1
            self._recent.append(t)
            if self.led_roi is not None:
                self.led_brightness = self.led_roi.brightness(frame)
            try:
                self._q.put_nowait((idx, t, frame))
            except queue.Full:
                self.dropped += 1
        self._q.put(None)

    def _write(self) -> None:
        writer = None
        while True:
            item = self._q.get()
            if item is None:
                break
            idx, t, frame = item
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*self.fourcc),
                                         self.fps, (w, h))
                if not writer.isOpened():
                    self.error = f"cannot open video writer ({self.fourcc}) for {self.video_path}"
                    self._running = False
                    break
            writer.write(frame)
            self.times.write(frame_index=self.written, capture_index=idx, host_time_ns=t)
            self.written += 1
        if writer is not None:
            writer.release()
        self.times.close()

    def measured_fps(self) -> float:
        r = list(self._recent)
        if len(r) < 2:
            return 0.0
        return (len(r) - 1) / ((r[-1] - r[0]) / 1e9)

    def stop(self) -> None:
        self._running = False
        self._cap_thread.join(3.0)
        self._wr_thread.join(30.0)   # let the writer drain the queue
        self.cap.release()

    def stats(self) -> dict:
        return {"frames_captured": self.captured, "frames_written": self.written,
                "frames_dropped": self.dropped, "read_failures": self.read_failures,
                "fps_measured": round(self.measured_fps(), 2), "error": self.error}
