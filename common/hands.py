"""MediaPipe Hand Landmarker (Tasks API, VIDEO mode) helpers.

Checked against mediapipe 1.0.1: the Tasks API names (HandLandmarker, HandLandmarkerOptions,
RunningMode.VIDEO, detect_for_video(image, timestamp_ms)) are unchanged from 0.10.x; the legacy
`mp.solutions` package (with drawing_utils) no longer exists, so drawing is done here with OpenCV
and vision.HandLandmarksConnections.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from common.config import REPO_ROOT, Config

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
FINGERTIPS = (4, 8, 12, 16, 20)
PALM = (0, 5, 9, 13, 17)


def model_path(cfg: Config) -> Path:
    p = REPO_ROOT / cfg["mediapipe.model_path"]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run python -m scripts.download_models")
    return p


def create_video_landmarker(cfg: Config) -> vision.HandLandmarker:
    m = cfg.section("mediapipe")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(cfg))),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(m["num_hands"]),
        min_hand_detection_confidence=float(m["min_hand_detection_confidence"]),
        min_hand_presence_confidence=float(m["min_hand_presence_confidence"]),
        min_tracking_confidence=float(m["min_tracking_confidence"]),
    )
    return vision.HandLandmarker.create_from_options(options)


class MonotonicMs:
    """detect_for_video needs strictly increasing integer millisecond timestamps."""

    def __init__(self) -> None:
        self.last = -1

    def __call__(self, t_ms: float) -> int:
        ts = max(int(round(t_ms)), self.last + 1)
        self.last = ts
        return ts


def to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_hands(frame: np.ndarray, result: vision.HandLandmarkerResult) -> None:
    h, w = frame.shape[:2]
    for i, lms in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for j, p in enumerate(pts):
            color = (0, 0, 255) if j in FINGERTIPS else (0, 200, 0)
            cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
        if result.handedness and i < len(result.handedness):
            cat = result.handedness[i][0]
            cv2.putText(frame, f"{cat.category_name} {cat.score:.2f}", (pts[0][0] + 10, pts[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
