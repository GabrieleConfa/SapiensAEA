"""Live webcam preview with the 21 hand landmarks, fps and detection rate.

For positioning the camera and the lighting before a session only; it never records and is never
used during a recording (the acquisition saves video only, MediaPipe runs offline).

    python -m tools.camera_preview --list      # find the external webcam index
    python -m tools.camera_preview             # camera.index from config
    python -m tools.camera_preview --index 1

Keys:  R  select the LED region (drag a box around the sync LED, ENTER to confirm)
       S  open the camera driver settings dialog (DirectShow) to lock exposure / white balance
       Q or ESC  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2

from common import camera as cam
from common import hands
from common.config import HardwareParamMissing, load_config

WINDOW = "Camera preview - NOT recording"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list camera indices and exit")
    ap.add_argument("--index", type=int, help="override camera.index")
    ap.add_argument("--video", help="play a video file instead of the camera (testing)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    ap.add_argument("--no-window", action="store_true", help="no display, print stats (testing)")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.list:
        for c in cam.list_cameras(cfg["camera.backend"]):
            print(f"index {c['index']}: frame {'ok' if c['ok'] else 'FAILED'} {c['shape']}")
        return 0

    try:
        if args.video:
            cap = cv2.VideoCapture(args.video)
            settings = {"index": args.video}
        else:
            cap, settings = cam.open_camera(cfg, args.index)
        landmarker = hands.create_video_landmarker(cfg)
    except (HardwareParamMissing, cam.CameraError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")
    print("camera settings read back:", settings)

    roi = cam.load_led_roi()
    ts = hands.MonotonicMs()
    t0 = time.perf_counter_ns()
    frame_times: deque = deque(maxlen=60)
    detected: deque = deque(maxlen=60)
    detect_ms: deque = deque(maxlen=60)
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    break
                print("camera read failed")
                time.sleep(0.05)
                continue
            now = time.perf_counter_ns()
            frame_times.append(now)
            t1 = time.perf_counter()
            result = landmarker.detect_for_video(hands.to_mp_image(frame), ts((now - t0) / 1e6))
            detect_ms.append((time.perf_counter() - t1) * 1000)
            detected.append(bool(result.hand_landmarks))
            n += 1

            fps = (len(frame_times) - 1) / ((frame_times[-1] - frame_times[0]) / 1e9) if len(frame_times) > 1 else 0.0
            rate = sum(detected) / len(detected)
            hands.draw_hands(frame, result)
            lines = [f"loop {fps:5.1f} fps   detection {100 * rate:5.1f}%   mediapipe {sum(detect_ms) / len(detect_ms):5.1f} ms",
                     f"{frame.shape[1]}x{frame.shape[0]}   R: LED region   S: driver settings   Q: quit"]
            if roi is not None:
                b = roi.brightness(frame)
                cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 2)
                lines.append(f"LED region brightness {b:5.1f}")
            else:
                lines.append("LED region not set (press R)")
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            if args.no_window:
                if n % 30 == 0:
                    print(lines[0])
            else:
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    x, y, w, h = cv2.selectROI(WINDOW, frame, showCrosshair=True)
                    if w > 0 and h > 0:
                        roi = cam.LedRoi(int(x), int(y), int(w), int(h))
                        cam.save_led_roi(roi, settings.get("index"))
                        print(f"LED region saved to {cam.LED_ROI_PATH}: {roi}")
                if key == ord("s"):
                    cap.set(cv2.CAP_PROP_SETTINGS, 1)
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
    print(f"{n} frames, detection rate over the last {len(detected)} frames: "
          f"{100 * sum(detected) / max(1, len(detected)):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
