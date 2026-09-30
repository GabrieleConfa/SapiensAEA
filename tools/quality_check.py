"""Live signal-quality check before each session. Works with the GIGA stream and the Cyton.

    python -m tools.quality_check --subject A --session 1                  # GIGA
    python -m tools.quality_check --subject A --session 1 --source cyton   # Cyton via BrainFlow
    python -m tools.quality_check --source sim --camera synthetic           # no hardware

Window: live spectrum per channel. Keys:
    ENTER  guided check: rest 5 s -> fist 5 s -> LED test -> PASS/FAIL (saved)
    R / F / L  repeat only the rest capture / the fist capture / the LED test
    E      evaluate again with the last captures
    Q      quit
Per channel: spectrum, share of 48-52 Hz power, fist/rest RMS ratio, flat or saturated channel;
plus whether the sync LED was seen (stream flag and camera). The report is written to the folder
of the next recording of that session (quality_check.json) and to the console.

Cyton: the OpenBCI GUI and this script cannot hold the dongle's serial port at the same time.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from common import camera as cam
from common import quality as ql
from common.config import Config, HardwareParamMissing, load_config
from common.paths import data_root, next_session_dir
from common.sources import SOURCE_KINDS, make_source
from common.sources.base import EmgSource


class CameraMonitor:
    """Grabs frames in a thread and keeps the LED-ROI brightness history. Records nothing."""

    def __init__(self, capture, roi: cam.LedRoi | None):
        self.capture, self.roi = capture, roi
        self.history: deque = deque(maxlen=600)
        self.frames = 0
        self._running = True
        self._t = threading.Thread(target=self._run, name="qc-camera", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while self._running:
            if not self.capture.grab():
                time.sleep(0.01)
                continue
            t = time.perf_counter_ns()
            ok, frame = self.capture.retrieve()
            if ok and frame is not None:
                self.frames += 1
                if self.roi is not None:
                    self.history.append((t, self.roi.brightness(frame)))

    def window(self, t0_ns: int, t1_ns: int) -> np.ndarray:
        return np.array([b for t, b in list(self.history) if t0_ns <= t <= t1_ns])

    def stop(self) -> None:
        self._running = False
        self._t.join(2.0)
        self.capture.release()


class QualityCheck:
    """Guided capture state machine, shared by the interactive window and headless mode."""

    def __init__(self, cfg: Config, source: EmgSource, camera: CameraMonitor | None, simulate: bool):
        self.cfg, self.source, self.camera, self.simulate = cfg, source, camera, simulate
        self.q = cfg.section("quality")
        self.fs = source.fs
        self.captures: dict[str, np.ndarray] = {}
        self.led: dict | None = None
        self.report: dict | None = None
        self.queue: list[str] = []
        self.step: str | None = None
        self._step_start_ns = 0
        self._step_start_total = 0
        self._flash_ns = 0
        self._baseline = float("nan")

    # -- steps --------------------------------------------------------------------------------
    def run(self, steps: list[str]) -> None:
        self.queue = list(steps)
        self._next()

    def _next(self) -> None:
        self.step = self.queue.pop(0) if self.queue else None
        self._step_start_ns = time.perf_counter_ns()
        self._step_start_total = self.source.buffer.total
        if self.simulate:
            self.source.send("SIM GRIP FIST 0.8" if self.step == "fist" else "SIM GRIP REST 0")
        if self.step == "led":
            self._baseline = self._camera_level(self._step_start_ns - int(0.5e9), self._step_start_ns)
            self._flash_ns = time.perf_counter_ns()
            if not self.source.flash():
                self.led = None  # source cannot drive the LED (Cyton)
                self._next()
        elif self.step == "evaluate":
            self.evaluate()
            self._next()

    def instruction(self) -> str:
        now = time.perf_counter_ns()
        if self.step in ("rest", "fist"):
            left = self.q["capture_s"] - (now - self._step_start_ns) / 1e9
            what = "RELAX the hand" if self.step == "rest" else "Make a FIST - strong, not maximal"
            return f"{what}   {max(0.0, left):.0f} s"
        if self.step == "led":
            return "LED test - camera must see the LED"
        if self.report:
            return f"{self.report['result']}   (ENTER: repeat the guided check, Q: quit)"
        return "ENTER: start the guided check (rest -> fist -> LED)"

    def poll(self) -> None:
        now = time.perf_counter_ns()
        elapsed = (now - self._step_start_ns) / 1e9
        if self.step in ("rest", "fist") and elapsed >= self.q["capture_s"]:
            uv, _, _ = self.source.buffer.since(self._step_start_total)
            # drop the first 0.5 s: reaction time after the instruction
            self.captures[self.step] = uv[int(0.5 * self.fs):]
            self._next()
        elif self.step == "led" and elapsed >= self.q["led_timeout_s"] + 0.5:
            self.led = self._led_result()
            self._next()

    def _camera_level(self, t0: int, t1: int) -> float:
        if self.camera is None:
            return float("nan")
        w = self.camera.window(t0, t1)
        return float(np.median(w)) if len(w) else float("nan")

    def _led_result(self) -> dict:
        timeout_ns = int(self.q["led_timeout_s"] * 1e9)
        edges = [e for e in self.source.buffer.edges_since(self._flash_ns)
                 if e.host_time_ns - self._flash_ns <= timeout_ns]
        res = {"stream_edge": bool(edges),
               "stream_latency_ms": round((edges[0].host_time_ns - self._flash_ns) / 1e6, 1) if edges else None,
               "camera_checked": False, "camera_step": None}
        msgs = [] if edges else ["no LED flag in the EMG stream"]
        if self.camera is None:
            msgs.append("camera not checked (--camera none)")
        elif self.camera.roi is None:
            msgs.append("no LED ROI set: run python -m tools.camera_preview and press R")
        else:
            w = self.camera.window(self._flash_ns, self._flash_ns + timeout_ns)
            step = float(np.max(w) - self._baseline) if len(w) else float("nan")
            res.update(camera_checked=True, camera_step=round(step, 1))
            if not step >= float(self.cfg["sync.video_threshold"]):
                msgs.append(f"camera brightness step {step:.0f} < {self.cfg['sync.video_threshold']}")
        # a missing camera is a warning, not a failure; everything else fails
        res["ok"] = bool(edges) and not any(m.startswith(("camera brightness", "no LED ROI")) for m in msgs)
        res["message"] = "; ".join(msgs) if msgs else "detected in stream and camera"
        return res

    def evaluate(self) -> dict | None:
        if "rest" not in self.captures or "fist" not in self.captures:
            return None
        led = self.led if self.source.supports_flash else None
        if self.source.supports_flash and led is None:
            led = {"ok": False, "message": "LED test not run"}
        self.report = ql.evaluate(self.captures["rest"], self.captures["fist"], self.fs, self.q,
                                  self.source.saturation_uv, led, list(self.cfg["emg.channel_labels"]))
        self.report["source"] = self.source.name
        self.report["device_info"] = self.source.info
        self.report["time"] = datetime.now().isoformat(timespec="seconds")
        return self.report


def format_report(rep: dict) -> str:
    lines = [f"QUALITY CHECK: {rep['result']}"]
    lines.append(f"{'ch':6s} {'50Hz fist':>9s} {'50Hz rest':>9s} {'rest µV':>8s} {'fist µV':>8s} "
                 f"{'ratio':>6s}  issues")
    for ch in rep["channels"]:
        lines.append(f"{ch['channel']:6s} {ch['line_share_48_52_fist']:9.3f} {ch['line_share_48_52_rest']:9.3f} "
                     f"{ch['rest_rms_uv']:8.2f} {ch['fist_rms_uv']:8.2f} {ch['rms_ratio_fist_rest']:6.2f}"
                     f"{'' if ch['rms_ratio_ok'] else '*'}  {', '.join(ch['issues'])}")
    led = rep["led"]
    lines.append("LED: " + ("not applicable (Cyton)" if led is None else
                            ("OK" if led["ok"] else "FAIL") + f" - {led.get('message', '')}"))
    for r in rep["reasons"]:
        lines.append(f"  - {r}")
    return "\n".join(lines)


def save_report(rep: dict, folder: Path, figure=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    text = json.dumps(rep, indent=2)
    (folder / f"quality_check_{stamp}.json").write_text(text, encoding="utf-8")
    (folder / "quality_check.json").write_text(text, encoding="utf-8")
    if figure is not None:
        figure.savefig(folder / f"quality_check_{stamp}.png", dpi=90)
    return folder / "quality_check.json"


def run_window(qc: QualityCheck, out_dir: Path, source: EmgSource) -> None:
    import matplotlib.pyplot as plt

    n_ch = source.n_channels
    labels = qc.cfg["emg.channel_labels"]
    fig, axes = plt.subplots(2, (n_ch + 1) // 2, figsize=(15, 8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    fig.canvas.manager.set_window_title("EMG quality check")
    lines = []
    for c in range(n_ch):
        ax = axes[c]
        (ln,) = ax.semilogy([], [], lw=0.8)
        ax.axvspan(48, 52, color="red", alpha=0.15)
        ax.set_title(labels[c], fontsize=9)
        ax.set_xlim(0, source.fs / 2)
        ax.set_ylim(1e-4, 1e3)
        ax.grid(True, which="both", alpha=0.2)
        lines.append(ln)
    for ax in axes[n_ch:]:
        ax.axis("off")
    fig.supxlabel("Hz")
    fig.supylabel("PSD (µV²/Hz)")
    title = fig.suptitle("", fontsize=14)
    fig.subplots_adjust(bottom=0.25)
    info = fig.text(0.01, 0.01, "", family="monospace", fontsize=8, va="bottom")

    def on_key(event):
        key = (event.key or "").lower()
        if key == "enter":
            qc.report = None
            qc.run(["rest", "fist", "led", "evaluate"])
        elif key in ("r", "f", "l"):
            qc.run([{"r": "rest", "f": "fist", "l": "led"}[key], "evaluate"])
        elif key == "e":
            qc.run(["evaluate"])
        elif key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    shown_report = None

    def update(_frame=None):
        nonlocal shown_report
        qc.poll()
        uv, _, _ = source.buffer.latest(int(2 * source.fs))
        if len(uv) > source.fs / 2:
            f, p = ql.spectrum(uv, source.fs)
            for c in range(n_ch):
                lines[c].set_data(f[1:], np.maximum(p[1:, c], 1e-6))
        s = source.stats()
        title.set_text(qc.instruction() + f"      [EMG {s['rate_hz']:.0f} Hz]")
        if qc.report is not None:
            title.set_color("green" if qc.report["result"] == "PASS" else "red")
            if qc.report is not shown_report:
                shown_report = qc.report
                info.set_text(format_report(qc.report))
                path = save_report(qc.report, out_dir, fig)
                print(format_report(qc.report))
                print(f"saved {path}")
        else:
            title.set_color("black")
        fig.canvas.draw_idle()

    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(fig, update, interval=300, cache_frame_data=False)  # noqa: F841
    plt.show()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", choices=["A", "B", "C"])
    ap.add_argument("--session", type=int, choices=[1, 2, 3])
    ap.add_argument("--source", default="giga", choices=SOURCE_KINDS)
    ap.add_argument("--port", help="override the serial port / simulator URL")
    ap.add_argument("--camera", default="config", help="'config', an OpenCV index, 'synthetic' or 'none'")
    ap.add_argument("--headless", action="store_true", help="run the guided check without a window")
    ap.add_argument("--out", type=Path, help="report folder (default: next session folder)")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.out:
        out_dir = args.out
    elif args.subject and args.session:
        out_dir = next_session_dir(cfg, args.subject, args.session)
    else:
        out_dir = data_root(cfg) / "quality_checks" / datetime.now().strftime("%Y-%m-%d_%H%M%S")

    try:
        source = make_source(cfg, args.source, args.port)
        source.open()
        camera = None
        if args.camera == "synthetic":
            capture = cam.SyntheticCamera(640, 360, float(cfg["camera.fps"]), cam.SYNTHETIC_ROI,
                                          lambda: source.buffer.led_on)
            camera = CameraMonitor(capture, cam.SYNTHETIC_ROI)
        elif args.camera != "none" and args.source != "cyton":
            index = None if args.camera == "config" else int(args.camera)
            capture, _ = cam.open_camera(cfg, index)
            camera = CameraMonitor(capture, cam.load_led_roi())
    except (HardwareParamMissing, cam.CameraError) as e:
        sys.exit(f"ERROR: {e}")
    except Exception as e:
        sys.exit(f"ERROR opening devices: {e}")

    source.start()
    qc = QualityCheck(cfg, source, camera, simulate=args.source == "sim")
    try:
        if args.headless:
            time.sleep(1.0)
            qc.run(["rest", "fist", "led", "evaluate"])
            while qc.step is not None:
                qc.poll()
                time.sleep(0.02)
            rep = qc.report
            print(format_report(rep))
            print(f"saved {save_report(rep, out_dir)}")
        else:
            run_window(qc, out_dir, source)
            rep = qc.report
    finally:
        source.stop()
        if camera is not None:
            camera.stop()
    return 0 if rep is not None and rep["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
