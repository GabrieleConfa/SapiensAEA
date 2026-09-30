"""Record one session: EMG stream + video + cues, on one laptop clock (time.perf_counter_ns).

    python -m acquisition --subject A --session 1
    python -m acquisition --subject A --session 1 --source sim --camera synthetic   # no hardware

Output folder data/<subject>_S<n>_<YYYY-MM-DD>/ (see docs/DATA_FORMAT.md):
    emg_raw.bin, emg_host_times.csv       (GIGA / simulator)   or   emg_cyton.csv (Cyton)
    video.mp4, frame_times.csv, cues.csv, sync_events.csv, session.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

from acquisition.runner import CUE_COLUMNS, SYNC_COLUMNS, SessionRunner
from acquisition.video import VideoRecorder
from acquisition.writers import CsvLog, RawStreamSink
from common import camera as cam
from common.config import REPO_ROOT, Config, HardwareParamMissing, load_config
from common.paths import next_session_dir
from common.sources import SOURCE_KINDS, make_source
from common.sources.base import EmgSource
from protocol.generate_sequences import get_session
from protocol.timeline import build_timeline

SCHEMA_VERSION = 1


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def ask(prompt: str, cast=str, optional: bool = False):
    while True:
        raw = input(f"{prompt}{' (Enter to skip)' if optional else ''}: ").strip()
        if not raw and optional:
            return None
        try:
            return cast(raw)
        except ValueError:
            print("  invalid value, try again")


def parse_impedances(text: str | None, n: int) -> list[float] | None:
    if text is None:
        return None
    values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(values) != n:
        raise ValueError(f"expected {n} impedance values, got {len(values)}")
    return values


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=["A", "B", "C"])
    ap.add_argument("--session", required=True, type=int, choices=[1, 2, 3])
    ap.add_argument("--source", default="giga", choices=SOURCE_KINDS)
    ap.add_argument("--port", help="override the serial port / simulator URL from config")
    ap.add_argument("--camera", default="config",
                    help="'config' (camera.index), an OpenCV index, 'synthetic' or 'none'")
    ap.add_argument("--forearm-cm", type=float, help="forearm length, cm")
    ap.add_argument("--band-pct", type=float, help="electrode band position, %% of forearm length from the elbow")
    ap.add_argument("--impedances", help="comma separated, kOhm, one per channel (optional)")
    ap.add_argument("--notes", help="free text")
    ap.add_argument("--no-prompt", action="store_true", help="never ask on the console")
    ap.add_argument("--headless", action="store_true", help="no cue screen, start immediately (tests)")
    ap.add_argument("--windowed", action="store_true", help="cue screen in a window, not full screen")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="timeline speed-up factor, simulator only (development)")
    ap.add_argument("--data-dir", type=Path, help="override project.data_dir (tests)")
    return ap.parse_args(argv)


def open_camera(args, cfg: Config, source: EmgSource):
    """Returns (capture or None, settings dict, led_roi or None)."""
    if args.camera == "none":
        return None, {"index": "none"}, None
    if args.camera == "synthetic":
        roi = cam.SYNTHETIC_ROI
        c = cam.SyntheticCamera(640, 360, float(cfg["camera.fps"]), roi, lambda: source.buffer.led_on)
        return c, cam.synthetic_settings(c), roi
    index = None if args.camera == "config" else int(args.camera)
    cap, settings = cam.open_camera(cfg, index)
    return cap, settings, cam.load_led_roi()


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    if args.speed != 1.0 and args.source != "sim":
        sys.exit("--speed is only allowed with --source sim")

    seq = get_session(f"{args.subject}{args.session}")
    timeline = build_timeline(cfg, seq)
    if args.data_dir:
        cfg.as_dict()["project"]["data_dir"] = str(args.data_dir)
    out = next_session_dir(cfg, args.subject, args.session, date.today())
    n_ch = int(cfg["emg.n_channels"])

    # ---- metadata ------------------------------------------------------------------------
    prompt = not (args.no_prompt or args.headless)
    forearm = args.forearm_cm if args.forearm_cm is not None else (
        ask("Forearm length [cm]", float) if prompt else None)
    band = args.band_pct if args.band_pct is not None else (
        ask("Band position [% of forearm length from the elbow]", float) if prompt else None)
    imp_text = args.impedances if args.impedances is not None else (
        ask(f"Impedances [kOhm, {n_ch} comma-separated values]", optional=True) if prompt else None)
    impedances = parse_impedances(imp_text, n_ch)
    notes = args.notes if args.notes is not None else (ask("Notes", optional=True) if prompt else None)

    # ---- devices -------------------------------------------------------------------------
    try:
        source = make_source(cfg, args.source, args.port)
        print(f"opening EMG source ({args.source}) ...")
        device_info = source.open()
        capture, cam_settings, led_roi = open_camera(args, cfg, source)
    except (HardwareParamMissing, cam.CameraError) as e:
        sys.exit(f"ERROR: {e}")
    except Exception as e:  # device errors are reported, never swallowed
        sys.exit(f"ERROR opening devices: {e}")
    if capture is not None and led_roi is None:
        print("WARNING: no LED ROI set (python -m tools.camera_preview, key R); "
              "the LED can still be found offline")

    out.mkdir(parents=True, exist_ok=True)
    print(f"recording into {out}")
    anchor = {"perf_counter_ns": time.perf_counter_ns(), "unix_time_ns": time.time_ns()}

    meta = {
        "schema_version": SCHEMA_VERSION,
        "status": "recording",
        "subject": args.subject,
        "session": args.session,
        "session_id": seq["id"],
        "day": seq["day"],
        "role": seq["role"],
        "date": date.today().isoformat(),
        "folder": out.name,
        "trial_order": seq["trials"],
        "ramp_order": seq["ramps"],
        "sampling_rate_hz": source.fs,
        "n_channels": n_ch,
        "channels": cfg["emg.channel_labels"],
        "front_end": cfg["emg.front_end"] if args.source != "cyton" else "OpenBCI Cyton via BrainFlow",
        "emg_source": args.source,
        "device_info": device_info,
        "uv_per_lsb": source.uv_per_lsb,
        "band_position_pct_forearm": band,
        "forearm_length_cm": forearm,
        "impedances_kohm": impedances,
        "notes": notes,
        "hand": cfg["subject.hand"],
        "hand_orientation": cfg["subject.hand_orientation"],
        "camera": cam_settings,
        "led_roi": None if led_roi is None else vars(led_roi),
        "protocol": cfg.section("protocol"),
        "flash_ms": cfg["sync.flash_ms"],
        "flash_times_planned_s": list(timeline.flash_times_s),
        "timeline_duration_s": timeline.duration_s,
        "speed": args.speed,
        "clock": {"name": "time.perf_counter_ns", **anchor},
        "config_path": str(cfg.path),
        "config_sha256": cfg.sha256,
        "software_commit": git_commit(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    session_json = out / "session.json"

    def save_meta() -> None:
        session_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_meta()

    # ---- recording -----------------------------------------------------------------------
    if args.source == "cyton":
        header = ("counter,host_time_ns," + ",".join(f"ch{i + 1}_uV" for i in range(n_ch)) + "\n").encode()
        sink = RawStreamSink(out / "emg_cyton.csv", out / "emg_host_times.csv", header=header)
    else:
        sink = RawStreamSink(out / "emg_raw.bin", out / "emg_host_times.csv")
    cues = CsvLog(out / "cues.csv", CUE_COLUMNS)
    sync = CsvLog(out / "sync_events.csv", SYNC_COLUMNS)
    runner = SessionRunner(timeline, source, cues, sync, speed=args.speed, simulate=args.source == "sim")

    video = None
    if capture is not None:
        video = VideoRecorder(capture, out / "video.mp4", out / "frame_times.csv",
                              float(cfg["camera.fps"]), cfg["camera.video_fourcc"], led_roi)
    source.start(sink)
    if video is not None:
        video.start()

    def status() -> tuple[str, bool]:
        s = source.stats()
        ok = s["rate_hz"] > 0.95 * source.fs if s["samples"] > source.fs else True
        parts = [f"EMG {s['rate_hz']:6.1f} Hz"]
        if "crc_errors" in s:
            parts.append(f"crc {s['crc_errors']}  lost {s['lost_samples']}  LED {s['led_edges']}")
            ok = ok and s["lost_samples"] < 0.01 * max(1, s["samples"])
        if source.error:
            parts.append(source.error)
            ok = False
        if video is not None:
            fps = video.measured_fps()
            parts.append(f"cam {fps:5.1f} fps  drop {video.dropped}")
            ok = ok and (video.captured < 30 or fps > 0.8 * float(cfg["camera.fps"])) and not video.error
            if video.error:
                parts.append(video.error)
        return "   ".join(parts), ok

    aborted = False
    try:
        if args.headless:
            time.sleep(1.0)
            runner.start(time.perf_counter_ns())
            while True:
                st = runner.tick(time.perf_counter_ns())
                if st.finished:
                    break
                time.sleep(0.01)
        else:
            from acquisition.ui import CueScreen

            result = {}
            screen = CueScreen(runner, seq["id"], status, lambda a: result.setdefault("aborted", a),
                               fullscreen=not args.windowed)
            screen.run()
            aborted = result.get("aborted", True)
            if runner.t0_ns is None:
                aborted = True
    except KeyboardInterrupt:
        runner.abort(time.perf_counter_ns())
        aborted = True
    finally:
        time.sleep(0.5)             # let the last packets and the final LED edge arrive
        source.stop()
        runner.poll_led_edges()
        if video is not None:
            print("finishing video file ...")
            video.stop()
        sink.close()
        cues.close()
        sync.close()

    s = source.stats()
    meta.update({
        "status": "aborted" if aborted or runner.aborted else "completed",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "session_start_host_ns": runner.t0_ns,
        "session_end_host_ns": runner.end_ns,
        "emg_stats": s,
        "video_stats": None if video is None else video.stats(),
        "device_messages": [m for _, m in source.messages][-50:],
    })
    save_meta()

    print(f"\n{meta['status'].upper()}: {out}")
    print(f"  EMG: {s.get('packets', s['samples'])} samples, "
          f"lost {s.get('lost_samples', 'n/a')}, crc errors {s.get('crc_errors', 'n/a')}, "
          f"LED edges {s.get('led_edges', 'n/a')} (planned {len(timeline.flash_times_s)})")
    if video is not None:
        v = video.stats()
        print(f"  video: {v['frames_written']} frames, dropped {v['frames_dropped']}, "
              f"{v['fps_measured']} fps at the end")
    return 0 if meta["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
