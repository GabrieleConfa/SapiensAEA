"""Guided EMG acquisition: metadata, quality check, calibration, two blocks of gesture trials,
or (--free) a free recording of the raw stream without any protocol.

    python -m emgacq                          # Cyton on [board] serial_port
    python -m emgacq --port COM3              # Cyton on another port
    python -m emgacq --synthetic              # BrainFlow synthetic board, no hardware
    python -m emgacq --synthetic --speed 10 --windowed
    python -m emgacq --synthetic --headless --speed 20 --subject T --session 1
    python -m emgacq --free                   # free recording: no protocol, raw data until Stop
    python -m emgacq --free --duration 60     # free recording that stops by itself after 60 s
    python -m emgacq --list-ports

Output: data/<subject>_S<session>_<YYYY-MM-DD_HHMM>/, free recordings
data/<subject>_S<session>_FREE_<YYYY-MM-DD_HHMM>/ (see docs/DATA_FORMAT.md).
"""

from __future__ import annotations

import argparse
import sys

from emgacq.config import HardwareParamMissing, load_config
from emgacq.sources import SOURCE_KINDS, SourceError, list_serial_ports, make_source
from emgacq.sources.base import GUI_HINT


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m emgacq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use the BrainFlow synthetic board")
    ap.add_argument("--source", choices=SOURCE_KINDS, help="data source (default: [board] source)")
    ap.add_argument("--port", help="serial port, overrides the config")
    ap.add_argument("--config", help="config file (default: config.toml)")
    ap.add_argument("--list-ports", action="store_true", help="list the serial ports and exit")
    ap.add_argument("--free", action="store_true",
                    help="free recording: no quality check, calibration or trials; the raw stream "
                         "is saved until the operator stops it (or --duration has passed)")
    ap.add_argument("--duration", type=float, default=0.0, metavar="SECONDS",
                    help="length of a free recording (required with --headless; default: until stopped)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="run the protocol this many times faster (synthetic/simulator only)")
    ap.add_argument("--windowed", action="store_true", help="no full screen (testing)")
    ap.add_argument("--single-screen", action="store_true", help="force single-screen mode")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI, operator decisions scripted (testing)")
    ap.add_argument("--auto", action="store_true",
                    help="GUI with the operator steps automated (synthetic/simulator only, demo)")
    g = ap.add_argument_group("session metadata (prefills the form; required with --headless)")
    g.add_argument("--subject")
    g.add_argument("--session", type=int)
    g.add_argument("--day", type=int, default=1)
    g.add_argument("--role", choices=("train", "test"), default="train")
    g.add_argument("--arm", choices=("right", "left"), default="right")
    g.add_argument("--band-position", type=float, default=None, help="%% of forearm length from the elbow")
    g.add_argument("--forearm-length", type=float, default=None, help="cm")
    g.add_argument("--electrode-type", default="")
    g.add_argument("--notes", default="")
    return ap.parse_args(argv)


def metadata_from_args(args, cfg, channels: list[int]) -> dict:
    positions = list(cfg["board.channel_positions"]) + [""] * 8
    return {
        "subject": args.subject or "TEST",
        "session": args.session or 1,
        "day": args.day,
        "role": args.role,
        "arm": args.arm,
        "band_position_pct": args.band_position,
        "forearm_length_cm": args.forearm_length,
        "channel_map": [{"channel": c, "column": f"ch{c}", "position": positions[i]}
                        for i, c in enumerate(channels)],
        "electrode_type": args.electrode_type,
        "notes": args.notes,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_ports:
        for dev, desc in list_serial_ports() or [("(none)", "")]:
            print(f"{dev:24} {desc}")
        return 0
    cfg = load_config(args.config)
    kind = "synthetic" if args.synthetic else (args.source or cfg["board.source"])
    simulated = kind in ("synthetic", "giga_sim")
    if (args.speed != 1.0 or args.auto) and not simulated:
        print("--speed and --auto are only allowed with the synthetic board or the GIGA simulator")
        return 2
    if args.duration and not args.free:
        print("--duration is only used with --free")
        return 2
    if args.duration < 0 or (args.free and args.headless and not args.duration):
        print("a headless free recording needs --duration SECONDS (> 0)")
        return 2
    try:
        source = make_source(cfg, kind, args.port)
    except HardwareParamMissing as e:
        print(f"ERROR: {e}")
        return 2
    if kind == "cyton":
        print(f"Cyton on {source.serial_port}. {GUI_HINT}")

    if args.headless:
        from emgacq.headless import run_headless

        try:
            ctrl = run_headless(cfg, source, metadata_from_args(args, cfg, source.channels),
                                speed=args.speed, free=args.free, duration_s=args.duration)
        except SourceError as e:
            print(f"ERROR: {e}")
            return 1
        except KeyboardInterrupt:
            print("interrupted")
            return 130
        print()
        print(ctrl.summary_text)
        print(f"files in {ctrl.folder}")
        return 0

    from emgacq.ui.app import run_gui

    return run_gui(cfg, args, source, kind)


if __name__ == "__main__":
    sys.exit(main())
