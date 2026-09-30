"""Read a recorded session folder back (EMG, frame times, cues, sync events).

    python -m common.recording data/A_S1_2026-10-05 --csv    # also writes emg.csv

Role locking for test sessions lives in the Stage 2 loader (pipeline/), not here: these are the
low-level readers used by the tools and by that loader.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.packet import FLAG_LED_RISE, ParserStats, parse_bytes


@dataclass
class EmgRecording:
    counters: np.ndarray       # (n,) int64, sample counter (EMG time base: counter / fs)
    flags: np.ndarray          # (n,) uint8
    uv: np.ndarray             # (n, ch) float32, µV at the electrodes
    host_time_ns: np.ndarray   # (n,) int64, laptop receive time of the USB read holding the packet
    fs: float
    stats: ParserStats | None

    @property
    def led_rise_counters(self) -> np.ndarray:
        return self.counters[(self.flags & FLAG_LED_RISE) != 0]


def load_meta(session_dir: Path) -> dict:
    return json.loads((Path(session_dir) / "session.json").read_text(encoding="utf-8"))


def load_emg(session_dir: Path) -> EmgRecording:
    session_dir = Path(session_dir)
    meta = load_meta(session_dir)
    n_ch = int(meta["n_channels"])
    fs = float(meta["sampling_rate_hz"])
    times = pd.read_csv(session_dir / "emg_host_times.csv")
    if (session_dir / "emg_raw.bin").exists():
        raw = (session_dir / "emg_raw.bin").read_bytes()
        block, stats = parse_bytes(raw, n_ch)
        info = meta["device_info"]
        mid = (int(info["max_code"]) + int(info["min_code"])) / 2 if int(info["min_code"]) >= 0 else 0.0
        uv = ((block.samples - mid) * float(meta["uv_per_lsb"])).astype(np.float32)
        idx = np.searchsorted(times["end_offset"].to_numpy(), block.end_offsets, side="left")
        host = times["host_time_ns"].to_numpy()[np.minimum(idx, len(times) - 1)]
        return EmgRecording(block.counters.astype(np.int64), block.flags, uv, host, fs, stats)
    df = pd.read_csv(session_dir / "emg_cyton.csv")
    uv = df[[c for c in df.columns if c.startswith("ch")]].to_numpy(np.float32)
    return EmgRecording(df["counter"].to_numpy(np.int64), np.zeros(len(df), np.uint8), uv,
                        df["host_time_ns"].to_numpy(np.int64), fs, None)


def load_frame_times(session_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(session_dir) / "frame_times.csv")


def load_cues(session_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(session_dir) / "cues.csv")


def load_sync_events(session_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(session_dir) / "sync_events.csv")


def export_csv(session_dir: Path) -> Path:
    rec = load_emg(session_dir)
    meta = load_meta(session_dir)
    df = pd.DataFrame(rec.uv, columns=[f"{c}_uV" for c in meta["channels"]])
    df.insert(0, "host_time_ns", rec.host_time_ns)
    df.insert(0, "flags", rec.flags)
    df.insert(0, "counter", rec.counters)
    out = Path(session_dir) / "emg.csv"
    df.to_csv(out, index=False, float_format="%.3f")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise (and optionally export) a session folder")
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--csv", action="store_true", help="write emg.csv next to emg_raw.bin")
    args = ap.parse_args()
    rec = load_emg(args.session_dir)
    dur = (rec.counters[-1] - rec.counters[0] + 1) / rec.fs if len(rec.counters) else 0
    print(f"{len(rec.counters)} samples, {dur:.1f} s, LED edges at counters {rec.led_rise_counters.tolist()}")
    if rec.stats:
        print(f"parser: {rec.stats}")
    if args.csv:
        print(f"written {export_csv(args.session_dir)}")


if __name__ == "__main__":
    main()
