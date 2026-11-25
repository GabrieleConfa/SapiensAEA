"""Live test of an exported gesture model on the Cyton (or the synthetic board).

Runs the same chain as the notebook (`models/emg_cnn1d.ipynb`) sample by sample: causal filter,
features every hop on the last frame, calibration baseline subtracted, standardised, model. The
filter, features, mean / SD and class order come from the model's `features.json`.

Steps:
1. connect and configure the board (same commands as the acquisition app);
2. calibration: rest for --cal seconds (the model is only valid with this baseline);
3. guided test (default): every gesture --reps times in random order, each trial = cue 1.5 s,
   hold 4 s, rest 4 s, with the live decision on screen; at the end, the score per posture;
   or free mode (--free): live decision until Ctrl+C.

Everything is saved to data/LIVE_<subject>_<date>/ (raw samples with markers and labels, every
decision, summary), so a test session can be looked at offline or added to the training data.

    python -m tools.live_test                          # 3-class model, Cyton on [board] serial_port
    python -m tools.live_test --port COM3 --reps 6
    python -m tools.live_test --free
    python -m tools.live_test --model ../models/emg_bin  # binary CLOSE model (OPEN and CLOSE trials)
    python -m tools.live_test --synthetic --reps 1     # dry run without hardware (predictions meaningless)

Place the electrodes as in the training sessions (2026-11-04): the model knows ch1-ch4 only by
their position at that time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from emgacq import markers as mk
from emgacq.config import APP_ROOT, load_config
from emgacq.sources import SourceError, make_source

DEFAULT_MODEL_DIR = APP_ROOT.parent / "models" / "emg_3cls"
AMPLITUDE = ("MAV", "RMS", "WL")          # log1p; ZC and SSC are divided by the frame length
CUE_S, HOLD_S, REST_S = 1.5, 4.0, 4.0     # trial timing of the acquisition protocol
SETTLE_S = 0.5                            # skipped after every label change when scoring (as in training)
CLASS_SHORT = {"OPEN": "OPEN ", "NOT_CLOSE": "OPEN ", "CLOSE": "CLOSE", "INDEX": "INDEX"}


# --------------------------------------------------------------------------- features

def frame_features(frame: np.ndarray, features: list[str], threshold: float) -> np.ndarray:
    """One frame (L, channels) of filtered µV -> (channels, len(features)).

    Same formulas as frame_feature_array in the notebook.
    """
    L = frame.shape[0]
    d = np.diff(frame, axis=0)
    out = []
    for f in features:
        if f == "MAV":
            v = np.log1p(np.abs(frame).mean(0))
        elif f == "RMS":
            v = np.log1p(np.sqrt((frame ** 2).mean(0)))
        elif f == "WL":
            v = np.log1p(np.abs(d).sum(0))
        elif f == "ZC":
            v = ((frame[:-1] * frame[1:] < 0) & (np.abs(d) >= threshold)).sum(0) / L
        elif f == "SSC":
            v = ((d[:-1] * -d[1:]) >= threshold ** 2).sum(0) / L
        else:
            raise ValueError(f"unknown feature {f!r}")
        out.append(v)
    return np.stack(out, axis=-1)


class StreamingFeatures:
    """Causal filter + one feature vector every `hop` samples, over the last `frame_len` samples.

    The filter state starts at the steady state of the first sample, as in the notebook. The
    vector is channel-major (ch1 features, ch2 features, ...), the order of features.json.
    """

    def __init__(self, sos: np.ndarray, n_channels: int, frame_len: int, hop: int,
                 features: list[str], threshold: float):
        self.sos = np.asarray(sos, float)
        self.n_channels, self.frame_len, self.hop = n_channels, frame_len, hop
        self.features, self.threshold = list(features), float(threshold)
        self.zi = None
        self.buf = np.zeros((0, n_channels))
        self.n_seen = 0                     # samples pushed so far
        self.next_end = frame_len           # sample count at which the next frame is complete

    def push(self, x: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Raw µV (n, channels) -> [(end sample index (exclusive), feature vector)] of the new frames."""
        x = np.asarray(x, float)
        if len(x) == 0:
            return []
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        self.buf = np.vstack([self.buf, y])
        self.n_seen += len(x)
        out = []
        while self.next_end <= self.n_seen:
            a = self.next_end - self.frame_len - (self.n_seen - len(self.buf))
            frame = self.buf[a:a + self.frame_len]
            out.append((self.next_end, frame_features(frame, self.features, self.threshold).reshape(-1)))
            self.next_end += self.hop
        keep = self.n_seen - (self.next_end - self.frame_len)   # samples still needed by the next frame
        self.buf = self.buf[-keep:] if keep > 0 else self.buf[:0]
        return out


# --------------------------------------------------------------------------- model

class Model:
    """TFLite model (int8 or float32) or Keras file, plus the parameters of features.json."""

    def __init__(self, model_dir: Path, kind: str = "int8"):
        self.dir = Path(model_dir)
        fj = self.dir / "features.json"
        if not fj.exists():
            raise FileNotFoundError(f"{fj} not found: run the export section of the notebook first")
        self.p = json.loads(fj.read_text(encoding="utf-8"))
        self.classes = list(self.p["classes"])
        self.features = list(self.p["features"])
        self.n_frames = int(self.p["n_frames"])
        self.mean = np.asarray(self.p["mean"], np.float32)
        self.sd = np.asarray(self.p["sd"], np.float32)
        self.threshold = float(self.p.get("decision_threshold", 0.5))
        names = [f"{ch}_{f}" for ch in self.p["channels"] for f in self.features]
        if names != list(self.p["feature_order"]):
            raise ValueError("feature_order in features.json is not channel-major; this script cannot rebuild it")
        stem = next(self.dir.glob("*.keras")).stem
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf      # slow import, only when a model is loaded

        self.kind = kind
        if kind == "keras":
            import keras

            self._keras = keras.models.load_model(self.dir / f"{stem}.keras")
            self.file = self.dir / f"{stem}.keras"
        else:
            self.file = self.dir / f"{stem}_{'int8' if kind == 'int8' else 'float32'}.tflite"
            self._interp = tf.lite.Interpreter(model_path=str(self.file))
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()[0]

    def probs(self, window: np.ndarray) -> np.ndarray:
        """Calibrated features (n_frames, n_feat) -> class probabilities (n_classes,)."""
        x = ((window - self.mean) / self.sd).astype(np.float32)[None]
        if self.kind == "keras":
            y = np.asarray(self._keras(x, training=False))[0]
        else:
            if self._in["dtype"] == np.int8:
                scale, zero = self._in["quantization"]
                x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
            self._interp.set_tensor(self._in["index"], x)
            self._interp.invoke()
            y = self._interp.get_tensor(self._out["index"])[0]
            if self._out["dtype"] == np.int8:
                scale, zero = self._out["quantization"]
                y = (y.astype(np.float32) - zero) * scale
        y = np.asarray(y, np.float32)
        if len(y) == 1:                      # binary model: one sigmoid unit, P(positive class)
            y = np.array([1.0 - y[0], y[0]], np.float32)
        return y

    def decide(self, p: np.ndarray) -> int:
        if len(self.classes) == 2:
            return int(p[1] >= self.threshold)
        return int(np.argmax(p))

    def target(self, posture: str) -> str | None:
        """Class the model should answer for a posture of the protocol (None: not scored)."""
        if posture in self.classes:
            return posture
        if posture in ("REST", "OPEN"):
            return self.classes[0]           # "hand open" class: OPEN or NOT_CLOSE
        return None


# --------------------------------------------------------------------------- scoring

def frame_ok_labels(labels: np.ndarray, settle: int) -> np.ndarray:
    """Per sample: samples since the last label change (to skip the first `settle`)."""
    change = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    idx = np.arange(len(labels))
    return idx - change[np.searchsorted(change, idx, "right") - 1]


def score(labels: np.ndarray, decisions: pd.DataFrame, model: Model, window_len: int, fs: float) -> dict:
    """Score the decisions whose whole input lies in one settled posture, as in training.

    labels: label of every sample; decisions: columns end (exclusive sample index), pred (class
    name), trial_id. Returns the tables printed at the end and written to summary.json.
    """
    since = frame_ok_labels(labels, 0)
    settle = int(SETTLE_S * fs)
    rows = []
    for r in decisions.itertuples():
        a, b = r.end - window_len, r.end
        if a < 0 or b > len(labels):
            continue
        lab = labels[a:b]
        if (lab != lab[0]).any() or since[a] < settle:
            continue
        tgt = model.target(lab[0])
        if tgt is not None:
            rows.append((lab[0], tgt, r.pred, r.trial_id))
    if not rows:
        return {"windows": 0}
    s = pd.DataFrame(rows, columns=["posture", "target", "pred", "trial_id"])
    postures = [p for p in ("REST", "OPEN", "CLOSE", "INDEX") if p in set(s["posture"])]
    share = (pd.crosstab(s["posture"], s["pred"], normalize="index")
             .reindex(index=postures, columns=model.classes, fill_value=0.0))
    recall = s.assign(ok=s["target"] == s["pred"]).groupby("target")["ok"].mean()
    trials = []
    for tid, g in s[s["posture"] != "REST"].groupby("trial_id"):
        vote = Counter(g["pred"]).most_common(1)[0][0]
        trials.append({"trial_id": int(tid), "posture": g["posture"].iloc[0], "vote": vote,
                       "ok": vote == g["target"].iloc[0]})
    return {"windows": len(s), "share": share, "recall": recall,
            "balanced_acc": float(recall.mean()), "trials": pd.DataFrame(trials)}


# --------------------------------------------------------------------------- session

def make_trials(gestures: list[str], reps: int, seed: int) -> list[str]:
    """Every gesture `reps` times, shuffled, no gesture twice in a row when avoidable."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        order = list(rng.permutation([g for g in gestures for _ in range(reps)]))
        if all(a != b for a, b in zip(order, order[1:])) or len(gestures) == 1:
            return order
    return order


def bar(p: float, width: int = 10) -> str:
    n = int(round(p * width))
    return "#" * n + "." * (width - n)


class LiveSession:
    def __init__(self, source, model: Model, fs: float, n_channels: int, vote: int, out_dir: Path):
        p = model.p
        self.source, self.model, self.fs = source, model, fs
        self.feats = StreamingFeatures(np.asarray(p["filter"]["sos"]), n_channels, int(p["frame_len"]),
                                       int(p["hop"]), model.features, float(p["threshold_uv"]))
        self.hop, self.frame_len = int(p["hop"]), int(p["frame_len"])
        self.window_len = (model.n_frames - 1) * self.hop + self.frame_len
        self.cal_skip = int(float(p["calibration"]["skip_s"]) * fs)
        self.vote = max(1, vote)
        self.out_dir = out_dir
        self.samples: list[np.ndarray] = []
        self.markers: list[np.ndarray] = []
        self.ts: list[np.ndarray] = []
        self.pkg: list[np.ndarray] = []
        self.frames: deque = deque(maxlen=model.n_frames)
        self.recent: deque = deque(maxlen=self.vote)
        self.cal_frames: list[np.ndarray] = []
        self.baseline: np.ndarray | None = None
        self.decisions: list[dict] = []
        self.calibrating = False
        self.trial_id = 0
        self.phase = "WAIT"
        self.last_line = ""

    # ---- data
    def pump(self) -> None:
        chunk = self.source.read()
        if len(chunk) == 0:
            return
        self.samples.append(chunk.data_uv)
        self.markers.append(chunk.marker)
        self.ts.append(chunk.timestamp)
        self.pkg.append(chunk.package_num)
        for end, f in self.feats.push(chunk.data_uv):
            self._on_frame(end, f)

    def _on_frame(self, end: int, f: np.ndarray) -> None:
        if self.calibrating:
            if end - self.frame_len >= self.cal_start + self.cal_skip:
                self.cal_frames.append(f)
            return
        if self.baseline is None:
            return
        self.frames.append(f - self.baseline)
        if len(self.frames) < self.model.n_frames:
            return
        p = self.model.probs(np.stack(self.frames))
        k = self.model.decide(p)
        self.recent.append(k)
        k_vote = Counter(self.recent).most_common(1)[0][0]
        self.decisions.append({"end": end, "trial_id": self.trial_id, "phase": self.phase,
                               "pred": self.model.classes[k], "pred_vote": self.model.classes[k_vote],
                               **{f"p_{c}": float(v) for c, v in zip(self.model.classes, p)}})
        probs = "  ".join(f"{CLASS_SHORT.get(c, c)} {bar(v)} {v:.2f}" for c, v in zip(self.model.classes, p))
        self.last_line = f"  -> {self.model.classes[k_vote]:<9} | {probs}"
        sys.stdout.write("\r" + self.last_line + " " * 4)
        sys.stdout.flush()

    def run_for(self, seconds: float, read_interval: float = 0.02) -> None:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            self.pump()
            time.sleep(read_interval)

    def step(self, text: str, marker: int | None, phase: str) -> None:
        if marker is not None:
            self.source.insert_marker(marker)
        self.phase = phase
        sys.stdout.write("\n" + text + "\n")
        sys.stdout.flush()

    # ---- protocol
    def calibrate(self, seconds: float) -> None:
        self.step(f"CALIBRATION: relax the arm completely, hand open and still ({seconds:.0f} s)",
                  mk.CAL_REST, "CALIBRATION")
        self.calibrating = True
        self.cal_start = self.feats.n_seen
        self.run_for(seconds)
        self.pump()
        self.calibrating = False
        if len(self.cal_frames) < 10:
            raise SourceError(f"calibration got only {len(self.cal_frames)} frames: is data arriving?")
        self.baseline = np.mean(self.cal_frames, axis=0)
        mav = [i for i, n in enumerate(self.feats_names()) if n.endswith("_MAV")]
        if mav:
            txt = ", ".join(f"{self.feats_names()[i].split('_')[0]} {np.expm1(self.baseline[i]):.1f}" for i in mav)
            print(f"calibration done: {len(self.cal_frames)} frames; resting MAV (µV, filtered): {txt}")
        else:
            print(f"calibration done: {len(self.cal_frames)} frames")

    def feats_names(self) -> list[str]:
        return list(self.model.p["feature_order"])

    def guided(self, trials: list[str]) -> None:
        for i, g in enumerate(trials, start=1):
            self.trial_id = i
            self.step(f"[{i}/{len(trials)}] get ready: {g}", mk.trial_marker(mk.TRIAL_CUE, i, g), "CUE")
            self.run_for(CUE_S)
            self.step(f"        >>> {g} <<<  (hold {HOLD_S:.0f} s)", mk.trial_marker(mk.TRIAL_HOLD, i, g), "HOLD")
            self.run_for(HOLD_S)
            self.step(f"        rest, hand relaxed ({REST_S:.0f} s)", mk.trial_marker(mk.TRIAL_REST, i, g), "REST")
            self.run_for(REST_S)
        self.trial_id = 0
        self.source.insert_marker(mk.SESSION_END)
        self.run_for(0.3)

    def free(self, duration: float | None) -> None:
        self.step("FREE: do any gesture; Ctrl+C to stop", mk.FREE_RECORDING, "FREE")
        try:
            self.run_for(duration if duration else 1e9)
        except KeyboardInterrupt:
            pass
        self.source.insert_marker(mk.SESSION_END)
        self.run_for(0.3)

    # ---- output
    def labels(self) -> tuple[np.ndarray, np.ndarray]:
        marker = np.concatenate(self.markers) if self.markers else np.zeros(0)
        lab = np.full(len(marker), "NONE", dtype=object)
        cur = "NONE"
        for i, m in enumerate(marker):
            if m != 0:
                try:
                    info = mk.decode(m)
                    cur = info.label or "NONE"
                except ValueError:
                    pass
            lab[i] = cur
        return marker, lab

    def save(self, meta: dict) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        marker, lab = self.labels()
        x = np.vstack(self.samples) if self.samples else np.zeros((0, self.feats.n_channels))
        emg = pd.DataFrame(x, columns=self.model.p["channels"])
        emg.insert(0, "package_num", np.concatenate(self.pkg) if self.pkg else [])
        emg.insert(0, "timestamp", np.concatenate(self.ts) if self.ts else [])
        emg.insert(0, "sample_index", np.arange(len(emg)))
        emg["marker"] = marker.astype(int)
        emg["label"] = lab
        emg.to_csv(self.out_dir / "emg.csv", index=False)
        dec = pd.DataFrame(self.decisions)
        dec.to_csv(self.out_dir / "decisions.csv", index=False)

        result = {"windows": 0}
        if len(dec) and meta["mode"] == "guided":
            result = score(lab.astype(str), dec, self.model, self.window_len, self.fs)
        pkg = np.concatenate(self.pkg) if self.pkg else np.zeros(0)
        lost = int(((np.diff(pkg) % 256) - 1).clip(min=0).sum()) if len(pkg) > 1 else 0
        summary = {**meta, "samples": len(emg), "duration_s": round(len(emg) / self.fs, 1),
                   "packets_lost": lost, "decisions": len(dec), "scored_windows": result["windows"],
                   "baseline": None if self.baseline is None else self.baseline.tolist()}
        if result["windows"]:
            summary.update({"balanced_acc": round(result["balanced_acc"], 3),
                            "recall": {k: round(float(v), 3) for k, v in result["recall"].items()},
                            "share": result["share"].round(3).to_dict(orient="index"),
                            "trials": result["trials"].to_dict(orient="records")})
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return result | {"lost": lost}


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, help="folder with features.json and the model files")
    ap.add_argument("--kind", choices=("int8", "float32", "keras"), default="int8", help="model file to run (default int8)")
    ap.add_argument("--port", help="serial port of the dongle (default: [board] serial_port)")
    ap.add_argument("--synthetic", action="store_true", help="BrainFlow synthetic board (plumbing check only)")
    ap.add_argument("--free", action="store_true", help="free mode: live decision until Ctrl+C")
    ap.add_argument("--duration", type=float, help="free mode: stop after this many seconds")
    ap.add_argument("--reps", type=int, default=5, help="guided mode: trials per gesture (default 5)")
    ap.add_argument("--gestures", nargs="+", choices=mk.GESTURES, help="guided mode: gestures to test (default: all the model knows)")
    ap.add_argument("--cal", type=float, default=10.0, help="calibration rest in seconds (default 10)")
    ap.add_argument("--vote", type=int, default=1, help="majority vote over the last N decisions on screen (default 1 = none)")
    ap.add_argument("--subject", default="LIVE", help="name used in the output folder")
    ap.add_argument("--seed", type=int, help="trial order seed (default: from the clock)")
    args = ap.parse_args(argv)

    cfg = load_config()
    model = Model(args.model, args.kind)
    fs = float(cfg["board.sampling_rate_hz"])
    n_ch = len(cfg["board.channels"])
    if abs(float(model.p["fs"]) - fs) > 1e-6:
        print(f"model was trained at {model.p['fs']} Hz, board is set to {fs} Hz", file=sys.stderr)
        return 2
    if len(model.p["channels"]) != n_ch:
        print(f"model expects {len(model.p['channels'])} channels, [board] channels has {n_ch}", file=sys.stderr)
        return 2

    gestures = args.gestures or [g for g in mk.GESTURES if model.target(g) is not None]
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    trials = make_trials(gestures, args.reps, seed)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = APP_ROOT / "data" / f"LIVE_{args.subject}_{stamp}"
    print(f"model: {model.file.relative_to(APP_ROOT.parent) if model.file.is_relative_to(APP_ROOT.parent) else model.file}"
          f" ({model.p.get('model', '?')}), classes {model.classes}, input {model.n_frames} frames x {len(model.mean)}")

    source = make_source(cfg, "synthetic" if args.synthetic else "cyton", args.port)
    session = LiveSession(source, model, fs, n_ch, args.vote, out_dir)
    meta = {"mode": "free" if args.free else "guided", "model_dir": str(args.model), "model_file": str(model.file),
            "model": model.p.get("model"), "classes": model.classes, "source": source.kind,
            "started": dt.datetime.now().isoformat(timespec="seconds"), "calibration_s": args.cal,
            "vote": args.vote, "trial_order": [] if args.free else trials, "seed": seed}
    try:
        info = source.connect()
        print(f"connected: {info.get('board', source.kind)}")
        source.start()
        session.step("starting the stream... keep still", None, "WAIT")
        session.run_for(2.0)
        session.calibrate(args.cal)
        if args.free:
            session.free(args.duration)
        else:
            print(f"\nguided test: {len(trials)} trials ({', '.join(gestures)} x {args.reps}); "
                  f"cue {CUE_S} s, hold {HOLD_S:.0f} s, rest {REST_S:.0f} s")
            session.run_for(2.0)
            session.guided(trials)
    except KeyboardInterrupt:
        print("\nstopped by the user")
        meta["stopped_early"] = True
    except SourceError as e:
        print(f"\nsource error: {e}", file=sys.stderr)
        meta["error"] = str(e)
    finally:
        source.release()

    result = session.save(meta)
    print(f"\n\nsaved to {out_dir}")
    print(f"{len(session.decisions)} decisions, {result['lost']} packets lost")
    if result.get("windows"):
        print(f"\nscored windows (whole input inside one settled posture): {result['windows']}")
        print("share of the windows of each posture given to each class:")
        print(result["share"].round(2).to_string())
        recall = {str(k): round(float(v), 2) for k, v in result["recall"].items()}
        print(f"\nrecall per class: {recall}   balanced accuracy {result['balanced_acc']:.2f}")
        t = result["trials"]
        if len(t):
            print(f"trials right by majority vote over the hold: {int(t['ok'].sum())}/{len(t)}")
            for p, g in t.groupby("posture"):
                print(f"  {p:<6} {int(g['ok'].sum())}/{len(g)}  votes: {dict(Counter(g['vote']))}")
    return 0 if "error" not in meta else 1


if __name__ == "__main__":
    raise SystemExit(main())
