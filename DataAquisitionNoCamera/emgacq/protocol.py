"""Trial order and session timeline.

Trial order: per block, every gesture `trials_per_gesture_per_block` times, shuffled, never the
same gesture twice in a row (also across the block boundary), and block 2 never equal to block 1.
The random generator is seeded from the subject ID and the session number, so the order is
reproducible on any machine:

    seed = first 8 bytes (big endian) of SHA-256("<subject>|<session>")

Timeline: the ordered list of steps the subject goes through; the session controller walks it.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from emgacq import markers as mk
from emgacq.config import Config

SEED_RULE = 'seed = int.from_bytes(sha256(f"{subject}|{session}").digest()[:8], "big")'


def session_seed(subject: str, session: int) -> int:
    digest = hashlib.sha256(f"{subject.strip()}|{int(session)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _block(rng: random.Random, gestures: list[str], per_gesture: int, previous: str | None) -> list[str]:
    """Random sequence with no two equal neighbours, drawn gesture by gesture.

    Each pick is weighted by the remaining count; a dead end (only the previous gesture left)
    restarts the block. Dead ends are rare, and the result depends only on the rng state.
    """
    while True:
        left = {g: per_gesture for g in gestures}
        seq: list[str] = []
        prev = previous
        while any(left.values()):
            options = [g for g in gestures if left[g] and g != prev]
            if not options:
                break
            g = rng.choices(options, weights=[left[o] for o in options])[0]
            seq.append(g)
            left[g] -= 1
            prev = g
        if len(seq) == per_gesture * len(gestures):
            return seq


def trial_order(subject: str, session: int, gestures: list[str], per_gesture: int,
                n_blocks: int) -> dict:
    seed = session_seed(subject, session)
    rng = random.Random(seed)
    blocks: list[list[str]] = []
    for _ in range(n_blocks):
        while True:
            seq = _block(rng, list(gestures), per_gesture, blocks[-1][-1] if blocks else None)
            if seq not in blocks:
                break
        blocks.append(seq)
    trials = []
    tid = 0
    for b, seq in enumerate(blocks, start=1):
        for g in seq:
            tid += 1
            trials.append({"trial_id": tid, "block": b, "gesture": g})
    return {"subject": subject, "session": int(session), "seed": seed, "seed_rule": SEED_RULE,
            "gestures": list(gestures), "trials_per_gesture_per_block": per_gesture,
            "blocks": blocks, "trials": trials}


def trial_order_from_config(cfg: Config, subject: str, session: int) -> dict:
    p = cfg.section("protocol")
    return trial_order(subject, session, p["gestures"], int(p["trials_per_gesture_per_block"]),
                       int(p["n_blocks"]))


# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    index: int
    stage: str             # what the operator sees: "Quality check", "Calibration", "Block 1", ...
    kind: str              # qc_rest qc_cue qc_fist qc_review cal_rest cal_cue cal_hold
                           # trial_cue trial_hold trial_rest block_pause closing_rest
    marker: int
    duration_s: float      # 0 for steps that wait for the operator
    gesture: str = ""
    block: int = 0
    trial_id: int = 0
    restart_index: int = 0  # where the protocol resumes if it is paused during this step
    wait_operator: bool = False
    min_s: float = 0.0     # minimum time before the operator can continue (wait steps)

    @property
    def info(self) -> mk.MarkerInfo:
        return mk.decode(self.marker)


def build_timeline(cfg: Config, order: dict) -> list[Step]:
    p = cfg.section("protocol")
    q = cfg.section("quality")
    steps: list[Step] = []

    def add(**kw) -> Step:
        kw.setdefault("restart_index", len(steps))
        s = Step(index=len(steps), **kw)
        steps.append(s)
        return s

    # Quality check: rest, cue, fist - `reps` times - then the operator reviews the result.
    # A pause restarts the whole check.
    qc_start = len(steps)
    for _ in range(int(q["reps"])):
        add(stage="Quality check", kind="qc_rest", marker=mk.QC_REST, duration_s=q["rest_s"],
            restart_index=qc_start)
        add(stage="Quality check", kind="qc_cue", marker=mk.QC_CUE, duration_s=q["cue_s"],
            gesture=mk.QC_GESTURE, restart_index=qc_start)
        add(stage="Quality check", kind="qc_fist", marker=mk.QC_FIST, duration_s=q["fist_s"],
            gesture=mk.QC_GESTURE, restart_index=qc_start)
    add(stage="Quality check", kind="qc_review", marker=mk.QC_REVIEW, duration_s=0.0,
        wait_operator=True)

    # Calibration: rest, then reference contractions with a rest after each one.
    # A pause during a cue or contraction repeats that contraction.
    add(stage="Calibration", kind="cal_rest", marker=mk.CAL_REST, duration_s=p["calibration_rest_s"])
    for g in p["calibration_gestures"]:
        for _ in range(int(p["calibration_reps"])):
            cue = add(stage="Calibration", kind="cal_cue", marker=mk.cal_cue_marker(g),
                      duration_s=p["calibration_cue_s"], gesture=g)
            add(stage="Calibration", kind="cal_hold", marker=mk.cal_hold_marker(g),
                duration_s=p["calibration_contraction_s"], gesture=g, restart_index=cue.index)
            add(stage="Calibration", kind="cal_rest", marker=mk.CAL_REST,
                duration_s=p["calibration_between_rest_s"])

    # Blocks of trials, with an operator-paced pause in between.
    # A pause during a cue or hold repeats that trial; a pause during its rest does not.
    n_blocks = len(order["blocks"])
    for b in range(1, n_blocks + 1):
        if b > 1:
            add(stage="Pause", kind="block_pause", marker=mk.BLOCK_PAUSE, duration_s=0.0,
                wait_operator=True, min_s=p["block_pause_min_s"])
        for t in (t for t in order["trials"] if t["block"] == b):
            g, tid = t["gesture"], t["trial_id"]
            cue = add(stage=f"Block {b}", kind="trial_cue",
                      marker=mk.trial_marker(mk.TRIAL_CUE, tid, g), duration_s=p["cue_s"],
                      gesture=g, block=b, trial_id=tid)
            add(stage=f"Block {b}", kind="trial_hold", marker=mk.trial_marker(mk.TRIAL_HOLD, tid, g),
                duration_s=p["hold_s"], gesture=g, block=b, trial_id=tid, restart_index=cue.index)
            add(stage=f"Block {b}", kind="trial_rest", marker=mk.trial_marker(mk.TRIAL_REST, tid, g),
                duration_s=p["rest_s"], gesture=g, block=b, trial_id=tid)

    add(stage="Closing", kind="closing_rest", marker=mk.CLOSING_REST, duration_s=p["closing_rest_s"])
    return steps


def planned_duration_s(steps: list[Step]) -> float:
    return sum(s.duration_s + s.min_s for s in steps)
