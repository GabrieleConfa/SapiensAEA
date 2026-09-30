"""Generate the trial and ramp orders of every session.

The block between the BEGIN/END markers is the reference code of the protocol and must not be
edited. Run:

    python -m protocol.generate_sequences

It writes protocol/sequences.json and prints the orders for checking against the lab map.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---- BEGIN reference code (verbatim from the protocol) ----
import random
GRIPS = ["OPEN", "FIST", "PINCH", "POINT"]
ORDER = [("A",1,1),("B",1,1),("C",1,1),("A",2,1),("B",2,1),("C",2,1),("A",3,2),("B",3,2),("C",3,2)]  # (subject, session, day); index k = 0..8, seed = 1000 + 17*k
def trials(seed):
    r = random.Random(seed)
    while True:
        s = GRIPS*8; r.shuffle(s)
        if all(s[i] != s[i+1] for i in range(len(s)-1)): return s
def ramps(seed):
    r = random.Random(seed+100); g = ["FIST", "PINCH", "POINT"]; r.shuffle(g); return g
# ---- END reference code ----

SEQUENCES_PATH = Path(__file__).with_name("sequences.json")
TRAIN_DAY = 1


def session_id(subject: str, session: int) -> str:
    return f"{subject}{session}"


def build() -> dict:
    sessions = []
    for k, (subject, session, day) in enumerate(ORDER):
        seed = 1000 + 17 * k
        sessions.append(
            {
                "id": session_id(subject, session),
                "k": k,
                "subject": subject,
                "session": session,
                "day": day,
                "role": "train" if day == TRAIN_DAY else "test",
                "seed": seed,
                "trials": trials(seed),
                "ramps": ramps(seed),
            }
        )
    return {
        "grips": GRIPS,
        "python_version": sys.version.split()[0],
        "seed_rule": "seed = 1000 + 17*k, ramps use seed + 100",
        "sessions": sessions,
    }


def format_table(data: dict) -> str:
    lines = []
    for s in data["sessions"]:
        lines.append(f"{s['id']}  k={s['k']}  day {s['day']} ({s['role']})  seed {s['seed']}")
        lines.append(f"  ramps  : {', '.join(s['ramps'])}")
        for block, (a, b) in enumerate([(0, 16), (16, 32)], start=1):
            items = [f"{i + 1:2d}:{g}" for i, g in enumerate(s["trials"][a:b], start=a)]
            lines.append(f"  block {block}: " + "  ".join(items[:8]))
            lines.append("           " + "  ".join(items[8:]))
    return "\n".join(lines)


def load_sequences(path: Path = SEQUENCES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_session(session_key: str, path: Path = SEQUENCES_PATH) -> dict:
    """Look up a session by id, e.g. 'A1'."""
    for s in load_sequences(path)["sessions"]:
        if s["id"] == session_key:
            return s
    raise KeyError(f"session {session_key!r} not in {path}")


def main() -> None:
    data = build()
    SEQUENCES_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(format_table(data))
    print(f"\nwritten {SEQUENCES_PATH}")


if __name__ == "__main__":
    main()
