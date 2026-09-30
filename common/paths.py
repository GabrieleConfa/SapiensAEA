"""Session folder naming: data/<subject>_S<n>_<YYYY-MM-DD>/ (plus _run2, _run3 for re-runs)."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from common.config import REPO_ROOT, Config


def data_root(cfg: Config) -> Path:
    p = Path(cfg["project.data_dir"])
    return p if p.is_absolute() else REPO_ROOT / p


def session_folder_name(subject: str, session: int, day: Date) -> str:
    return f"{subject}_S{session}_{day.isoformat()}"


def next_session_dir(cfg: Config, subject: str, session: int, day: Date | None = None) -> Path:
    """Folder for the next recording of this session today.

    A folder that already holds a session.json is a finished (or failed) run and is never reused:
    a re-run the same day goes to <name>_run2, _run3, ...
    The quality check and the acquisition both call this, so the quality report lands in the
    folder of the run that follows it.
    """
    day = day or Date.today()
    root = data_root(cfg)
    name = session_folder_name(subject, session, day)
    candidate = root / name
    run = 2
    while (candidate / "session.json").exists():
        candidate = root / f"{name}_run{run}"
        run += 1
    return candidate
