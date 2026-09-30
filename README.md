# EMG grip

Hand grip estimation (grip class + closure 0–1) from forearm surface EMG, running stand-alone on an
Arduino GIGA R1.

This README grows with each stage; the full pipeline description is written in Stage 4.

## Setup (Windows, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

All programs are run from the repository root as modules, e.g. `python -m tools.camera_preview`.

## Configuration

Everything lives in `config/config.toml`. Values that depend on hardware not yet known are set to
`"TODO_HW"`; a program that needs one of them stops and names the key.
