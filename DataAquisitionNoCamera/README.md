# Data acquisition without camera (prototype)

Guided forearm surface-EMG recording for the gesture prototype. The application connects to an
**OpenBCI Cyton** (8 channels, RF dongle, 250 Hz) through **BrainFlow**, takes the subject through a
timed protocol on screen and saves a clean, labelled CSV.

- **Gestures:** OPEN (fingers extended and spread - not rest), CLOSE (fist), INDEX (pointing),
  with REST between trials.
- **Session:** metadata form → guided quality check → calibration → block 1 (18 trials) → pause →
  block 2 (18 trials) → closing rest.
- **Two views:** the subject view shows only the task; the operator view shows live filtered
  signals, RMS, flat/railed flags, packet loss, controls and the bad-trial key.
- **Markers:** every phase change is stamped into the EMG stream (`board.insert_marker`) and
  logged in `events.csv`.
- **Swappable source:** the rest of the application only sees `emgacq.sources.SampleSource`.
  Implemented: Cyton and BrainFlow synthetic board; the Arduino GIGA stream of the main project
  and its simulator (for the future PCB).

This folder is self-contained (its own config, tests and docs). The only link to the main
project is the optional GIGA source, which imports `common.sources.giga` from `..`.

```
config.toml          the single configuration file (port, channels, timings, thresholds)
emgacq/              the application (python -m emgacq)
  sources/           SampleSource interface: BrainFlow (Cyton, synthetic), GIGA
  protocol.py        trial order (seeded) and session timeline
  markers.py         marker code table          labels.py   labels from markers
  session.py         session controller         recorder.py acquisition thread
  writer.py          emg.csv / events.csv       quality.py  quality-check metrics
  ui/                metadata form, subject view, operator view (PySide6 + pyqtgraph)
tools/inspect_session.py   check a recorded session (trials, RMS per gesture, plots)
tools/stream_quality.py    is the stream good, noisy or disturbed? (+ stream_quality.ipynb)
docs/DATA_FORMAT.md  files, columns, marker codes      docs/LAB_GUIDE.md  operator checklist
tests/               pytest suite (includes full sessions on the synthetic board)
data/                recordings (git-ignored)
```

## Setup from scratch

Python 3.13 (tested on Windows 11; 3.11+ should work with matching package versions).

**Windows (PowerShell)**
```powershell
cd DataAquisitionNoCamera
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest            # ~35 s
```

**macOS / Linux**
```bash
cd DataAquisitionNoCamera
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

(In the main repository the root `.venv` can be used as well: install `PySide6` and `pyqtgraph`
into it.)

### Finding the dongle's serial port

Plug in the dongle (switch on **GPC**), then `python -m emgacq --list-ports`, or:

| OS      | Where                                                                                      |
|---------|--------------------------------------------------------------------------------------------|
| Windows | Device Manager → Ports (COM & LPT) → "USB Serial Port (COM*n*)". Set its latency timer to 1 ms (Properties → Port Settings → Advanced), otherwise the data arrive in large bursts. |
| macOS   | `ls /dev/cu.usbserial-*` → e.g. `/dev/cu.usbserial-DM00Q8QM`                                |
| Linux   | `ls /dev/ttyUSB*` → e.g. `/dev/ttyUSB0`; add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, log out and in) |

Write it into `config.toml` → `[board] serial_port`, or pass `--port`. **Close the OpenBCI GUI**
first: two programs cannot hold the port at the same time.

## Running

```powershell
python -m emgacq                                   # Cyton, port from config.toml
python -m emgacq --port COM3                       # another port
python -m emgacq --synthetic                       # no hardware: BrainFlow synthetic board
python -m emgacq --synthetic --speed 10 --windowed # quick look, windows instead of full screen
python -m emgacq --synthetic --headless --speed 20 --subject T --session 1   # no GUI
python -m emgacq --free                            # free recording: raw stream, no protocol
python -m emgacq --free --duration 60              # free recording that stops after 60 s
python -m tools.inspect_session data/<folder>      # after the session
python -m tools.stream_quality [data/<folder>]     # stream quality verdict (default: latest)
python -m tools.stream_quality --all               # one line per session
```

1. **Form**: subject ID, session number, day, role, arm, band position, forearm length, channel
   map, electrode type, notes → **Connect**. Each channel in use is set to bipolar
   (`x<ch>060100X`: on, gain 24, normal input, in BIAS, SRB2 and SRB1 off) and the unused ones are
   powered down; every board reply is logged.
2. **Operator view**: check the live signals, then **Start session**.
3. **Quality check** (subject view guides the subject), then review: **Proceed**, or **Repeat**;
   after a FAIL, proceeding needs a note.
4. Calibration and block 1 run by themselves. **B** flags the last trial bad, **P** pauses.
5. **Pause** between blocks: at least 60 s, then **Space**.
6. Block 2, closing rest; the files are closed and `summary.txt` is written.

**Free recording** (`--free`). Same form, same channel setup and live view, but no quality
check, calibration or trials: after **Start recording** the raw stream is saved until **Stop
recording [Esc]** (or until `--duration` seconds have passed). Every row has phase `FREE` and
label `NONE`; the folder is `data/<subject>_S<session>_FREE_<YYYY-MM-DD_HHMM>/` and has no
`trial_order.json` or `quality_check.json`. Gaps are logged but nothing pauses. Use it for
recordings you want to drive yourself (e.g. testing electrodes, noise floor, ad-hoc gestures);
`--headless --free` needs `--duration`.

**Screens.** With a second monitor the subject view goes full screen there (`[display]
subject_screen`). With only the laptop screen: the operator view during setup, quality-check
review and data gaps; during the protocol the subject view with a one-line text status strip for
the operator (no signals). **O** brings the operator view back while the protocol waits.
Optional gesture images: `assets/OPEN.png`, `CLOSE.png`, `INDEX.png`, `REST.png`.

**Safety.** No samples for more than 1 s: the protocol pauses, the operator view says **NO
DATA**, the gap is logged; **Reconnect** reopens the board and the recording continues in the
same files. The stream is stopped and the board released on every exit path (end, abort, Ctrl+C,
closed window, exception).

## Configuration

Everything is in [`config.toml`](config.toml): source, port, channels in use, sampling rate,
timings, number of trials, quality thresholds (first guesses - tune them on the pilot), display.
Values that depend on unknown hardware are `"TODO_HW"`; a program that needs one stops and names
the key. Currently open: `board.serial_port`, `board.channel_positions`, `giga.url`.

## Output

`data/<subject>_S<session>_<YYYY-MM-DD_HHMM>/`: `emg.csv` (one row per sample: raw µV, BrainFlow
timestamp, package number, marker, phase, label, block, trial, bad flag), `events.csv`,
`session.json`, `quality_check.json`, `trial_order.json`, `summary.txt`. Free recordings:
`data/<subject>_S<session>_FREE_<YYYY-MM-DD_HHMM>/`, without the quality-check and trial-order
files. The saved channels are never filtered by this program. Details and the marker
code table: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Adding a source (e.g. the Arduino PCB)

Subclass `emgacq.sources.base.SampleSource` (`connect`, `start`, `read` → `Chunk`,
`insert_marker`, `release`, and `package_modulus`), add a branch in `emgacq/sources/__init__.py`
and a config section. `read()` must return the channels in use in µV with a timestamp and a
package counter per sample; a marker inserted on the host goes on the next sample delivered. See
`giga_source.py` for a source whose device cannot stamp markers itself.

## Testing

```powershell
python -m pytest                 # everything (~35 s)
python -m pytest -m "not slow"   # without the full synthetic sessions and the GIGA simulator
```

Trial-order generator (counts, no repeats, reproducibility), labels from markers (including
interrupted and repeated trials), CSV writer and bad-trial update, packet-loss detection,
quality metrics, the GIGA source against the simulator, and complete sessions on the BrainFlow
synthetic board with a bad trial, an operator pause, a simulated radio stall and a reconnection.
