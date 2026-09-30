# SapiensAEA — EMG grip estimation

Estimation of hand grips from forearm surface EMG, running stand-alone on an **Arduino GIGA R1**.
The system outputs a **grip class** (OPEN, FIST, PINCH, POINT, REST) and a **closure level** (0 = open
hand, 1 = fist) to a hand prototype.

The training data are recorded through **the same signal chain that runs the model**: a single
firmware with two modes.

- **RECORD** — the GIGA samples 8 EMG channels at 1 kHz on a hardware clock and streams raw samples
  to a laptop over USB, together with the sync-LED events used to align EMG and video.
- **INFER** — same sampling and filters; features, classifier and closure regressor run on the
  board, after a short on-device calibration (planned, see [Roadmap](#roadmap)).

```
 electrodes ─► OpenBCI front-end (ADS1299, 8 ch) ─► Arduino GIGA R1 ─USB─► laptop
                                                      │   sync LED            ├─ acquisition (EMG + video + cues)
                                                      │      ▼                ├─ quality check
                                                      │   webcam ─────────────┘  offline pipeline: MediaPipe, features, models
                                                      └─► prototype (INFER mode)
```

## Roadmap

| Stage | Content                                                                                   | Status      |
|-------|-------------------------------------------------------------------------------------------|-------------|
| 1     | Firmware RECORD mode + LED sync, acquisition, quality check, simulator, MediaPipe setup    | done        |
| 2     | Offline hand tracking, closure, EMG–video alignment, labels, features, models, evaluation | planned     |
| 3     | Portable C features, model export, INFER mode, on-device calibration, latency report      | planned     |
| 4     | Operator lab guide, full pipeline documentation                                           | planned     |

Hardware-dependent values (pins, ports, camera index, prototype interface) are still open; see
[Configuration](#configuration).

## Repository layout

```
config/config.toml     single configuration file
firmware/emg_giga/     GIGA sketch: sampling-source interface, ADS1299 (SPI) and GIGA-ADC sources, LED sync, packets
common/                config loader, packet parser, stream sources (GIGA, simulator, Cyton), camera, quality metrics
protocol/              trial/ramp sequence generator (sequences.json) and session timeline
acquisition/           session recorder: EMG stream + video + cues on one laptop clock
simulator/             GIGA stream simulator (synthetic EMG + LED events) for work without hardware
tools/                 quality_check.py (pre-session signal check), camera_preview.py (camera positioning)
scripts/               download_models.py (MediaPipe model), gen_firmware_config.py (config -> firmware header)
docs/                  SERIAL_PROTOCOL.md, DATA_FORMAT.md
tests/                 pytest suite (runs the whole laptop side against the simulator)
```

## Getting started

Requirements: Windows 10/11, Python 3.13, an external USB webcam; for the firmware, `arduino-cli`.

```powershell
git clone https://github.com/GabrieleConfa/SapiensAEA.git
cd SapiensAEA
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.download_models      # MediaPipe Hand Landmarker -> models/ (SHA-256 logged in models/MODELS.md)
python -m pytest                       # ~45 s
```

All programs are run from the repository root as modules (`python -m ...`).

> `opencv-contrib-python` is used instead of `opencv-python`: MediaPipe depends on the contrib
> package, and having both installed makes them overwrite each other's `cv2` module.

### Try it without hardware

```powershell
python -m simulator.giga_sim                                        # terminal 1
python -m tools.quality_check --source sim --camera synthetic       # terminal 2
python -m acquisition --subject A --session 1 --source sim --camera synthetic --speed 20 --windowed
```

The simulator speaks the same serial protocol as the firmware over `socket://localhost:8765`, and the
synthetic camera draws the sync LED, so the full recording chain — including EMG–video
synchronisation — can be exercised on any laptop.

### Firmware

```powershell
winget install ArduinoSA.CLI
arduino-cli core install arduino:mbed_giga
arduino-cli lib install Arduino_AdvancedAnalog
python -m scripts.gen_firmware_config          # config/config.toml -> firmware/emg_giga/config_gen.h
arduino-cli compile --fqbn arduino:mbed_giga:giga firmware/emg_giga
arduino-cli upload  --fqbn arduino:mbed_giga:giga -p COMx firmware/emg_giga
```

While pins are unset the generator refuses to run; `--allow-todo` writes placeholders for a compile
check, and that firmware answers `INFO` but refuses to `START`, so it never drives unknown pins.

## Recording a session

| Step                            | Command                                                                  |
|---------------------------------|--------------------------------------------------------------------------|
| Print the trial and ramp orders | `python -m protocol.generate_sequences`                                  |
| Find the webcam index           | `python -m tools.camera_preview --list`                                  |
| Position camera, mark the LED   | `python -m tools.camera_preview` (R = LED region, S = driver settings)   |
| Quality check (GIGA)            | `python -m tools.quality_check --subject A --session 1`                  |
| Quality check (Cyton reference) | `python -m tools.quality_check --subject A --session 1 --source cyton`   |
| Record                          | `python -m acquisition --subject A --session 1`                          |
| Inspect / export a recording    | `python -m common.recording data/A_S1_2026-10-05 --csv`                  |

The quality check shows per channel the spectrum, the share of 48–52 Hz power, the fist/rest RMS
ratio and flat or saturated channels, checks that the sync LED is seen in both the EMG stream and
the video, and ends with PASS/FAIL. MediaPipe never runs during recording: the acquisition only saves
the video and a capture timestamp per frame; hand tracking runs offline.

The OpenBCI GUI and a BrainFlow script cannot hold the Cyton dongle's serial port at the same time:
stop the GUI session before using `--source cyton`.

## Protocol

Four grips — **OPEN** (fingers actively extended and spread, not rest), **FIST** (power grip),
**PINCH** (thumb tip to index tip), **POINT** (index extended, other fingers flexed) — plus a relaxed
**REST** state between trials. Three subjects (A, B, C); Day 1 training sessions A1, B1, C1, A2, B2, C2
(electrodes re-applied between a subject's two sessions), Day 2 test sessions A3, B3, C3 one week later.

Each session (≈ 14.5 min): LED sync → calibration (rest, reference contractions, reference poses) →
block 1 (trials 1–16) → pause → closure ramps for FIST, PINCH, POINT → pause → block 2 (trials 17–32) →
closing (rest, reference poses, LED sync). A trial is a 1 s cue, 4 s hold, 4 s rest. The sync LED
flashes at the start, at the end and every 60 s in between. Trial and ramp orders are generated
deterministically in `protocol/generate_sequences.py`; timings are in `config/config.toml`.

## Data

One folder per recording, `data/<subject>_S<n>_<YYYY-MM-DD>/`, with the byte-exact EMG stream,
video with per-frame timestamps, cue and sync-event logs and a `session.json` with all metadata.
See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) and [docs/SERIAL_PROTOCOL.md](docs/SERIAL_PROTOCOL.md).
Recorded data are not part of the repository.

## Configuration

Everything is set in [`config/config.toml`](config/config.toml). Values that depend on hardware not
yet fixed are set to `"TODO_HW"`; any program that needs one stops and names the key. Currently open:
`emg.ads1299.pin_*`, `sync.led_pin`, `calibration.button_pin`, `serial.port`, `camera.index`,
`cyton.serial_port`, `output.kind`.

## Testing

```powershell
python -m pytest            # full suite
python -m pytest -m "not slow"   # skip the real-time simulator tests
```

The suite covers packet parsing and resynchronisation, the sequence generator and session timeline,
the firmware config generator, quality metrics, the MediaPipe setup, and end-to-end recordings with
the simulator in which the LED seen in the video is matched to the LED sample marked in the EMG
stream.
