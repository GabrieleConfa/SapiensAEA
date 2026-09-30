# EMG grip

Hand grip estimation (grip class + closure 0–1) from forearm surface EMG, running stand-alone on an
Arduino GIGA R1. Training data are recorded through the same signal chain that runs the model: one
firmware with a RECORD mode (stream raw samples to the laptop) and an INFER mode (features and
model on board, Stage 3).

This README grows with each stage; the full pipeline description is written in Stage 4.

## Layout

```
config/config.toml     the single config file ("TODO_HW" = hardware detail not known yet)
firmware/emg_giga/     GIGA sketch: sampling source interface, ADS1299 / GIGA ADC sources, LED sync
common/                config, packet parser, stream sources (GIGA, simulator, Cyton), camera, quality metrics
protocol/              sequence generator (sequences.json) and session timeline
acquisition/           session recorder: EMG + video + cues on one clock
simulator/             GIGA stream simulator (synthetic EMG + LED) for work without hardware
tools/                 quality_check.py, camera_preview.py
scripts/               download_models.py, gen_firmware_config.py
docs/                  SERIAL_PROTOCOL.md, DATA_FORMAT.md (LAB_GUIDE.md in Stage 4)
tests/                 pytest suite
```

## Setup from scratch (Windows, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.download_models      # MediaPipe hand model -> models/, hash logged in models/MODELS.md
python -m pytest                       # ~45 s, runs the simulator end to end
```

`opencv-contrib-python` is pinned instead of `opencv-python`: mediapipe depends on the contrib
package, and having both installed makes them overwrite each other's `cv2`.

All programs are run from the repository root as modules (`python -m ...`).

### Firmware

```powershell
winget install ArduinoSA.CLI
arduino-cli core install arduino:mbed_giga
arduino-cli lib install Arduino_AdvancedAnalog
python -m scripts.gen_firmware_config          # config.toml -> firmware/emg_giga/config_gen.h
arduino-cli compile --fqbn arduino:mbed_giga:giga firmware/emg_giga
arduino-cli upload  --fqbn arduino:mbed_giga:giga -p COMx firmware/emg_giga
```

While pins are `TODO_HW` the generator refuses to run; `--allow-todo` writes placeholders for a
compile check, and that firmware answers `INFO` but refuses to `START`.

## Stage 1 tools

| Task                        | Command                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| Print the trial/ramp orders | `python -m protocol.generate_sequences`                                  |
| Find the webcam index       | `python -m tools.camera_preview --list`                                  |
| Position camera, set LED ROI| `python -m tools.camera_preview` (R = LED region, S = driver settings)   |
| Quality check (GIGA)        | `python -m tools.quality_check --subject A --session 1`                  |
| Quality check (Cyton)       | `python -m tools.quality_check --subject A --session 1 --source cyton`   |
| Record a session            | `python -m acquisition --subject A --session 1`                          |
| Inspect / export a recording| `python -m common.recording data/A_S1_2026-10-05 --csv`                  |

Without hardware, start `python -m simulator.giga_sim` in one terminal and add
`--source sim --camera synthetic` to the quality check or the acquisition
(`--speed 20 --windowed` makes an acquisition dry run take under a minute).

The OpenBCI GUI and a BrainFlow script cannot hold the Cyton dongle's serial port at the same time:
stop the GUI session before running the quality check or the acquisition with `--source cyton`.

## Configuration

Everything lives in `config/config.toml`. Values that depend on hardware not yet known are set to
`"TODO_HW"`; a program that needs one of them stops and names the key. Currently open:
`emg.ads1299.pin_*`, `sync.led_pin`, `calibration.button_pin`, `serial.port`, `camera.index`,
`cyton.serial_port`, `output.kind`, `emg.giga_adc.front_end_gain`.
