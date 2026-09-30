# Session data format

One folder per recording: `data/<subject>_S<n>_<YYYY-MM-DD>/`. A re-run of the same session on the
same day goes to `..._run2`, `..._run3`; a folder holding a `session.json` is never overwritten.

All laptop times are `time.perf_counter_ns()` of the single acquisition process (one clock for
EMG, video and cues). `session.json → clock` pairs it with the wall clock once.

| File                  | Content                                                                                         |
|-----------------------|-------------------------------------------------------------------------------------------------|
| `emg_raw.bin`         | Byte-exact copy of the GIGA serial stream (packets + `#` reply lines), see `SERIAL_PROTOCOL.md` |
| `emg_host_times.csv`  | `end_offset,host_time_ns`: stream length after each USB read, and when that read returned       |
| `emg_cyton.csv`       | Cyton runs only, instead of `emg_raw.bin`: `counter,host_time_ns,ch1_uV..ch8_uV`                |
| `video.mp4`           | Camera video (container fps is nominal; use `frame_times.csv`)                                  |
| `frame_times.csv`     | `frame_index,capture_index,host_time_ns` (capture time right after `grab()`)                    |
| `cues.csv`            | One row per timeline segment, see below                                                         |
| `sync_events.csv`     | `event,host_time_ns,emg_counter,planned_t_s,note`, see below                                    |
| `quality_check.json`  | Report of `tools/quality_check.py`, if run before the session                                  |
| `session.json`        | Metadata, see below                                                                             |

## Why raw bytes

`emg_raw.bin` is exactly what came over USB, so any later parser fix can reprocess it. Read it with
`common.recording.load_emg(folder)`, which returns counters, flags, µV and a host time per sample,
or export to CSV with `python -m common.recording <folder> --csv`.

The EMG time base is the sample counter (`t = counter / fs`), clocked on the GIGA. Host receive
times are jittery (USB buffering) and are used only to relate the counter to the laptop clock
coarsely; the precise EMG ↔ video link is the LED.

## cues.csv

`segment_index, phase, kind, grip, trial, rep, text, planned_start_s, planned_duration_s,
closure_from, closure_to, planned_host_time_ns, host_time_ns, emg_last_counter`

- `kind`: `sync`, `cal_rest`, `cal_cue`, `cal_contraction`, `cal_contraction_rest`, `ref_cue`,
  `ref_pose`, `ref_rest`, `trial_cue`, `trial_hold`, `trial_rest`, `pause`, `ramp_intro`,
  `ramp_move`, `ramp_hold`, `ramp_rest`, `closing_rest`.
- `grip`: the class the subject is asked to perform (`OPEN FIST PINCH POINT EXTENSION REST`),
  empty for cues and sync.
- `planned_host_time_ns` is the schedule; `host_time_ns` is when the cue was drawn (≤ one UI tick
  later). Every segment is logged, in order, even if a tick was late.
- `closure_from/to`: target closure for reference poses and ramps (0 open, 1 fist).

## sync_events.csv

| event             | meaning                                                                         |
|-------------------|---------------------------------------------------------------------------------|
| `session_start`   | SPACE pressed, timeline starts                                                  |
| `flash_request`   | laptop asked for a flash (`planned_t_s` = timeline time)                        |
| `led_rise`        | GIGA reported the LED on at sample `emg_counter` (exact), received at host time |
| `session_end`     | timeline finished                                                               |
| `session_aborted` | operator aborted                                                                |

## session.json

Subject, session, day, role (`train`/`test`), trial and ramp order, sampling rate, channels,
front-end, device INFO, µV per code, band position (% of forearm length from the elbow), forearm
length, impedances (kΩ, optional), notes, hand and orientation, camera settings read back from the
driver, LED ROI, protocol timings, planned flash times, clock anchor, config SHA-256, software
commit, status (`recording` → `completed` / `aborted`), EMG and video statistics.
