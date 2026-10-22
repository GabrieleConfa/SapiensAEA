# Session data format

One folder per session: `data/<subject>_S<session>_<YYYY-MM-DD_HHMM>/` (a second folder in the
same minute gets `_2`). The folder is created when the operator presses **Start session**.
A free recording (`--free`, no protocol) goes to `data/<subject>_S<session>_FREE_<YYYY-MM-DD_HHMM>/`
and has no `trial_order.json` or `quality_check.json`.

| File                 | Written                                    | Content                                       |
|----------------------|--------------------------------------------|-----------------------------------------------|
| `trial_order.json`   | at start, before the first step            | seed, seed rule, the two blocks, trial list   |
| `session.json`       | at start (`recording`), again at the end   | metadata, source, settings, status, stats     |
| `emg.csv`            | continuously, flushed (fsync) every ≤ 1 s  | one row per sample, raw µV, labels            |
| `events.csv`         | continuously, flushed on every row         | every phase change and operator action        |
| `quality_check.json` | after each quality check                   | per-channel metrics, PASS/FAIL, decision, note |
| `summary.txt`        | at the end                                 | duration, samples, loss, trials, QC result    |
| `inspect/*.png`      | by `python -m tools.inspect_session`       | signal and RMS plots                          |

A crash loses at most one second of `emg.csv`. If the session did not close normally,
`session.json` still says `"status": "recording"`; `tools/inspect_session.py` then applies the
bad-trial flags from `events.csv` itself.

## emg.csv

| Column          | Meaning                                                                                  |
|-----------------|------------------------------------------------------------------------------------------|
| `sample_index`  | row number, 0-based, continuous (also across a reconnection)                             |
| `timestamp`     | BrainFlow timestamp channel: laptop wall clock (Unix s) when the packet arrived. Arrives in bursts; for timing within the session use `sample_index / fs`. GIGA: wall clock of the first sample + counter / fs. |
| `package_num`   | board package counter (Cyton and synthetic: 0-255, wraps; GIGA: sample counter). A step > 1 = lost samples; it restarts after a reconnection |
| `ch<k>`         | raw µV of board channel k (only the channels in use), unfiltered                          |
| `marker`        | 0, or the marker code that starts at this sample (table below)                           |
| `phase`         | `QUALITY` `CALIBRATION` `CUE` `HOLD` `REST` `PAUSE`; `FREE` in a free recording          |
| `label`         | `OPEN` `CLOSE` `INDEX` `REST` `TRANSITION`; `NONE` in a free recording                   |
| `block`         | 1 or 2 during trials, else 0                                                             |
| `trial_id`      | 1-36 during a trial (cue, hold, rest), else 0                                            |
| `trial_bad`     | 1 for the trials the operator flagged bad and for interrupted attempts, else 0 (set when the session closes) |

The file starts at the sample carrying the first protocol marker and ends with the
`SESSION_END` (or `SESSION_ABORT`) row. The labels of every row can be rebuilt from the marker
column alone (`emgacq.labels.Labeler`).

### Labels by step

| Step                                   | phase       | label                |
|----------------------------------------|-------------|----------------------|
| quality check rest / cue / fist        | QUALITY     | REST / TRANSITION / CLOSE |
| quality-check review (operator)        | PAUSE       | REST                 |
| calibration rest (start, between)      | CALIBRATION | REST                 |
| calibration cue / contraction          | CALIBRATION | TRANSITION / gesture |
| trial cue ("Get ready: CLOSE")         | CUE         | TRANSITION           |
| trial hold                             | HOLD        | gesture              |
| trial rest                             | REST        | REST                 |
| pause between blocks                   | PAUSE       | REST                 |
| pause by the operator or for a data gap | PAUSE      | TRANSITION           |
| closing rest                           | REST        | REST                 |

### Interruptions

A pause (operator, or no data for more than 1 s) during a trial's cue or hold repeats the trial
from its cue on resume; during a rest, only the rest is repeated. The interrupted attempt keeps
its rows (label as recorded) with `trial_bad = 1`; the repetition carries the same `trial_id`. A
pause during the quality check restarts the check; during a calibration cue or contraction it
repeats that contraction.

While no sample has arrived for 0.25 s, the protocol does not move to the next step.

## Marker codes

Stamped into the EMG stream with `board.insert_marker(code)` at every phase change (BrainFlow puts
it on the next sample received; several queued markers land on consecutive samples). Gesture
codes: OPEN = 1, CLOSE = 2, INDEX = 3. `python -m emgacq.markers` prints this table.

| Code   | Name             | phase       | label      |
|--------|------------------|-------------|------------|
| 2      | SESSION_END      | -           | -          |
| 3      | SESSION_ABORT    | -           | -          |
| 10     | QC_REST          | QUALITY     | REST       |
| 11     | QC_CUE           | QUALITY     | TRANSITION |
| 12     | QC_FIST          | QUALITY     | CLOSE      |
| 13     | QC_REVIEW        | PAUSE       | REST       |
| 20     | CAL_REST         | CALIBRATION | REST       |
| 21-23  | CAL_CUE_\<g\>    | CALIBRATION | TRANSITION |
| 31-33  | CAL_HOLD_\<g\>   | CALIBRATION | \<g\>      |
| 40     | BLOCK_PAUSE      | PAUSE       | REST       |
| 41     | OPERATOR_PAUSE   | PAUSE       | TRANSITION |
| 42     | GAP_PAUSE        | PAUSE       | TRANSITION |
| 50     | CLOSING_REST     | REST        | REST       |
| 60     | FREE_RECORDING   | FREE        | NONE       |
| 1TTgg  | TRIAL_CUE        | CUE         | TRANSITION |
| 2TTgg  | TRIAL_HOLD       | HOLD        | \<g\>      |
| 3TTgg  | TRIAL_REST       | REST        | REST       |

Trial markers: `10000 * kind + 100 * trial_id + gesture`, e.g. `20702` = HOLD of trial 7, CLOSE.

## events.csv

`wall_time, unix_time, sample_index, marker, event, phase, gesture, block, trial_id, detail`

`sample_index` is the number of rows written when the event was logged (the marker itself lands a
few samples later, on the next packet). Events: the marker names above, plus `SESSION_START`,
`PAUSE`, `RESUME`, `GAP_START`, `GAP_END`, `RECONNECT_START/OK/FAILED`, `PACKET_LOSS` (at most one
row per second), `QC_RESULT`, `QC_PROCEED`, `QC_REPEAT`, `TRIAL_BAD`, `TRIAL_BAD_UNDO`,
`BLOCK_PAUSE_END`, `BOARD_CONFIG` (each channel command and the board reply), `DISPLAY`,
`BAD_ROWS_UPDATED`, `SESSION_CLOSED`, `ERROR`, `STOP` (free recording stopped by the operator).

## session.json

Subject, session, day, role (`train`/`test`), arm, band position (% of forearm length from the
elbow), forearm length (cm), channel map (channel, CSV column, position), electrode type, notes;
source (board, port, sampling rate, channels, full scale, every configuration command with the
board's reply); protocol timings, quality thresholds, seed, speed factor, planned duration; the
whole config and its SHA-256; app version, git commit, Python and OS; status (`recording`,
`completed`, `aborted`), `mode` (`protocol` or `free`; in a free recording `protocol`,
`quality_thresholds` and `seed` are `null`), start and end time; at the end, stats (rows, packets received and lost,
gaps, reconnections, bad trials, interrupted attempts, last quality-check result).

## quality_check.json

`final_result`, `final_note`, and `attempts`: one entry per check with, per channel, the share of
48-52 Hz power at rest and during the fist, rest and fist RMS (20-120 Hz), fist/rest ratio, flat,
railed fraction and the issues found; overall `result` (`PASS`, `FAIL`, or `NOT_JUDGED` when
`[quality] judge = false`: same findings, nothing fails) with `reasons`, packet loss during the
check, the thresholds, the operator's `decision` (`proceed` / `repeat`) and `operator_note`
(required to proceed after a FAIL).

## trial_order.json

`seed` (from `SHA-256("<subject>|<session>")`, first 8 bytes), `seed_rule`, `blocks` (two lists of
18 gestures, 6 each, never the same twice in a row) and `trials` (`trial_id`, `block`, `gesture`).
