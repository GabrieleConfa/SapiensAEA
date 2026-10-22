# Lab day - operator checklist

One session ≈ 9 min of protocol (plus the pause you choose and setup). Keys: **P** pause/resume, **B** flag last trial bad,
**Space** proceed after the quality check / start the next block, **R** repeat the quality
check, **Esc** abort (asks first), **O** operator view (single screen).

## Before the subject arrives
- [ ] Laptop on mains power, sleep and screen saver off, notifications off.
- [ ] OpenBCI GUI **closed** (it holds the dongle's port). Dongle plugged in, switch on **GPC**.
- [ ] Cyton battery charged/fresh; board switch on **PC**.
- [ ] Port in `config.toml` (`[board] serial_port`) matches `python -m emgacq --list-ports`.
- [ ] Second monitor for the subject connected (or plan the single-screen mode, see below).
- [ ] Dry run: `python -m emgacq --synthetic --speed 10` - the windows open, the subject screen
      appears on the right monitor.
- [ ] Free disk space, `data/` backed up from the previous session.

## Electrode setup
- [ ] Measure forearm length (elbow crease to wrist crease) and mark the band position
      (% from the elbow) - both go into the form.
- [ ] Clean the skin (alcohol), let it dry; shave if needed.
- [ ] Electrode pairs along the muscle fibres, fixed spacing; reference/bias on a bony spot
      (elbow); cables strain-relieved and taped, not touching each other more than necessary.
- [ ] Write the channel → position map into the form; it is saved in `session.json`.
- [ ] Subject seated, forearm supported, same posture for the whole session.

## Running the quality check
- [ ] Fill the form, **Connect**. The operator view shows live signals: every channel moves when
      the subject contracts lightly, none flat or railed, packet loss ~0.
- [ ] Explain the gestures before starting (OPEN = fingers extended and spread - not rest;
      CLOSE = fist; INDEX = point, other fingers flexed, thumb on them) and ask the subject to
      **vary force naturally between light and firm** from trial to trial, never maximal.
- [ ] **Start session**. The subject follows the screen: rest 15 s, firm comfortable fist 5 s, twice.
- [ ] Read the table: 50 Hz share, rest RMS, fist/rest ratio, flat/railed, packet loss.
      **Proceed** [Space] on PASS. With `[quality] judge = false` the result is NOT_JUDGED: the
      findings are only listed and Proceed never needs a note. To type a note, click the note
      box (Enter = Proceed, Esc = back to the keys).

## On a FAIL
- [ ] Find the channel(s) named in the reasons: high 50 Hz share or rest RMS → electrode contact
      or cable movement (re-press/replace the electrode, check the bias/reference); flat → cable
      unplugged or channel off; railed → electrode off the skin; low fist/rest → electrode not
      over an active muscle.
- [ ] Fix, then **Repeat the check** [R]. To go on anyway, type a note (why) and **Proceed** - the
      note is saved in `quality_check.json`.

## Packet loss or disconnection
- [ ] A few lost packets per minute are fine (they are counted and saved). Loss growing: move
      the dongle closer (USB extension cable, line of sight), away from Wi-Fi routers and other
      2.4 GHz devices; check the battery.
- [ ] **NO DATA** banner: the protocol has paused itself. Check the board switch, battery and
      dongle; press **Reconnect**. When the signals look right, **P** resumes - the interrupted
      trial is repeated, and nothing recorded before is lost.
- [ ] If reconnecting keeps failing: Esc to abort (the data so far are kept), power-cycle the
      board and the dongle, and start the session again (a new folder is created); write what
      happened in the notes.

## When to flag a bad trial (B)
Press **B** right after (during the hold or the following rest) when the subject:
- performed the wrong gesture, or started it late / released it early;
- moved the arm, talked, coughed, touched the cables, or was clearly distracted;
- you saw anything else that makes the trial not representative.

**B** again on the same trial removes the flag. Only the last trial can be flagged, so press
immediately; for anything later, write it in the notes at the end. There is no camera: this flag
is the only record of what the subject actually did.

## Single screen
Before the start, during the quality-check review and on a data gap, the operator view covers
the screen - turn the laptop away from the subject. During the protocol the subject sees only the
task, and the operator a text strip at the bottom (no signals).

## Free recording (no protocol)
`python -m emgacq --free` (optionally `--duration 60`): same form and live view, then **Start
recording**; the subject view only says "Recording", so give the instructions yourself and write
what was done in the notes (rows carry no gesture label). **Stop recording [Esc]** closes the files.

## After the session
- [ ] `python -m tools.inspect_session data/<folder>`: trials per gesture, losses, gaps, and the
      plots in `<folder>/inspect/` - the gestures should differ in RMS on several channels.
- [ ] Copy `data/<folder>` to the backup location.
