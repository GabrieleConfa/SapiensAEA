import numpy as np
import pandas as pd

from emgacq.writer import EmgCsvWriter, EventLog, mark_bad_trials


def write_rows(path, trials, flush_s=1.0):
    n = len(trials)
    w = EmgCsvWriter(path, ["ch1", "ch3"], flush_interval_s=flush_s)
    data = np.column_stack([np.arange(n) * 1.25, -np.arange(n) * 0.5])
    marker = np.zeros(n)
    marker[0] = 10
    w.write(np.arange(n) / 250 + 1000.0, np.arange(n) % 256, data, marker,
            np.array(["HOLD"] * n, object), np.array(["OPEN"] * n, object), np.ones(n, int),
            np.asarray(trials))
    return w


def test_columns_and_values(tmp_path):
    p = tmp_path / "emg.csv"
    w = write_rows(p, [1, 1, 2, 2, 3])
    w.close()
    df = pd.read_csv(p)
    assert list(df.columns) == ["sample_index", "timestamp", "package_num", "ch1", "ch3", "marker",
                                "phase", "label", "block", "trial_id", "trial_bad"]
    assert df["sample_index"].tolist() == [0, 1, 2, 3, 4]
    assert df["ch1"].tolist() == [0.0, 1.25, 2.5, 3.75, 5.0]
    assert df["marker"].tolist() == [10, 0, 0, 0, 0]
    assert df["trial_bad"].sum() == 0


def test_rows_readable_before_close(tmp_path):
    """What has been flushed is on disk while the session is still running (crash safety)."""
    p = tmp_path / "emg.csv"
    w = write_rows(p, [1] * 10, flush_s=0.0)
    assert len(pd.read_csv(p)) == 10
    w.close()


def test_bad_trial_update(tmp_path):
    p = tmp_path / "emg.csv"
    write_rows(p, [1, 1, 2, 2, 3, 3, 0, 4]).close()
    before = p.read_text()
    n = mark_bad_trials(p, {2}, ranges=[(6, 8)])
    df = pd.read_csv(p)
    assert n == 4
    assert df["trial_bad"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    # nothing else changes, byte for byte
    strip = lambda text: [line.rsplit(",", 1)[0] for line in text.splitlines()]
    assert strip(p.read_text()) == strip(before)
    assert not (tmp_path / "emg.tmp").exists()


def test_bad_trial_update_noop(tmp_path):
    p = tmp_path / "emg.csv"
    write_rows(p, [1, 2]).close()
    assert mark_bad_trials(p, set()) == 0


def test_event_log(tmp_path):
    log = EventLog(tmp_path / "events.csv")
    log.log("TRIAL_HOLD", marker=20702, phase="HOLD", gesture="CLOSE", block=1, trial_id=7,
            sample_index=123)
    df = pd.read_csv(tmp_path / "events.csv")      # flushed immediately
    assert df.loc[0, "marker"] == 20702 and df.loc[0, "trial_id"] == 7
    assert df.loc[0, "wall_time"].startswith("20")
    log.close()
