from pathlib import Path

import numpy as np
import pandas as pd

from emgacq import analysis as an


def test_summary_of_session_aborted_before_any_trial():
    n = 500
    emg = pd.DataFrame({"sample_index": np.arange(n), "timestamp": np.arange(n) / 250.0,
                        "package_num": np.arange(n) % 256, "ch1": np.zeros(n), "marker": 0.0,
                        "phase": "QUALITY", "label": "REST", "block": 0, "trial_id": 0,
                        "trial_bad": 0})
    s = an.Session(Path("."), emg, pd.DataFrame(), {"sampling_rate_hz": 250.0, "status": "aborted"},
                   None, {"trials": [{"trial_id": 1, "block": 1, "gesture": "OPEN"}]})
    trials = an.trial_table(s)
    assert trials.empty and trials["bad"].dtype == bool
    text = an.summary_text(s)
    assert "Bad trials: none" in text and "Trials recorded: 0 of 1 planned" in text
