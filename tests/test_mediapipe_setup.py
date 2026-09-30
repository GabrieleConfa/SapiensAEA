"""MediaPipe setup: API usage, timestamps, model hash tracking. Needs the downloaded model for the
landmarker test (skipped otherwise)."""

import numpy as np
import pytest

from common import hands
from common.config import load_config
from scripts import download_models as dm


def test_monotonic_timestamps():
    ts = hands.MonotonicMs()
    assert [ts(t) for t in (0.2, 0.4, 5.0, 4.0, 33.4)] == [0, 1, 5, 6, 33]


def test_landmarker_runs_in_video_mode():
    cfg = load_config()
    try:
        lm = hands.create_video_landmarker(cfg)
    except FileNotFoundError:
        pytest.skip("model not downloaded")
    ts = hands.MonotonicMs()
    with lm:
        for i in range(3):
            frame = np.full((360, 640, 3), 90, np.uint8)
            res = lm.detect_for_video(hands.to_mp_image(frame), ts(i * 33.3))
            assert res.hand_landmarks == [] and res.hand_world_landmarks == []


def test_model_hash_change_is_detected(tmp_path, monkeypatch):
    md = tmp_path / "MODELS.md"
    model = tmp_path / "hand_landmarker.task"
    monkeypatch.setattr(dm, "MODELS_MD", md)
    monkeypatch.setattr(dm, "REPO_ROOT", tmp_path)
    cfg = load_config()
    monkeypatch.setattr(dm, "load_config", lambda: _Cfg(cfg, "hand_landmarker.task"))
    payload = {"data": b"model-v1"}

    def fake_download(url, dest):
        part = dest.with_suffix(".part")
        part.write_bytes(payload["data"])
        return part

    monkeypatch.setattr(dm, "download", fake_download)
    assert dm.main([]) == 0                       # first download
    assert dm.main([]) == 0                       # unchanged
    payload["data"] = b"model-v2"
    assert dm.main([]) == 2                       # changed: warned, local file kept
    assert model.read_bytes() == b"model-v1"
    assert dm.main(["--accept-new"]) == 2
    assert model.read_bytes() == b"model-v2"
    statuses = [line.split("|")[5].strip() for line in md.read_text().splitlines() if "hand_landmarker" in line]
    assert statuses == ["first download", "unchanged", "CHANGED, not installed", "CHANGED, installed"]


class _Cfg:
    def __init__(self, cfg, path):
        self.cfg, self.path = cfg, path

    def __getitem__(self, key):
        return self.path if key == "mediapipe.model_path" else self.cfg[key]
