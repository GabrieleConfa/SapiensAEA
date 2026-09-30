import pytest

from common.config import HardwareParamMissing, load_config, todo_hw_keys


def test_repo_config_loads():
    cfg = load_config()
    assert cfg["emg.n_channels"] == 8
    assert cfg["emg.fs_hz"] == 1000
    assert len(cfg["emg.channel_labels"]) == cfg["emg.n_channels"]
    assert len(cfg.sha256) == 64


def test_todo_hw_refused(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[sync]\nled_pin = "TODO_HW"\nflash_ms = 200\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.hw("sync.flash_ms") == 200
    assert not cfg.is_set("sync.led_pin")
    with pytest.raises(HardwareParamMissing, match="sync.led_pin"):
        cfg.hw("sync.led_pin")
    assert todo_hw_keys(cfg) == ["sync.led_pin"]
