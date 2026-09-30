from common.config import load_config
from scripts.gen_firmware_config import render


def _cfg(tmp_path, replacements):
    text = load_config().path.read_text(encoding="utf-8")
    for a, b in replacements:
        assert a in text
        text = text.replace(a, b)
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return load_config(p)


def test_placeholders_block_start(tmp_path):
    cfg = load_config()
    text, missing = render(cfg, allow_todo=True)
    assert "sync.led_pin" in missing
    assert "#define HW_PINS_CONFIGURED 0" in text
    assert "#define SYNC_LED_PIN -1" in text
    assert f'#define CONFIG_SHA256 "{cfg.sha256}"' in text


def test_giga_adc_only_needs_its_own_pins(tmp_path):
    cfg = _cfg(tmp_path, [('source = "ads1299_spi"', 'source = "giga_adc"'),
                          ('led_pin = "TODO_HW"', "led_pin = 2"),
                          ('front_end_gain = "TODO_HW"', "front_end_gain = 1000.0")])
    text, missing = render(cfg, allow_todo=False)
    assert missing == []
    assert "#define HW_PINS_CONFIGURED 1" in text
    assert "#define GIGA_ADC_PINS A0, A1, A2, A3, A4, A5, A6, A7" in text
    assert "#define EMG_SOURCE_GIGA_ADC 1" in text
