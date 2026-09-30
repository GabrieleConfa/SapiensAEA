// EMG grip firmware for the Arduino GIGA R1.
//
// RECORD mode: samples the EMG channels at a hardware-clocked rate and streams one binary packet
// per sample to the laptop over USB (docs/SERIAL_PROTOCOL.md), with the sync LED marked in the
// packet flags. INFER mode (on-board features, model and output to the prototype) comes in Stage 3.
//
// Build: python -m scripts.gen_firmware_config   (writes config_gen.h from config/config.toml)
//        arduino-cli compile --fqbn arduino:mbed_giga:giga firmware/emg_giga

#include <hal/us_ticker_api.h>

#include "config_gen.h"
#include "src/frame.h"
#include "src/packet.h"
#include "src/sample_source.h"
#include "src/source_ads1299.h"
#include "src/source_giga_adc.h"
#include "src/sync_led.h"

#define FW_VERSION "0.1.0"

#if EMG_SOURCE_ADS1299
static Ads1299Source g_hw_source(ADS_PIN_CS, ADS_PIN_DRDY, ADS_PIN_RESET, ADS_PIN_START, ADS_PIN_PWDN,
                                 EMG_N_CHANNELS, EMG_FS_HZ, ADS_GAIN, ADS_VREF_V, ADS_SPI_HZ);
#elif EMG_SOURCE_GIGA_ADC
static const pin_size_t kAdcPins[] = {GIGA_ADC_PINS};
static GigaAdcSource g_hw_source(kAdcPins, EMG_N_CHANNELS, EMG_FS_HZ, GIGA_ADC_VREF_V,
                                 GIGA_ADC_FRONT_END_GAIN);
#else
#error "no EMG source selected in config_gen.h"
#endif
static SampleSource& g_source = g_hw_source;

static SyncLed g_led;

enum Mode { MODE_RECORD, MODE_INFER };
static Mode g_mode = MODE_RECORD;
static bool g_source_ok = false;
static bool g_streaming = false;

static char g_line[64];
static size_t g_line_len = 0;
static uint8_t g_tx[512];
static size_t g_tx_len = 0;

static void flush_tx() {
  if (g_tx_len > 0) {
    Serial.write(g_tx, g_tx_len);
    g_tx_len = 0;
  }
}

static bool button_pressed() {
  if (CAL_BUTTON_PIN < 0) return false;
  bool level = digitalRead(CAL_BUTTON_PIN) == HIGH;
  return CAL_BUTTON_ACTIVE_LOW ? !level : level;
}

static void print_info() {
  Serial.print("#INFO {\"fw\":\"" FW_VERSION "\",\"mode\":\"");
  Serial.print(g_mode == MODE_RECORD ? "RECORD" : "INFER");
  Serial.print("\",\"source\":\"");
  Serial.print(g_source.name());
  Serial.print("\",\"fs\":");
  Serial.print(EMG_FS_HZ);
  Serial.print(",\"n_ch\":");
  Serial.print(EMG_N_CHANNELS);
  Serial.print(",\"bits\":");
  Serial.print(g_source.bits());
  Serial.print(",\"uv_per_lsb\":");
  Serial.print(g_source.uvPerLsb(), 8);
  Serial.print(",\"min_code\":");
  Serial.print(g_source.minCode());
  Serial.print(",\"max_code\":");
  Serial.print(g_source.maxCode());
  Serial.print(",\"flash_ms\":");
  Serial.print(SYNC_FLASH_MS);
  Serial.print(",\"pins_configured\":");
  Serial.print(HW_PINS_CONFIGURED ? "true" : "false");
  Serial.print(",\"source_ok\":");
  Serial.print(g_source_ok ? "true" : "false");
  Serial.print(",\"streaming\":");
  Serial.print(g_streaming ? "true" : "false");
  Serial.print(",\"config_sha256\":\"" CONFIG_SHA256 "\"}\n");
}

static void print_error(const char* what, const char* detail) {
  Serial.print("#ERR ");
  Serial.print(what);
  if (detail && detail[0]) {
    Serial.print(": ");
    Serial.print(detail);
  }
  Serial.print("\n");
}

static void handle_command(char* cmd) {
  if (strcmp(cmd, "PING") == 0) {
    Serial.print("#PONG\n");
  } else if (strcmp(cmd, "INFO") == 0) {
    print_info();
  } else if (strcmp(cmd, "START") == 0) {
    if (!g_source_ok) { print_error("source not ready", g_source.error()); return; }
    if (g_streaming) g_source.stop();
    g_tx_len = 0;
    Serial.print("#OK START\n");
    g_streaming = g_source.start();
    if (!g_streaming) print_error("start failed", g_source.error());
  } else if (strcmp(cmd, "STOP") == 0) {
    if (g_streaming) g_source.stop();
    g_streaming = false;
    flush_tx();
    Serial.print("#OK STOP\n");
  } else if (strcmp(cmd, "FLASH") == 0) {
    bool ok = g_led.request(us_ticker_read());
    // while streaming the LED_RISE flag is the acknowledgement
    if (!g_streaming) {
      if (ok) Serial.print("#OK FLASH\n"); else print_error("flash busy", "");
    }
  } else if (strcmp(cmd, "MODE RECORD") == 0) {
    g_mode = MODE_RECORD;
    Serial.print("#OK MODE RECORD\n");
  } else if (strcmp(cmd, "MODE INFER") == 0) {
    print_error("INFER mode not implemented yet", "");
  } else {
    print_error("unknown command", cmd);
  }
}

static void read_commands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      g_line[g_line_len] = '\0';
      if (g_line_len > 0) {
        flush_tx();  // keep replies between whole packets
        handle_command(g_line);
      }
      g_line_len = 0;
    } else if (g_line_len < sizeof(g_line) - 1) {
      g_line[g_line_len++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  g_led.begin(SYNC_LED_PIN, SYNC_LED_ACTIVE_HIGH, (uint32_t)SYNC_FLASH_MS * 1000UL);
  if (CAL_BUTTON_PIN >= 0) pinMode(CAL_BUTTON_PIN, CAL_BUTTON_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
#if HW_PINS_CONFIGURED
  g_source_ok = g_source.begin();
#else
  g_source_ok = false;  // never drive unknown pins
#endif
}

void loop() {
  read_commands();
  g_led.update(us_ticker_read());
  if (!g_streaming) return;

  Frame f;
  while (g_source.read(f)) {
    f.flags |= g_led.flagsFor(f.t_us);
    if (button_pressed()) f.flags |= FLAG_BUTTON;
    if (g_tx_len + PACKET_MAX_LEN > sizeof(g_tx)) flush_tx();
    g_tx_len += encode_packet(f, EMG_N_CHANNELS, g_tx + g_tx_len);
  }
  flush_tx();
}
