#include "source_ads1299.h"

#include <Arduino.h>
#include <SPI.h>
#include <hal/us_ticker_api.h>

// ADS1299 opcodes and registers (datasheet SBAS499).
enum : uint8_t {
  CMD_WAKEUP = 0x02, CMD_RESET = 0x06, CMD_START = 0x08, CMD_STOP = 0x0A,
  CMD_RDATAC = 0x10, CMD_SDATAC = 0x11, CMD_RREG = 0x20, CMD_WREG = 0x40,
};
enum : uint8_t {
  REG_ID = 0x00, REG_CONFIG1 = 0x01, REG_CONFIG2 = 0x02, REG_CONFIG3 = 0x03, REG_CH1SET = 0x05,
  REG_BIAS_SENSP = 0x0D, REG_BIAS_SENSN = 0x0E, REG_MISC1 = 0x15,
};

// Written by the DRDY interrupt only.
static volatile uint32_t g_drdy_count = 0;
static volatile uint32_t g_drdy_us = 0;

void Ads1299Source::onDrdy() {
  g_drdy_us = us_ticker_read();
  g_drdy_count = g_drdy_count + 1;
}

Ads1299Source::Ads1299Source(int pin_cs, int pin_drdy, int pin_reset, int pin_start, int pin_pwdn,
                             uint8_t n_channels, uint32_t fs_hz, uint8_t gain, float vref_v,
                             uint32_t spi_hz)
    : cs_(pin_cs), drdy_(pin_drdy), reset_(pin_reset), start_(pin_start), pwdn_(pin_pwdn),
      n_ch_(n_channels), fs_(fs_hz), gain_(gain), vref_(vref_v), spi_hz_(spi_hz) {}

float Ads1299Source::uvPerLsb() const { return vref_ / gain_ / 8388607.0f * 1e6f; }

static uint8_t data_rate_code(uint32_t fs) {
  switch (fs) {
    case 16000: return 0; case 8000: return 1; case 4000: return 2; case 2000: return 3;
    case 1000: return 4; case 500: return 5; case 250: return 6; default: return 0xFF;
  }
}

static uint8_t gain_code(uint8_t gain) {
  switch (gain) {
    case 1: return 0; case 2: return 1; case 4: return 2; case 6: return 3;
    case 8: return 4; case 12: return 5; case 24: return 6; default: return 0xFF;
  }
}

void Ads1299Source::command(uint8_t cmd) {
  SPI.beginTransaction(SPISettings(spi_hz_, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_, LOW);
  SPI.transfer(cmd);
  delayMicroseconds(3);  // >= 4 tCLK at 2.048 MHz before CS goes high
  digitalWrite(cs_, HIGH);
  SPI.endTransaction();
}

void Ads1299Source::writeReg(uint8_t reg, uint8_t value) {
  SPI.beginTransaction(SPISettings(spi_hz_, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_, LOW);
  SPI.transfer(CMD_WREG | reg);
  SPI.transfer(0x00);  // one register
  SPI.transfer(value);
  delayMicroseconds(3);
  digitalWrite(cs_, HIGH);
  SPI.endTransaction();
}

uint8_t Ads1299Source::readReg(uint8_t reg) {
  SPI.beginTransaction(SPISettings(spi_hz_, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_, LOW);
  SPI.transfer(CMD_RREG | reg);
  SPI.transfer(0x00);
  uint8_t v = SPI.transfer(0x00);
  delayMicroseconds(3);
  digitalWrite(cs_, HIGH);
  SPI.endTransaction();
  return v;
}

bool Ads1299Source::begin() {
  uint8_t dr = data_rate_code(fs_), g = gain_code(gain_);
  if (dr == 0xFF) { error_ = "unsupported sampling rate for ADS1299"; return false; }
  if (g == 0xFF) { error_ = "unsupported ADS1299 gain"; return false; }
  if (n_ch_ > 8) { error_ = "ADS1299 has at most 8 channels"; return false; }

  pinMode(cs_, OUTPUT);
  digitalWrite(cs_, HIGH);
  pinMode(start_, OUTPUT);
  digitalWrite(start_, LOW);
  pinMode(pwdn_, OUTPUT);
  pinMode(reset_, OUTPUT);
  pinMode(drdy_, INPUT);
  SPI.begin();

  // Power-up sequence: PWDN and RESET high, wait for the oscillator and tPOR, then reset pulse.
  digitalWrite(pwdn_, HIGH);
  digitalWrite(reset_, HIGH);
  delay(150);
  digitalWrite(reset_, LOW);
  delayMicroseconds(10);
  digitalWrite(reset_, HIGH);
  delay(1);
  command(CMD_SDATAC);  // registers are writable only in SDATAC mode
  delay(1);

  chip_id_ = readReg(REG_ID);
  if ((chip_id_ & 0x1C) != 0x1C) { error_ = "ADS1299 not found (bad ID register)"; return false; }

  writeReg(REG_CONFIG3, 0xEC);  // internal reference buffer on, bias buffer on, bias ref internal
  delay(10);
  writeReg(REG_CONFIG1, 0x90 | dr);
  writeReg(REG_CONFIG2, 0xC0);  // test signals off
  uint8_t mask = 0;
  for (uint8_t c = 0; c < 8; ++c) {
    if (c < n_ch_) {
      // on, gain, normal electrode input, SRB2 off: bipolar, as the Cyton "x<ch>060100X"
      writeReg(REG_CH1SET + c, (uint8_t)(g << 4));
      mask |= (uint8_t)(1 << c);
    } else {
      writeReg(REG_CH1SET + c, 0x81);  // powered down, input shorted
    }
  }
  writeReg(REG_BIAS_SENSP, mask);  // bias derived from both inputs of every used channel
  writeReg(REG_BIAS_SENSN, mask);
  writeReg(REG_MISC1, 0x00);       // SRB1 off
  if (readReg(REG_CONFIG1) != (0x90 | dr)) { error_ = "ADS1299 register write failed"; return false; }
  return true;
}

bool Ads1299Source::start() {
  counter_ = 0;
  noInterrupts();
  g_drdy_count = 0;
  interrupts();
  handled_ = 0;
  command(CMD_RDATAC);
  attachInterrupt(digitalPinToInterrupt(drdy_), onDrdy, FALLING);
  digitalWrite(start_, HIGH);
  running_ = true;
  return true;
}

void Ads1299Source::stop() {
  digitalWrite(start_, LOW);
  detachInterrupt(digitalPinToInterrupt(drdy_));
  command(CMD_SDATAC);
  running_ = false;
}

bool Ads1299Source::read(Frame& f) {
  if (!running_) return false;
  noInterrupts();
  uint32_t count = g_drdy_count;
  uint32_t t_us = g_drdy_us;
  interrupts();
  if (count == handled_) return false;

  // Only the latest conversion can be read; older ones not read in time are lost.
  uint32_t missed = count - handled_ - 1;
  handled_ = count;

  uint8_t raw[27];
  SPI.beginTransaction(SPISettings(spi_hz_, MSBFIRST, SPI_MODE1));
  digitalWrite(cs_, LOW);
  for (int i = 0; i < 27; ++i) raw[i] = SPI.transfer(0x00);
  digitalWrite(cs_, HIGH);
  SPI.endTransaction();

  bool torn = (g_drdy_count != count);  // a new conversion arrived during the read
  f.flags = 0;
  if (missed > 0 || torn) f.flags |= FLAG_OVERRUN;
  counter_ += missed;
  f.counter = counter_++;
  f.t_us = t_us;
  for (uint8_t c = 0; c < n_ch_; ++c) {
    const uint8_t* p = raw + 3 + 3 * c;  // first 3 bytes are the status word
    int32_t v = ((int32_t)p[0] << 16) | ((int32_t)p[1] << 8) | p[2];
    if (v & 0x800000) v -= 0x1000000;
    f.ch[c] = v;
  }
  return true;
}
