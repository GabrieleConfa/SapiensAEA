// ADS1299 over SPI, GIGA as SPI master. Sampling is clocked by the ADS1299 itself: the DRDY
// falling edge interrupt timestamps and counts each conversion; the data are read over SPI from
// the main loop (mbed's SPI cannot be used inside an interrupt).
#pragma once

#include "sample_source.h"

class Ads1299Source : public SampleSource {
 public:
  Ads1299Source(int pin_cs, int pin_drdy, int pin_reset, int pin_start, int pin_pwdn,
                uint8_t n_channels, uint32_t fs_hz, uint8_t gain, float vref_v, uint32_t spi_hz);
  bool begin() override;
  bool start() override;
  void stop() override;
  bool read(Frame& f) override;
  const char* name() const override { return "ads1299_spi"; }
  uint8_t bits() const override { return 24; }
  int32_t minCode() const override { return -8388608; }
  int32_t maxCode() const override { return 8388607; }
  float uvPerLsb() const override;
  const char* error() const override { return error_; }
  uint8_t chipId() const { return chip_id_; }

 private:
  void command(uint8_t cmd);
  void writeReg(uint8_t reg, uint8_t value);
  uint8_t readReg(uint8_t reg);
  static void onDrdy();

  int cs_, drdy_, reset_, start_, pwdn_;
  uint8_t n_ch_;
  uint32_t fs_;
  uint8_t gain_;
  float vref_;
  uint32_t spi_hz_;
  uint8_t chip_id_ = 0;
  const char* error_ = "";
  bool running_ = false;
  uint32_t handled_ = 0;   // DRDY count already turned into frames
  uint32_t counter_ = 0;
};
