// GIGA on-board ADC: timer-triggered scan of the analog pins, moved by DMA (Arduino_AdvancedAnalog),
// 16-bit, one sample instant per DMA buffer. The DMA-complete interrupt timestamps each buffer.
#pragma once

#include <Arduino_AdvancedAnalog.h>

#include "sample_source.h"

class GigaAdcSource : public SampleSource {
 public:
  GigaAdcSource(const pin_size_t* pins, uint8_t n_channels, uint32_t fs_hz, float vref_v,
                float front_end_gain);
  bool begin() override;
  bool start() override;
  void stop() override;
  bool read(Frame& f) override;
  const char* name() const override { return "giga_adc"; }
  uint8_t bits() const override { return 16; }
  int32_t minCode() const override { return 0; }
  int32_t maxCode() const override { return 65535; }
  float uvPerLsb() const override { return vref_ / 65535.0f / gain_ * 1e6f; }
  const char* error() const override { return error_; }

 private:
  AdvancedADC adc_;
  pin_size_t pins_[EMG_MAX_CHANNELS];
  uint8_t n_ch_;
  uint32_t fs_;
  float vref_, gain_;
  const char* error_ = "";
  bool running_ = false;
  bool first_ = true;
  uint32_t counter_ = 0;
  uint32_t last_us_ = 0;
};
