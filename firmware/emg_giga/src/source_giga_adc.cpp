#include "source_giga_adc.h"

// 64 one-sample buffers = 64 ms of slack for the main loop before samples are lost.
static const size_t kBuffers = 64;

GigaAdcSource::GigaAdcSource(const pin_size_t* pins, uint8_t n_channels, uint32_t fs_hz,
                             float vref_v, float front_end_gain)
    : n_ch_(n_channels), fs_(fs_hz), vref_(vref_v), gain_(front_end_gain) {
  for (uint8_t c = 0; c < n_channels && c < EMG_MAX_CHANNELS; ++c) pins_[c] = pins[c];
}

bool GigaAdcSource::begin() {
  if (n_ch_ > EMG_MAX_CHANNELS) { error_ = "too many channels"; return false; }
  return true;  // the ADC is configured in start(), since stop() releases it
}

bool GigaAdcSource::start() {
  counter_ = 0;
  first_ = true;
  // All pins must belong to the same ADC instance; begin() fails otherwise.
  if (!adc_.begin(AN_RESOLUTION_16, fs_, 1, kBuffers, n_ch_, pins_, true, AN_ADC_SAMPLETIME_64_5)) {
    error_ = "AdvancedADC begin failed (pins not on one ADC?)";
    return false;
  }
  running_ = true;
  return true;
}

void GigaAdcSource::stop() {
  if (running_) adc_.stop();
  running_ = false;
}

bool GigaAdcSource::read(Frame& f) {
  if (!running_ || !adc_.available()) return false;
  SampleBuffer buf = adc_.read();
  uint32_t t_us = buf.timestamp();
  f.flags = 0;
  if (!first_) {
    // Samples lost when the pool was full: recover the count from the buffer timestamps.
    uint32_t period_us = 1000000UL / fs_;
    uint32_t steps = (t_us - last_us_ + period_us / 2) / period_us;
    if (steps > 1) {
      counter_ += steps - 1;
      f.flags |= FLAG_OVERRUN;
    }
  }
  if (buf.get_flags(DMA_BUFFER_DISCONT)) f.flags |= FLAG_OVERRUN;
  first_ = false;
  last_us_ = t_us;
  f.counter = counter_++;
  f.t_us = t_us;
  for (uint8_t c = 0; c < n_ch_; ++c) f.ch[c] = buf[c];  // interleaved, one sample per channel
  buf.release();
  return true;
}
