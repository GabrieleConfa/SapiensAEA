// Sync LED for EMG-video alignment.
//
// The LED is switched from the main loop, and the switching times are taken with us_ticker. Every
// sample carries the us_ticker time of its conversion, captured in the sampling clock context
// (DMA-complete or DRDY interrupt). A sample is flagged LED_ON if it was converted while the LED
// was on, and the first such sample also gets LED_RISE. The marked sample is therefore exact,
// whatever the latency between conversion and processing.
#pragma once

#include <stdint.h>

class SyncLed {
 public:
  void begin(int pin, bool active_high, uint32_t flash_us);
  bool request(uint32_t now_us);          // switch on now; false if a flash is still in progress
  void update(uint32_t now_us);           // switch off when due; call before processing samples
  uint8_t flagsFor(uint32_t sample_us);   // flags of a sample converted at sample_us (in order)
  bool on() const { return on_; }

 private:
  void write(bool on);

  int pin_ = -1;
  bool active_high_ = true;
  uint32_t flash_us_ = 200000;
  bool on_ = false;
  bool have_flash_ = false;
  bool rise_pending_ = false;
  uint32_t t_on_ = 0;
  uint32_t t_off_ = 0;
};
