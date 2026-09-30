// One sample instant of all EMG channels, plus its bookkeeping.
#pragma once

#include <stdint.h>

#define EMG_MAX_CHANNELS 8

// Packet flag bits (docs/SERIAL_PROTOCOL.md).
enum : uint8_t {
  FLAG_LED_ON = 0x01,    // sync LED on at this sample
  FLAG_LED_RISE = 0x02,  // first sample with the LED on
  FLAG_OVERRUN = 0x04,   // at least one sample was lost before this one
  FLAG_BUTTON = 0x08,    // calibration button pressed
};

struct Frame {
  uint32_t counter;                // sample index since START
  uint32_t t_us;                   // us_ticker time of the conversion (sampling clock context)
  uint8_t flags;
  int32_t ch[EMG_MAX_CHANNELS];    // raw ADC codes
};
