// Interface of the EMG sampling source. The rest of the firmware only sees this, so the GIGA ADC
// can be swapped for an ADS1299 over SPI (or anything else) without touching packets, LED sync,
// features or the model.
//
// Contract:
//  - sampling is clocked by hardware (ADC timer, ADS1299 DRDY), never by polling micros();
//  - read() returns samples in order; counter restarts at 0 on start() and advances by one per
//    sample instant, also over lost samples (which set FLAG_OVERRUN on the next frame);
//  - t_us is the us_ticker time captured in the sampling clock context, used to mark the LED.
#pragma once

#include "frame.h"

class SampleSource {
 public:
  virtual ~SampleSource() {}
  virtual bool begin() = 0;                 // configure the hardware once
  virtual bool start() = 0;                 // start sampling, counter = 0
  virtual void stop() = 0;
  virtual bool read(Frame& f) = 0;          // next sample if ready: counter, t_us, ch, FLAG_OVERRUN
  virtual const char* name() const = 0;
  virtual uint8_t bits() const = 0;
  virtual int32_t minCode() const = 0;
  virtual int32_t maxCode() const = 0;
  virtual float uvPerLsb() const = 0;       // µV at the electrodes per ADC code
  virtual const char* error() const { return ""; }
};
