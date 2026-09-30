#include "sync_led.h"

#include <Arduino.h>
#include <hal/us_ticker_api.h>

#include "frame.h"

// Minimum time between the end of a flash and the next one, so that samples of the previous flash
// still waiting in the queue are flagged against the right on/off times.
static const uint32_t kMinGapUs = 200000;

static inline bool at_or_after(uint32_t t, uint32_t ref) { return (int32_t)(t - ref) >= 0; }

void SyncLed::begin(int pin, bool active_high, uint32_t flash_us) {
  pin_ = pin;
  active_high_ = active_high;
  flash_us_ = flash_us;
  if (pin_ >= 0) pinMode(pin_, OUTPUT);
  write(false);
}

void SyncLed::write(bool on) {
  if (pin_ >= 0) digitalWrite(pin_, (on == active_high_) ? HIGH : LOW);
}

bool SyncLed::request(uint32_t now_us) {
  if (on_) return false;
  if (have_flash_ && !at_or_after(now_us, t_off_ + kMinGapUs)) return false;
  t_on_ = now_us;
  write(true);
  // take the time again after switching, so t_on_ is not earlier than the real edge
  t_on_ = us_ticker_read();
  on_ = true;
  have_flash_ = true;
  rise_pending_ = true;
  return true;
}

void SyncLed::update(uint32_t now_us) {
  if (on_ && at_or_after(now_us, t_on_ + flash_us_)) {
    write(false);
    t_off_ = us_ticker_read();
    on_ = false;
  }
}

uint8_t SyncLed::flagsFor(uint32_t sample_us) {
  if (!have_flash_ || !at_or_after(sample_us, t_on_)) return 0;
  if (!on_ && at_or_after(sample_us, t_off_)) return 0;
  uint8_t f = FLAG_LED_ON;
  if (rise_pending_) {
    f |= FLAG_LED_RISE;
    rise_pending_ = false;
  }
  return f;
}
