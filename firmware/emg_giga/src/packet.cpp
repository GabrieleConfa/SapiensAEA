#include "packet.h"

uint16_t crc16_ccitt(const uint8_t* data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

size_t encode_packet(const Frame& f, uint8_t n_channels, uint8_t* out) {
  size_t k = 0;
  out[k++] = 0xA5;
  out[k++] = 0x5A;
  out[k++] = (uint8_t)(f.counter);
  out[k++] = (uint8_t)(f.counter >> 8);
  out[k++] = (uint8_t)(f.counter >> 16);
  out[k++] = (uint8_t)(f.counter >> 24);
  out[k++] = f.flags;
  for (uint8_t c = 0; c < n_channels; ++c) {
    uint32_t v = (uint32_t)f.ch[c] & 0xFFFFFF;
    out[k++] = (uint8_t)(v);
    out[k++] = (uint8_t)(v >> 8);
    out[k++] = (uint8_t)(v >> 16);
  }
  uint16_t crc = crc16_ccitt(out + 2, k - 2);
  out[k++] = (uint8_t)(crc);
  out[k++] = (uint8_t)(crc >> 8);
  return k;
}
