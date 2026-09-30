// Binary sample packet, see docs/SERIAL_PROTOCOL.md (common/packet.py is the laptop side).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "frame.h"

#define PACKET_MAX_LEN (9 + 3 * EMG_MAX_CHANNELS)

uint16_t crc16_ccitt(const uint8_t* data, size_t n);

// Encodes f into out (at least PACKET_MAX_LEN bytes); returns the packet length.
size_t encode_packet(const Frame& f, uint8_t n_channels, uint8_t* out);
