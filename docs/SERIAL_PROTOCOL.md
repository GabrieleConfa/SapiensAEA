# GIGA ↔ laptop serial protocol

USB CDC (virtual COM port). The baud rate setting is ignored. The same protocol is spoken by the
firmware (`firmware/emg_giga`) and by the simulator (`simulator/giga_sim.py`).

## Laptop → GIGA: commands

ASCII lines terminated by `\n`. Replies are text lines starting with `#` and an uppercase keyword.

| Command            | Reply                                   | Notes                                                    |
|--------------------|-----------------------------------------|----------------------------------------------------------|
| `PING`             | `#PONG`                                 |                                                          |
| `INFO`             | `#INFO {json}`                          | fields below                                             |
| `START`            | `#OK START`, then packets               | sample counter restarts at 0                             |
| `STOP`             | `#OK STOP` after the last packet        |                                                          |
| `FLASH`            | none while streaming, else `#OK FLASH`  | LED on for `flash_ms` from the next sample               |
| `MODE RECORD`      | `#OK MODE RECORD`                       | `MODE INFER` arrives in Stage 3                          |
| `SIM GRIP <G> <l>` | none                                    | simulator only: synthetic activity for grip G, level 0–1 |
| anything else      | `#ERR <reason>`                         |                                                          |

`#INFO` JSON fields: `fw` (version), `mode`, `source` (`ads1299_spi`, `giga_adc`, `simulator`),
`fs` (Hz), `n_ch`, `bits` (ADC resolution), `uv_per_lsb` (µV per code at the electrodes),
`min_code`/`max_code` (saturation limits), `flash_ms`, `pins_configured` (false while pins are
TODO_HW), `streaming`.

## GIGA → laptop: sample packets

One packet per sample instant, little endian. For N channels the length is `9 + 3N` bytes (33 for 8
channels, i.e. 33 kB/s at 1 kHz).

| Offset | Size | Field                                                                 |
|--------|------|-----------------------------------------------------------------------|
| 0      | 2    | sync `0xA5 0x5A`                                                      |
| 2      | 4    | sample counter, uint32, 0 at `START`                                  |
| 6      | 1    | flags                                                                 |
| 7      | 3N   | N samples, signed 24-bit (ADS1299 native; GIGA ADC codes 0–65535 fit) |
| 7+3N   | 2    | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over bytes 2 … 7+3N−1   |

Flags:

| Bit | Name       | Meaning                                                               |
|-----|------------|-----------------------------------------------------------------------|
| 0   | `LED_ON`   | sync LED is on at this sample                                         |
| 1   | `LED_RISE` | the LED was switched on at this sample (first sample with `LED_ON`)   |
| 2   | `OVERRUN`  | the firmware lost at least one sample before this one                 |
| 3   | `BUTTON`   | calibration button pressed                                            |

Text reply lines may appear between packets. They contain only printable ASCII and so never
contain `0xA5`; the parser (`common/packet.py`) separates them from the binary stream and
resynchronises on the sync word + CRC after any corruption.

## Timing

- The sample counter is the EMG time base: `t = counter / fs`. The counter is incremented in the
  sampling clock context (ADS1299 DRDY interrupt, or the ADC timer for the GIGA ADC), never by
  polling `micros()`.
- The LED is switched in the same context, so `LED_RISE` marks the sample at which the LED turned on.
- Gaps in the counter mean lost samples; they are counted by the parser and never silently filled.
