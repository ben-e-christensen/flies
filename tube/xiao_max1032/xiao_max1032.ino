/*
 * XIAO ESP32-C6 -> MAX1032 (14-bit SAR, SPI) -> high-rate binary stream over USB.
 *
 * Every 1/rate seconds the firmware converts CH0..CH3 back to back (external
 * clock mode). Samples go out in packets of PACKET_SAMPLES:
 *
 *     A5 5A | seq (u32 LE) | t0_us (u32 LE) | N x [dt_us (u16), code0..code3 (u16)] | xor8
 *
 *   seq    index of the packet's first sample since streaming started. A jump
 *          means packets were dropped (host not keeping up).
 *   t0_us  micros() when the packet's first sample was taken
 *   dt_us  when this sample was actually taken, in us after t0_us. Every
 *          sample carries its real time, so timing stays exact even if the
 *          loop runs late (the schedule is a target, not an assumption).
 *   code   14-bit offset binary, 0x2000 = 0 V.  V = (code - 8192) * 375 uV
 *   xor8   XOR of every byte between the sync and the checksum
 *
 * MAX1032 limit: 115 ksps total across channels (32 SCLKs per conversion at
 * the 3.67 MHz max SCLK), so ~28 kHz per channel with 4 channels. The real
 * limit is measured at startup (tick_us = time to convert all channels) and
 * reported as max_hz.
 *
 * Commands from the host (text):
 *   ?          stop streaming, print the header (lines start with '#', ends "# end").
 *              The "# cfg key=value ..." line has everything the host needs.
 *   r<hz>\n    set the per-channel sample rate (rounded to a whole-us period,
 *              capped at max_hz), then print the header
 *   g<code>\n  set the input range for all channels: 1 = +/-3.072 V,
 *              4 = +/-6.144 V, 7 = +/-12.288 V (default), then print the header
 *   s          start streaming
 *   x          stop streaming
 *
 * Wiring: D8/GPIO19 SCLK, D9/GPIO20 DOUT(MISO), D10/GPIO18 DIN(MOSI), D0/GPIO0 CS.
 * CH0..CH3 = Electro1..4, single-ended, range set by g (default +/-12.288 V). CH4..7 grounded.
 *   CH0 = Electro1 top ring, CH1 = Electro2 (not connected, floating),
 *   CH2 = Electro3 middle ring, CH3 = Electro4 bottom ring.
 *
 * Flash from the tube folder:
 *   arduino-cli compile --upload --fqbn esp32:esp32:XIAO_ESP32C6:CDCOnBoot=cdc -p /dev/ttyACM0 xiao_max1032
 * If the port enumerates then disappears: hold BOOT, tap RESET, flash again.
 */

#include <SPI.h>

// ---------------- CONFIG ----------------
const int      PIN_CS     = D0;
const bool     HW_CS      = true;      // SPI peripheral drives CS per transfer (faster
                                       // than toggling it in software); false = digitalWrite
const uint32_t SPI_HZ     = 3636364;   // 80 MHz / 22; must be <= 3.67 MHz (external clock mode)
const uint32_t DEFAULT_HZ = 20000;     // per channel; capped at max_hz, so in practice
                                       // "as fast as this board can" (~17.9 kHz measured)
const int      N_CH       = 4;
// Input range (same for all channels), set with the g command. Symmetric
// single-ended ranges, R[2:0] code -> span (VREF = 4.096 V):
//   1: +/-3 x VREF/4 = +/-3.072 V   375 uV/LSB
//   4: +/-3 x VREF/2 = +/-6.144 V   750 uV/LSB
//   7: +/-3 x VREF   = +/-12.288 V  1.5 mV/LSB   (inputs tolerate +/-16.5 V)
// Codes are offset binary in all three: 8192 = 0 V, 0 = -range, 16383 = +range.
const uint8_t  DEFAULT_RANGE_R = 7;
const int      PACKET_SAMPLES = 64;
const float    TICK_MARGIN = 1.20f;    // keep 20% slack per period for packet/USB work, so
                                       // the schedule holds and samples stay evenly spaced
// ----------------------------------------

const uint8_t CMD_RESET   = 0xC8;
const uint8_t CMD_EXT_CLK = 0x88;
const int     HDR_BYTES   = 10;        // sync + seq + t0_us
const int     SAMPLE_BYTES = 2 + 2 * N_CH;  // dt_us + codes
const int     PACKET_BYTES = HDR_BYTES + PACKET_SAMPLES * SAMPLE_BYTES + 1;

SPISettings adcSPI(SPI_HZ, MSBFIRST, SPI_MODE0);

bool     streaming = false;
uint32_t periodUs, nextUs, lastReconfigMs, pktT0;
uint32_t seq;              // samples taken since 's'
uint32_t droppedPackets;
float    tickUs;
uint32_t maxHz;
uint8_t  pkt[PACKET_BYTES];
int      pktN;             // samples in the current packet
uint8_t  pktXor;
uint8_t  rangeR = DEFAULT_RANGE_R;
char     cmdBuf[16];
int      cmdLen;

inline void csLow()  { if (!HW_CS) digitalWrite(PIN_CS, LOW); }
inline void csHigh() { if (!HW_CS) digitalWrite(PIN_CS, HIGH); }

void adcCmd(uint8_t b) {
  SPI.beginTransaction(adcSPI);
  csLow();
  SPI.transfer(b);
  csHigh();
  SPI.endTransaction();
}

float rangeV() { return rangeR == 7 ? 12.288f : rangeR == 4 ? 6.144f : 3.072f; }
float lsbV()   { return 2 * rangeV() / 16384.0f; }

void configureChannels() {
  for (uint8_t ch = 0; ch < N_CH; ch++) adcCmd(0x80 | (ch << 4) | rangeR);
}

void setupAdc() {
  delay(20);                // internal reference settle (~10 ms)
  adcCmd(CMD_RESET);
  delay(2);
  adcCmd(CMD_EXT_CLK);
  configureChannels();
}

// Convert all channels into dst (u16 LE). Caller holds the SPI transaction.
// One 32-bit transfer per conversion: start byte, then three 0x00 bytes; the
// result is in the low 16 bits (B13..B0 then 2 don't-care bits).
inline void convertAll(uint8_t *dst) {
  for (uint8_t ch = 0; ch < N_CH; ch++) {
    csLow();
    uint32_t rx = SPI.transfer32((uint32_t)(0x80 | (ch << 4)) << 24);
    csHigh();
    uint16_t code = (rx & 0xFFFF) >> 2;
    dst[2 * ch] = code & 0xFF;
    dst[2 * ch + 1] = code >> 8;
  }
}

float measureTickUs() {
  uint8_t tmp[2 * N_CH];
  const int n = 2000;
  SPI.beginTransaction(adcSPI);
  uint32_t t0 = micros();
  for (int i = 0; i < n; i++) convertAll(tmp);
  uint32_t t1 = micros();
  SPI.endTransaction();
  return (t1 - t0) / (float)n;
}

void setRate(uint32_t hz) {
  if (hz < 1) hz = 1;
  if (hz > maxHz) hz = maxHz;
  periodUs = (1000000UL + hz / 2) / hz;  // whole microseconds
  uint32_t minPeriod = (uint32_t)ceilf(tickUs * TICK_MARGIN);
  if (periodUs < minPeriod) periodUs = minPeriod;
}

void printHeader() {
  Serial.println();
  Serial.println("# xiao_max1032: MAX1032 on XIAO ESP32-C6, packet stream");
  Serial.printf("# cfg n_ch=%d sample_hz=%.3f period_us=%u max_hz=%u lsb_v=%.9f zero=8192 "
                "range_v=%.3f spi_hz=%u tick_us=%.2f packet_samples=%d packet_bytes=%d\n",
                N_CH, 1e6 / periodUs, periodUs, maxHz, lsbV(), rangeV(), SPI_HZ, tickUs,
                PACKET_SAMPLES, PACKET_BYTES);
  Serial.printf("# dropped_packets=%u (since boot) hw_cs=%d\n", droppedPackets, HW_CS);
  Serial.println("# packet: A5 5A seq(u32) t0_us(u32) N x [dt_us(u16) code0..3(u16, offset binary)] xor8");
  Serial.println("# commands: ? header, r<hz>\\n set rate, g<1|4|7>\\n set range, s start, x stop");
  Serial.println("# end");
}

void startPacket() {
  pktN = 0;
  pktXor = 0;
  pkt[0] = 0xA5;
  pkt[1] = 0x5A;
}

void stopStreaming() {
  if (streaming) SPI.endTransaction();
  streaming = false;
}

void handleCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (cmdLen > 0) {                        // collecting "r<digits>\n" or "g<code>\n"
      if (c == '\n' || c == '\r') {
        cmdBuf[cmdLen] = 0;
        uint32_t val = strtoul(cmdBuf + 1, nullptr, 10);
        if (cmdBuf[0] == 'r') setRate(val);
        else if (val == 1 || val == 4 || val == 7) { rangeR = val; configureChannels(); }
        cmdLen = 0;
        printHeader();
      } else if (cmdLen < (int)sizeof(cmdBuf) - 1) {
        cmdBuf[cmdLen++] = c;
      }
      continue;
    }
    if (c == 'r' || c == 'g') { stopStreaming(); cmdBuf[0] = c; cmdLen = 1; }
    else if (c == '?') { stopStreaming(); printHeader(); }
    else if (c == 's' && !streaming) {
      streaming = true;
      seq = 0;
      startPacket();
      SPI.beginTransaction(adcSPI);          // held for the whole stream
      nextUs = micros() + 100;
    }
    else if (c == 'x') stopStreaming();
  }
}

void setup() {
  Serial.setTxBufferSize(16384);
  Serial.setTxTimeoutMs(0);
  Serial.begin(921600);                      // baud is ignored on native USB
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}

  // loop() never yields while streaming; keep the idle-task watchdog from firing.
  disableCore0WDT();

  SPI.begin(SCK, MISO, MOSI, PIN_CS);        // XIAO: SCK=D8, MISO=D9, MOSI=D10, CS=D0
  if (HW_CS) {
    SPI.setHwCs(true);
  } else {
    pinMode(PIN_CS, OUTPUT);
    digitalWrite(PIN_CS, HIGH);
  }

  setupAdc();
  tickUs = measureTickUs();
  maxHz = (uint32_t)(1e6f / (tickUs * TICK_MARGIN));
  setRate(DEFAULT_HZ);
  printHeader();
}

void loop() {
  // Stay in here for the whole stream. Returning from loop() lets the Arduino
  // core run yieldIfNecessary(), which on single-core chips (C6) pauses this
  // task for 5 ms every 2 s: a 5 ms hole in the data.
  do {
    handleCommands();
    if (streaming) takeSample();
  } while (streaming);
}

void takeSample() {
  // Busy-wait for the next scheduled sample. If the loop ran late (USB write),
  // take it right away; if more than 2 periods late, re-anchor the schedule
  // instead of bursting. Either way each sample records its real time.
  while ((int32_t)(micros() - nextUs) < 0) {}
  nextUs += periodUs;
  if ((int32_t)(micros() - nextUs) > (int32_t)(2 * periodUs)) nextUs = micros() + periodUs;

  uint32_t now = micros();                   // actual sample time
  if (pktN == 0) {
    pktT0 = now;
    memcpy(pkt + 2, &seq, 4);
    memcpy(pkt + 6, &pktT0, 4);
    for (int i = 2; i < HDR_BYTES; i++) pktXor ^= pkt[i];
  }
  uint8_t *dst = pkt + HDR_BYTES + pktN * SAMPLE_BYTES;
  uint16_t dt = (uint16_t)(now - pktT0);     // packet spans a few ms, fits easily
  dst[0] = dt & 0xFF;
  dst[1] = dt >> 8;
  convertAll(dst + 2);
  for (int i = 0; i < SAMPLE_BYTES; i++) pktXor ^= dst[i];
  seq++;

  if (++pktN == PACKET_SAMPLES) {
    pkt[PACKET_BYTES - 1] = pktXor;
    // Never block: if the host isn't keeping up, drop the packet (seq shows the gap).
    if (Serial.availableForWrite() >= PACKET_BYTES) Serial.write(pkt, PACKET_BYTES);
    else droppedPackets++;
    startPacket();
  }

  // Re-send the channel config once a second (no readback on the MAX1032);
  // covers a supply glitch resetting it. Costs a few microseconds.
  if (millis() - lastReconfigMs > 1000) {
    lastReconfigMs = millis();
    for (uint8_t ch = 0; ch < N_CH; ch++) {  // transaction is already held
      csLow();
      SPI.transfer(0x80 | (ch << 4) | rangeR);
      csHigh();
    }
  }
}
