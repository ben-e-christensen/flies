/*
 * Adafruit Feather ESP32-S3 -> 3x ADS1115 -> serial.
 *
 * Each ADS1115 runs in continuous-conversion mode at 860 SPS. Every
 * 1/SAMPLE_HZ seconds the ESP reads the latest conversion from all three and
 * prints one CSV line:
 *
 *     t_us,raw0,raw1,raw2    (signed ADS1115 counts; "nan" if that ADC isn't responding)
 *
 * Counts -> volts happens on the host (adc_serial.py), see tube/claude.md.
 *
 * Lines starting with '#' are info/config messages; the host ignores them.
 *
 * Commands (send one character):
 *   ?   re-print the config header
 *   a   read A0-A3 on every ADC once and print the pin voltages. Use it to find
 *       which input the signal is on (an idle INA159 output sits near 1.25 V).
 *
 * Wiring (ADDR pin sets the I2C address):
 *     ADC0  ADDR->GND  0x48   bottom ring
 *     ADC1  ADDR->VDD  0x49   middle ring
 *     ADC2  ADDR->SDA  0x4A   top ring
 *   All share SDA/SCL (Feather: SDA=3, SCL=4, or the STEMMA QT port),
 *   VDD 3.3 V, GND. The board variant turns on I2C power (GPIO 7) at boot.
 *
 * Flash (arduino-cli, from the tube folder):
 *   arduino-cli compile --fqbn esp32:esp32:adafruit_feather_esp32s3 esp32_ads
 *   arduino-cli upload  --fqbn esp32:esp32:adafruit_feather_esp32s3 -p /dev/ttyACM0 esp32_ads
 * Arduino IDE: board "Adafruit Feather ESP32-S3 2MB PSRAM".
 * If upload can't connect: hold BOOT, tap RESET, release BOOT, upload again,
 * then tap RESET afterwards.
 */

#include <Wire.h>

// ---------------- CONFIG ----------------
const int      SDA_PIN   = SDA;      // board defaults (Feather ESP32-S3: 3 / 4)
const int      SCL_PIN   = SCL;
const uint32_t I2C_HZ    = 400000;
const uint32_t BAUD      = 921600;   // ignored over native USB, matters over UART bridge
const uint32_t SAMPLE_HZ = 500;      // keep <= 860 (ADS1115 max data rate)

const uint8_t ADC_ADDR[3] = {0x48, 0x49, 0x4A};

// Input for each ADC (config register MUX bits):
//   0x4000 AIN0 vs GND   0x5000 AIN1   0x6000 AIN2   0x7000 AIN3
//   0x0000 AIN0 - AIN1 (differential)  0x3000 AIN2 - AIN3
const uint16_t ADC_MUX[3] = {0x4000, 0x4000, 0x4000};

// Full-scale range (PGA bits) -- must match FSR_V below AND ADS_FSR_V in adc_serial.py.
//   0x0000 ±6.144V  0x0200 ±4.096V  0x0400 ±2.048V
//   0x0600 ±1.024V  0x0800 ±0.512V  0x0A00 ±0.256V
const uint16_t PGA   = 0x0200;
const float    FSR_V = 4.096f;
// ----------------------------------------

const uint8_t  REG_CONV   = 0x00;
const uint8_t  REG_CONFIG = 0x01;
const uint16_t MODE_CONT  = 0x0000;  // bit 8 = 0 -> continuous
const uint16_t MODE_SINGLE = 0x0100;
const uint16_t OS_START   = 0x8000;  // start a single-shot conversion
const uint16_t DR_860     = 0x00E0;
const uint16_t COMP_OFF   = 0x0003;

const uint32_t CHECK_MS = 20;  // verify one ADC's config every 20 ms (all 3 within 60 ms)

bool     adcOk[3];
uint32_t periodUs;
uint32_t nextUs;
uint32_t nextCheckMs;
int      checkIdx = 0;

bool writeReg(uint8_t addr, uint8_t reg, uint16_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val >> 8);
  Wire.write(val & 0xFF);
  return Wire.endTransmission() == 0;
}

bool readConv(uint8_t addr, int16_t &out) {
  Wire.beginTransmission(addr);
  Wire.write(REG_CONV);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)2) != 2) return false;
  out = (int16_t)((Wire.read() << 8) | Wire.read());
  return true;
}

bool readReg(uint8_t addr, uint8_t reg, uint16_t &out) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)2) != 2) return false;
  out = (Wire.read() << 8) | Wire.read();
  return true;
}

uint16_t adcConfig(int i) {
  return ADC_MUX[i] | PGA | MODE_CONT | DR_860 | COMP_OFF;
}

bool setupAdc(int i) {
  return writeReg(ADC_ADDR[i], REG_CONFIG, adcConfig(i));
}

// An ADS1115 that loses power comes back in its default state (powered down,
// conversion register = 0) but still ACKs, so reads "succeed" with 0 forever.
// Read the config back and re-apply it if it doesn't match.
void checkAdc(int i) {
  uint16_t cfg;
  if (!readReg(ADC_ADDR[i], REG_CONFIG, cfg)) {
    adcOk[i] = false;
    return;
  }
  if ((cfg & 0x7FFF) != (adcConfig(i) & 0x7FFF)) {  // bit 15 is a status bit on read
    adcOk[i] = setupAdc(i);
    Serial.printf("# ADC%d 0x%02X was reset (config 0x%04X), reconfigured\n",
                  i, ADC_ADDR[i], cfg);
  }
}

void printConfig() {
  Serial.printf("# esp32_ads: %u Hz, FSR +/-%.3f V, SDA=%d SCL=%d\n",
                SAMPLE_HZ, FSR_V, SDA_PIN, SCL_PIN);
  for (int i = 0; i < 3; i++) {
    Serial.printf("# ADC%d 0x%02X mux=0x%04X %s\n", i, ADC_ADDR[i], ADC_MUX[i],
                  adcOk[i] ? "ok" : "NOT FOUND");
  }
  Serial.println("# t_us,raw0,raw1,raw2");
}

// Single-shot read of all four inputs (vs GND) on each ADC, then back to
// continuous mode. Stalls sampling for ~50 ms.
void printAllInputs() {
  Serial.println("# pin voltages, each input vs GND:");
  for (int i = 0; i < 3; i++) {
    char buf[96];
    int n = snprintf(buf, sizeof(buf), "# ADC%d 0x%02X", i, ADC_ADDR[i]);
    for (int ch = 0; ch < 4; ch++) {
      uint16_t mux = 0x4000 | (ch << 12);
      int16_t raw;
      bool ok = writeReg(ADC_ADDR[i], REG_CONFIG,
                         OS_START | mux | PGA | MODE_SINGLE | DR_860 | COMP_OFF);
      delay(3);  // one conversion at 860 SPS is ~1.2 ms
      if (ok && readConv(ADC_ADDR[i], raw)) {
        n += snprintf(buf + n, sizeof(buf) - n, "  A%d=%.3fV", ch, raw * FSR_V / 32768.0f);
      } else {
        n += snprintf(buf + n, sizeof(buf) - n, "  A%d=--", ch);
      }
    }
    Serial.println(buf);
    adcOk[i] = setupAdc(i);
  }
}

void setup() {
  Serial.begin(BAUD);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}  // give USB CDC a moment

  Wire.begin(SDA_PIN, SCL_PIN, I2C_HZ);
  for (int i = 0; i < 3; i++) adcOk[i] = setupAdc(i);
  delay(5);  // first conversion at 860 SPS takes ~1.2 ms

  printConfig();
  periodUs = 1000000UL / SAMPLE_HZ;
  nextUs = micros();
  nextCheckMs = millis();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '?') printConfig();
    if (c == 'a') { printAllInputs(); nextUs = micros(); }
  }

  if ((int32_t)(millis() - nextCheckMs) >= 0) {
    nextCheckMs += CHECK_MS;
    checkAdc(checkIdx);
    checkIdx = (checkIdx + 1) % 3;
  }

  uint32_t now = micros();
  if ((int32_t)(now - nextUs) < 0) return;
  nextUs += periodUs;
  if ((int32_t)(now - nextUs) > (int32_t)(10 * periodUs)) nextUs = now;  // fell way behind, resync

  char line[64];
  int n = snprintf(line, sizeof(line), "%lu", (unsigned long)now);
  for (int i = 0; i < 3; i++) {
    int16_t raw;
    if (adcOk[i] && readConv(ADC_ADDR[i], raw)) {
      n += snprintf(line + n, sizeof(line) - n, ",%d", raw);
    } else {
      n += snprintf(line + n, sizeof(line) - n, ",nan");
      adcOk[i] = setupAdc(i);  // try to bring it back (e.g. loose wire)
    }
  }
  Serial.println(line);
}
