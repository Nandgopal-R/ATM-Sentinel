/*
  INMP441 I2S microphone test

  Standalone sketch - do NOT merge into atm_sentinel.ino yet. This only
  verifies the mic captures audio. The model has no audio features, so the
  mic feeds nothing in the main firmware.

  Wiring (I2S, not I2C - separate from the sensor bus):
    VDD -> 3.3V rail   (3.3V ONLY, no onboard regulator, 5V destroys it)
    GND -> ground rail
    SCK -> GPIO32      bit clock
    WS  -> GPIO25      word select / LRCLK
    SD  -> GPIO33      data out from mic
    L/R -> ground rail (selects LEFT channel - must be tied, not floating)

  Serial Monitor: 115200 baud

  Expected levels (roughly):
    quiet room     -70 to -55 dBFS
    normal speech  -40 to -25 dBFS
    clap nearby    -10 to 0 dBFS
*/

#include <ESP_I2S.h>

// ---------- pins ----------
static const int8_t PIN_I2S_SCK = 32;   // bit clock
static const int8_t PIN_I2S_WS  = 25;   // word select
static const int8_t PIN_I2S_SD  = 33;   // data in (from mic)

// ---------- audio config ----------
static const uint32_t SAMPLE_RATE = 16000;
static const size_t   BLOCK       = 512;   // samples per read (~32 ms at 16 kHz)

// INMP441 sends 24-bit samples left-justified in a 32-bit slot. Read 32-bit
// and shift down by 8 to recover the signed 24-bit value. Reading as 16-bit
// gives noise.
static const int32_t FULL_SCALE_24 = 8388608L;   // 2^23

I2SClass i2s;
static int32_t buf[BLOCK];

// ---------- setup ----------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\n\n=== INMP441 microphone test ==="));

  // setPins(bclk, ws, dout, din, mclk). dout = -1 because we only receive.
  i2s.setPins(PIN_I2S_SCK, PIN_I2S_WS, -1, PIN_I2S_SD, -1);

  // Mono, left slot - matches L/R tied to ground. If L/R is tied to VDD
  // instead, change this to I2S_STD_SLOT_RIGHT or you will read silence.
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE,
                 I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_MONO,
                 I2S_STD_SLOT_LEFT)) {
    Serial.println(F("I2S begin FAILED"));
    Serial.println(F("  -> check the ESP32 core version supports ESP_I2S.h (core 3.x)"));
    while (true) delay(1000);
  }

  Serial.println(F("I2S started. Make some noise.\n"));
  Serial.println(F("  dBFS   RMS      peak     level"));
}

// ---------- loop ----------
void loop() {
  size_t bytesRead = i2s.readBytes((char*)buf, sizeof(buf));
  size_t n = bytesRead / sizeof(int32_t);
  if (n == 0) {
    Serial.println(F("read returned nothing"));
    delay(200);
    return;
  }

  // Shift to 24-bit signed, and accumulate for the DC mean. The INMP441 has a
  // standing DC offset that has to come out before RMS means anything.
  static int32_t s[BLOCK];
  int64_t sum = 0;
  bool    allSame = true;

  for (size_t i = 0; i < n; i++) {
    s[i] = buf[i] >> 8;
    sum += s[i];
    if (i > 0 && s[i] != s[0]) allSame = false;
  }

  // Every sample identical means no real data is arriving.
  if (allSame) {
    Serial.printf("all %u samples identical (%ld) - check SD on GPIO%d, "
                  "VDD on 3.3V, and L/R tied to GND\n",
                  (unsigned)n, (long)s[0], PIN_I2S_SD);
    delay(500);
    return;
  }

  int32_t mean = (int32_t)(sum / (int64_t)n);

  // RMS and peak about the DC mean.
  double  acc  = 0.0;
  int32_t peak = 0;

  for (size_t i = 0; i < n; i++) {
    int32_t v = s[i] - mean;
    acc += (double)v * (double)v;
    int32_t a = (v < 0) ? -v : v;
    if (a > peak) peak = a;
  }

  double rms  = sqrt(acc / (double)n);
  double dbfs = (rms > 1.0) ? 20.0 * log10(rms / (double)FULL_SCALE_24) : -120.0;

  // Simple level bar: -80 dBFS (empty) to 0 dBFS (full), 40 columns.
  int bars = (int)((dbfs + 80.0) / 2.0);
  if (bars < 0)  bars = 0;
  if (bars > 40) bars = 40;

  Serial.printf("%6.1f  %8.0f %8ld  ", dbfs, rms, (long)peak);
  for (int i = 0; i < bars; i++)  Serial.print('#');
  for (int i = bars; i < 40; i++) Serial.print('.');
  Serial.println();

  delay(100);   // ~10 lines/sec, readable in the monitor
}
