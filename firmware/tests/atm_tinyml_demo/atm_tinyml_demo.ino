/*
  ATM Sentinel - TinyML audio + sensor-fusion demo / integration sketch (ESP32, Arduino core 3.x)

  What it does
    * Streams INMP441 audio at 16 kHz, extracts 36 compact features per ~1 s window
      (no raw audio is stored), classifies BACKGROUND / DISPENSE / ABNORMAL (INT8 MLP).
    * Counts SW-420 edges per second, samples the PIR once per second.
    * Segments interactions, runs the fusion decision tree + safety rules, keeps a rolling
      window of recent events and prints one compact JSON summary per event.
    * At boot: self-test against reference vectors from the Python pipeline.
    * Every 30 s: prints measured frame/inference timing, CPU load and heap. Copy those
      numbers into esp32_benchmark.md - they are the on-device measurements.

  It does NOT replace the existing two-stage malfunction/tamper cascade. Feed that cascade's
  tamper decision into cascadeTamperFlag() below; it always takes priority.

  Wiring (INMP441 as in the earlier mic test sketch):
    INMP441 VDD->3.3V  GND->GND  SCK->GPIO32  WS->GPIO25  SD->GPIO33  L/R->GND
  SW-420 DO -> GPIO26, HC-SR501 OUT -> GPIO27 (HARDWARE.md).

  Build: Arduino IDE / arduino-cli, board "ESP32 Dev Module", esp32 core 3.x.
  All .cpp/.h files in this folder are compiled automatically.
  Optional TFLite Micro backend: install a TFLM library for ESP32 and add
  #define ATM_USE_TFLM before the includes (or -DATM_USE_TFLM in build flags).
*/
#include <Arduino.h>
#include <ESP_I2S.h>

#include "esp32_inference.h"
#include "esp32_preprocessing.h"

// ---------------------------------------------------------------- configuration
static const int8_t PIN_I2S_SCK = 32;
static const int8_t PIN_I2S_WS = 25;
static const int8_t PIN_I2S_SD = 33;
static const int PIN_SW420 = 26;   // HARDWARE.md
static const int PIN_PIR = 27;     // HARDWARE.md
static const char* ATM_ID = "ATM_001";

// Same counting as firmware/atm_sentinel (schema.yaml vibration): falling edges, 5 ms debounce,
// count per 1 s capped at 200. ATM_TAMPER_VIB_THRESHOLD (75) is in these units.
static const int SW420_EDGE = FALLING;
static const uint32_t SW420_DEBOUNCE_US = 5000;
static const uint32_t SW420_MAX_COUNT = 200;

static const size_t I2S_BLOCK = 256;  // samples per I2S read (16 ms)

// ---------------------------------------------------------------- globals
I2SClass i2s;
static atm::AudioFeatureExtractor fx;   // ~17 KB static RAM
static atm::AudioClassifier clf;
static atm::EventInterpreter interp;

static volatile uint32_t gVibEdges = 0, gVibLastUs = 0;
static portMUX_TYPE gMux = portMUX_INITIALIZER_UNLOCKED;
static atm::AudioResult gLatest;
static volatile bool gLatestFresh = false;

// timing (written by audio task, read by loop)
static volatile uint32_t gPushUsSum = 0, gPushBlocks = 0, gPushUsMax = 0;
static volatile uint32_t gClfUsSum = 0, gClfCount = 0, gClfUsMax = 0;
static volatile uint32_t gWindows = 0, gI2sErrors = 0;
static TaskHandle_t gAudioTask = nullptr;

void IRAM_ATTR onVibEdge() {
  uint32_t now = micros();
  if (now - gVibLastUs >= SW420_DEBOUNCE_US) { gVibEdges = gVibEdges + 1; gVibLastUs = now; }
}

// Hook for the existing cascade's tamper decision (safety priority).
// Replace the body with the flag your main firmware already computes.
bool cascadeTamperFlag() { return false; }

// Hook for the transport (ESP-NOW -> gateway -> MQTT). Default: serial only.
void publishSummary(const char* json) { Serial.println(json); }

// ---------------------------------------------------------------- audio task
static void audioTask(void*) {
  static int32_t raw[I2S_BLOCK];
  static float pcm[I2S_BLOCK];
  for (;;) {
    size_t got = i2s.readBytes((char*)raw, sizeof(raw));
    if (got != sizeof(raw)) {
      gI2sErrors = gI2sErrors + 1;
      continue;
    }
    for (size_t i = 0; i < I2S_BLOCK; ++i) pcm[i] = atm::inmp441ToFloat(raw[i]);

    uint32_t t0 = micros();
    size_t n = I2S_BLOCK;
    const float* p = pcm;
    while (n) {
      bool ready = false;
      size_t used = fx.push(p, n, &ready);
      p += used;
      n -= used;
      if (ready) {
        uint32_t c0 = micros();
        atm::AudioResult r;
        if (clf.classify(fx.features(), &r)) {
          uint32_t dt = micros() - c0;
          gClfUsSum = gClfUsSum + dt;
          gClfCount = gClfCount + 1;
          if (dt > gClfUsMax) gClfUsMax = dt;
          portENTER_CRITICAL(&gMux);
          gLatest = r;
          gLatestFresh = true;
          portEXIT_CRITICAL(&gMux);
          gWindows = gWindows + 1;
        }
      }
    }
    uint32_t dt = micros() - t0;
    gPushUsSum = gPushUsSum + dt;
    gPushBlocks = gPushBlocks + 1;
    if (dt > gPushUsMax) gPushUsMax = dt;
  }
}

// ---------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.printf("\n=== ATM Sentinel TinyML (%s) ===\n", ATM_MODEL_VERSION);

  pinMode(PIN_PIR, INPUT_PULLDOWN);
  pinMode(PIN_SW420, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_SW420), onVibEdge, SW420_EDGE);

  fx.begin();
  if (!clf.begin()) {
    Serial.println("audio classifier init FAILED");
    while (true) delay(1000);
  }
  float err = 0;
  int ok = atm::selfTest(clf, &err);
  const int nTv = atm::selfTestVectorCount();
  Serial.printf("self-test: %d/%d argmax match, max |prob diff| %.4f -> %s\n", ok, nTv, err,
                (ok == nTv && err < 0.02f) ? "PASS" : "CHECK");

  // micro-benchmark on synthetic noise (repeatable, independent of the mic)
  {
    static float frame[ATM_FRAME];
    uint32_t seed = 1;
    for (int i = 0; i < ATM_FRAME; ++i) {
      seed = seed * 1664525u + 1013904223u;
      frame[i] = ((int32_t)(seed >> 8) - (1 << 23)) / 8388608.0f * 0.1f;
    }
    float f[ATM_N_FRAME_FEATURES];
    uint32_t t0 = micros();
    for (int i = 0; i < 50; ++i) fx.computeFrame(frame, f);
    uint32_t perFrame = (micros() - t0) / 50;
    static float feats[ATM_N_AUDIO_FEATURES];
    for (int i = 0; i < ATM_N_AUDIO_FEATURES; ++i) feats[i] = f[i % ATM_N_FRAME_FEATURES];
    atm::AudioResult r;
    t0 = micros();
    for (int i = 0; i < 200; ++i) clf.classify(feats, &r);
    uint32_t perClf = (micros() - t0) / 200;
    Serial.printf("bench: frame features %lu us/frame (budget 32000 us), classify %lu us, "
                  "CPU load ~%.1f%%\n", (unsigned long)perFrame, (unsigned long)perClf,
                  100.0f * (perFrame + perClf / 31.0f) / 32000.0f);
    fx.reset();
  }

  i2s.setPins(PIN_I2S_SCK, PIN_I2S_WS, -1, PIN_I2S_SD, -1);
  if (!i2s.begin(I2S_MODE_STD, ATM_SR, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO,
                 I2S_STD_SLOT_LEFT)) {
    Serial.println("I2S begin FAILED (needs Arduino-ESP32 core 3.x)");
    while (true) delay(1000);
  }
  interp.begin();
  // 8 KB stack: extractor state is static, the task only holds two 1 KB blocks
  xTaskCreatePinnedToCore(audioTask, "atm_audio", 8192, nullptr, 3, &gAudioTask, 0);
  Serial.printf("free heap after init: %lu bytes\n", (unsigned long)ESP.getFreeHeap());
}

void loop() {
  static uint32_t nextTick = millis() + 1000;
  static uint32_t nowS = 0;
  static uint32_t nextStats = 30;
  static char json[640];

  if ((int32_t)(millis() - nextTick) < 0) {
    delay(5);
    return;
  }
  nextTick += 1000;
  ++nowS;

  noInterrupts();
  uint32_t edges = gVibEdges;
  gVibEdges = 0;
  interrupts();
  bool pir = digitalRead(PIN_PIR) == HIGH;

  atm::AudioResult a;
  bool fresh = false;
  portENTER_CRITICAL(&gMux);
  if (gLatestFresh) {
    a = gLatest;
    gLatestFresh = false;
    fresh = true;
  }
  portEXIT_CRITICAL(&gMux);

  interp.tick(nowS, (float)min(edges, SW420_MAX_COUNT), pir, fresh ? &a : nullptr, cascadeTamperFlag());

  atm::EventSummary e;
  while (interp.pop(&e)) {
    atm::formatEventJson(e, interp.window().stats(nowS), ATM_ID, json, sizeof json);
    publishSummary(json);
  }

  if (nowS >= nextStats) {
    nextStats = nowS + 30;
    const uint32_t blocks = gPushBlocks ? gPushBlocks : 1;
    const uint32_t clfs = gClfCount ? gClfCount : 1;
    const float blockUs = (float)gPushUsSum / blocks;
    Serial.printf(
        "[stats] windows=%lu  push %.0f us/block avg (%lu max, block=%u samples=%.0f us)  "
        "classify %.0f us avg (%lu max)  audio CPU ~%.1f%%  heap free %lu (min %lu)  "
        "audio stack free %u B  i2s errors %lu\n",
        (unsigned long)gWindows, blockUs, (unsigned long)gPushUsMax, (unsigned)I2S_BLOCK,
        I2S_BLOCK * 1e6f / ATM_SR, (float)gClfUsSum / clfs, (unsigned long)gClfUsMax,
        100.0f * blockUs / (I2S_BLOCK * 1e6f / ATM_SR), (unsigned long)ESP.getFreeHeap(),
        (unsigned long)ESP.getMinFreeHeap(),
        (unsigned)(uxTaskGetStackHighWaterMark(gAudioTask) * sizeof(StackType_t)),
        (unsigned long)gI2sErrors);
  }
}
