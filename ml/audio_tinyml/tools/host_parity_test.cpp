// Host (PC) build of the ESP32 firmware modules, for parity testing only.
//
//   (from ml/audio_tinyml/; FW = ../../firmware/atm_sentinel)
//   g++ -O2 -std=c++17 -I$FW tools/host_parity_test.cpp $FW/esp32_preprocessing.cpp $FW/esp32_inference.cpp
//       -o build/host_parity_test
//
//   host_parity_test features <pcm_f32le_16k_mono.raw>   -> CSV: window, 36 features, 3 probs
//   host_parity_test selftest                            -> embedded test vectors
//   host_parity_test scenario                            -> scripted event-interpreter run (JSON)
//   host_parity_test fusion <events.csv>                 -> decide() over fusion feature rows
//   host_parity_test bench <pcm_f32le_16k_mono.raw>      -> host timing (NOT ESP32 timing)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "esp32_inference.h"
#include "esp32_preprocessing.h"
#include "test_vectors.h"

using namespace atm;

static std::vector<float> readF32(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  std::vector<float> v;
  float buf[4096];
  size_t n;
  while ((n = fread(buf, sizeof(float), 4096, f)) > 0) v.insert(v.end(), buf, buf + n);
  fclose(f);
  return v;
}

static AudioFeatureExtractor fx;  // ~17 KB, static like on the ESP32
static AudioClassifier clf;

static int cmdFeatures(const char* path) {
  std::vector<float> x = readF32(path);
  fx.begin();
  clf.begin();
  const size_t kBlock = 256;  // mimic I2S DMA block size
  int win = 0;
  for (size_t i = 0; i < x.size(); i += kBlock) {
    size_t n = std::min(kBlock, x.size() - i);
    const float* p = &x[i];
    while (n) {
      bool ready;
      size_t used = fx.push(p, n, &ready);
      p += used;
      n -= used;
      if (ready) {
        AudioResult r;
        clf.classify(fx.features(), &r);
        printf("%d", win++);
        for (int k = 0; k < ATM_N_AUDIO_FEATURES; ++k) printf(",%.9g", fx.features()[k]);
        for (int c = 0; c < ATM_N_AUDIO_CLASSES; ++c) printf(",%.6f", r.prob[c]);
        printf("\n");
      }
    }
  }
  return 0;
}

static int cmdSelfTest() {
  clf.begin();
  float err = 0;
  int ok = selfTest(clf, &err);
  printf("selftest: %d/%d argmax match, max |prob diff| = %.4f  (TFLM arena used: %zu B)\n", ok,
         ATM_N_TEST_VECTORS, err, clf.arenaUsedBytes());
  return ok == ATM_N_TEST_VECTORS ? 0 : 2;
}

static int cmdFusion(const char* path) {
  // expects atm_fusion_events.csv: first 6 columns metadata, then 13 fusion features
  FILE* f = fopen(path, "r");
  if (!f) { perror(path); return 1; }
  char line[4096];
  if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }  // header
  while (fgets(line, sizeof line, f)) {
    std::vector<std::string> cols;
    char* save = nullptr;
    for (char* t = strtok_r(line, ",\n", &save); t; t = strtok_r(nullptr, ",\n", &save)) cols.push_back(t);
    if (cols.size() < 6 + ATM_N_FUSION_FEATURES) continue;
    float feat[ATM_N_FUSION_FEATURES];
    for (int i = 0; i < ATM_N_FUSION_FEATURES; ++i) feat[i] = strtof(cols[6 + i].c_str(), nullptr);
    EventClass c;
    DecisionRule r;
    float conf;
    EventInterpreter::decide(feat, &c, &r, &conf);
    printf("%s,%s,%s,%s,%.3f\n", cols[0].c_str(), cols[1].c_str(), cols[4].c_str(), eventClassName(c), conf);
  }
  fclose(f);
  return 0;
}

// Scripted scenario: idle -> short visit -> withdrawal with dispense audio -> abnormal
// sounds -> vibration attack. Audio results are injected directly (the extractor is tested
// separately), so this checks segmentation, rules, rolling window and JSON.
static int cmdScenario() {
  EventInterpreter ei;
  ei.begin();
  AudioResult bg = {{0.97f, 0.02f, 0.01f}, AUDIO_BACKGROUND};
  AudioResult disp = {{0.05f, 0.90f, 0.05f}, AUDIO_DISPENSE};
  AudioResult abn = {{0.05f, 0.15f, 0.80f}, AUDIO_ABNORMAL};
  uint32_t t = 0;
  char json[640];
  auto drain = [&]() {
    EventSummary e;
    while (ei.pop(&e)) {
      formatEventJson(e, ei.window().stats(t), "ATM-TEST-01", json, sizeof json);
      printf("t=%4u  %s\n", t, json);
    }
  };
  auto run = [&](int secs, bool pir, float vib, const AudioResult* a, bool cascade = false) {
    for (int i = 0; i < secs; ++i, ++t) {
      ei.tick(t, vib, pir, a, cascade);
      drain();
    }
  };
  run(75, false, 0.0f, &bg);              // idle (one heartbeat at 60 s)
  for (int k = 0; k < 4; ++k) {           // four short visits ~22 s, no dispense
    run(22, true, 2.0f, &bg);
    run(40, false, 0.0f, &bg);
  }
  run(30, true, 3.0f, &bg);               // withdrawal: keypad...
  run(8, true, 12.0f, &disp);             // ...dispense sound + small vibration
  run(15, true, 2.0f, &bg);
  run(30, false, 0.0f, &bg);
  run(40, true, 8.0f, &abn);              // abnormal mechanism while customer waits
  run(30, false, 0.0f, &bg);
  run(10, true, 3.0f, &disp);             // dispense sound...
  run(1, true, 140.0f, &disp);            // ...with a hard hit: tamper must win
  run(20, true, 4.0f, &bg);
  run(30, false, 0.0f, &bg);
  run(3, false, 0.0f, &bg);
  return 0;
}

static int cmdBench(const char* path) {
  std::vector<float> x = readF32(path);
  fx.begin();
  clf.begin();
  float f[ATM_N_FRAME_FEATURES];
  const int frames = (int)(x.size() / ATM_FRAME);
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int rep = 0; rep < 20; ++rep)
    for (int i = 0; i < frames; ++i) fx.computeFrame(&x[i * ATM_FRAME], f);
  auto t1 = std::chrono::high_resolution_clock::now();
  AudioResult r;
  float feat[ATM_N_AUDIO_FEATURES] = {0};
  for (int i = 0; i < 100000; ++i) { feat[0] = (float)(i % 7); clf.classify(feat, &r); }
  auto t2 = std::chrono::high_resolution_clock::now();
  printf("host frame features: %.2f us/frame;  host classify: %.3f us\n",
         std::chrono::duration<double, std::micro>(t1 - t0).count() / (20.0 * frames),
         std::chrono::duration<double, std::micro>(t2 - t1).count() / 100000.0);
  printf("sizeof(AudioFeatureExtractor)=%zu  sizeof(EventInterpreter)=%zu  sizeof(EventSummary)=%zu\n",
         sizeof(AudioFeatureExtractor), sizeof(EventInterpreter), sizeof(EventSummary));
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 1; }
  std::string cmd = argv[1];
  if (cmd == "features" && argc > 2) return cmdFeatures(argv[2]);
  if (cmd == "selftest") return cmdSelfTest();
  if (cmd == "scenario") return cmdScenario();
  if (cmd == "fusion" && argc > 2) return cmdFusion(argv[2]);
  if (cmd == "bench" && argc > 2) return cmdBench(argv[2]);
  fprintf(stderr, "bad command\n");
  return 1;
}
