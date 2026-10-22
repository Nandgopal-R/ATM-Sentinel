// ATM Sentinel - ESP32 TinyML inference: audio classifier, sensor-fusion decision layer,
// event segmentation and rolling event window.
//
// Pipeline (one call per second to EventInterpreter::tick):
//   audio window features -> AudioClassifier -> per-window class probabilities
//   + SW-420 counts/s + PIR state + existing cascade tamper flag
//   -> event segmentation -> fusion decision tree + safety rules -> EventSummary
//   -> RollingWindow (confirmed dispenses, repeated short visits)
//   -> compact JSON for the backend (no raw audio is ever sent).
//
// Backends for the audio model:
//   default            native INT8 dense engine (bit-compatible with TFLite integer
//                      FULLY_CONNECTED; no library needed)
//   -DATM_USE_TFLM      TensorFlow Lite Micro (esp-tflite-micro) running the same
//                      atm_audio_model.tflite flatbuffer embedded in atm_model_data.h
//
// The existing two-stage malfunction/tamper cascade is NOT replaced. Its tamper decision is
// passed in as `cascadeTamper` and always wins.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "atm_model_data.h"
#include "esp32_preprocessing.h"

namespace atm {

enum AudioClass : uint8_t { AUDIO_BACKGROUND = 0, AUDIO_DISPENSE = 1, AUDIO_ABNORMAL = 2 };

struct AudioResult {
  float prob[ATM_N_AUDIO_CLASSES];
  uint8_t cls;
};

class AudioClassifier {
 public:
  bool begin();  // false if the TFLM interpreter could not be created
  // rawFeatures: the 36 un-normalised window features from AudioFeatureExtractor.
  bool classify(const float* rawFeatures, AudioResult* out);
  // TFLM backend only: bytes of the tensor arena actually used (0 for the native engine).
  size_t arenaUsedBytes() const;

 private:
  bool ready_ = false;
};

enum EventClass : uint8_t {
  EV_IDLE = 0,
  EV_NORMAL_DISPENSE = 1,
  EV_SHORT_VISIT = 2,
  EV_VISIT_NO_DISPENSE = 3,
  EV_ABNORMAL_MECHANICAL = 4,
  EV_SUSPECTED_TAMPER = 5,
  EV_UNKNOWN = 6,
};
const char* eventClassName(EventClass c);

enum DecisionRule : uint8_t {
  RULE_MODEL = 0,                  // fusion tree, confident
  RULE_TAMPER_PRIORITY = 1,        // cascade tamper flag or vib peak >= threshold
  RULE_LOW_CONFIDENCE = 2,         // tree confidence below kMinConfidence -> UNKNOWN
  RULE_DISPENSE_NO_PRESENCE = 3,   // dispense sound with nobody at the ATM -> UNKNOWN
  RULE_TAMPER_EARLY_ALERT = 4,     // immediate alert raised mid-event
  RULE_CONFIRMED = 5,
};
const char* ruleName(DecisionRule r);

struct EventSummary {
  uint32_t startS;          // device seconds at event start
  uint32_t durationS;
  EventClass cls;
  DecisionRule rule;
  float confidence;         // 0..1
  bool final;               // false for an early tamper alert (event still running)
  bool countedDispense;     // true only for a confirmed NORMAL_DISPENSE
  float fusion[ATM_N_FUSION_FEATURES];  // order = model_metadata.json fusion_model.features
};

struct RollingStats {
  uint32_t dispenseCount;           // confirmed dispenses since boot
  uint8_t shortVisitsRecent;        // SHORT_VISIT events in the last kShortVisitWindowS
  bool repeatedShortVisits;         // >= kRepeatedShortVisits in that window
  float noDispenseVisitScore;       // consecutive visits without dispense (short = 0.5)
  int32_t lastDispenseAgeS;         // -1 if none since boot
};

// Fixed-size ring buffer of recent event summaries (no raw sensor history).
class RollingWindow {
 public:
  static constexpr int kCapacity = 16;
  static constexpr uint32_t kShortVisitWindowS = 600;
  static constexpr int kRepeatedShortVisits = 3;

  void reset();
  void add(const EventSummary& e);
  RollingStats stats(uint32_t nowS) const;

 private:
  struct Slim {
    uint32_t endS;
    EventClass cls;
  };
  Slim ring_[kCapacity];
  int head_ = 0, count_ = 0;
  uint32_t dispenseCount_ = 0;
  uint32_t lastDispenseEndS_ = 0;
  bool anyDispense_ = false;
};

class EventInterpreter {
 public:
  static constexpr float kMinConfidence = 0.6f;
  static constexpr uint32_t kIdleHeartbeatS = 60;      // IDLE summary period
  static constexpr uint32_t kMaxEventS = 300;          // force-close long events
  static constexpr uint32_t kAudioQuietToCloseS = 3;   // background audio needed to close
  static constexpr float kAudioActiveProb = 0.6f;      // non-background prob to open an event
  static constexpr int kQueue = 4;

  void begin();

  // Call once per second.
  //   vibCount      SW-420 edges counted in the last second
  //   pirHigh       PIR output sampled this second
  //   audio         result of an audio window that completed since the last tick, or nullptr
  //   cascadeTamper tamper decision from the existing cascade (safety priority)
  void tick(uint32_t nowS, float vibCount, bool pirHigh, const AudioResult* audio,
            bool cascadeTamper);

  // Ends the current event now as NORMAL_DISPENSE (dispensed) or ABNORMAL_MECHANICAL, with its
  // measurements. The rest of that visit is then absorbed, so it does not produce a second event.
  // Tamper still wins.
  void forceEvent(uint32_t nowS, bool dispensed);

  // Pops the next finished summary (or early tamper alert). Returns false if none.
  bool pop(EventSummary* out);

  const RollingWindow& window() const { return window_; }

  // Classifies a fusion feature vector with tree + rules (exposed for testing).
  static void decide(const float* f, EventClass* cls, DecisionRule* rule, float* conf);

 private:
  void openEvent(uint32_t nowS, bool interaction);
  void closeEvent(uint32_t nowS);
  void fillFusion(float* f) const;
  void enqueue(const EventSummary& e);

  bool open_ = false;
  bool interaction_ = false;
  bool tamperSeen_ = false;
  bool earlyAlertSent_ = false;
  bool absorbing_ = false;
  uint32_t startS_ = 0, lastS_ = 0;
  uint32_t quietAudioS_ = 0;

  // running audio statistics over the event (no per-window storage)
  uint32_t nWin_ = 0;
  float pSum_[ATM_N_AUDIO_CLASSES];
  float pDispMax_ = 0, pAbnMax_ = 0;
  uint32_t nDispWin_ = 0, nAbnWin_ = 0;

  // sensor accumulators (same maths as the training script)
  VibrationAccumulator vib_;
  PirTracker pir_;

  EventSummary q_[kQueue];
  int qHead_ = 0, qCount_ = 0;
  RollingWindow window_;
};

// Compact JSON for the backend (ESP-NOW/MQTT payload). Returns bytes written (excl. NUL).
int formatEventJson(const EventSummary& e, const RollingStats& s, const char* atmId, char* buf,
                    size_t len);

// Runs the embedded reference vectors through the classifier.
// Returns the number of vectors whose argmax matches and max |prob diff| in *maxAbsErr.
int selfTest(AudioClassifier& clf, float* maxAbsErr);
int selfTestVectorCount();

}  // namespace atm
