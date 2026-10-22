// ATM Sentinel - ESP32 TinyML inference. See esp32_inference.h.
#include "esp32_inference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "test_vectors.h"

#ifdef ATM_USE_TFLM
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#if __has_include("tensorflow/lite/micro/schema/schema_generated.h")
#include "tensorflow/lite/micro/schema/schema_generated.h"  // TFLM 2025+
#else
#include "tensorflow/lite/schema/schema_generated.h"        // esp-tflite-micro / older TFLM
#endif
#ifndef ATM_TFLM_ARENA_BYTES
#define ATM_TFLM_ARENA_BYTES (6 * 1024)
#endif
#endif

namespace atm {

// =====================================================================================
// Integer kernels (same arithmetic as TFLite reference int8 FULLY_CONNECTED)
// =====================================================================================
namespace {

inline int32_t saturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  const bool overflow = (a == b) && (a == INT32_MIN);
  const int64_t ab = (int64_t)a * (int64_t)b;
  const int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  const int32_t r = (int32_t)((ab + nudge) / (1LL << 31));
  return overflow ? INT32_MAX : r;
}

inline int32_t roundingDivideByPOT(int32_t x, int exponent) {
  if (exponent <= 0) return x;
  const int32_t mask = (int32_t)((1LL << exponent) - 1);
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
  return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

inline int32_t multiplyByQuantizedMultiplier(int32_t x, int32_t mult, int shift) {
  const int left = shift > 0 ? shift : 0;
  const int right = shift > 0 ? 0 : -shift;
  return roundingDivideByPOT(saturatingRoundingDoublingHighMul(x * (1 << left), mult), right);
}

inline int8_t clampI8(int32_t v) { return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); }

// Per-output-channel requantisation (TFLite converter default for FULLY_CONNECTED weights).
void denseInt8(const int8_t* in, int nIn, int8_t* out, int nOut, const int8_t* W,
               const int32_t* B, int inZp, int outZp, const int32_t* mult, const int8_t* shift,
               bool relu) {
  for (int o = 0; o < nOut; ++o) {
    int32_t acc = B[o];
    const int8_t* w = &W[o * nIn];
    for (int i = 0; i < nIn; ++i) acc += (int32_t)w[i] * ((int32_t)in[i] - inZp);
    int32_t v = multiplyByQuantizedMultiplier(acc, mult[o], shift[o]) + outZp;
    if (relu && v < outZp) v = outZp;
    out[o] = clampI8(v);
  }
}

void softmax(const float* logits, float* p, int n) {
  float m = logits[0];
  for (int i = 1; i < n; ++i) m = logits[i] > m ? logits[i] : m;
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    p[i] = expf(logits[i] - m);
    s += p[i];
  }
  for (int i = 0; i < n; ++i) p[i] /= s;
}

// Generic flattened tree ensemble: averages leaf class probabilities over trees.
// sklearn convention: x[feature] <= threshold goes left.
void treePredict(const int8_t* feat, const float* thr, const int16_t* left, const int16_t* right,
                 const uint8_t* leafProb, const int16_t* roots, int nTrees, int nOut,
                 const float* x, float* out) {
  for (int c = 0; c < nOut; ++c) out[c] = 0.0f;
  for (int t = 0; t < nTrees; ++t) {
    int n = roots[t];
    while (feat[n] >= 0) n = (x[feat[n]] <= thr[n]) ? left[n] : right[n];
    for (int c = 0; c < nOut; ++c) out[c] += leafProb[n * nOut + c] * (1.0f / 255.0f);
  }
  float s = 0.0f;
  for (int c = 0; c < nOut; ++c) s += out[c];
  if (s > 0.0f)
    for (int c = 0; c < nOut; ++c) out[c] /= s;
}

void normalizeAudio(const float* raw, float* z) {
  for (int i = 0; i < ATM_N_AUDIO_FEATURES; ++i) z[i] = (raw[i] - ATM_FEAT_MEAN[i]) / ATM_FEAT_STD[i];
}

#ifdef ATM_USE_TFLM
alignas(16) uint8_t gArena[ATM_TFLM_ARENA_BYTES];
tflite::MicroInterpreter* gInterp = nullptr;
#endif

}  // namespace

// =====================================================================================
// AudioClassifier
// =====================================================================================
bool AudioClassifier::begin() {
#ifdef ATM_USE_TFLM
#ifndef ATM_AUDIO_BACKEND_DENSE_INT8
#error "ATM_USE_TFLM requires the dense INT8 audio model (retrain/export with a dense backend)"
#endif
  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  const tflite::Model* model = tflite::GetModel(ATM_AUDIO_TFLITE);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  static tflite::MicroInterpreter interp(model, resolver, gArena, sizeof(gArena));
  if (interp.AllocateTensors() != kTfLiteOk) return false;
  gInterp = &interp;
#endif
  ready_ = true;
  return true;
}

size_t AudioClassifier::arenaUsedBytes() const {
#ifdef ATM_USE_TFLM
  return gInterp ? gInterp->arena_used_bytes() : 0;
#else
  return 0;
#endif
}

bool AudioClassifier::classify(const float* raw, AudioResult* out) {
  if (!ready_) return false;
  float z[ATM_N_AUDIO_FEATURES];
  normalizeAudio(raw, z);

#if defined(ATM_AUDIO_BACKEND_DENSE_INT8)
  int8_t q0[ATM_N_AUDIO_FEATURES];
  for (int i = 0; i < ATM_N_AUDIO_FEATURES; ++i)
    q0[i] = clampI8((int32_t)lrintf(z[i] / ATM_IN_SCALE) + ATM_IN_ZP);  // round-half-even
#ifdef ATM_USE_TFLM
  TfLiteTensor* in = gInterp->input(0);
  memcpy(in->data.int8, q0, sizeof(q0));
  if (gInterp->Invoke() != kTfLiteOk) return false;
  TfLiteTensor* o = gInterp->output(0);
  for (int c = 0; c < ATM_N_AUDIO_CLASSES; ++c)
    out->prob[c] = ((int32_t)o->data.int8[c] - o->params.zero_point) * o->params.scale;
#else
  float logits[ATM_N_AUDIO_CLASSES];
#if ATM_DENSE_N_LAYERS == 1
  int8_t q1[ATM_L0_OUT];
  denseInt8(q0, ATM_L0_IN, q1, ATM_L0_OUT, ATM_L0_W, ATM_L0_B, ATM_L0_IN_ZP, ATM_L0_OUT_ZP,
            ATM_L0_MULT, ATM_L0_SHIFT, false);
  const int8_t* qLogits = q1;
#elif ATM_DENSE_N_LAYERS == 2
  int8_t q1[ATM_L0_OUT];
  int8_t q2[ATM_L1_OUT];
  denseInt8(q0, ATM_L0_IN, q1, ATM_L0_OUT, ATM_L0_W, ATM_L0_B, ATM_L0_IN_ZP, ATM_L0_OUT_ZP,
            ATM_L0_MULT, ATM_L0_SHIFT, true);
  denseInt8(q1, ATM_L1_IN, q2, ATM_L1_OUT, ATM_L1_W, ATM_L1_B, ATM_L1_IN_ZP, ATM_L1_OUT_ZP,
            ATM_L1_MULT, ATM_L1_SHIFT, false);
  const int8_t* qLogits = q2;
#elif ATM_DENSE_N_LAYERS == 3
  int8_t q1[ATM_L0_OUT];
  int8_t q2[ATM_L1_OUT];
  int8_t q3[ATM_L2_OUT];
  denseInt8(q0, ATM_L0_IN, q1, ATM_L0_OUT, ATM_L0_W, ATM_L0_B, ATM_L0_IN_ZP, ATM_L0_OUT_ZP,
            ATM_L0_MULT, ATM_L0_SHIFT, true);
  denseInt8(q1, ATM_L1_IN, q2, ATM_L1_OUT, ATM_L1_W, ATM_L1_B, ATM_L1_IN_ZP, ATM_L1_OUT_ZP,
            ATM_L1_MULT, ATM_L1_SHIFT, true);
  denseInt8(q2, ATM_L2_IN, q3, ATM_L2_OUT, ATM_L2_W, ATM_L2_B, ATM_L2_IN_ZP, ATM_L2_OUT_ZP,
            ATM_L2_MULT, ATM_L2_SHIFT, false);
  const int8_t* qLogits = q3;
#else
#error "unsupported ATM_DENSE_N_LAYERS"
#endif
  for (int c = 0; c < ATM_N_AUDIO_CLASSES; ++c)
    logits[c] = ((int32_t)qLogits[c] - ATM_LOGIT_ZP) * ATM_LOGIT_SCALE;
  softmax(logits, out->prob, ATM_N_AUDIO_CLASSES);
#endif  // ATM_USE_TFLM

#elif defined(ATM_AUDIO_BACKEND_TREES)
  treePredict(ATM_AT_FEATURE, ATM_AT_THRESHOLD, ATM_AT_LEFT, ATM_AT_RIGHT, ATM_AT_LEAF_PROB,
              ATM_AT_ROOTS, ATM_AT_N_TREES, ATM_N_AUDIO_CLASSES, z, out->prob);
#else
#error "atm_model_data.h defines no audio backend"
#endif

  uint8_t best = 0;
  for (uint8_t c = 1; c < ATM_N_AUDIO_CLASSES; ++c)
    if (out->prob[c] > out->prob[best]) best = c;
  out->cls = best;
  return true;
}

// =====================================================================================
// Names
// =====================================================================================
const char* eventClassName(EventClass c) {
  static const char* const names[] = {"IDLE", "NORMAL_DISPENSE", "SHORT_VISIT",
                                      "VISIT_NO_DISPENSE", "ABNORMAL_MECHANICAL",
                                      "SUSPECTED_TAMPER", "UNKNOWN"};
  return (c <= EV_UNKNOWN) ? names[c] : "?";
}

const char* ruleName(DecisionRule r) {
  static const char* const names[] = {"MODEL", "TAMPER_PRIORITY", "LOW_CONFIDENCE",
                                      "DISPENSE_NO_PRESENCE", "TAMPER_EARLY_ALERT", "CONFIRMED"};
  return (r <= RULE_CONFIRMED) ? names[r] : "?";
}

// =====================================================================================
// RollingWindow
// =====================================================================================
void RollingWindow::reset() {
  head_ = count_ = 0;
  dispenseCount_ = 0;
  lastDispenseEndS_ = 0;
  anyDispense_ = false;
}

void RollingWindow::add(const EventSummary& e) {
  if (!e.final) return;               // early alerts are not events
  if (e.countedDispense) {
    ++dispenseCount_;
    anyDispense_ = true;
    lastDispenseEndS_ = e.startS + e.durationS;
  }
  if (e.cls == EV_IDLE) return;       // heartbeats would flush the ring
  ring_[head_] = Slim{e.startS + e.durationS, e.cls};
  head_ = (head_ + 1) % kCapacity;
  if (count_ < kCapacity) ++count_;
}

RollingStats RollingWindow::stats(uint32_t nowS) const {
  RollingStats s;
  memset(&s, 0, sizeof(s));
  s.dispenseCount = dispenseCount_;
  s.lastDispenseAgeS = !anyDispense_ ? -1 : nowS > lastDispenseEndS_ ? (int32_t)(nowS - lastDispenseEndS_) : 0;
  bool stopScore = false;
  for (int k = 0; k < count_; ++k) {
    const Slim& e = ring_[(head_ - 1 - k + kCapacity) % kCapacity];  // newest first
    if (e.cls == EV_SHORT_VISIT && nowS - e.endS <= kShortVisitWindowS) ++s.shortVisitsRecent;
    if (!stopScore) {
      if (e.cls == EV_NORMAL_DISPENSE) stopScore = true;
      else if (e.cls == EV_VISIT_NO_DISPENSE) s.noDispenseVisitScore += 1.0f;
      else if (e.cls == EV_SHORT_VISIT) s.noDispenseVisitScore += 0.5f;  // not a failed withdrawal
    }
  }
  s.repeatedShortVisits = s.shortVisitsRecent >= kRepeatedShortVisits;
  return s;
}

// =====================================================================================
// EventInterpreter
// =====================================================================================
void EventInterpreter::begin() {
  open_ = false;
  absorbing_ = false;
  qHead_ = qCount_ = 0;
  window_.reset();
}

void EventInterpreter::openEvent(uint32_t nowS, bool interaction) {
  open_ = true;
  interaction_ = interaction;
  tamperSeen_ = false;
  earlyAlertSent_ = false;
  startS_ = lastS_ = nowS;
  quietAudioS_ = 0;
  nWin_ = 0;
  for (int c = 0; c < ATM_N_AUDIO_CLASSES; ++c) pSum_[c] = 0.0f;
  pDispMax_ = pAbnMax_ = 0.0f;
  nDispWin_ = nAbnWin_ = 0;
  vib_.reset();
  pir_.reset();
}

void EventInterpreter::fillFusion(float* f) const {
  const float n = nWin_ ? (float)nWin_ : 1.0f;
  // order must match FUSION_FEATURES in train_esp32_model.py
  f[0] = nWin_ ? pSum_[AUDIO_BACKGROUND] / n : 1.0f;
  f[1] = pSum_[AUDIO_DISPENSE] / n;
  f[2] = pSum_[AUDIO_ABNORMAL] / n;
  f[3] = pDispMax_;
  f[4] = pAbnMax_;
  f[5] = nDispWin_ / n;
  f[6] = nAbnWin_ / n;
  f[7] = vib_.rms();
  f[8] = vib_.peak();
  f[9] = vib_.variance();
  f[10] = pir_.activeFraction();
  f[11] = pir_.interactionSeconds();
  f[12] = tamperSeen_ ? 1.0f : 0.0f;
}

void EventInterpreter::decide(const float* f, EventClass* cls, DecisionRule* rule, float* conf) {
  // 1. Safety priority: the existing tamper decision / threshold always wins.
  if (f[12] >= 0.5f || f[8] >= ATM_TAMPER_VIB_THRESHOLD) {
    *cls = EV_SUSPECTED_TAMPER;
    *rule = RULE_TAMPER_PRIORITY;
    *conf = 1.0f;
    return;
  }
  // 2. Fusion tree.
  float p[ATM_N_FUSION_CLASSES];
  treePredict(ATM_FT_FEATURE, ATM_FT_THRESHOLD, ATM_FT_LEFT, ATM_FT_RIGHT, ATM_FT_LEAF_PROB,
              ATM_FT_ROOTS, ATM_FT_N_TREES, ATM_N_FUSION_CLASSES, f, p);
  int best = 0;
  for (int c = 1; c < ATM_N_FUSION_CLASSES; ++c)
    if (p[c] > p[best]) best = c;
  *cls = (EventClass)best;
  *conf = p[best];
  *rule = RULE_MODEL;
  // 3. Uncertain or contradictory evidence -> UNKNOWN (never guessed into a class).
  if (*conf < kMinConfidence) {
    *cls = EV_UNKNOWN;
    *rule = RULE_LOW_CONFIDENCE;
  } else if (*cls == EV_NORMAL_DISPENSE && f[10] < 0.05f) {
    *cls = EV_UNKNOWN;
    *rule = RULE_DISPENSE_NO_PRESENCE;
  }
  // A tree-only SUSPECTED_TAMPER (threshold not reached) is kept: safety-biased.
}

void EventInterpreter::forceEvent(uint32_t nowS, bool dispensed) {
  if (!open_) openEvent(nowS, true);
  EventSummary e;
  memset(&e, 0, sizeof(e));
  e.startS = startS_;
  e.durationS = lastS_ - startS_ + 1;
  e.final = true;
  fillFusion(e.fusion);
  if (e.fusion[12] >= 0.5f || e.fusion[8] >= ATM_TAMPER_VIB_THRESHOLD) {
    e.cls = EV_SUSPECTED_TAMPER;
    e.rule = RULE_TAMPER_PRIORITY;
  } else {
    e.cls = dispensed ? EV_NORMAL_DISPENSE : EV_ABNORMAL_MECHANICAL;
    e.rule = RULE_CONFIRMED;
  }
  e.confidence = 1.0f;
  e.countedDispense = (e.cls == EV_NORMAL_DISPENSE);
  window_.add(e);
  enqueue(e);
  openEvent(nowS, true);
  absorbing_ = true;
}

void EventInterpreter::enqueue(const EventSummary& e) {
  if (qCount_ == kQueue) {  // drop oldest rather than block the sensor loop
    qHead_ = (qHead_ + 1) % kQueue;
    --qCount_;
  }
  q_[(qHead_ + qCount_) % kQueue] = e;
  ++qCount_;
}

bool EventInterpreter::pop(EventSummary* out) {
  if (!qCount_) return false;
  *out = q_[qHead_];
  qHead_ = (qHead_ + 1) % kQueue;
  --qCount_;
  return true;
}

void EventInterpreter::closeEvent(uint32_t nowS) {
  EventSummary e;
  memset(&e, 0, sizeof(e));
  e.startS = startS_;
  e.durationS = lastS_ - startS_ + 1;
  e.final = true;
  fillFusion(e.fusion);
  decide(e.fusion, &e.cls, &e.rule, &e.confidence);
  // The cashflow counter only counts a confirmed, tamper-free dispense.
  e.countedDispense = (e.cls == EV_NORMAL_DISPENSE) && !tamperSeen_;
  open_ = false;
  (void)nowS;
  window_.add(e);
  enqueue(e);
}

void EventInterpreter::tick(uint32_t nowS, float vibCount, bool pirHigh, const AudioResult* audio,
                            bool cascadeTamper) {
  const bool tamperNow = cascadeTamper || vibCount >= ATM_TAMPER_VIB_THRESHOLD;
  const bool audioActive = audio && (1.0f - audio->prob[AUDIO_BACKGROUND]) >= kAudioActiveProb;
  const bool trigger = pirHigh || audioActive || tamperNow;

  if (!open_) openEvent(nowS, trigger);

  if (!interaction_ && trigger) {
    // idle -> interaction: flush a meaningful idle stretch, drop a tiny one
    if (lastS_ - startS_ + 1 >= 10) closeEvent(nowS);
    openEvent(nowS, true);
  }

  // accumulate this second
  lastS_ = nowS;
  vib_.add(vibCount);
  pir_.add(pirHigh);
  if (tamperNow) tamperSeen_ = true;
  if (audio) {
    ++nWin_;
    for (int c = 0; c < ATM_N_AUDIO_CLASSES; ++c) pSum_[c] += audio->prob[c];
    if (audio->prob[AUDIO_DISPENSE] > pDispMax_) pDispMax_ = audio->prob[AUDIO_DISPENSE];
    if (audio->prob[AUDIO_ABNORMAL] > pAbnMax_) pAbnMax_ = audio->prob[AUDIO_ABNORMAL];
    if (audio->cls == AUDIO_DISPENSE) ++nDispWin_;
    if (audio->cls == AUDIO_ABNORMAL) ++nAbnWin_;
    quietAudioS_ = audioActive ? 0 : quietAudioS_ + 1;
  } else {
    ++quietAudioS_;
  }

  // Safety priority: raise the tamper alert immediately, don't wait for the event to end.
  if (tamperNow && !earlyAlertSent_) {
    EventSummary a;
    memset(&a, 0, sizeof(a));
    a.startS = startS_;
    a.durationS = nowS - startS_ + 1;
    a.cls = EV_SUSPECTED_TAMPER;
    a.rule = RULE_TAMPER_EARLY_ALERT;
    a.confidence = 1.0f;
    a.final = false;
    fillFusion(a.fusion);
    enqueue(a);
    earlyAlertSent_ = true;
  }

  const uint32_t dur = nowS - startS_ + 1;
  if (!interaction_) {
    if (dur >= kIdleHeartbeatS) closeEvent(nowS);
    return;
  }
  const bool pirQuiet = pir_.secondsSinceActive() > (uint32_t)ATM_PIR_GAP_TOLERANCE_S;
  const bool audioQuiet = quietAudioS_ >= kAudioQuietToCloseS;
  if ((pirQuiet && audioQuiet) || dur >= kMaxEventS) {
    if (absorbing_ && !tamperSeen_) {  // rest of a forced visit: already reported
      absorbing_ = false;
      openEvent(nowS, false);
      return;
    }
    absorbing_ = false;
    // a lone PIR blip with no sound and no vibration is not a visit
    const bool spurious = pir_.interactionSeconds() < 3.0f && nDispWin_ == 0 && nAbnWin_ == 0 &&
                          !tamperSeen_ && dur < kMaxEventS;
    if (spurious) {
      openEvent(nowS, false);  // back to idle accumulation
    } else {
      closeEvent(nowS);
    }
  }
}

// =====================================================================================
// JSON + self test
// =====================================================================================
int formatEventJson(const EventSummary& e, const RollingStats& s, const char* atmId, char* buf,
                    size_t len) {
  const float* f = e.fusion;
  return snprintf(
      buf, len,
      "{\"atm\":\"%s\",\"t\":%lu,\"dur\":%lu,\"ev\":\"%s\",\"final\":%d,\"conf\":%.2f,"
      "\"rule\":\"%s\",\"audio\":{\"pd\":%.2f,\"pa\":%.2f,\"pdmax\":%.2f,\"pamax\":%.2f,"
      "\"df\":%.2f,\"af\":%.2f},\"vib\":{\"rms\":%.1f,\"pk\":%.0f,\"var\":%.1f},"
      "\"pir\":{\"frac\":%.2f,\"int\":%.0f},\"tamper\":%d,\"dispensed\":%d,"
      "\"roll\":{\"disp\":%lu,\"sv10m\":%u,\"rsv\":%d,\"nd\":%.1f,\"lastDisp\":%ld},\"mv\":\"%s\"}",
      atmId, (unsigned long)e.startS, (unsigned long)e.durationS, eventClassName(e.cls),
      e.final ? 1 : 0, e.confidence, ruleName(e.rule), f[1], f[2], f[3], f[4], f[5], f[6], f[7],
      f[8], f[9], f[10], f[11], f[12] >= 0.5f ? 1 : 0, e.countedDispense ? 1 : 0,
      (unsigned long)s.dispenseCount, (unsigned)s.shortVisitsRecent, s.repeatedShortVisits ? 1 : 0,
      s.noDispenseVisitScore, (long)s.lastDispenseAgeS, ATM_MODEL_VERSION);
}

int selfTestVectorCount() { return ATM_N_TEST_VECTORS; }

int selfTest(AudioClassifier& clf, float* maxAbsErr) {
  int ok = 0;
  float worst = 0.0f;
  for (int v = 0; v < ATM_N_TEST_VECTORS; ++v) {
    AudioResult r;
    if (!clf.classify(&ATM_TV_FEATURES[v * ATM_N_AUDIO_FEATURES], &r)) continue;
    const float* exp = &ATM_TV_EXPECTED_PROB[v * ATM_N_AUDIO_CLASSES];
    int eb = 0;
    for (int c = 0; c < ATM_N_AUDIO_CLASSES; ++c) {
      const float d = fabsf(r.prob[c] - exp[c]);
      if (d > worst) worst = d;
      if (exp[c] > exp[eb]) eb = c;
    }
    if (eb == r.cls) ++ok;
  }
  if (maxAbsErr) *maxAbsErr = worst;
  return ok;
}

}  // namespace atm
