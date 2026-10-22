// ATM Sentinel - ESP32 streaming feature extraction (audio, vibration, PIR).
//
// Mirrors FrameDSP / window_compact / pir_interaction_s in train_esp32_model.py exactly:
// same frame size, window, FFT, mel formula, log floor, DCT, feature order and aggregation.
// If you change anything here, change the Python reference and retrain.
//
// Memory: one AudioFeatureExtractor is ~17 KB of static RAM (tables + one frame of work
// buffers). No dynamic allocation, no raw-audio history, no spectrogram storage.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "atm_model_data.h"

namespace atm {

// INMP441: 24-bit sample left-justified in a 32-bit I2S slot.
inline float inmp441ToFloat(int32_t raw) {
  return (float)(raw >> 8) * (1.0f / 8388608.0f);
}

class AudioFeatureExtractor {
 public:
  // Builds window/FFT/mel/DCT tables. Call once at boot.
  void begin();
  // Drops any partial frame/window (e.g. after an I2S restart).
  void reset();

  // Streams samples (float, nominal range [-1, 1)) into the extractor.
  // Consumes samples until either all n are used or a 1 s window completes.
  // Returns the number of samples consumed; *windowReady is set when a window completed,
  // in which case features() holds the 36 raw (un-normalised) window features. Call again
  // with the remaining samples.
  size_t push(const float* x, size_t n, bool* windowReady);

  const float* features() const { return feat_; }
  float windowRmsDbMean() const { return feat_[ATM_N_MFCC]; }  // rms_db_mean

  // Per-frame features of one FRAME-length buffer (exposed for testing).
  void computeFrame(const float* frame, float out[ATM_N_FRAME_FEATURES]);

 private:
  void fft512();
  void accumulate(const float* f);
  void finishWindow();

  // streaming state
  float frameBuf_[ATM_FRAME];
  size_t frameFill_ = 0;
  int framesInWindow_ = 0;
  double mean_[ATM_N_FRAME_FEATURES];
  double m2_[ATM_N_FRAME_FEATURES];
  float feat_[ATM_N_AUDIO_FEATURES];

  // tables
  float hann_[ATM_FRAME];
  float cosTab_[ATM_N_FFT / 2];
  float sinTab_[ATM_N_FFT / 2];
  uint16_t bitrev_[ATM_N_FFT];
  uint16_t melStart_[ATM_N_MEL];
  uint16_t melLen_[ATM_N_MEL];
  uint16_t melOff_[ATM_N_MEL];
  float melW_[2 * (ATM_N_FFT / 2 + 1)];
  float dct_[ATM_N_MFCC * ATM_N_MEL];

  // per-frame work buffers
  float re_[ATM_N_FFT];
  float im_[ATM_N_FFT];
  float pow_[ATM_N_FFT / 2 + 1];
};

// SW-420 vibration statistics over an event. One sample = edge count in a 1 s bin
// (the same unit as the existing tamper rule threshold, ATM_TAMPER_VIB_THRESHOLD).
class VibrationAccumulator {
 public:
  void reset();
  void add(float countPerSecond);
  float rms() const;
  float peak() const { return n_ ? peak_ : 0.0f; }
  float variance() const;  // population variance
  uint32_t samples() const { return n_; }

 private:
  uint32_t n_ = 0;
  double sum_ = 0, sumSq_ = 0;
  float peak_ = 0;
};

// HC-SR501 PIR statistics over an event, one sample per second.
// interactionSeconds() = longest active span, bridging gaps <= ATM_PIR_GAP_TOLERANCE_S
// (identical to pir_interaction_s() in the training script).
class PirTracker {
 public:
  void reset();
  void add(bool high);
  float activeFraction() const { return n_ ? (float)active_ / (float)n_ : 0.0f; }
  float interactionSeconds() const { return best_; }
  uint32_t secondsSinceActive() const { return gap_; }
  bool everActive() const { return everActive_; }

 private:
  uint32_t n_ = 0, active_ = 0;
  float best_ = 0, cur_ = 0;
  uint32_t gap_ = 1000000;
  bool everActive_ = false;
};

}  // namespace atm
