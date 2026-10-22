// ATM Sentinel - ESP32 streaming feature extraction. See esp32_preprocessing.h.
#include "esp32_preprocessing.h"

#include <math.h>
#include <string.h>

namespace atm {

namespace {
constexpr int kN = ATM_N_FFT;
constexpr int kBins = ATM_N_FFT / 2 + 1;
constexpr int kLog2N = 9;  // 512
static_assert(ATM_N_FFT == 512, "fft512() is written for a 512-point FFT");
static_assert(ATM_FRAME == ATM_N_FFT, "frame length must equal FFT size");
static_assert(ATM_N_FRAME_FEATURES == ATM_N_MFCC + 5, "frame feature layout changed");
static_assert(ATM_N_AUDIO_FEATURES == 2 * ATM_N_FRAME_FEATURES, "window feature layout changed");

constexpr float kRolloffFrac = 0.85f;
constexpr double kEps = 1e-12;

inline double htkMel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
inline double htkMelInv(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }
}  // namespace

void AudioFeatureExtractor::begin() {
  // periodic Hann
  for (int n = 0; n < ATM_FRAME; ++n)
    hann_[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / ATM_FRAME));
  // FFT twiddles + bit reversal
  for (int k = 0; k < kN / 2; ++k) {
    cosTab_[k] = (float)cos(2.0 * M_PI * k / kN);
    sinTab_[k] = (float)sin(2.0 * M_PI * k / kN);
  }
  for (int i = 0; i < kN; ++i) {
    unsigned r = 0;
    for (int b = 0; b < kLog2N; ++b) r |= ((i >> b) & 1u) << (kLog2N - 1 - b);
    bitrev_[i] = (uint16_t)r;
  }
  // HTK mel filterbank (unnormalised triangles), stored sparsely
  double hz[ATM_N_MEL + 2];
  const double mLo = htkMel(ATM_FMIN), mHi = htkMel(ATM_FMAX);
  for (int i = 0; i < ATM_N_MEL + 2; ++i)
    hz[i] = htkMelInv(mLo + (mHi - mLo) * i / (double)(ATM_N_MEL + 1));
  int off = 0;
  for (int m = 0; m < ATM_N_MEL; ++m) {
    const double lo = hz[m], ce = hz[m + 1], hi = hz[m + 2];
    int first = -1, last = -1;
    for (int k = 0; k < kBins; ++k) {
      const double f = k * (double)ATM_SR / kN;
      double w = fmin((f - lo) / (ce - lo), (hi - f) / (hi - ce));
      if (w > 0.0) {
        if (first < 0) first = k;
        last = k;
      }
    }
    if (first < 0) first = last = 0;
    melStart_[m] = (uint16_t)first;
    melLen_[m] = (uint16_t)(last - first + 1);
    melOff_[m] = (uint16_t)off;
    for (int k = first; k <= last; ++k) {
      const double f = k * (double)ATM_SR / kN;
      double w = fmin((f - lo) / (ce - lo), (hi - f) / (hi - ce));
      melW_[off++] = (float)(w > 0.0 ? w : 0.0);
    }
  }
  // orthonormal DCT-II
  for (int k = 0; k < ATM_N_MFCC; ++k) {
    const double s = (k == 0) ? sqrt(1.0 / ATM_N_MEL) : sqrt(2.0 / ATM_N_MEL);
    for (int n = 0; n < ATM_N_MEL; ++n)
      dct_[k * ATM_N_MEL + n] = (float)(s * cos(M_PI * k * (n + 0.5) / ATM_N_MEL));
  }
  reset();
}

void AudioFeatureExtractor::reset() {
  frameFill_ = 0;
  framesInWindow_ = 0;
  memset(mean_, 0, sizeof(mean_));
  memset(m2_, 0, sizeof(m2_));
}

// In-place iterative radix-2 complex FFT on re_/im_ (512 points).
void AudioFeatureExtractor::fft512() {
  for (int i = 0; i < kN; ++i) {
    const int j = bitrev_[i];
    if (j > i) {
      float t = re_[i]; re_[i] = re_[j]; re_[j] = t;
      t = im_[i]; im_[i] = im_[j]; im_[j] = t;
    }
  }
  for (int len = 2; len <= kN; len <<= 1) {
    const int half = len >> 1;
    const int step = kN / len;
    for (int i = 0; i < kN; i += len) {
      for (int k = 0; k < half; ++k) {
        const float wr = cosTab_[k * step];
        const float wi = -sinTab_[k * step];  // e^{-j 2 pi k / len}
        const int a = i + k, b = a + half;
        const float xr = re_[b] * wr - im_[b] * wi;
        const float xi = re_[b] * wi + im_[b] * wr;
        re_[b] = re_[a] - xr;
        im_[b] = im_[a] - xi;
        re_[a] += xr;
        im_[a] += xi;
      }
    }
  }
}

void AudioFeatureExtractor::computeFrame(const float* frame, float out[ATM_N_FRAME_FEATURES]) {
  // Hot loops are single-precision on purpose: the ESP32 (LX6) FPU has no double support,
  // so double maths here would be software-emulated. Parity vs float64 Python is ~1e-5.
  // DC removal
  float mean = 0.0f;
  for (int n = 0; n < ATM_FRAME; ++n) mean += frame[n];
  mean /= (float)ATM_FRAME;
  float sq = 0.0f;
  int zc = 0;
  bool prevPos = false;
  for (int n = 0; n < ATM_FRAME; ++n) {
    const float v = frame[n] - mean;
    sq += v * v;
    const bool pos = v >= 0.0f;
    if (n > 0 && pos != prevPos) ++zc;
    prevPos = pos;
    re_[n] = v * hann_[n];
    im_[n] = 0.0f;
  }
  const float rms = sqrtf(sq / (float)ATM_FRAME);
  const float rmsDb = 20.0f * log10f(rms + 1e-9f);
  const float zcr = (float)zc / (float)(ATM_FRAME - 1);

  fft512();

  float tot = 0.0f, wsum = 0.0f, psum = 0.0f, lsum = 0.0f;
  const float invBins = 1.0f / (float)(kBins - 1);
  for (int k = 0; k < kBins; ++k) {
    const float p = (re_[k] * re_[k] + im_[k] * im_[k]) * (1.0f / (float)kN);
    pow_[k] = p;
    tot += p;
    wsum += p * (float)k * invBins;
    if (k > 0) {
      psum += p;
      lsum += logf(p + ATM_LOG_FLOOR);
    }
  }
  const float centroid = wsum / (tot + (float)kEps);
  int roll = 0;
  {
    const float target = kRolloffFrac * tot;
    float c = 0.0f;
    for (int k = 0; k < kBins; ++k) {
      c += pow_[k];
      if (c >= target) { roll = k; break; }
    }
  }
  const float rolloff = (float)roll / (float)(kBins - 1);
  const int nk = kBins - 1;
  const float flat = expf(lsum / (float)nk) / (psum / (float)nk + ATM_LOG_FLOOR);

  float logmel[ATM_N_MEL];
  for (int m = 0; m < ATM_N_MEL; ++m) {
    float e = 0.0f;
    const float* w = &melW_[melOff_[m]];
    const float* p = &pow_[melStart_[m]];
    for (int i = 0; i < melLen_[m]; ++i) e += w[i] * p[i];
    logmel[m] = logf(e > ATM_LOG_FLOOR ? e : ATM_LOG_FLOOR);
  }
  for (int k = 0; k < ATM_N_MFCC; ++k) {
    float c = 0.0f;
    const float* d = &dct_[k * ATM_N_MEL];
    for (int m = 0; m < ATM_N_MEL; ++m) c += d[m] * logmel[m];
    out[k] = c;
  }
  out[ATM_N_MFCC + 0] = rmsDb;
  out[ATM_N_MFCC + 1] = zcr;
  out[ATM_N_MFCC + 2] = centroid;
  out[ATM_N_MFCC + 3] = rolloff;
  out[ATM_N_MFCC + 4] = flat;
}

// Welford running mean / M2 per frame feature (double: avoids float cancellation in std).
void AudioFeatureExtractor::accumulate(const float* f) {
  ++framesInWindow_;
  const double n = framesInWindow_;
  for (int i = 0; i < ATM_N_FRAME_FEATURES; ++i) {
    const double d = f[i] - mean_[i];
    mean_[i] += d / n;
    m2_[i] += d * (f[i] - mean_[i]);
  }
}

void AudioFeatureExtractor::finishWindow() {
  for (int i = 0; i < ATM_N_FRAME_FEATURES; ++i) {
    feat_[i] = (float)mean_[i];
    const double var = m2_[i] / framesInWindow_;  // population variance (numpy std default)
    feat_[ATM_N_FRAME_FEATURES + i] = (float)sqrt(var > 0.0 ? var : 0.0);
  }
  framesInWindow_ = 0;
  memset(mean_, 0, sizeof(mean_));
  memset(m2_, 0, sizeof(m2_));
}

size_t AudioFeatureExtractor::push(const float* x, size_t n, bool* windowReady) {
  *windowReady = false;
  size_t used = 0;
  while (used < n) {
    size_t take = ATM_FRAME - frameFill_;
    if (take > n - used) take = n - used;
    memcpy(&frameBuf_[frameFill_], &x[used], take * sizeof(float));
    frameFill_ += take;
    used += take;
    if (frameFill_ == ATM_FRAME) {
      float f[ATM_N_FRAME_FEATURES];
      computeFrame(frameBuf_, f);
      accumulate(f);
      frameFill_ = 0;
      if (framesInWindow_ == ATM_FRAMES_PER_WINDOW) {
        finishWindow();
        *windowReady = true;
        return used;
      }
    }
  }
  return used;
}

// ---------------------------------------------------------------- vibration / PIR
void VibrationAccumulator::reset() {
  n_ = 0;
  sum_ = sumSq_ = 0.0;
  peak_ = 0.0f;
}

void VibrationAccumulator::add(float v) {
  if (n_ == 0 || v > peak_) peak_ = v;
  ++n_;
  sum_ += v;
  sumSq_ += (double)v * v;
}

float VibrationAccumulator::rms() const { return n_ ? (float)sqrt(sumSq_ / n_) : 0.0f; }

float VibrationAccumulator::variance() const {
  if (!n_) return 0.0f;
  const double m = sum_ / n_;
  const double v = sumSq_ / n_ - m * m;
  return (float)(v > 0.0 ? v : 0.0);
}

void PirTracker::reset() {
  n_ = active_ = 0;
  best_ = cur_ = 0.0f;
  gap_ = 1000000;
  everActive_ = false;
}

void PirTracker::add(bool high) {
  ++n_;
  if (high) {
    ++active_;
    if (everActive_ && gap_ <= (uint32_t)ATM_PIR_GAP_TOLERANCE_S)
      cur_ += (float)gap_ + 1.0f;
    else
      cur_ = 1.0f;
    everActive_ = true;
    gap_ = 0;
    if (cur_ > best_) best_ = cur_;
  } else {
    ++gap_;
  }
}

}  // namespace atm
