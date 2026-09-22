// ATM Sentinel — window feature extraction + scaler. Pure C++, no Arduino deps,
// so it compiles on the host for checking against atmsim/features.py.
//
// Mirrors features.py exactly: population std, closed-form least-squares slope
// in units/min, delta = last - first, linear null interpolation edge-filled,
// all-null window -> zeros, non-finite feature -> 0.
//
// FIXED CONTRACT: changing anything here invalidates the trained models.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

static const uint16_t WINDOW_S        = 300;
static const uint8_t  N_FEATURES      = 57;
static const float    NORMALIZED_CLIP = 8.0f;
static const float    SAG_V = 207.0f, SWELL_V = 253.0f, DROPOUT_V = 165.0f;
static const double   SLOPE_DENOM = 2249975.0;  // sum((t - 149.5)^2), t = 0..299

enum Chan { CH_T, CH_RH, CH_P, CH_LUX, CH_V, CH_I, CH_VIB, CH_MOT, N_CHAN };
static const uint8_t N_FLOAT_CHAN = 6;
static const char* CHAN_NAMES[N_CHAN] = {"temperature", "humidity", "pressure", "light", "voltage", "current", "vibration", "motion"};
// schema.yaml range / resolution for the six nullable float channels
static const float CH_LO[N_FLOAT_CHAN]  = {-40.0f, 0.0f,   300.0f,  0.0f,      0.0f,   -3.2f};
static const float CH_HI[N_FLOAT_CHAN]  = { 85.0f, 100.0f, 1100.0f, 120000.0f, 300.0f,  3.2f};
static const float CH_RES[N_FLOAT_CHAN] = { 0.01f, 0.01f,  0.01f,   0.1f,      0.1f,    0.0008f};

// data/manifest.json feature_columns — this order is the model input order.
static const char* FEATURE_NAMES[N_FEATURES] = {
  "temperature_null_frac", "temperature_mean", "temperature_std", "temperature_min", "temperature_max", "temperature_slope", "temperature_delta",
  "humidity_null_frac", "humidity_mean", "humidity_std", "humidity_min", "humidity_max", "humidity_slope", "humidity_delta",
  "pressure_null_frac", "pressure_mean", "pressure_std", "pressure_min", "pressure_max", "pressure_slope", "pressure_delta",
  "light_null_frac", "light_mean", "light_std", "light_min", "light_max", "light_slope", "light_delta",
  "voltage_null_frac", "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "voltage_slope", "voltage_delta",
  "current_null_frac", "current_mean", "current_std", "current_min", "current_max", "current_slope", "current_delta",
  "voltage_sag_frac", "voltage_swell_frac", "voltage_dropout_frac",
  "vibration_sum", "vibration_max", "vibration_active_frac", "vibration_burst_max_s",
  "motion_frac", "motion_transitions", "motion_longest_run_s",
  "abs_humidity_mean", "abs_humidity_std", "abs_humidity_slope", "dewpoint_mean", "temp_rh_corr",
};

// data/scaler.json — fitted on train only. Generated, do not hand-edit.
static const float FEATURE_MEAN[N_FEATURES] = {
  0.0002971014492753623f,  // temperature_null_frac
  30.32781929951691f,  // temperature_mean
  0.09353448956551037f,  // temperature_std
  30.14292028985507f,  // temperature_min
  30.51271739130435f,  // temperature_max
  0.018907433348293246f,  // temperature_slope
  0.09357246376811597f,  // temperature_delta
  5.845410628019324e-05f,  // humidity_null_frac
  55.38085407487923f,  // humidity_mean
  0.3302763522984206f,  // humidity_std
  54.75190869565217f,  // humidity_min
  56.00973333333334f,  // humidity_max
  -0.004725689367563021f,  // humidity_slope
  -0.024124637681159435f,  // humidity_delta
  0.0005946859903381643f,  // pressure_null_frac
  1007.885724705314f,  // pressure_mean
  0.060693572913890916f,  // pressure_std
  1007.7429884057971f,  // pressure_min
  1008.0287217391306f,  // pressure_max
  -0.0009138378252511413f,  // pressure_slope
  -0.0053369565217375224f,  // pressure_delta
  2.8019323671497584e-05f,  // light_null_frac
  246.5543938888889f,  // light_mean
  5.165162132511819f,  // light_std
  235.64727536231882f,  // light_min
  254.52308695652178f,  // light_max
  -0.005551413663050374f,  // light_slope
  -0.0660289855072466f,  // light_delta
  6.231884057971015e-05f,  // voltage_null_frac
  225.21015903381644f,  // voltage_mean
  1.655736063377891f,  // voltage_std
  219.60647826086955f,  // voltage_min
  228.85784057971017f,  // voltage_max
  0.0024021051441634298f,  // voltage_slope
  0.052623188405797286f,  // voltage_delta
  1.835748792270531e-05f,  // current_null_frac
  1.4039969659903384f,  // current_mean
  0.007664764285945187f,  // current_std
  1.3820064927536233f,  // current_min
  1.4278266666666668f,  // current_max
  0.0004007619075960747f,  // current_slope
  0.0016100869565217394f,  // current_delta
  0.06639082125603865f,  // voltage_sag_frac
  0.03759371980676328f,  // voltage_swell_frac
  0.019478743961352657f,  // voltage_dropout_frac
  1648.1450724637682f,  // vibration_sum
  15.621884057971014f,  // vibration_max
  0.11655265700483092f,  // vibration_active_frac
  25.367101449275363f,  // vibration_burst_max_s
  0.24070048309178746f,  // motion_frac
  1.122608695652174f,  // motion_transitions
  58.633043478260866f,  // motion_longest_run_s
  14.704274430423007f,  // abs_humidity_mean
  0.050810328912604806f,  // abs_humidity_std
  0.009190011186862367f,  // abs_humidity_slope
  19.711705242684044f,  // dewpoint_mean
  -0.7424594189668302f,  // temp_rh_corr
};
static const float FEATURE_STD[N_FEATURES] = {
  0.003842340768423635f,  // temperature_null_frac
  3.859379269215118f,  // temperature_mean
  0.07357684366015967f,  // temperature_std
  3.8350802885641455f,  // temperature_min
  3.8834199470762516f,  // temperature_max
  0.07664404843146255f,  // temperature_slope
  0.3724932133937001f,  // temperature_delta
  0.0006228061217466241f,  // humidity_null_frac
  15.380462826173982f,  // humidity_mean
  0.3037855221884583f,  // humidity_std
  15.27099890131086f,  // humidity_min
  15.487807346188712f,  // humidity_max
  0.29681294778355705f,  // humidity_slope
  1.4534047688605567f,  // humidity_delta
  0.006070888596198474f,  // pressure_null_frac
  1.2900817187516118f,  // pressure_mean
  0.023303033958013447f,  // pressure_std
  1.2918398731408594f,  // pressure_min
  1.2904368952187868f,  // pressure_max
  0.03329876315754861f,  // pressure_slope
  0.156253320695652f,  // pressure_delta
  0.00030432328710213824f,  // light_null_frac
  96.62889702496292f,  // light_mean
  4.838631203338721f,  // light_std
  94.91497586419642f,  // light_min
  97.99776036500863f,  // light_max
  2.487299060675415f,  // light_slope
  12.523583764961227f,  // light_delta
  0.0005959988177607795f,  // voltage_null_frac
  32.898001820350146f,  // voltage_mean
  3.264749724385367f,  // voltage_std
  34.15232438547466f,  // voltage_min
  33.41146229123103f,  // voltage_max
  0.9889438500189808f,  // voltage_slope
  7.584144239366777f,  // voltage_delta
  0.00024668731026582106f,  // current_null_frac
  0.26512826367128156f,  // current_mean
  0.012229750445316358f,  // current_std
  0.2628322118295919f,  // current_min
  0.26912420908379064f,  // current_max
  0.0047700570493768345f,  // current_slope
  0.036707811311719296f,  // current_delta
  0.23899806879633032f,  // voltage_sag_frac
  0.18491420866454322f,  // voltage_swell_frac
  0.13750190318940683f,  // voltage_dropout_frac
  4407.988463841503f,  // vibration_sum
  33.80930001441642f,  // vibration_max
  0.2478305675706871f,  // vibration_active_frac
  57.470490388645956f,  // vibration_burst_max_s
  0.2894632638921282f,  // motion_frac
  1.2706424851789653f,  // motion_transitions
  69.73386142949259f,  // motion_longest_run_s
  3.0202310739542537f,  // abs_humidity_mean
  0.06594393073481891f,  // abs_humidity_std
  0.05246659591808406f,  // abs_humidity_slope
  2.830791616192814f,  // dewpoint_mean
  0.3769862598851921f,  // temp_rh_corr
};

// ----------------------------------------------------------------------------

// Linear interpolation across NaN, edge-filled; all-NaN -> zeros. In place.
static void fillNulls(float* a) {
  int first = -1, last = -1;
  for (int i = 0; i < WINDOW_S; i++) if (!isnan(a[i])) { if (first < 0) first = i; last = i; }
  if (first < 0) { for (int i = 0; i < WINDOW_S; i++) a[i] = 0.0f; return; }
  for (int i = 0; i < first; i++) a[i] = a[first];
  for (int i = last + 1; i < WINDOW_S; i++) a[i] = a[last];
  int prev = first;
  for (int i = first + 1; i <= last; i++) {
    if (isnan(a[i])) continue;
    for (int j = prev + 1; j < i; j++) a[j] = a[prev] + (a[i] - a[prev]) * (float)(j - prev) / (float)(i - prev);
    prev = i;
  }
}

// out[0..5] = mean, std, min, max, slope (/min), delta
static void basicStats(const float* a, float* out) {
  double sum = 0, sxy = 0; float mn = a[0], mx = a[0];
  for (int i = 0; i < WINDOW_S; i++) {
    sum += a[i]; sxy += (double)a[i] * (i - 149.5);
    if (a[i] < mn) mn = a[i];
    if (a[i] > mx) mx = a[i];
  }
  float mean = sum / WINDOW_S;
  double ss = 0;
  for (int i = 0; i < WINDOW_S; i++) { double d = a[i] - mean; ss += d * d; }
  out[0] = mean; out[1] = sqrt(ss / WINDOW_S); out[2] = mn; out[3] = mx;
  out[4] = sxy / SLOPE_DENOM * 60.0; out[5] = a[WINDOW_S - 1] - a[0];
}

static uint16_t longestRun(const float* a, bool (*pred)(float)) {
  uint16_t run = 0, best = 0;
  for (int i = 0; i < WINDOW_S; i++) { run = pred(a[i]) ? run + 1 : 0; if (run > best) best = run; }
  return best;
}
static bool isNull(float v)   { return isnan(v); }
static bool isActive(float v) { return v > 0.0f; }
static bool isOn(float v)     { return v > 0.5f; }

// atmsim/psychro.py: Magnus-Tetens (Alduchov & Eskridge 1996)
static float satVapourPressure(float t) { return 6.1094f * expf(17.625f * t / (243.04f + t)); }
static float mixingRatio(float t, float rh, float p) {
  float e = rh / 100.0f * satVapourPressure(t);
  float d = p - e; if (d < 1e-6f) d = 1e-6f;
  return 621.97f * e / d;
}
static float dewpoint(float t, float rh) {
  rh = fminf(fmaxf(rh, 0.1f), 100.0f);
  float g = logf(rh / 100.0f) + 17.625f * t / (243.04f + t);
  return 243.04f * g / (17.625f - g);
}

// chan[c] = WINDOW_S samples of channel c in time order, NaN = null (chan is
// NOT modified). Fills feat[N_FEATURES]. Returns a bitmask of float channels
// that failed a deterministic check (SENSOR_VALIDATION.md): a null run of at
// least missingRunS samples, or every non-null sample equal to one
// resolution-quantised value (SENSOR_STUCK).
static uint8_t extractFeatures(const float* const chan[N_CHAN], float* feat, uint16_t missingRunS) {
  static float scratch[WINDOW_S], fillT[WINDOW_S], fillRH[WINDOW_S], fillP[WINDOW_S];
  uint8_t faultMask = 0;
  float st[6];

  for (uint8_t ch = 0; ch < N_FLOAT_CHAN; ch++) {
    memcpy(scratch, chan[ch], sizeof(scratch));

    uint16_t nulls = 0, valid = 0; bool stuck = true; long q0 = 0;
    for (int i = 0; i < WINDOW_S; i++) {
      if (isnan(scratch[i])) { nulls++; continue; }
      long q = lroundf(scratch[i] / CH_RES[ch]);
      if (valid == 0) q0 = q; else if (q != q0) stuck = false;
      valid++;
    }
    if (longestRun(scratch, isNull) >= missingRunS) faultMask |= (1 << ch);
    if (valid > 0 && stuck)                         faultMask |= (1 << ch);

    fillNulls(scratch);
    basicStats(scratch, st);
    feat[ch * 7] = (float)nulls / WINDOW_S;
    memcpy(&feat[ch * 7 + 1], st, sizeof(st));

    if (ch == CH_T)  memcpy(fillT,  scratch, sizeof(fillT));
    if (ch == CH_RH) memcpy(fillRH, scratch, sizeof(fillRH));
    if (ch == CH_P)  memcpy(fillP,  scratch, sizeof(fillP));
    if (ch == CH_V) {
      uint16_t sag = 0, swell = 0, drop = 0;
      for (int i = 0; i < WINDOW_S; i++) { sag += scratch[i] < SAG_V; swell += scratch[i] > SWELL_V; drop += scratch[i] < DROPOUT_V; }
      feat[42] = (float)sag / WINDOW_S; feat[43] = (float)swell / WINDOW_S; feat[44] = (float)drop / WINDOW_S;
    }
  }

  memcpy(scratch, chan[CH_VIB], sizeof(scratch)); fillNulls(scratch);
  { float sum = 0, mx = 0; uint16_t active = 0;
    for (int i = 0; i < WINDOW_S; i++) { sum += scratch[i]; if (scratch[i] > mx) mx = scratch[i]; active += scratch[i] > 0; }
    feat[45] = sum; feat[46] = mx; feat[47] = (float)active / WINDOW_S; feat[48] = longestRun(scratch, isActive); }

  memcpy(scratch, chan[CH_MOT], sizeof(scratch)); fillNulls(scratch);
  { uint16_t on = 0, trans = 0;
    for (int i = 0; i < WINDOW_S; i++) { bool b = scratch[i] > 0.5f; on += b; if (i && b != (scratch[i - 1] > 0.5f)) trans++; }
    feat[49] = (float)on / WINDOW_S; feat[50] = trans; feat[51] = longestRun(scratch, isOn); }

  for (int i = 0; i < WINDOW_S; i++) scratch[i] = mixingRatio(fillT[i], fillRH[i], fillP[i]);
  basicStats(scratch, st);
  feat[52] = st[0]; feat[53] = st[1]; feat[54] = st[4];
  { double dsum = 0; for (int i = 0; i < WINDOW_S; i++) dsum += dewpoint(fillT[i], fillRH[i]); feat[55] = dsum / WINDOW_S; }

  { double tm = 0, hm = 0;
    for (int i = 0; i < WINDOW_S; i++) { tm += fillT[i]; hm += fillRH[i]; }
    tm /= WINDOW_S; hm /= WINDOW_S;
    double sth = 0, stt = 0, shh = 0;
    for (int i = 0; i < WINDOW_S; i++) { double tc = fillT[i] - tm, hc = fillRH[i] - hm; sth += tc * hc; stt += tc * tc; shh += hc * hc; }
    double den = sqrt(stt * shh);
    feat[56] = den > 1e-12 ? sth / den : 0.0; }

  for (uint8_t i = 0; i < N_FEATURES; i++) if (!isfinite(feat[i])) feat[i] = 0.0f;  // np.nan_to_num
  return faultMask;
}

// clip((x - mean) / std, -8, 8) — ml/stage-*/common.py
static void normaliseFeatures(const float* feat, float* out) {
  for (uint8_t i = 0; i < N_FEATURES; i++) {
    float z = (feat[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
    out[i] = fminf(fmaxf(z, -NORMALIZED_CLIP), NORMALIZED_CLIP);
  }
}
