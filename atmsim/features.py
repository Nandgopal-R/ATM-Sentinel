"""Window feature extraction.

300 raw 1 Hz observations -> one ML sample.

Design notes:

* Slopes are in **units per minute**, computed by closed-form least squares.
  At a 300 s window a cooling degradation moves temperature 0.5-1.0 degC, which
  is 25-50x the BME280 short-term noise. At the originally proposed 30 s window
  the same fault moves it 0.05 degC, which is inside the noise, and the model
  would have been forced to rely on absolute level instead of trend.
* `abs_humidity_mean` / `abs_humidity_slope` are the features that cleanly
  separate HIGH_HUMIDITY from COOLING_FAILURE. Both faults move relative
  humidity; only moisture ingress moves the mixing ratio. Computed from the
  *observed* temperature and RH, so a faulty temperature sensor corrupts them
  the same way it would in the field.
* `temp_rh_corr` captures the sign of the coupling: strongly negative during a
  thermal fault, near zero or positive during ingress.
* `*_null_frac` is how SENSOR_STUCK with dropouts becomes visible at all.

Everything is computed with stride tricks over the whole run at once, so the
extractor is vectorised rather than a per-window Python loop.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import psychro

FLOAT_CHANNELS = ["temperature", "humidity", "pressure", "light", "voltage", "current"]

SAG_THRESHOLD_V = 207.0      # 230 V - 10%
SWELL_THRESHOLD_V = 253.0    # 230 V + 10%
DROPOUT_THRESHOLD_V = 165.0


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linear interpolation across nulls, edge-filled. This is what the edge
    device would do before running inference; the nulls themselves are retained
    as a separate feature."""
    out = a.astype(float).copy()
    bad = np.isnan(out)
    if not bad.any():
        return out
    if bad.all():
        return np.zeros_like(out)
    idx = np.arange(len(out))
    out[bad] = np.interp(idx[bad], idx[~bad], out[~bad])
    return out


def _windows(a: np.ndarray, size: int, stride: int) -> np.ndarray:
    return sliding_window_view(a, size)[::stride]


def _slope_per_min(w: np.ndarray, dt: float) -> np.ndarray:
    """Least-squares slope of each row, converted to units per minute."""
    L = w.shape[1]
    x = np.arange(L, dtype=float) * dt
    x_c = x - x.mean()
    denom = (x_c**2).sum()
    y_c = w - w.mean(axis=1, keepdims=True)
    return (y_c @ x_c) / denom * 60.0


def _longest_true_run(b: np.ndarray) -> np.ndarray:
    """Longest consecutive run of True in each row, without a per-row loop."""
    run = np.zeros(b.shape[0], dtype=np.int32)
    best = np.zeros(b.shape[0], dtype=np.int32)
    for j in range(b.shape[1]):
        col = b[:, j]
        run = (run + 1) * col
        np.maximum(best, run, out=best)
    return best


def extract(telemetry, window_size: int, stride: int, dt: float = 1.0) -> dict:
    """Returns a dict of feature-name -> array, one entry per window."""
    feats: dict[str, np.ndarray] = {}
    filled = {}

    for ch in FLOAT_CHANNELS:
        raw = telemetry[ch].to_numpy(dtype=float)
        null_w = _windows(np.isnan(raw).astype(float), window_size, stride)
        feats[f"{ch}_null_frac"] = null_w.mean(axis=1)

        f = _interp_nan(raw)
        filled[ch] = f
        w = _windows(f, window_size, stride)

        feats[f"{ch}_mean"] = w.mean(axis=1)
        feats[f"{ch}_std"] = w.std(axis=1)
        feats[f"{ch}_min"] = w.min(axis=1)
        feats[f"{ch}_max"] = w.max(axis=1)
        feats[f"{ch}_slope"] = _slope_per_min(w, dt)
        feats[f"{ch}_delta"] = w[:, -1] - w[:, 0]

    # ------------------------------------------------------------- power ----
    vw = _windows(filled["voltage"], window_size, stride)
    feats["voltage_sag_frac"] = (vw < SAG_THRESHOLD_V).mean(axis=1)
    feats["voltage_swell_frac"] = (vw > SWELL_THRESHOLD_V).mean(axis=1)
    feats["voltage_dropout_frac"] = (vw < DROPOUT_THRESHOLD_V).mean(axis=1)

    # --------------------------------------------------------- vibration ----
    vib = _interp_nan(telemetry["vibration"].to_numpy(dtype=float))
    vibw = _windows(vib, window_size, stride)
    feats["vibration_sum"] = vibw.sum(axis=1)
    feats["vibration_max"] = vibw.max(axis=1)
    feats["vibration_active_frac"] = (vibw > 0).mean(axis=1)
    feats["vibration_burst_max_s"] = _longest_true_run(vibw > 0).astype(float)

    # ------------------------------------------------------------ motion ----
    mot = _interp_nan(telemetry["motion"].to_numpy(dtype=float))
    motw = _windows(mot, window_size, stride) > 0.5
    feats["motion_frac"] = motw.mean(axis=1)
    feats["motion_transitions"] = np.abs(np.diff(motw.astype(np.int8), axis=1)).sum(axis=1).astype(float)
    feats["motion_longest_run_s"] = _longest_true_run(motw).astype(float)

    # ----------------------------------------------------------- derived ----
    w_g = psychro.mixing_ratio(filled["temperature"], filled["humidity"], filled["pressure"])
    dew = psychro.dewpoint(filled["temperature"], filled["humidity"])
    ahw = _windows(w_g, window_size, stride)
    feats["abs_humidity_mean"] = ahw.mean(axis=1)
    feats["abs_humidity_std"] = ahw.std(axis=1)
    feats["abs_humidity_slope"] = _slope_per_min(ahw, dt)
    feats["dewpoint_mean"] = _windows(dew, window_size, stride).mean(axis=1)

    tw = _windows(filled["temperature"], window_size, stride)
    hw = _windows(filled["humidity"], window_size, stride)
    tc = tw - tw.mean(axis=1, keepdims=True)
    hc = hw - hw.mean(axis=1, keepdims=True)
    denom = np.sqrt((tc**2).sum(axis=1) * (hc**2).sum(axis=1))
    feats["temp_rh_corr"] = np.where(denom > 1e-12, (tc * hc).sum(axis=1) / np.maximum(denom, 1e-12), 0.0)

    for k, v in feats.items():
        feats[k] = np.nan_to_num(np.asarray(v, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return feats


FEATURE_ORDER = None  # populated on first extract by the caller
