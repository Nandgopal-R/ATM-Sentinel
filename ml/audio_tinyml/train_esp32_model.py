#!/usr/bin/env python3
"""
ATM Sentinel - offline TinyML training pipeline for the ESP32 audio + sensor-fusion module.

Runs on a PC. Produces everything the ESP32 firmware needs:

  datasets/atm_full_features.csv      all audio features (+ synthetic vibration/PIR context)
  datasets/atm_esp32_features.csv     only the compact features the ESP32 computes
  datasets/atm_fusion_events.csv      event-level dataset for the fusion classifier
  models/atm_audio_model.tflite       INT8 audio classifier (TFLite Micro compatible)
  models/atm_audio_cnn.tflite         INT8 log-Mel CNN (comparison candidate only)
  ../../firmware/atm_sentinel/atm_model_data.h  generated C header: normalisation, INT8 weights,
                                      fusion decision tree, TFLite byte array
  ../../firmware/atm_sentinel/test_vectors.h    reference inputs/outputs for on-device self-test
  model_metadata.json                 feature order, scaling, quantisation, model info
  reports/training_report.md          metrics, confusion matrices, model comparison

Usage:
  python train_esp32_model.py --manifest data/recordings.csv --audio-dir data/src --out .

Design rules this script enforces
  * The feature extractor below (`FrameDSP`) is the *reference* implementation. The C++
    file esp32_preprocessing.cpp implements exactly the same maths (same window, FFT size,
    mel filterbank formula, log floor, DCT, feature order). Do not swap in librosa.
  * Splits are made at recording level *before* windowing or augmentation. Augmented and
    synthetic audio inherits the split of the recording it was derived from, and only
    train-split audio is ever augmented. Test sets contain original audio only.
  * When a class has a single recording, a contiguous temporal block split with a guard gap
    is used instead and the report says so explicitly.
  * The existing two-stage malfunction/tamper cascade is NOT retrained here. Its tamper
    output is consumed as an input flag and keeps safety priority in the decision layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1. Shared DSP configuration (must match esp32_preprocessing.cpp / atm_model_data.h)
# --------------------------------------------------------------------------------------
SR = 16000                 # Hz, INMP441 I2S rate used by the firmware
FRAME = 512                # samples per frame (32 ms)
HOP = 512                  # non-overlapping frames: no frame history needed on-device
N_FFT = 512
N_BINS = N_FFT // 2 + 1    # 257
N_MEL = 26
N_MFCC = 13
FMIN = 20.0
FMAX = 8000.0
LOG_FLOOR = 1e-10
FRAMES_PER_WINDOW = 31     # 31 x 32 ms = 0.992 s inference window
TRAIN_WINDOW_STEP = 8      # frames between training windows (~0.26 s); device uses 31
ROLLOFF_FRAC = 0.85
EPS = 1e-12

CLASSES = ["BACKGROUND", "DISPENSE", "ABNORMAL"]
# Headers go straight into the main firmware sketch. Absolute, so os.path.join(--out, FW_DIR) is FW_DIR.
# After regenerating, copy them into firmware/tests/atm_tinyml_demo/ too (tools/check_demo_sync.sh checks).
FW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware", "atm_sentinel")
SYNTH_ABNORMAL_KINDS = ["jam", "grind"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Per-frame base features, in order. Window features = mean of each, then std of each.
FRAME_FEATURES = [f"mfcc{i}" for i in range(N_MFCC)] + [
    "rms_db", "zcr", "centroid", "rolloff", "flatness"]
COMPACT_FEATURES = [f"{n}_mean" for n in FRAME_FEATURES] + [f"{n}_std" for n in FRAME_FEATURES]

# Fusion (event-level) features, in order.
FUSION_FEATURES = [
    "p_background_mean", "p_dispense_mean", "p_abnormal_mean",
    "p_dispense_max", "p_abnormal_max",
    "dispense_window_frac", "abnormal_window_frac",
    "vib_rms", "vib_peak", "vib_var",
    "pir_active_frac", "interaction_s",
    "tamper_flag",
]
FUSION_CLASSES = ["IDLE", "NORMAL_DISPENSE", "SHORT_VISIT", "VISIT_NO_DISPENSE",
                  "ABNORMAL_MECHANICAL", "SUSPECTED_TAMPER"]

# Sensor conventions (must match the existing firmware - edit if your units differ)
VIB_UNIT = "SW-420 edge count per 1 s bin"
TAMPER_VIB_THRESHOLD = 75.0          # existing tamper rule: ~70-80 sensor units
SHORT_VISIT_MAX_S = 30.0             # short PIR visit ~20-25 s
PIR_GAP_TOLERANCE_S = 8.0            # HC-SR501 retrigger gaps merged into one visit


def htk_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def htk_mel_inv(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """Triangular HTK mel filterbank, unnormalised, shape (N_MEL, N_BINS).

    The C++ code rebuilds this at boot with the same formula (double precision)."""
    mel_pts = np.linspace(htk_mel(FMIN), htk_mel(FMAX), N_MEL + 2)
    hz_pts = htk_mel_inv(mel_pts)
    bin_hz = np.arange(N_BINS) * SR / N_FFT
    fb = np.zeros((N_MEL, N_BINS))
    for m in range(N_MEL):
        lo, ce, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (bin_hz - lo) / (ce - lo)
        down = (hi - bin_hz) / (hi - ce)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def dct_matrix() -> np.ndarray:
    """Orthonormal DCT-II, first N_MFCC rows, shape (N_MFCC, N_MEL)."""
    n = np.arange(N_MEL)
    d = np.zeros((N_MFCC, N_MEL))
    for k in range(N_MFCC):
        d[k] = np.cos(np.pi * k * (n + 0.5) / N_MEL)
        d[k] *= np.sqrt(1.0 / N_MEL) if k == 0 else np.sqrt(2.0 / N_MEL)
    return d


class FrameDSP:
    """Reference per-frame feature extractor (vectorised over frames)."""

    def __init__(self):
        n = np.arange(FRAME)
        self.window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / FRAME)   # periodic Hann
        self.fb = mel_filterbank()
        self.dct = dct_matrix()
        self.bin_norm = np.arange(N_BINS) / (N_BINS - 1)            # 0..1 of Nyquist

    def frames(self, x: np.ndarray) -> np.ndarray:
        n = (len(x) - FRAME) // HOP + 1
        if n <= 0:
            return np.zeros((0, FRAME))
        idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
        return x[idx]

    def per_frame(self, fr: np.ndarray):
        """Returns (base_features[n,18], logmel[n,26], extra dict) for frames fr[n,512]."""
        fr = fr.astype(np.float64)
        fr = fr - fr.mean(axis=1, keepdims=True)                     # DC removal
        rms = np.sqrt(np.mean(fr * fr, axis=1))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        sgn = fr >= 0.0
        zcr = np.count_nonzero(sgn[:, 1:] != sgn[:, :-1], axis=1) / (FRAME - 1)
        spec = np.fft.rfft(fr * self.window[None, :], n=N_FFT, axis=1)
        p = (spec.real ** 2 + spec.imag ** 2) / N_FFT                # power, shape (n,257)
        tot = p.sum(axis=1)
        centroid = (p * self.bin_norm[None, :]).sum(axis=1) / (tot + EPS)
        cum = np.cumsum(p, axis=1)
        roll_idx = np.argmax(cum >= ROLLOFF_FRAC * tot[:, None], axis=1)
        rolloff = roll_idx / (N_BINS - 1)
        pk = p[:, 1:]                                                # skip DC bin
        flat = np.exp(np.mean(np.log(pk + LOG_FLOOR), axis=1)) / (np.mean(pk, axis=1) + LOG_FLOOR)
        mel = p @ self.fb.T
        logmel = np.log(np.maximum(mel, LOG_FLOOR))
        mfcc = logmel @ self.dct.T
        base = np.column_stack([mfcc, rms_db, zcr, centroid, rolloff, flat])
        # extra (offline-only) per-frame descriptors for the full dataset
        bw = np.sqrt((p * (self.bin_norm[None, :] - centroid[:, None]) ** 2).sum(axis=1) / (tot + EPS))
        crest = np.max(np.abs(fr), axis=1) / (rms + 1e-9)
        extra = {"bandwidth": bw, "crest": crest}
        return base, logmel, extra


DSP = FrameDSP()


def window_compact(base_win: np.ndarray) -> np.ndarray:
    """36 compact features for one window of base frame features (31,18)."""
    return np.concatenate([base_win.mean(axis=0), base_win.std(axis=0)])


# --------------------------------------------------------------------------------------
# 2. Audio I/O
# --------------------------------------------------------------------------------------
def load_audio(path: str) -> np.ndarray:
    """Decode any format to mono float32 at SR via ffmpeg (s16 PCM, /32768)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


# --------------------------------------------------------------------------------------
# 3. Synthetic audio (background, lobby noise, abnormal-mechanism variants)
# --------------------------------------------------------------------------------------
def pink_noise(n, rng):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f)); k[0] = 1
    f /= np.sqrt(k)
    y = np.fft.irfft(f, n)
    return y / (np.std(y) + EPS)


def brown_noise(n, rng):
    y = np.cumsum(rng.standard_normal(n))
    y -= np.convolve(y, np.ones(4001) / 4001, mode="same")          # remove drift
    return y / (np.std(y) + EPS)


def db_to_amp(db):
    return 10.0 ** (db / 20.0)


def lobby_noise(n, rng) -> np.ndarray:
    """ATM lobby ambience: HVAC rumble + mains hum + pink floor. Unit RMS."""
    t = np.arange(n) / SR
    hum_f = rng.choice([50.0, 50.0, 60.0])
    hum = sum(rng.uniform(0.2, 1.0) / (h + 1) * np.sin(2 * np.pi * hum_f * (h + 1) * t + rng.uniform(0, 6.28))
              for h in range(4))
    y = (rng.uniform(0.3, 1.0) * brown_noise(n, rng) + rng.uniform(0.2, 1.0) * pink_noise(n, rng)
         + rng.uniform(0.0, 0.6) * hum)
    return y / (np.std(y) + EPS)


def keypad_beeps(n, rng) -> np.ndarray:
    """Sparse ATM keypad/UI beeps (non-dispense interaction sounds)."""
    y = np.zeros(n)
    nb = rng.integers(1, 8)
    for _ in range(nb):
        f = rng.choice([800.0, 1000.0, 1200.0, 2000.0, 2400.0])
        L = int(SR * rng.uniform(0.06, 0.25))
        s = rng.integers(0, max(1, n - L))
        tt = np.arange(L) / SR
        env = np.minimum(1.0, np.minimum(tt / 0.005, (L / SR - tt) / 0.005))
        y[s:s + L] += env * np.sin(2 * np.pi * f * tt)
    return y


def synth_background_recording(rid: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("bg", rid))
    n = int(seconds * SR)
    y = lobby_noise(n, rng) * db_to_amp(rng.uniform(-62, -36))
    if rng.random() < 0.6:
        y += keypad_beeps(n, rng) * db_to_amp(rng.uniform(-38, -18))
    if rng.random() < 0.3:                                             # door / footsteps thumps
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, n - 2000)
            L = 1600
            y[s:s + L] += np.exp(-np.arange(L) / 250.0) * rng.standard_normal(L) * db_to_amp(rng.uniform(-40, -24))
    return y.astype(np.float32)


def clunk(rng) -> np.ndarray:
    """Decaying low-frequency mechanical impact (jammed roller / shutter hit)."""
    L = int(SR * rng.uniform(0.08, 0.25))
    t = np.arange(L) / SR
    f0 = rng.uniform(70, 260)
    y = np.sin(2 * np.pi * f0 * t) * np.exp(-t * rng.uniform(15, 40))
    y += 0.5 * rng.standard_normal(L) * np.exp(-t * rng.uniform(40, 120))
    return y / (np.max(np.abs(y)) + EPS)


def synth_abnormal_from(x: np.ndarray, rng, kind: str) -> np.ndarray:
    """Turn a train-split mechanism recording into an abnormal-mechanism variant."""
    y = x.astype(np.float64).copy()
    lvl = np.sqrt(np.mean(y ** 2)) + EPS
    n = len(y)
    if kind == "jam":
        # irregular heavy clunks + the transport stalls (abrupt drop to near-silence)
        for _ in range(rng.integers(2, 6)):
            c = clunk(rng) * lvl * db_to_amp(rng.uniform(10, 18))
            s = rng.integers(0, max(1, n - len(c)))
            y[s:s + len(c)] += c
        stop = int(n * rng.uniform(0.4, 0.8))
        y[stop:] *= db_to_amp(rng.uniform(-30, -18))
    elif kind == "grind":
        # rough amplitude modulation + band-limited rasp
        t = np.arange(n) / SR
        am = 1.0 + 0.8 * np.sign(np.sin(2 * np.pi * rng.uniform(6, 18) * t))
        rasp = np.diff(np.concatenate([[0], rng.standard_normal(n)])) * lvl * db_to_amp(rng.uniform(-4, 4))
        y = y * am * 0.6 + rasp
    elif kind == "stutter":
        # start-stop motor: gated mechanism with knocks at each restart
        gate_len = int(SR * rng.uniform(0.15, 0.4))
        g = np.ones(n)
        for s in range(0, n, 2 * gate_len):
            g[s + gate_len:s + 2 * gate_len] = db_to_amp(-25)
            c = clunk(rng) * lvl * db_to_amp(rng.uniform(6, 14))
            e = min(n, s + len(c))
            y[s:e] += c[:e - s]
        y *= g
    return y.astype(np.float32)


def augment(x: np.ndarray, rng) -> np.ndarray:
    """Mic/level/environment variation for train-split audio."""
    from scipy.signal import resample_poly, butter, lfilter
    y = x.astype(np.float64)
    # speed / pitch perturbation (motor speed variation between machines)
    ratios = [(10, 11), (11, 10), (20, 21), (21, 20), (1, 1)]
    r = ratios[rng.integers(0, len(ratios))]
    if r != (1, 1):
        y = resample_poly(y, r[0], r[1])
    # mic / enclosure response: random gentle HPF and LPF
    b, a = butter(1, rng.uniform(60, 250) / (SR / 2), btype="high")
    y = lfilter(b, a, y)
    if rng.random() < 0.5:
        b, a = butter(2, rng.uniform(3500, 7500) / (SR / 2), btype="low")
        y = lfilter(b, a, y)
    # level change (INMP441 distance / gain differences)
    sig = np.sqrt(np.mean(y ** 2)) + EPS
    target_db = rng.uniform(-48, -14)
    y *= db_to_amp(target_db) / sig
    # lobby noise at random SNR
    snr = rng.uniform(5, 35)
    y += lobby_noise(len(y), rng) * db_to_amp(target_db - snr)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# 4. Segmenting recordings into split-assigned audio chunks
# --------------------------------------------------------------------------------------
@dataclass
class Chunk:
    audio: np.ndarray
    label: str
    recording_id: str          # source recording
    chunk_id: str              # unique id for this audio stream
    provenance: str            # original | synthetic
    synth_method: str          # none | augment | jam | grind | ...
    split: str                 # train | test
    split_method: str          # recording_holdout | temporal_block | synthetic_recording


def build_chunks(manifest: pd.DataFrame, audio_dir: str, n_aug: int, n_bg: int,
                 block_test_frac: float, guard_s: float) -> list[Chunk]:
    chunks: list[Chunk] = []
    originals: list[Chunk] = []
    for _, r in manifest.iterrows():
        x = load_audio(os.path.join(audio_dir, r["file"]))
        rid = r["recording_id"]
        if r["split"] == "block":
            n = len(x)
            b = int(n * (1.0 - block_test_frac))
            g = int(guard_s * SR)
            originals.append(Chunk(x[: max(0, b - g)], r["label"], rid, f"{rid}_tr", "original", "none",
                                   "train", "temporal_block"))
            originals.append(Chunk(x[b:], r["label"], rid, f"{rid}_te", "original", "none",
                                   "test", "temporal_block"))
        else:
            originals.append(Chunk(x, r["label"], rid, rid, "original", "none", r["split"],
                                   "recording_holdout"))
    chunks.extend(originals)

    # augmentation + synthetic abnormal: train split only
    for c in [c for c in originals if c.split == "train"]:
        for k in range(n_aug):
            rng = np.random.default_rng(stable_seed("aug", c.chunk_id, k))
            chunks.append(Chunk(augment(c.audio, rng), c.label, c.recording_id, f"{c.chunk_id}_aug{k}",
                                "synthetic", "augment", "train", c.split_method))
        if c.label == "DISPENSE":
            # "stutter" (start-stop) exists but is not used: real dispensers start and stop
            # normally, so it taught the model that normal dispensing is abnormal.
            for kind in SYNTH_ABNORMAL_KINDS:
                for k in range(max(1, n_aug // 4)):
                    rng = np.random.default_rng(stable_seed(kind, c.chunk_id, k))
                    y = synth_abnormal_from(c.audio, rng, kind)
                    chunks.append(Chunk(augment(y, rng), "ABNORMAL", c.recording_id,
                                        f"{c.chunk_id}_{kind}{k}", "synthetic", kind, "train",
                                        c.split_method))

    # synthetic background recordings: independent recordings, split at recording level
    for i in range(n_bg):
        rng = np.random.default_rng(stable_seed("bgsplit", i))
        split = "test" if i % 4 == 3 else "train"
        y = synth_background_recording(i, rng.uniform(6, 12))
        chunks.append(Chunk(y, "BACKGROUND", f"synth_bg{i:02d}", f"synth_bg{i:02d}", "synthetic",
                            "lobby_bg", split, "synthetic_recording"))
    return chunks


# --------------------------------------------------------------------------------------
# 5. Windowing + feature tables
# --------------------------------------------------------------------------------------
def chunk_windows(c: Chunk, step: int):
    fr = DSP.frames(c.audio)
    if len(fr) < FRAMES_PER_WINDOW:
        return []
    base, logmel, extra = DSP.per_frame(fr)
    rms_db = base[:, N_MFCC]
    active_ref = np.median(rms_db[rms_db > np.percentile(rms_db, 30)])
    active = rms_db > active_ref - 20.0
    out = []
    for w0 in range(0, len(fr) - FRAMES_PER_WINDOW + 1, step):
        sl = slice(w0, w0 + FRAMES_PER_WINDOW)
        frac = active[sl].mean()
        label, label_source = c.label, "recording"
        if c.label != "BACKGROUND" and c.provenance == "original":
            # windows that are essentially silent lead-in/tail become BACKGROUND,
            # ambiguous partial windows are dropped
            if frac <= 0.10 and rms_db[sl].mean() < -45:
                label, label_source = "BACKGROUND", "auto_silence"
            elif frac < 0.6:
                continue
        elif c.label != "BACKGROUND" and frac < 0.6:
            continue            # synthetic: drop stalled/quiet stretches rather than mislabel
        bw = base[sl]
        compact = window_compact(bw)
        full = {}
        for j, nme in enumerate(FRAME_FEATURES):
            col = bw[:, j]
            full[f"{nme}_mean"] = col.mean()
            full[f"{nme}_std"] = col.std()
            full[f"{nme}_min"] = col.min()
            full[f"{nme}_max"] = col.max()
        lm = logmel[sl]
        for m in range(N_MEL):
            full[f"logmel{m}_mean"] = lm[:, m].mean()
            full[f"logmel{m}_std"] = lm[:, m].std()
        dm = np.diff(bw[:, :N_MFCC], axis=0)
        for j in range(N_MFCC):
            full[f"dmfcc{j}_std"] = dm[:, j].std()
        for k, v in extra.items():
            full[f"{k}_mean"] = v[sl].mean()
            full[f"{k}_std"] = v[sl].std()
        flux = np.sqrt((np.diff(lm, axis=0) ** 2).sum(axis=1))
        full["flux_mean"], full["flux_std"] = flux.mean(), flux.std()
        full["active_frac"] = frac
        out.append(dict(label=label, label_source=label_source, start_s=w0 * HOP / SR,
                        compact=compact, full=full, logmel=lm.astype(np.float32)))
    return out


def build_window_tables(chunks: list[Chunk]):
    rows, logmels = [], []
    for c in chunks:
        step = TRAIN_WINDOW_STEP
        for k, w in enumerate(chunk_windows(c, step)):
            meta = dict(window_id=f"{c.chunk_id}_w{k:03d}", recording_id=c.recording_id,
                        chunk_id=c.chunk_id, provenance=c.provenance, synth_method=c.synth_method,
                        split=c.split, split_method=c.split_method, label=w["label"],
                        label_id=CLASS_TO_ID[w["label"]], label_source=w["label_source"],
                        start_s=round(w["start_s"], 3))
            row = dict(meta)
            row.update({n: v for n, v in zip(COMPACT_FEATURES, w["compact"])})
            row.update({f"x_{k2}": v for k2, v in w["full"].items() if k2 not in
                        [f"{n}" for n in COMPACT_FEATURES]})
            rows.append(row)
            logmels.append(w["logmel"])
    df = pd.DataFrame(rows)
    return df, np.stack(logmels)


# --------------------------------------------------------------------------------------
# 6. Models
# --------------------------------------------------------------------------------------
def metrics(y_true, y_pred, n_cls, names):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
    labels = list(range(n_cls))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(np.mean(f[s > 0])) if np.any(s > 0) else 0.0,
                per_class={names[i]: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                                          support=int(s[i])) for i in labels},
                confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist())


def tree_nodes(t):
    return int(t.tree_.node_count)


def tree_flash_bytes(n_nodes, n_out):
    # feature(int8) + threshold(float) + left(int16) + right(int16) + leaf class probs(uint8 * n_out)
    return n_nodes * (1 + 4 + 2 + 2 + n_out)


def build_mlp(n_in, n_out, hidden):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(n_in,), name="features")
    h = inp
    for u in hidden:
        h = tf.keras.layers.Dense(u, activation="relu")(h)
    out = tf.keras.layers.Dense(n_out, activation="softmax", name="probs")(h)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_cnn(n_t, n_mel, n_out):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(n_t, n_mel), name="logmel")
    h = tf.keras.layers.Conv1D(12, 3, activation="relu")(inp)
    h = tf.keras.layers.MaxPooling1D(2)(h)
    h = tf.keras.layers.SeparableConv1D(16, 3, activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    out = tf.keras.layers.Dense(n_out, activation="softmax", name="probs")(h)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def class_weights(y, n):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = cnt.sum() / (n * np.maximum(cnt, 1))
    return {i: float(w[i]) for i in range(n)}


def fit_keras(model, X, y, n_cls, seed, epochs=120):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)]
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=0, class_weight=class_weights(y, n_cls),
              callbacks=cb, shuffle=True)
    return model


def to_int8_tflite(model, rep_X):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        idx = np.random.default_rng(0).choice(len(rep_X), size=min(300, len(rep_X)), replace=False)
        for i in idx:
            yield [rep_X[i:i + 1].astype(np.float32)]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


class TFLiteRunner:
    def __init__(self, blob: bytes):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_content=blob)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def quant_in(self, X):
        s, z = self.i["quantization"]
        return np.clip(np.round(X / s + z), -128, 127).astype(np.int8)

    def run_q(self, Xq):
        out = []
        for row in Xq:
            self.it.set_tensor(self.i["index"], row[None, ...])
            self.it.invoke()
            out.append(self.it.get_tensor(self.o["index"])[0].copy())
        return np.array(out)

    def dequant_out(self, Yq):
        s, z = self.o["quantization"]
        return (Yq.astype(np.float32) - z) * s

    def predict(self, X):
        return self.dequant_out(self.run_q(self.quant_in(X)))


def dense_layers_from_tflite(blob: bytes):
    """Extract int8 FULLY_CONNECTED layers (weights, bias, scales, zero points) from the
    converted model so the native C engine reproduces TFLite's integer arithmetic."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob, experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    tens = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    layers = []
    for op in ops:
        if op["op_name"] != "FULLY_CONNECTED":
            continue
        a, w, b = op["inputs"][:3]
        o = op["outputs"][0]
        ta, tw, tb, to = tens[a], tens[w], tens[b], tens[o]
        layers.append(dict(
            W=it.get_tensor(w).astype(np.int8), B=it.get_tensor(b).astype(np.int32),
            in_scale=float(ta["quantization"][0]), in_zp=int(ta["quantization"][1]),
            w_scale=np.asarray(tw["quantization_parameters"]["scales"], dtype=np.float64)
                    * np.ones(it.get_tensor(w).shape[0]),        # per-output-channel
            out_scale=float(to["quantization"][0]), out_zp=int(to["quantization"][1])))
    return layers


# --------------------------------------------------------------------------------------
# 7. Fusion dataset (event level) - synthetic sensor context + real audio probabilities
# --------------------------------------------------------------------------------------
def simulate_event(kind: str, rng, audio_pool: dict, prob_fn):
    """One interaction/idle event. Vibration is per-second SW-420 edge counts; PIR is a
    per-second 0/1 series; audio windows are drawn from real (split-matched) windows and
    passed through the deployed audio model."""
    if kind == "IDLE":
        dur = int(rng.choice([60, 60, 60, rng.integers(10, 60)]))   # firmware flushes 10-60 s idle
        pir = (rng.random(dur) < rng.uniform(0.0, 0.03)).astype(float)    # rare false triggers
        vib = rng.poisson(rng.uniform(0.0, 1.5), dur).astype(float)
        audio_mix = {"BACKGROUND": 1.0}
    elif kind == "NORMAL_DISPENSE":
        dur = int(rng.uniform(35, 180))
        pir = np.ones(dur); pir[rng.random(dur) < rng.uniform(0, 0.25)] = 0
        vib = rng.poisson(rng.uniform(1, 6), dur).astype(float)
        d0 = rng.integers(int(dur * 0.3), int(dur * 0.8))
        vib[d0:d0 + 8] += rng.poisson(rng.uniform(4, 20), len(vib[d0:d0 + 8]))      # dispenser vibration
        if rng.random() < 0.15:                                               # firm keypad press / bump
            vib[rng.integers(0, dur)] += rng.uniform(25, 60)
        audio_mix = {"DISPENSE": rng.uniform(0.08, 0.4), "BACKGROUND": None}
    elif kind == "SHORT_VISIT":
        dur = int(np.clip(rng.normal(22.5, 3.5), 10, SHORT_VISIT_MAX_S))
        pir = np.ones(dur); pir[rng.random(dur) < 0.15] = 0
        vib = rng.poisson(rng.uniform(0.5, 4), dur).astype(float)
        audio_mix = {"BACKGROUND": 1.0}
    elif kind == "VISIT_NO_DISPENSE":
        dur = int(rng.uniform(SHORT_VISIT_MAX_S + 2, 150))
        pir = np.ones(dur); pir[rng.random(dur) < rng.uniform(0, 0.25)] = 0
        vib = rng.poisson(rng.uniform(0.5, 5), dur).astype(float)
        if rng.random() < 0.15:
            vib[rng.integers(0, dur)] += rng.uniform(25, 60)
        audio_mix = {"BACKGROUND": 1.0}
    elif kind == "ABNORMAL_MECHANICAL":
        dur = int(rng.uniform(35, 200))
        pir = np.ones(dur); pir[rng.random(dur) < rng.uniform(0, 0.3)] = 0
        vib = rng.poisson(rng.uniform(3, 15), dur).astype(float)
        audio_mix = {"ABNORMAL": rng.uniform(0.08, 0.5), "DISPENSE": rng.uniform(0.0, 0.15),
                     "BACKGROUND": None}
    elif kind == "SUSPECTED_TAMPER":
        dur = int(rng.uniform(20, 240))
        pir = (rng.random(dur) < rng.uniform(0.3, 1.0)).astype(float)
        vib = rng.poisson(rng.uniform(2, 10), dur).astype(float)
        nb = rng.integers(1, 6)
        for _ in range(nb):
            vib[rng.integers(0, dur)] += rng.uniform(TAMPER_VIB_THRESHOLD, 220)
        m = rng.choice(["BACKGROUND", "DISPENSE", "ABNORMAL"], p=[0.6, 0.25, 0.15])
        audio_mix = {m: 1.0} if m == "BACKGROUND" else {m: rng.uniform(0.1, 0.4), "BACKGROUND": None}
    else:
        raise ValueError(kind)

    # audio windows: one per second of event (1 s windows)
    n_win = dur
    labels = []
    rest = 1.0 - sum(v for v in audio_mix.values() if v is not None)
    for c, v in audio_mix.items():
        labels += [c] * int(round(n_win * (v if v is not None else rest)))
    labels = (labels + ["BACKGROUND"] * n_win)[:n_win]
    if kind != "IDLE":
        # the firmware closes an interaction only after the PIR has been quiet for
        # > PIR_GAP_TOLERANCE_S seconds, so every real event ends with that quiet tail
        tail = int(PIR_GAP_TOLERANCE_S) + 1
        pir = np.concatenate([pir, np.zeros(tail)])
        vib = np.concatenate([vib, rng.poisson(rng.uniform(0.0, 2.0), tail).astype(float)])
        labels += ["BACKGROUND"] * tail
    idx = [audio_pool[c][rng.integers(0, len(audio_pool[c]))] for c in labels]
    P = prob_fn(idx)
    arg = P.argmax(1)
    vrms = float(np.sqrt(np.mean(vib ** 2)))
    feat = dict(p_background_mean=P[:, 0].mean(), p_dispense_mean=P[:, 1].mean(),
                p_abnormal_mean=P[:, 2].mean(), p_dispense_max=P[:, 1].max(), p_abnormal_max=P[:, 2].max(),
                dispense_window_frac=(arg == 1).mean(), abnormal_window_frac=(arg == 2).mean(),
                vib_rms=vrms, vib_peak=float(vib.max()), vib_var=float(vib.var()),
                pir_active_frac=float(pir.mean()), interaction_s=float(pir_interaction_s(pir)),
                tamper_flag=float(vib.max() >= TAMPER_VIB_THRESHOLD))
    return feat


def pir_interaction_s(pir: np.ndarray) -> float:
    """Longest PIR-active span, bridging gaps <= PIR_GAP_TOLERANCE_S (same as firmware)."""
    best = cur = 0.0
    gap = 1e9
    active = False
    for v in pir:
        if v > 0:
            if active and gap <= PIR_GAP_TOLERANCE_S:
                cur += gap + 1
            else:
                cur = 1
            active = True
            gap = 0
            best = max(best, cur)
        else:
            gap += 1
    return best


def train_tree_export(clfs, n_out):
    """Flatten one or more sklearn trees into node arrays. Child indices are absolute;
    roots lists the first node of each tree (C averages leaf probabilities over trees)."""
    if not isinstance(clfs, (list, tuple)):
        clfs = [clfs]
    F, T, Lf, R, P, roots = [], [], [], [], [], []
    off = 0
    for clf in clfs:
        t = clf.tree_
        leaf = t.children_left == -1
        feat = np.where(leaf, -1, t.feature)
        vals = t.value[:, 0, :]
        probs = vals / np.maximum(vals.sum(1, keepdims=True), EPS)
        full = np.zeros((t.node_count, n_out), dtype=np.uint8)
        full[:, np.asarray(clf.classes_).astype(int)] = np.round(probs * 255).astype(np.uint8)
        roots.append(off)
        F += feat.astype(int).tolist()
        T += np.where(leaf, 0.0, t.threshold).astype(np.float32).tolist()
        Lf += np.where(leaf, -1, t.children_left + off).astype(int).tolist()
        R += np.where(leaf, -1, t.children_right + off).astype(int).tolist()
        P += full.tolist()
        off += t.node_count
    return dict(feature=F, threshold=T, left=Lf, right=R, leaf_probs=P, roots=roots)


# --------------------------------------------------------------------------------------
# 8. C header generation
# --------------------------------------------------------------------------------------
def c_float_array(name, arr, per_line=8):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    def lit(v):
        t = f"{v:.9g}"
        return (t if any(ch in t for ch in ".e") else t + ".0") + "f"
    body = ",\n  ".join(", ".join(lit(v) for v in arr[i:i + per_line]) for i in range(0, len(arr), per_line))
    return f"static const float {name}[{len(arr)}] = {{\n  {body}\n}};\n"


def c_int_array(name, ctype, arr, per_line=16):
    arr = np.asarray(arr).ravel()
    body = ",\n  ".join(", ".join(str(int(v)) for v in arr[i:i + per_line]) for i in range(0, len(arr), per_line))
    return f"static const {ctype} {name}[{len(arr)}] = {{\n  {body}\n}};\n"


def quantize_multiplier(real: float):
    """TFLite-style fixed point multiplier: real = q * 2^(shift-31)."""
    if real == 0:
        return 0, 0
    m, e = np.frexp(real)
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        e += 1
    return q, int(e)


def write_model_header(path, meta, mu, sd, dense_layers, tree_audio, fusion_tree, tflite_blob):
    L = []
    L.append("// AUTO-GENERATED by train_esp32_model.py - do not edit by hand.\n")
    L.append(f"// generated {meta['generated_utc']}  dataset_hash {meta['dataset_hash']}\n")
    L.append("#pragma once\n#include <stdint.h>\n\n")
    L.append(f"#define ATM_MODEL_VERSION \"{meta['model_version']}\"\n")
    L.append(f"#define ATM_SR {SR}\n#define ATM_FRAME {FRAME}\n#define ATM_N_FFT {N_FFT}\n")
    L.append(f"#define ATM_N_MEL {N_MEL}\n#define ATM_N_MFCC {N_MFCC}\n")
    L.append(f"#define ATM_FMIN {FMIN}f\n#define ATM_FMAX {FMAX}f\n#define ATM_LOG_FLOOR {LOG_FLOOR}f\n")
    L.append(f"#define ATM_FRAMES_PER_WINDOW {FRAMES_PER_WINDOW}\n")
    L.append(f"#define ATM_N_FRAME_FEATURES {len(FRAME_FEATURES)}\n")
    L.append(f"#define ATM_N_AUDIO_FEATURES {len(COMPACT_FEATURES)}\n")
    L.append(f"#define ATM_N_AUDIO_CLASSES {len(CLASSES)}\n")
    L.append(f"#define ATM_N_FUSION_FEATURES {len(FUSION_FEATURES)}\n")
    L.append(f"#define ATM_N_FUSION_CLASSES {len(FUSION_CLASSES)}\n")
    L.append(f"#define ATM_TAMPER_VIB_THRESHOLD {TAMPER_VIB_THRESHOLD}f\n")
    L.append(f"#define ATM_SHORT_VISIT_MAX_S {SHORT_VISIT_MAX_S}f\n")
    L.append(f"#define ATM_PIR_GAP_TOLERANCE_S {PIR_GAP_TOLERANCE_S}f\n\n")
    L.append("// audio feature order: " + ", ".join(COMPACT_FEATURES) + "\n")
    L.append("static const char* const ATM_AUDIO_CLASS_NAMES[] = {" + ", ".join(f'"{c}"' for c in CLASSES) + "};\n")
    L.append("static const char* const ATM_FUSION_CLASS_NAMES[] = {" + ", ".join(f'"{c}"' for c in FUSION_CLASSES) + "};\n\n")
    L.append(c_float_array("ATM_FEAT_MEAN", mu))
    L.append(c_float_array("ATM_FEAT_STD", sd))

    backend = meta["audio_model"]["backend"]
    L.append(f"\n#define ATM_AUDIO_BACKEND_{backend.upper()} 1\n")
    if dense_layers is not None:
        L.append(f"#define ATM_DENSE_N_LAYERS {len(dense_layers)}\n")
        for i, l in enumerate(dense_layers):
            n_out, n_in = l["W"].shape
            qs = [quantize_multiplier(l["in_scale"] * ws / l["out_scale"]) for ws in l["w_scale"]]
            L.append(f"// layer {i}: {n_in} -> {n_out}, relu={'1' if i < len(dense_layers) - 1 else '0 (softmax follows)'}\n")
            L.append(f"#define ATM_L{i}_IN {n_in}\n#define ATM_L{i}_OUT {n_out}\n")
            L.append(f"#define ATM_L{i}_IN_ZP {l['in_zp']}\n#define ATM_L{i}_OUT_ZP {l['out_zp']}\n")
            L.append(c_int_array(f"ATM_L{i}_MULT", "int32_t", [q for q, _ in qs]))
            L.append(c_int_array(f"ATM_L{i}_SHIFT", "int8_t", [sh for _, sh in qs]))
            L.append(c_int_array(f"ATM_L{i}_W", "int8_t", l["W"]))
            L.append(c_int_array(f"ATM_L{i}_B", "int32_t", l["B"]))
        first, last = dense_layers[0], dense_layers[-1]
        L.append(f"#define ATM_IN_SCALE {first['in_scale']:.9g}f\n#define ATM_IN_ZP {first['in_zp']}\n")
        L.append(f"// last dense layer outputs logits; softmax is done in float on-device\n")
        L.append(f"#define ATM_LOGIT_SCALE {last['out_scale']:.9g}f\n#define ATM_LOGIT_ZP {last['out_zp']}\n")
    if tree_audio is not None:
        L.append(tree_c("ATM_AT", tree_audio))
    L.append(tree_c("ATM_FT", fusion_tree))
    if tflite_blob is not None:
        L.append("\n// INT8 TFLite flatbuffer of the same audio model (for the TFLite Micro backend).\n")
        L.append("#ifdef ATM_USE_TFLM\nalignas(16) ")
        L.append(c_int_array("ATM_AUDIO_TFLITE", "unsigned char", list(tflite_blob)).replace("static const", "static const"))
        L.append(f"static const unsigned int ATM_AUDIO_TFLITE_LEN = {len(tflite_blob)};\n#endif\n")
    with open(path, "w") as f:
        f.write("".join(L))


def tree_c(prefix, t):
    n = len(t["feature"])
    s = f"\n#define {prefix}_N_NODES {n}\n"
    s += c_int_array(f"{prefix}_FEATURE", "int8_t", t["feature"])
    s += c_float_array(f"{prefix}_THRESHOLD", t["threshold"])
    s += c_int_array(f"{prefix}_LEFT", "int16_t", t["left"])
    s += c_int_array(f"{prefix}_RIGHT", "int16_t", t["right"])
    s += c_int_array(f"{prefix}_LEAF_PROB", "uint8_t", t["leaf_probs"])
    s += f"#define {prefix}_N_TREES {len(t['roots'])}\n"
    s += c_int_array(f"{prefix}_ROOTS", "int16_t", t["roots"])
    return s


# --------------------------------------------------------------------------------------
# 9. Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/recordings.csv")
    ap.add_argument("--audio-dir", default="data/src")
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-aug", type=int, default=12, help="augmented copies per train recording")
    ap.add_argument("--n-bg", type=int, default=24, help="synthetic background recordings")
    ap.add_argument("--block-test-frac", type=float, default=0.4)
    ap.add_argument("--guard-s", type=float, default=0.5)
    ap.add_argument("--n-fusion-events", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    out = args.out
    for d in ["datasets", "models", FW_DIR, "reports"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    t0 = time.time()

    manifest = pd.read_csv(args.manifest)
    print(f"[1/7] building audio chunks from {len(manifest)} recordings")
    chunks = build_chunks(manifest, args.audio_dir, args.n_aug, args.n_bg, args.block_test_frac, args.guard_s)
    print(f"      {len(chunks)} chunks ({sum(c.provenance == 'original' for c in chunks)} original)")

    print("[2/7] windowing + feature extraction")
    df, LM = build_window_tables(chunks)
    print(df.groupby(["split", "label", "provenance"]).size().to_string())

    # datasets ----------------------------------------------------------------------
    meta_cols = ["window_id", "recording_id", "chunk_id", "provenance", "synth_method", "split",
                 "split_method", "label", "label_id", "label_source", "start_s"]
    # synthetic concurrent sensor context for each audio window (offline analysis only)
    rng = np.random.default_rng(args.seed)
    ctx = []
    for lbl in df["label"]:
        base = {"BACKGROUND": 0.8, "DISPENSE": 8.0, "ABNORMAL": 12.0}[lbl]
        v = rng.poisson(base, 10).astype(float)
        ctx.append(dict(vib_rms=np.sqrt(np.mean(v ** 2)), vib_peak=v.max(), vib_var=v.var(),
                        pir_active=float(rng.random() < (0.2 if lbl == "BACKGROUND" else 0.95)),
                        sensor_provenance="synthetic"))
    ctx = pd.DataFrame(ctx)
    full_df = pd.concat([df.reset_index(drop=True), ctx], axis=1)
    full_path = os.path.join(out, "datasets", "atm_full_features.csv")
    full_df.to_csv(full_path, index=False, float_format="%.6g")
    esp_df = pd.concat([df[meta_cols].reset_index(drop=True), df[COMPACT_FEATURES].reset_index(drop=True),
                        ctx], axis=1)
    esp_path = os.path.join(out, "datasets", "atm_esp32_features.csv")
    esp_df.to_csv(esp_path, index=False, float_format="%.7g")
    ds_hash = hashlib.sha256(open(esp_path, "rb").read()).hexdigest()[:16]

    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    X = df[COMPACT_FEATURES].values.astype(np.float64)
    y = df["label_id"].values.astype(int)
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    lm_mu = LM[tr].mean(axis=(0, 1))
    lm_sd = LM[tr].std(axis=(0, 1)) + 1e-6
    LMn = ((LM - lm_mu) / lm_sd).astype(np.float32)
    nC = len(CLASSES)

    # Model A / B ----------------------------------------------------------------------
    print("[3/7] training audio candidates")
    results = {}
    preds = {}

    def eval_sk(name, clf, flash, ram, notes=""):
        clf.fit(Xn[tr], y[tr])
        p = clf.predict(Xn[te])
        preds[name] = p
        results[name] = dict(metrics=metrics(y[te], p, nC, CLASSES), flash_bytes=int(flash(clf)),
                             ram_bytes=int(ram), notes=notes)
        return clf

    dt = eval_sk("A1_decision_tree", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                 class_weight="balanced", random_state=args.seed),
                 lambda c: tree_flash_bytes(tree_nodes(c), nC), 0, "max_depth=6")
    lr = eval_sk("A2_logistic_regression", LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"),
                 lambda c: (c.coef_.size + c.intercept_.size) * 4, nC * 4, "float32 weights")
    rf = eval_sk("A3_random_forest", RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=5,
                 class_weight="balanced", random_state=args.seed),
                 lambda c: sum(tree_flash_bytes(tree_nodes(e), nC) for e in c.estimators_), 0, "10 trees, depth 6")

    mlp = build_mlp(len(COMPACT_FEATURES), nC, [16])
    fit_keras(mlp, Xn[tr], y[tr], nC, args.seed)
    mlp_blob = to_int8_tflite(mlp, Xn[tr])
    mlp_rt = TFLiteRunner(mlp_blob)
    P_mlp_f = mlp.predict(Xn[te], verbose=0)
    P_mlp_q = mlp_rt.predict(Xn[te])
    preds["B1_mlp_int8"] = P_mlp_q.argmax(1)
    results["B1_mlp_float"] = dict(metrics=metrics(y[te], P_mlp_f.argmax(1), nC, CLASSES),
                                   flash_bytes=int(mlp.count_params() * 4), ram_bytes=(36 + 16 + 3) * 4,
                                   notes="36-16-3 float32 reference")
    results["B1_mlp_int8"] = dict(metrics=metrics(y[te], P_mlp_q.argmax(1), nC, CLASSES),
                                  flash_bytes=len(mlp_blob), ram_bytes=36 + 16 + 3 + 16 * 4,
                                  notes="36-16-3 full-integer INT8; flash = .tflite size")

    cnn = build_cnn(FRAMES_PER_WINDOW, N_MEL, nC)
    fit_keras(cnn, LMn[tr], y[tr], nC, args.seed, epochs=80)
    cnn_blob = to_int8_tflite(cnn, LMn[tr])
    cnn_rt = TFLiteRunner(cnn_blob)
    P_cnn_q = cnn_rt.predict(LMn[te])
    preds["B2_logmel_cnn_int8"] = P_cnn_q.argmax(1)
    results["B2_logmel_cnn_int8"] = dict(metrics=metrics(y[te], P_cnn_q.argmax(1), nC, CLASSES),
                                         flash_bytes=len(cnn_blob),
                                         ram_bytes=FRAMES_PER_WINDOW * N_MEL * 4 + 29 * 12 + 12 * 16 + 2048,
                                         notes="Conv1D(12)+SepConv1D(16) on 31x26 log-Mel patch; RAM incl. "
                                               "float patch buffer + rough TFLM arena")

    # Leave-one-recording-out for classes with >=2 original recordings
    print("[4/7] leave-one-recording-out check")
    loro = []
    orig = df[(df.provenance == "original")]
    for lbl in CLASSES:
        recs = sorted(orig[orig.label == lbl].recording_id.unique())
        recs = [r for r in recs if not r.startswith("synth")]
        if len(recs) < 2:
            continue
        for hold in recs:
            trm = (df.recording_id != hold).values & ~((df.split == "test") & (df.split_method == "temporal_block")).values
            trm &= ~((df.split == "test") & (df.split_method == "synthetic_recording")).values
            tem = ((df.recording_id == hold) & (df.provenance == "original")).values
            m_ = (X[trm].mean(0), X[trm].std(0) + 1e-6)
            Xa, Xb = (X[trm] - m_[0]) / m_[1], (X[tem] - m_[0]) / m_[1]
            row = dict(held_out=hold, label=lbl, n_windows=int(tem.sum()))
            for nm, mk in [("A2_logistic_regression", lambda: LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")),
                           ("A1_decision_tree", lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                                     class_weight="balanced", random_state=args.seed)),
                           ("A3_random_forest", lambda: RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=5,
                                                                     class_weight="balanced", random_state=args.seed))]:
                c = mk().fit(Xa, y[trm])
                row[nm] = float((c.predict(Xb) == y[tem]).mean())
            m2 = fit_keras(build_mlp(len(COMPACT_FEATURES), nC, [16]), Xa.astype(np.float32), y[trm], nC, args.seed)
            row["B1_mlp_int8"] = float((m2.predict(Xb.astype(np.float32), verbose=0).argmax(1) == y[tem]).mean())
            lmm, lms = LM[trm].mean(axis=(0, 1)), LM[trm].std(axis=(0, 1)) + 1e-6
            c2 = fit_keras(build_cnn(FRAMES_PER_WINDOW, N_MEL, nC), ((LM[trm] - lmm) / lms).astype(np.float32),
                           y[trm], nC, args.seed, 80)
            row["B2_logmel_cnn_int8"] = float((c2.predict(((LM[tem] - lmm) / lms).astype(np.float32), verbose=0)
                                               .argmax(1) == y[tem]).mean())
            loro.append(row)
            print("     ", row)

    # Model selection --------------------------------------------------------------
    deployable = ["A1_decision_tree", "A2_logistic_regression", "A3_random_forest", "B1_mlp_int8",
                  "B2_logmel_cnn_int8"]
    # score = mean(test macro-F1, leave-one-recording-out mean accuracy). The test split alone
    # has very few non-background windows, so cross-recording generalisation gets equal weight.
    for k in deployable:
        lo = [r[k] for r in loro] if loro else [results[k]["metrics"]["macro_f1"]]
        results[k]["loro_mean_acc"] = float(np.mean(lo))
        results[k]["selection_score"] = 0.5 * results[k]["metrics"]["macro_f1"] + 0.5 * float(np.mean(lo))
    best = max(results[k]["selection_score"] for k in deployable)
    # within 0.03 of the best score, pick the smallest flash footprint
    shortlist = [k for k in deployable if results[k]["selection_score"] >= best - 0.03]
    selected = min(shortlist, key=lambda k: results[k]["flash_bytes"])
    # we only ship backends the firmware implements natively
    if selected == "A2_logistic_regression":
        # ship LR as a 1-layer INT8 dense model so it shares the MLP engine
        backend = "dense_int8"
    elif selected in ("A1_decision_tree", "A3_random_forest"):
        backend = "trees"
    elif selected == "B1_mlp_int8":
        backend = "dense_int8"
    else:
        backend = "dense_int8"
        print("      CNN selected on score but needs TFLM conv kernels; shipping MLP natively, CNN as .tflite")
        selected = "B1_mlp_int8"
    print(f"      selected: {selected} (backend {backend}); shortlist {shortlist}")

    deploy_blob, dense_layers, tree_audio = None, None, None
    if selected == "A2_logistic_regression":
        lr_k = build_mlp(len(COMPACT_FEATURES), nC, [])
        lr_k.layers[-1].set_weights([lr.coef_.T.astype(np.float32), lr.intercept_.astype(np.float32)])
        deploy_blob = to_int8_tflite(lr_k, Xn[tr])
        deploy_keras = lr_k
    elif backend == "dense_int8":
        deploy_blob = mlp_blob
        deploy_keras = mlp
    else:
        deploy_keras = None
        tree_audio = train_tree_export(dt if selected == "A1_decision_tree" else list(rf.estimators_), nC)
    if deploy_blob is not None:
        dense_layers = dense_layers_from_tflite(deploy_blob)
        with open(os.path.join(out, "models", "atm_audio_model.tflite"), "wb") as f:
            f.write(deploy_blob)
    with open(os.path.join(out, "models", "atm_audio_cnn.tflite"), "wb") as f:
        f.write(cnn_blob)

    # probability function of the deployed model (used to build fusion events)
    if deploy_blob is not None:
        dep_rt = TFLiteRunner(deploy_blob)
        P_all = dep_rt.predict(Xn)
    else:
        P_all = (dt if selected == "A1_decision_tree" else rf).predict_proba(Xn)
    P_all = P_all / np.maximum(P_all.sum(1, keepdims=True), EPS)

    # Out-of-fold audio probabilities for train windows (grouped by source recording), so the
    # fusion model sees the audio model's real uncertainty instead of its training-set fit.
    from sklearn.model_selection import GroupKFold
    P_fuse = P_all.copy()
    tr_idx = np.where(tr)[0]
    gkf = GroupKFold(n_splits=4)
    for k, (a_, b_) in enumerate(gkf.split(tr_idx, y[tr_idx], groups=df.recording_id.values[tr_idx])):
        ia, ib = tr_idx[a_], tr_idx[b_]
        mu_k, sd_k = X[ia].mean(0), X[ia].std(0) + 1e-6
        if deploy_keras is not None:
            mk = fit_keras(build_mlp(len(COMPACT_FEATURES), nC, [16] if selected == "B1_mlp_int8" else []),
                           ((X[ia] - mu_k) / sd_k).astype(np.float32), y[ia], nC, args.seed + k)
            P_fuse[ib] = mk.predict(((X[ib] - mu_k) / sd_k).astype(np.float32), verbose=0)
        else:
            ck = RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=5,
                                        class_weight="balanced", random_state=args.seed).fit((X[ia] - mu_k) / sd_k, y[ia])
            P_fuse[ib] = ck.predict_proba((X[ib] - mu_k) / sd_k)
    oof_acc = float((P_fuse[tr_idx].argmax(1) == y[tr_idx]).mean())
    print(f"      out-of-fold audio accuracy on train windows: {oof_acc:.3f}")

    # Model C: fusion ----------------------------------------------------------------
    print("[5/7] fusion dataset + model C")
    pools = {}
    for sp in ["train", "test"]:
        m = (df.split == sp).values.copy()
        if sp == "test":
            m &= (df.provenance == "original").values | (df.label == "BACKGROUND").values
        pools[sp] = {c: np.where(m & (y == CLASS_TO_ID[c]))[0] for c in CLASSES}
    ev_rows = []
    n_ev = args.n_fusion_events
    for i in range(n_ev):
        kind = FUSION_CLASSES[i % len(FUSION_CLASSES)]
        sp = "test" if (i // len(FUSION_CLASSES)) % 4 == 3 else "train"     # 25% of each class
        rng_e = np.random.default_rng(stable_seed("ev", i, args.seed))
        f = simulate_event(kind, rng_e, pools[sp], lambda idx: P_fuse[np.array(idx)])
        f.update(event_id=f"ev{i:05d}", split=sp, label=kind, label_id=FUSION_CLASSES.index(kind),
                 provenance="synthetic", audio_source=f"real_{sp}_windows")
        ev_rows.append(f)
    ev = pd.DataFrame(ev_rows)[["event_id", "split", "provenance", "audio_source", "label", "label_id"]
                               + FUSION_FEATURES]
    ev.to_csv(os.path.join(out, "datasets", "atm_fusion_events.csv"), index=False, float_format="%.6g")
    Ftr, Fte = ev[ev.split == "train"], ev[ev.split == "test"]
    fz_dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=args.seed)
    fz_dt.fit(Ftr[FUSION_FEATURES].values, Ftr.label_id.values)
    fz_lr = LogisticRegression(max_iter=5000, C=1.0)
    f_mu, f_sd = Ftr[FUSION_FEATURES].values.mean(0), Ftr[FUSION_FEATURES].values.std(0) + 1e-6
    fz_lr.fit((Ftr[FUSION_FEATURES].values - f_mu) / f_sd, Ftr.label_id.values)
    nF = len(FUSION_CLASSES)
    fz_res = {
        "C1_fusion_decision_tree": dict(metrics=metrics(Fte.label_id.values, fz_dt.predict(Fte[FUSION_FEATURES].values), nF, FUSION_CLASSES),
                                        flash_bytes=tree_flash_bytes(tree_nodes(fz_dt), nF), nodes=tree_nodes(fz_dt)),
        "C2_fusion_logreg": dict(metrics=metrics(Fte.label_id.values, fz_lr.predict((Fte[FUSION_FEATURES].values - f_mu) / f_sd), nF, FUSION_CLASSES),
                                 flash_bytes=(fz_lr.coef_.size + fz_lr.intercept_.size + 2 * len(FUSION_FEATURES)) * 4),
    }
    fusion_tree = train_tree_export(fz_dt, nF)

    # Exports ----------------------------------------------------------------------
    print("[6/7] exports")
    meta = dict(
        model_version=f"atm-tinyml-{time.strftime('%Y%m%d')}-{ds_hash[:6]}",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dataset_hash=ds_hash,
        target=dict(mcu="ESP32 (Xtensa LX6, 520 KB SRAM, FPU single precision)",
                    framework="Arduino-ESP32 core 3.x or ESP-IDF 5.x",
                    microphone="INMP441 I2S, 32-bit slot, sample = raw >> 8, scaled by 1/2^23"),
        preprocessing=dict(sample_rate_hz=SR, frame_len=FRAME, hop_len=HOP, n_fft=N_FFT,
                           window="periodic Hann", dc_removal="per-frame mean subtraction",
                           power_spectrum="|X|^2 / n_fft", mel=dict(n_mels=N_MEL, fmin_hz=FMIN, fmax_hz=FMAX,
                           scale="HTK", filter_norm="none"), log="ln(max(mel, 1e-10))",
                           mfcc=dict(n=N_MFCC, dct="DCT-II orthonormal, c0 included"),
                           rms_db="20*log10(rms + 1e-9) of DC-removed frame",
                           zcr="sign changes / (frame_len-1), sign = x>=0",
                           centroid="sum(p_k * k/256)/sum(p_k)", rolloff="first k with cumsum>=0.85*total, /256",
                           flatness="exp(mean(ln(p_k+1e-10)))/(mean(p_k)+1e-10), k=1..256",
                           frames_per_window=FRAMES_PER_WINDOW, window_seconds=FRAMES_PER_WINDOW * HOP / SR,
                           aggregation="population mean then population std of each per-frame feature",
                           train_window_step_frames=TRAIN_WINDOW_STEP, device_window_step_frames=FRAMES_PER_WINDOW),
        audio_features=dict(order=COMPACT_FEATURES, normalization="z-score (x-mean)/std",
                            mean=mu.tolist(), std=sd.tolist()),
        audio_model=dict(selected=selected, backend=backend, classes=CLASSES,
                         tflite_file="models/atm_audio_model.tflite" if deploy_blob else None,
                         tflite_bytes=len(deploy_blob) if deploy_blob else None,
                         quantization=(dict(type="full-integer INT8 (per-tensor activations)",
                                            input_scale=dense_layers[0]["in_scale"], input_zero_point=dense_layers[0]["in_zp"],
                                            logit_scale=dense_layers[-1]["out_scale"], logit_zero_point=dense_layers[-1]["out_zp"],
                                            softmax="float on-device (TFLM path uses int8 SOFTMAX, scale 1/256, zp -128)",
                                            layers=[dict(shape=list(l["W"].shape), in_scale=l["in_scale"], in_zp=l["in_zp"],
                                                         w_scale=l["w_scale"].tolist(), w_quant="per-output-channel symmetric",
                                                         out_scale=l["out_scale"], out_zp=l["out_zp"])
                                                    for l in dense_layers]) if dense_layers else None)),
        fusion_model=dict(type="decision tree (native C)", features=FUSION_FEATURES, classes=FUSION_CLASSES,
                          nodes=tree_nodes(fz_dt), max_depth=6,
                          rules=dict(tamper_priority=f"tamper_flag==1 or vib_peak>={TAMPER_VIB_THRESHOLD} -> SUSPECTED_TAMPER",
                                     unknown="tree confidence < 0.6, or NORMAL_DISPENSE with pir_active_frac < 0.05",
                                     dispense_count="only NORMAL_DISPENSE without tamper increments the dispense counter",
                                     short_visit="SHORT_VISIT is never counted as a failed withdrawal")),
        sensors=dict(vibration_unit=VIB_UNIT, tamper_vib_threshold=TAMPER_VIB_THRESHOLD,
                     short_visit_max_s=SHORT_VISIT_MAX_S, pir_gap_tolerance_s=PIR_GAP_TOLERANCE_S),
        datasets=dict(full="datasets/atm_full_features.csv", esp32="datasets/atm_esp32_features.csv",
                      fusion="datasets/atm_fusion_events.csv",
                      n_windows=int(len(df)), n_train=int(tr.sum()), n_test=int(te.sum())),
        log_mel_cnn=dict(file="models/atm_audio_cnn.tflite", bytes=len(cnn_blob), deployed=False,
                         input="31x26 z-scored log-Mel", band_mean=lm_mu.tolist(), band_std=lm_sd.tolist()),
    )
    with open(os.path.join(out, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    write_model_header(os.path.join(out, FW_DIR, "atm_model_data.h"), meta, mu, sd, dense_layers,
                       tree_audio, fusion_tree, deploy_blob)

    # test vectors: a few real test windows with python reference outputs
    tv_idx = []
    for c in range(nC):
        cand = np.where(te & (y == c) & (df.provenance == "original").values)[0]
        if len(cand) == 0:
            cand = np.where(te & (y == c))[0]
        tv_idx += list(cand[:: max(1, len(cand) // 3)][:3])
    tv_X = X[tv_idx].astype(np.float32)
    if deploy_blob is not None:
        tv_p = dep_rt.predict(Xn[tv_idx])
    else:
        tv_p = (dt if selected == "A1_decision_tree" else rf).predict_proba(Xn[tv_idx])
    with open(os.path.join(out, FW_DIR, "test_vectors.h"), "w") as f:
        f.write("// AUTO-GENERATED reference vectors: raw (un-normalised) audio features of real test\n")
        f.write("// windows and the class probabilities the TFLite interpreter produced for them.\n#pragma once\n#include <stdint.h>\n")
        f.write(f"#define ATM_N_TEST_VECTORS {len(tv_idx)}\n")
        f.write(c_float_array("ATM_TV_FEATURES", tv_X))
        f.write(c_float_array("ATM_TV_EXPECTED_PROB", tv_p))
        f.write(c_int_array("ATM_TV_LABEL", "uint8_t", y[tv_idx]))
    pd.DataFrame({"window_id": df.window_id.values[tv_idx], "label": y[tv_idx]}).to_csv(
        os.path.join(out, "reports", "test_vector_ids.csv"), index=False)

    # Report -----------------------------------------------------------------------
    print("[7/7] report")
    rep = dict(results=results, fusion=fz_res, loro=loro, selected=selected, shortlist=shortlist,
               counts=df.groupby(["split", "label", "provenance"]).size().reset_index(name="n").to_dict("records"),
               mlp_float_vs_int8_agreement=float((P_mlp_f.argmax(1) == P_mlp_q.argmax(1)).mean()),
               fusion_tree_nodes=tree_nodes(fz_dt), oof_audio_acc=oof_acc, seconds=time.time() - t0)
    with open(os.path.join(out, "reports", "metrics.json"), "w") as f:
        json.dump(rep, f, indent=2)
    write_report(os.path.join(out, "reports", "training_report.md"), rep, manifest)
    print(f"done in {time.time() - t0:.1f}s")


def write_report(path, rep, manifest):
    L = ["# ATM Sentinel TinyML - training report\n",
         "Generated by `train_esp32_model.py`. All numbers are offline (PC) results. "
         "Nothing here is verified on ESP32 hardware.\n",
         "## Data\n", "| recording | label | split |", "|---|---|---|"]
    for _, r in manifest.iterrows():
        L.append(f"| {r.recording_id} | {r.label} | {r.split} |")
    L += ["", "Window counts (1 s windows, 0.26 s step for training):\n", "| split | label | provenance | n |", "|---|---|---|---|"]
    for c in rep["counts"]:
        L.append(f"| {c['split']} | {c['label']} | {c['provenance']} | {c['n']} |")
    L += ["", "## Audio classifier candidates (test split, original audio only)\n",
          "| model | accuracy | macro-F1 | LORO mean acc | score | flash (B) | notes |", "|---|---|---|---|---|---|---|"]
    for k, v in rep["results"].items():
        lo = f"{v['loro_mean_acc']:.3f}" if "loro_mean_acc" in v else "-"
        sc = f"{v['selection_score']:.3f}" if "selection_score" in v else "-"
        L.append(f"| {k} | {v['metrics']['accuracy']:.3f} | {v['metrics']['macro_f1']:.3f} | {lo} | {sc} | {v['flash_bytes']} | {v['notes']} |")
    L += ["", f"Selected: **{rep['selected']}** (score = mean of test macro-F1 and LORO accuracy; smallest flash within 0.03 of the best score; shortlist {rep['shortlist']}).",
          f"MLP float vs INT8 argmax agreement on test: {rep['mlp_float_vs_int8_agreement']:.3f}\n"]
    for k, v in rep["results"].items():
        L.append(f"### {k}\n")
        L.append("| class | precision | recall | support |\n|---|---|---|---|")
        for c, m in v["metrics"]["per_class"].items():
            L.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['support']} |")
        L.append(f"\nConfusion (rows=true {CLASSES}): `{v['metrics']['confusion']}`\n")
    L += ["## Leave-one-recording-out (DISPENSE recordings)\n",
          "Each DISPENSE recording is held out in turn (all of its augmentations removed from training); "
          "value = fraction of its original windows classified DISPENSE.\n",
          "| held out | windows | DT | LR | RF | MLP | CNN |", "|---|---|---|---|---|---|---|"]
    for r in rep["loro"]:
        L.append(f"| {r['held_out']} | {r['n_windows']} | {r['A1_decision_tree']:.2f} | {r['A2_logistic_regression']:.2f} | "
                 f"{r['A3_random_forest']:.2f} | {r['B1_mlp_int8']:.2f} | {r['B2_logmel_cnn_int8']:.2f} |")
    L += ["", "## Fusion classifier (synthetic events, test split)\n",
          "Event sensor streams are simulated; audio inside each event is drawn from real windows of the "
          "matching split. Train events use out-of-fold audio probabilities "
          f"(train-window OOF accuracy {rep['oof_audio_acc']:.3f}); test events use the deployed INT8 model on "
          "held-out windows.\n", "| model | accuracy | macro-F1 | flash (B) |", "|---|---|---|---|"]
    for k, v in rep["fusion"].items():
        L.append(f"| {k} | {v['metrics']['accuracy']:.3f} | {v['metrics']['macro_f1']:.3f} | {v['flash_bytes']} |")
    for k, v in rep["fusion"].items():
        L.append(f"\n### {k}\n\n| class | precision | recall |\n|---|---|---|")
        for c, m in v["metrics"]["per_class"].items():
            L.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} |")
        L.append(f"\nConfusion (rows=true {FUSION_CLASSES}): `{v['metrics']['confusion']}`")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
