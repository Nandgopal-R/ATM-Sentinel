"""Layer 6: turn physical truth into what the hardware would actually report.

Four things happen here, in order, and all four matter:

1. **Per-device calibration bias.** Drawn once per ATM from a seed derived from
   its id, then held for that unit's entire life across every run. Two BME280s
   in the same room do not read the same number. Without this the model can
   memorise absolute values and will fall apart on a unit it has not seen,
   which is precisely the failure mode a fleet deployment hits. It is also why
   ATM identities are kept disjoint across train/validation/test.
2. **Measurement noise** at the datasheet short-term RMS.
3. **Quantisation and range clipping.** A BME280 cannot report 0.0034 degC of
   change and a ZMPT101B cannot report a negative RMS.
4. **Dropouts** as nulls, never as zeros.

Sensor faults follow the Bruijn et al. (2016) taxonomy — stuck, bias, drift,
spike/random, malfunction (variance inflation) — so the implementation has a
published reference to cite and their released fault-injected Intel Lab subsets
can be used to sanity-check it.
"""

from __future__ import annotations

import hashlib

import numpy as np

FLOAT_CHANNELS = ["temperature", "humidity", "pressure", "light", "voltage", "current"]


def device_bias(atm_id: str, cfg, schema) -> dict:
    """Fixed calibration offset for one physical unit, stable across all runs."""
    digest = hashlib.sha256(f"device::{atm_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    spec = cfg["sensors"]["device_bias_sigma"]
    bias = {
        "temperature": rng.normal(0.0, spec["temperature"]),
        "humidity": rng.normal(0.0, spec["humidity"]),
        "pressure": rng.normal(0.0, spec["pressure"]),
        "voltage": rng.normal(0.0, spec["voltage"]),
        "current": rng.normal(0.0, spec["current"]),
        # The VEML7700's error is multiplicative (gain calibration), not additive.
        "light_gain": 1.0 + rng.normal(0.0, spec["light_pct"] / 100.0),
    }
    return bias


def _apply_sensor_fault(values, fault, sigma, n, rng):
    kind = fault["kind"]
    onset = fault["onset"]
    out = values.copy()

    if kind == "SENSOR_STUCK":
        out[onset:] = out[onset]
        nr = fault.get("null_rate", 0.0)
        if nr > 0:
            mask = rng.random(n) < nr
            mask[:onset] = False
            out[mask] = np.nan

    elif kind == "SENSOR_BIAS":
        out[onset:] += fault["sigmas"] * sigma

    elif kind == "SENSOR_DRIFT":
        ramp = np.zeros(n)
        tail = n - onset
        if tail > 1:
            ramp[onset:] = np.linspace(0.0, 1.0, tail)
        out += ramp * fault["sigmas"] * sigma

    elif kind == "SENSOR_SPIKE":
        mask = rng.random(n) < fault["rate"]
        mask[:onset] = False
        signs = rng.choice([-1.0, 1.0], size=n)
        out[mask] += signs[mask] * fault["sigmas"] * sigma * rng.uniform(0.5, 1.5, mask.sum())

    elif kind == "SENSOR_NOISE":
        extra = rng.normal(0.0, sigma * fault["inflation"], n)
        extra[:onset] = 0.0
        out += extra

    return out


def emulate(truth: dict, atm_id: str, cfg, schema, sensor_fault, rng) -> dict:
    """truth: physical values. Returns the observed telemetry channels."""
    fields = schema["fields"]
    bias = device_bias(atm_id, cfg, schema)
    n = len(truth["temperature"])
    dropout_rate = cfg["sensors"]["dropout_rate"]
    out = {}

    for ch in FLOAT_CHANNELS:
        spec = fields[ch]
        sigma = spec["noise_sigma"]
        v = np.asarray(truth[ch], dtype=float)

        if ch == "light":
            v = v * bias["light_gain"]
        else:
            v = v + bias[ch]

        v = v + rng.normal(0.0, sigma, n)

        if sensor_fault is not None and sensor_fault["channel"] == ch:
            v = _apply_sensor_fault(v, sensor_fault, sigma, n, rng)

        lo, hi = spec["range"]
        v = np.clip(v, lo, hi)
        res = spec["resolution"]
        v = np.round(v / res) * res

        drop = rng.random(n) < dropout_rate
        v[drop] = np.nan
        out[ch] = v

    vib = np.asarray(truth["vibration"], dtype=np.int32)
    out["vibration"] = np.clip(vib, *fields["vibration"]["range"]).astype("float64")
    out["motion"] = np.asarray(truth["motion"], dtype=float)

    # Digital lines drop out too, though much more rarely than an I2C read.
    for ch in ("vibration", "motion"):
        drop = rng.random(n) < dropout_rate * 0.3
        out[ch][drop] = np.nan

    return out
