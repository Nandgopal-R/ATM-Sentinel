"""Fault injection with ground truth.

Each scenario produces *modifier arrays* that the runner feeds into the physics
layers, plus a state timeline. Nothing here writes sensor values directly — a
fault expresses itself only through the physics, which is why the cross-channel
signatures come out coherent.

Fine-grained simulator states and how they surface:

  COOLING_DEGRADATION  cooling health decays over 30-90 min; fan current rises
                       first (strain), cabinet temperature follows on a ~43 min
                       time constant, RH falls as a consequence of the physics.
                       Relabelled COOLING_FAILURE once the cabinet crosses the
                       failure threshold, so the run contains the precursor and
                       the failure as distinct ground-truth phases.
  FAN_STALL            cooling health collapses in seconds and fan current DROPS.
                       Same temperature outcome, opposite current signature.
  HIGH_HUMIDITY        absolute humidity multiplier ramps up (seal failure,
                       monsoon ingress) at flat temperature.
  POWER_UNDERVOLTAGE   mains ramps down to 185-207 V, above SMPS dropout.
  POWER_OVERVOLTAGE    mains ramps up to 253-271 V.
  POWER_INSTABILITY    dense random sags; the signature is voltage_std, not mean.
  POWER_OUTAGE         mains collapses; fan stops; cabinet begins to heat.
  TAMPER               dense SW-420 bursts, biased toward low-footfall hours,
                       with PIR activity that does not match a transaction shape.
  SENSOR_*             stuck / bias / drift / spike / noise on one channel,
                       applied at the sensor-emulation stage only.

Severity is recorded per run so a "train on mild, test on severe" split is
available later. Reporting 99% accuracy on faults you injected yourself is only
meaningful if you can show the model generalises across severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STAGE2_SUBTYPES = {
    "NORMAL": ["NORMAL"],
    "COOLING_FAILURE": ["COOLING_DEGRADATION", "FAN_STALL"],
    "HIGH_HUMIDITY": ["HIGH_HUMIDITY"],
    "POWER_FAILURE": [
        "POWER_UNDERVOLTAGE",
        "POWER_OVERVOLTAGE",
        "POWER_INSTABILITY",
        "POWER_OUTAGE",
    ],
    "TAMPER": ["TAMPER"],
    "SENSOR_FAULT": [
        "SENSOR_STUCK",
        "SENSOR_BIAS",
        "SENSOR_DRIFT",
        "SENSOR_SPIKE",
        "SENSOR_NOISE",
    ],
}

SUBTYPE_WEIGHTS = {"COOLING_DEGRADATION": 0.65, "FAN_STALL": 0.35}


@dataclass
class ScenarioResult:
    state: np.ndarray                      # fine-grained ground truth, per sample
    cooling_health: np.ndarray
    fan_running: np.ndarray
    ah_multiplier: np.ndarray
    mains_delta: np.ndarray
    mains_override: np.ndarray             # NaN where no override
    extra_vibration: np.ndarray
    extra_motion: np.ndarray
    sensor_fault: dict | None = None
    onset_idx: int | None = None
    severity: float = 0.0
    params: dict = field(default_factory=dict)


def _ramp(n, onset, develop, dt):
    """Logistic 0 -> 1 transition starting at `onset`, completing over `develop`."""
    x = (np.arange(n) - onset) * dt / max(develop, 1e-6)
    s = 1.0 / (1.0 + np.exp(-9.0 * np.clip(x - 0.5, -50.0, 50.0)))
    s[x < 0.0] = 0.0
    s[x > 1.0] = 1.0
    return s


def _blank(n):
    return ScenarioResult(
        state=np.full(n, "NORMAL", dtype=object),
        cooling_health=np.ones(n),
        fan_running=np.ones(n, dtype=bool),
        ah_multiplier=np.ones(n),
        mains_delta=np.zeros(n),
        mains_override=np.full(n, np.nan),
        extra_vibration=np.zeros(n, dtype=np.int32),
        extra_motion=np.zeros(n, dtype=bool),
    )


def choose_subtype(stage2_class, rng):
    options = STAGE2_SUBTYPES[stage2_class]
    if stage2_class == "COOLING_FAILURE":
        w = np.array([SUBTYPE_WEIGHTS[o] for o in options])
        return str(rng.choice(options, p=w / w.sum()))
    return str(rng.choice(options))


def build_scenario(subtype, cfg, n, hour_of_day, rng, dt=1.0) -> ScenarioResult:
    res = _blank(n)
    if subtype == "NORMAL":
        return res

    sc = cfg["scenarios"]
    lo, hi = sc["onset_fraction_range"]
    onset = int(n * rng.uniform(lo, hi))
    res.onset_idx = onset

    if subtype == "COOLING_DEGRADATION":
        s = sc["COOLING_DEGRADATION"]
        develop = rng.uniform(*s["develop_s"])
        floor = rng.uniform(*s["health_floor"])
        res.cooling_health = 1.0 - (1.0 - floor) * _ramp(n, onset, develop, dt)
        res.state[onset:] = "COOLING_DEGRADATION"   # refined later using temperature
        res.severity = 1.0 - floor
        res.params = {"develop_s": develop, "health_floor": floor}

    elif subtype == "FAN_STALL":
        s = sc["FAN_STALL"]
        develop = rng.uniform(*s["develop_s"])
        floor = rng.uniform(*s["health_floor"])
        res.cooling_health = 1.0 - (1.0 - floor) * _ramp(n, onset, develop, dt)
        res.fan_running[onset + int(develop / dt) :] = False
        res.state[onset:] = "FAN_STALL"
        res.severity = 1.0 - floor
        res.params = {"develop_s": develop, "health_floor": floor}

    elif subtype == "HIGH_HUMIDITY":
        s = sc["HIGH_HUMIDITY"]
        develop = rng.uniform(*s["develop_s"])
        mult = rng.uniform(*s["ah_multiplier"])
        res.ah_multiplier = 1.0 + (mult - 1.0) * _ramp(n, onset, develop, dt)
        res.state[onset:] = "HIGH_HUMIDITY"
        res.severity = mult - 1.0
        res.params = {"develop_s": develop, "ah_multiplier": mult}

    elif subtype in ("POWER_UNDERVOLTAGE", "POWER_OVERVOLTAGE"):
        s = sc[subtype]
        develop = rng.uniform(*s["develop_s"])
        target = rng.uniform(*s["target_v"])
        delta = target - cfg["power"]["mains_nominal_v"]
        res.mains_delta = delta * _ramp(n, onset, develop, dt)
        res.state[onset:] = subtype
        res.severity = abs(delta)
        res.params = {"develop_s": develop, "target_v": target}

    elif subtype == "POWER_INSTABILITY":
        s = sc["POWER_INSTABILITY"]
        rate = rng.uniform(*s["event_rate_per_hour"])
        span = (n - onset) * dt / 3600.0
        n_ev = int(rng.poisson(rate * span))
        depths = []
        for _ in range(n_ev):
            start = int(rng.integers(onset, n))
            dur = max(1, int(rng.uniform(*s["duration_s"]) / dt))
            depth = rng.uniform(*s["depth_v"])
            depths.append(depth)
            end = min(n, start + dur)
            sign = -1.0 if rng.random() < 0.8 else 1.0   # mostly sags, some swells
            res.mains_delta[start:end] += sign * depth
        res.state[onset:] = "POWER_INSTABILITY"
        res.severity = float(np.mean(depths)) if depths else 0.0
        res.params = {"event_rate_per_hour": rate, "n_events": n_ev}

    elif subtype == "POWER_OUTAGE":
        s = sc["POWER_OUTAGE"]
        dur = int(rng.uniform(*s["outage_duration_s"]) / dt)
        end = min(n, onset + dur)
        res.mains_override[onset:end] = rng.uniform(0.0, 3.0)
        res.fan_running[onset:end] = False
        res.cooling_health[onset:end] = 0.0
        res.state[onset:end] = "POWER_OUTAGE"
        res.severity = float(dur * dt)
        res.params = {"outage_duration_s": dur * dt}

    elif subtype == "TAMPER":
        s = sc["TAMPER"]
        # Several incidents per run. This is deliberate incident density so the
        # rarest class yields enough windows; it is recorded in the manifest
        # rather than hidden.
        n_episodes = int(rng.integers(*s["episodes"]))
        intens = []
        for _ in range(n_episodes):
            if s.get("prefer_low_footfall", True):
                cands = np.flatnonzero((hour_of_day < 5.5) | (hour_of_day > 22.5))
                ep_start = int(rng.choice(cands)) if len(cands) > 60 else int(rng.integers(0, n))
            else:
                ep_start = int(rng.integers(0, n))
            cursor = ep_start
            n_bursts = int(rng.integers(*s["burst_count"]))
            for _b in range(n_bursts):
                bdur = max(1, int(rng.uniform(*s["burst_duration_s"]) / dt))
                lam = rng.uniform(*s["intensity_counts_per_s"])
                intens.append(lam)
                bend = min(n, cursor + bdur)
                if bend <= cursor:
                    break
                res.extra_vibration[cursor:bend] += rng.poisson(lam, bend - cursor).astype(
                    np.int32
                )
                res.extra_motion[cursor:bend] = True
                cursor = bend + max(1, int(rng.uniform(*s["burst_gap_s"]) / dt))
                if cursor >= n:
                    break
            ep_end = min(n, cursor)
            res.state[ep_start:ep_end] = "TAMPER"   # the whole incident envelope
        res.severity = float(np.mean(intens)) if intens else 0.0
        res.params = {"n_episodes": n_episodes}

    elif subtype.startswith("SENSOR_"):
        s = sc["SENSOR_FAULT"]
        channel = str(rng.choice(s["channels"]))
        spec = {"kind": subtype, "channel": channel, "onset": onset}
        if subtype == "SENSOR_BIAS":
            spec["sigmas"] = rng.uniform(*s["bias_sigma_multiple"]) * rng.choice([-1, 1])
            res.severity = abs(spec["sigmas"])
        elif subtype == "SENSOR_DRIFT":
            spec["sigmas"] = rng.uniform(*s["drift_total_sigma_multiple"]) * rng.choice([-1, 1])
            res.severity = abs(spec["sigmas"])
        elif subtype == "SENSOR_SPIKE":
            spec["rate"] = rng.uniform(*s["spike_rate_per_s"])
            spec["sigmas"] = rng.uniform(*s["spike_sigma_multiple"])
            res.severity = spec["rate"] * spec["sigmas"]
        elif subtype == "SENSOR_NOISE":
            spec["inflation"] = rng.uniform(*s["noise_inflation"])
            res.severity = spec["inflation"]
        elif subtype == "SENSOR_STUCK":
            spec["null_rate"] = rng.uniform(*s["null_rate"])
            res.severity = 1.0
        res.sensor_fault = spec
        res.state[onset:] = subtype
        res.params = {k: v for k, v in spec.items() if k != "onset"}

    else:
        raise ValueError(f"unknown subtype {subtype}")

    return res


def refine_cooling_states(state, subtype, cabinet_temp, cfg):
    """Split COOLING_DEGRADATION into precursor and failed phases.

    The threshold crossing is the ground-truth 'failure' instant used for the
    detection-lead-time metric.
    """
    if subtype != "COOLING_DEGRADATION":
        return state, None
    thr = cfg["scenarios"]["COOLING_DEGRADATION"]["failure_threshold_c"]
    over = np.flatnonzero((cabinet_temp >= thr) & (state == "COOLING_DEGRADATION"))
    if len(over) == 0:
        return state, None
    first = int(over[0])
    state = state.copy()
    state[first:][state[first:] == "COOLING_DEGRADATION"] = "COOLING_FAILURE"
    return state, first
