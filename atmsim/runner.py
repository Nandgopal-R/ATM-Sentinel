"""Orchestrates one simulation run: environment -> physics -> events -> sensors.

Output is a DataFrame holding exactly the canonical schema columns, plus a
sidecar ground-truth record. The truth columns (`state`, severity, onset) are
kept OUT of the telemetry frame and written separately, so the raw stream is
byte-for-byte the shape an ESP32 will publish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import events as ev_mod
from . import power, psychro, scenarios, sensors, thermal
from .environment import build_environment, ou_process

# 2026-10-01 00:00:00 IST, as a UTC epoch. Runs are spaced a day apart so
# timestamps are globally unique and sortable.
BASE_EPOCH_UTC = 1790793000
IST_OFFSET_S = 19800


def simulate_run(spec, cfg, schema):
    rng = np.random.default_rng(spec.seed)
    dt = 1.0 / cfg["run"]["sample_rate_hz"]
    n = int(cfg["run"]["hours"] * 3600.0 / dt)

    env = build_environment(cfg, n, spec.start_hour, rng, dt)
    events = ev_mod.build_events(cfg, n, env.hour_of_day, rng, dt)
    scen = scenarios.build_scenario(spec.subtype, cfg, n, env.hour_of_day, rng, dt)

    # ----------------------------------------------------------- thermal ----
    th = cfg["thermal"]
    q_elec = (
        th["q_electronics_w"]
        + ou_process(n, th["q_electronics_jitter_w"], 600.0, rng, dt)
        + events.q_transaction_w
    )
    cabinet_temp = thermal.integrate_cabinet_temperature(
        env.lobby_temp_c, q_elec, scen.cooling_health, cfg, dt
    )
    state, failure_idx = scenarios.refine_cooling_states(
        scen.state, spec.subtype, cabinet_temp, cfg
    )

    # ---------------------------------------------------------- humidity ----
    cabinet_w = thermal.lag_mixing_ratio(
        env.lobby_w_g_per_kg, scen.ah_multiplier, th["humidity_tau_s"], dt
    )
    cabinet_rh = psychro.rh_from_mixing_ratio(cabinet_temp, cabinet_w, env.pressure_hpa)

    # ------------------------------------------------------------- power ----
    mains = power.baseline_mains(cfg, n, rng, dt) + scen.mains_delta
    override = ~np.isnan(scen.mains_override)
    mains[override] = scen.mains_override[override]
    mains = np.clip(mains, 0.0, 300.0)
    current, _rail_v, collapsed = power.aux_rail_current(
        mains, scen.cooling_health, scen.fan_running, cfg, rng
    )

    # ------------------------------------------------------------- light ----
    light = env.ambient_lux + cfg["lobby"]["occupancy_shadow_lux"] * events.presence
    # A mains failure also takes out the lobby lighting.
    light = np.where(collapsed, light * 0.06, light)
    light = np.maximum(light, 0.0)

    # ------------------------------------------------- discrete channels ----
    vibration = events.vibration + scen.extra_vibration
    motion = ev_mod.latch(
        events.presence | scen.extra_motion, int(cfg["events"]["pir_hold_s"] / dt)
    ).astype(np.int8)

    truth = {
        "temperature": cabinet_temp,
        "humidity": cabinet_rh,
        "pressure": env.pressure_hpa,
        "light": light,
        "voltage": mains,
        "current": current,
        "vibration": vibration,
        "motion": motion,
    }
    obs = sensors.emulate(truth, spec.atm_id, cfg, schema, scen.sensor_fault, rng)

    t0 = BASE_EPOCH_UTC + spec.run_index * 86400 + int((spec.start_hour) * 3600) - IST_OFFSET_S
    timestamps = t0 + (np.arange(n) * dt).astype(np.int64)

    telemetry = pd.DataFrame(
        {
            "atm_id": spec.atm_id,
            "timestamp": timestamps,
            "temperature": obs["temperature"],
            "humidity": obs["humidity"],
            "pressure": obs["pressure"],
            "light": obs["light"],
            "voltage": obs["voltage"],
            "current": obs["current"],
            "vibration": obs["vibration"],
            "motion": obs["motion"],
        }
    )

    truth_frame = pd.DataFrame({"state": state.astype(str)})

    onset_ts = int(timestamps[scen.onset_idx]) if scen.onset_idx is not None else None
    failure_ts = int(timestamps[failure_idx]) if failure_idx is not None else None
    record = {
        "run_id": spec.run_id,
        "atm_id": spec.atm_id,
        "split": spec.split,
        "stage2_class": spec.stage2_class,
        "subtype": spec.subtype,
        "start_hour_ist": round(spec.start_hour, 3),
        "t_start": int(timestamps[0]),
        "t_end": int(timestamps[-1]),
        "t_onset": onset_ts,
        "t_failure": failure_ts,
        "severity": round(float(scen.severity), 4),
        "n_transactions": events.n_transactions,
        "peak_temperature_c": round(float(np.nanmax(cabinet_temp)), 2),
        "min_voltage_v": round(float(np.nanmin(mains)), 2),
        "fault_channel": (scen.sensor_fault or {}).get("channel"),
        "params": scen.params,
        "seed": spec.seed,
    }
    return telemetry, truth_frame, record
