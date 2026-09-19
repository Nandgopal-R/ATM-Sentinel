"""Layer 1-2: outdoor climate and the air-conditioned ATM lobby.

Generic Indian metro. The outdoor profile matters even though the ATM is
indoors: the cabinet inherits its *absolute* humidity from the lobby, which
inherits it from outdoors through infiltration minus whatever the AC removes.
The AC controls temperature; it does not control the moisture baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from . import psychro

SECONDS_PER_DAY = 86400.0


def ou_process(n, sigma, tau_s, rng, dt=1.0):
    """Ornstein-Uhlenbeck process, stationary std = sigma, correlation time tau.

    Implemented as a first-order IIR filter so it is vectorised rather than
    looped. a = exp(-dt/tau); innovation std chosen to make the stationary
    variance exactly sigma^2.
    """
    a = float(np.exp(-dt / tau_s))
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - a * a), size=n)
    innov[0] = rng.normal(0.0, sigma)  # start in the stationary distribution
    return lfilter([1.0], [1.0, -a], innov)


@dataclass
class Environment:
    hour_of_day: np.ndarray      # local hour, float, wraps at 24
    outdoor_temp_c: np.ndarray
    outdoor_rh_pct: np.ndarray
    pressure_hpa: np.ndarray
    lobby_temp_c: np.ndarray
    lobby_w_g_per_kg: np.ndarray  # lobby absolute humidity (mixing ratio)
    ambient_lux: np.ndarray       # lobby lighting before occupancy shadowing


def build_environment(cfg, n, start_hour, rng, dt=1.0):
    env_cfg = cfg["environment"]
    lob_cfg = cfg["lobby"]

    t = np.arange(n, dtype=float) * dt
    hour = (start_hour + t / 3600.0) % 24.0

    # --- outdoor temperature: diurnal sinusoid peaking mid-afternoon --------
    phase = 2.0 * np.pi * (hour - env_cfg["outdoor_temp_peak_hour"]) / 24.0
    outdoor_temp = (
        env_cfg["outdoor_temp_mean_c"]
        + env_cfg["outdoor_temp_amplitude_c"] * np.cos(phase)
        + ou_process(n, 0.45, 5400.0, rng, dt)
        + rng.normal(0.0, 0.03, n)
    )

    # --- pressure: semidiurnal atmospheric tide + slow synoptic wander ------
    # The S2 tide peaks near 10:00 and 22:00 local time and is a real, citable
    # feature of barometric records. It also gives the pressure channel enough
    # structure that a stuck-sensor fault is genuinely detectable.
    tide = env_cfg["pressure_semidiurnal_amp_hpa"] * np.cos(
        2.0 * np.pi * (hour - 10.0) / 12.0
    )
    pressure = (
        env_cfg["pressure_mean_hpa"]
        + tide
        + ou_process(
            n,
            env_cfg["pressure_synoptic_sigma_hpa"],
            env_cfg["pressure_synoptic_tau_s"],
            rng,
            dt,
        )
    )

    # --- outdoor RH: anti-phase with temperature ---------------------------
    outdoor_rh = np.clip(
        env_cfg["outdoor_rh_mean_pct"]
        - env_cfg["outdoor_rh_amplitude_pct"] * np.cos(phase)
        + ou_process(n, 2.0, 5400.0, rng, dt),
        8.0,
        99.0,
    )

    # --- lobby: AC hysteresis cycling + infiltration of the outdoor swing ---
    # A split AC does not hold a setpoint; it cycles across a deadband. That
    # cycling is a real, periodic signal the model will see in temperature_std
    # and it is why a naive "temperature is unstable" rule produces false
    # positives.
    cycle = np.sin(2.0 * np.pi * t / lob_cfg["ac_cycle_period_s"] + rng.uniform(0, 2 * np.pi))
    # Triangle-ish rather than pure sine: compressor on/off is not sinusoidal.
    cycle = np.arcsin(np.clip(cycle, -1, 1)) * (2.0 / np.pi)
    lobby_temp = (
        lob_cfg["ac_setpoint_c"]
        + lob_cfg["ac_hysteresis_c"] * cycle
        + lob_cfg["infiltration_gain"] * (outdoor_temp - env_cfg["outdoor_temp_mean_c"])
        + ou_process(n, 0.12, 900.0, rng, dt)
    )

    # --- lobby absolute humidity -------------------------------------------
    w_out = psychro.mixing_ratio(outdoor_temp, outdoor_rh, pressure)
    lobby_w = w_out * lob_cfg["ac_dehumid_factor"]

    # --- lighting: 24/7 artificial + daylight through the glazing ----------
    daylight = np.zeros(n)
    day_mask = (hour > 6.0) & (hour < 18.5)
    span = np.clip((hour[day_mask] - 6.0) / 12.5, 0.0, 1.0)
    daylight[day_mask] = lob_cfg["daylight_peak_lux"] * np.sin(np.pi * span) ** 1.6
    ambient_lux = (
        lob_cfg["lighting_lux"]
        + daylight
        + rng.normal(0.0, lob_cfg["lighting_jitter_sigma"], n)
    )

    return Environment(
        hour_of_day=hour,
        outdoor_temp_c=outdoor_temp,
        outdoor_rh_pct=outdoor_rh,
        pressure_hpa=pressure,
        lobby_temp_c=lobby_temp,
        lobby_w_g_per_kg=lobby_w,
        ambient_lux=ambient_lux,
    )
