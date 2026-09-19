"""Layer 4: mains supply and the DC auxiliary rail.

Two separate circuits, coupled through a switch-mode power supply:

    ZMPT101B  ->  `voltage`  ->  AC mains RMS, nominal 230 V
    INA219    ->  `current`  ->  12 V auxiliary rail (controller + cooling fan)

The coupling is what makes the power classes learnable in a non-trivial way:

  * A mains sag ABOVE the SMPS dropout voltage is invisible on `current`. The
    supply regulates the output rail. POWER_UNDERVOLTAGE is therefore a
    genuinely single-channel fault, which is the honest physics and makes it
    harder than it looks.
  * Below dropout the rail droops and, because the load is roughly constant
    power, `current` RISES as `voltage` falls.
  * In a full outage the node runs on its backup and reports a near-zero mains
    reading with a collapsed rail current, while the fan stops and the cabinet
    starts heating. That cascade across three channels is the kind of
    correlation a real deployment shows and a row-wise random generator cannot.

Fan current carries the cooling story:

    COOLING_DEGRADATION -> bearing friction / dust loading: the fan draws MORE
                           current while moving less air. Current rises BEFORE
                           temperature moves. This is the precursor.
    FAN_STALL           -> the fan stops: current DROPS.

Both raise temperature. The current channel is what separates them.
"""

from __future__ import annotations

import numpy as np

from .environment import ou_process


def baseline_mains(cfg, n, rng, dt=1.0):
    """Healthy mains: slow grid wander, fast noise, occasional load sags."""
    p = cfg["power"]
    v = (
        p["mains_nominal_v"]
        + ou_process(n, p["mains_drift_sigma_v"], p["mains_drift_tau_s"], rng, dt)
        + rng.normal(0.0, p["mains_noise_sigma_v"], n)
    )

    # Background sags from other loads on the same distribution transformer.
    # These occur in HEALTHY runs too, which is exactly the point: they are the
    # nuisance events a naive voltage threshold would flag.
    hours = n * dt / 3600.0
    n_sags = rng.poisson(p["sag_rate_per_hour"] * hours)
    for _ in range(int(n_sags)):
        start = rng.integers(0, n)
        dur = int(rng.uniform(*p["sag_duration_s"]) / dt)
        depth = rng.uniform(*p["sag_depth_v"])
        end = min(n, start + dur)
        if end <= start:
            continue
        # Sag then recover with a short ramp, not a rectangular notch.
        profile = np.ones(end - start)
        ramp = max(1, (end - start) // 5)
        profile[:ramp] = np.linspace(0.0, 1.0, ramp)
        profile[-ramp:] = np.linspace(1.0, 0.0, ramp)
        v[start:end] -= depth * profile
    return v


def aux_rail_current(mains_v, cooling_health, fan_running, cfg, rng):
    """INA219 reading on the 12 V auxiliary rail."""
    p = cfg["power"]
    n = len(mains_v)

    fan_w = np.where(
        fan_running,
        p["p_fan_nominal_w"] * (1.0 + p["fan_strain_beta"] * (1.0 - cooling_health)),
        0.0,
    )
    load_w = p["p_aux_base_w"] + fan_w

    nominal_rail = p["rail_voltage_v"]
    dropout = p["smps_dropout_v"]
    rail_v = np.where(
        mains_v >= dropout, nominal_rail, nominal_rail * np.maximum(mains_v, 0.0) / dropout
    )
    collapsed = rail_v < 0.5 * nominal_rail

    current = np.where(
        collapsed,
        p["p_node_only_w"] / nominal_rail,       # running on backup
        load_w / np.maximum(rail_v, 1e-3),
    )
    # Load jitter: the controller's duty cycle is not perfectly flat.
    current = current * (1.0 + rng.normal(0.0, 0.004, n))
    return current, rail_v, collapsed
