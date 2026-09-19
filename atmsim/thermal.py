"""Layer 3: lumped cabinet thermal model.

    C dT/dt = (T_lobby - T)/R + Q_elec - k_cool * health * (T - T_lobby)

Rearranged, this is a first-order lag toward an equilibrium:

    G(t)    = 1/R + k_cool * health(t)          effective conductance, W/K
    T_eq(t) = T_lobby(t) + Q_elec(t) / G(t)
    tau(t)  = C / G(t)
    T(t+dt) = T_eq + (T(t) - T_eq) * exp(-dt / tau)

The exact exponential update is used instead of Euler so the integration stays
correct even when tau changes abruptly (FAN_STALL).

Why this matters more than a linear ramp: a real cooling fault does not make
temperature climb at a constant rate. Losing cooling capacity raises T_eq AND
lengthens tau, so the cabinet makes a decelerating exponential approach to a
new, higher equilibrium. With the default constants:

    health = 1.0  ->  +4.1 K over lobby,  tau ~  8 min
    health = 0.0  ->  +22  K over lobby,  tau ~ 43 min

The slow tau is exactly why early detection is a real problem rather than a
threshold check, and it is the physical basis of the predictive-maintenance
claim in the report.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def integrate_cabinet_temperature(
    lobby_temp_c: np.ndarray,
    q_elec_w: np.ndarray,
    cooling_health: np.ndarray,
    cfg,
    dt: float = 1.0,
) -> np.ndarray:
    th = cfg["thermal"]
    C = th["heat_capacity_j_per_k"]
    g_passive = th["passive_conductance_w_per_k"]
    k_cool = th["cooling_conductance_w_per_k"]

    g = g_passive + k_cool * cooling_health
    t_eq = lobby_temp_c + q_elec_w / g
    decay = np.exp(-dt * g / C)

    n = len(lobby_temp_c)
    temp = np.empty(n, dtype=float)
    temp[0] = t_eq[0]  # start at steady state, not at an arbitrary value
    for i in range(1, n):
        temp[i] = t_eq[i] + (temp[i - 1] - t_eq[i]) * decay[i]
    return temp


def lag_mixing_ratio(
    lobby_w: np.ndarray, ingress_multiplier: np.ndarray, tau_s: float, dt: float = 1.0
) -> np.ndarray:
    """Cabinet absolute humidity follows the lobby with a first-order lag.

    tau is constant here, so this is a plain IIR filter and needs no loop.
    ``ingress_multiplier`` is the HIGH_HUMIDITY driver (water ingress, a failed
    door seal in monsoon) applied to the target before the lag.
    """
    target = lobby_w * ingress_multiplier
    a = float(np.exp(-dt / tau_s))
    return lfilter([1.0 - a], [1.0, -a], target, zi=[target[0] * a])[0]


def steady_state_rise(cfg, health: float) -> float:
    """Helper used by the tests and the calibration notes."""
    th = cfg["thermal"]
    g = th["passive_conductance_w_per_k"] + th["cooling_conductance_w_per_k"] * health
    return th["q_electronics_w"] / g


def time_constant_s(cfg, health: float) -> float:
    th = cfg["thermal"]
    g = th["passive_conductance_w_per_k"] + th["cooling_conductance_w_per_k"] * health
    return th["heat_capacity_j_per_k"] / g
