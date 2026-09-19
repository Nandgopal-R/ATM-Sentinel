"""Psychrometrics.

This module is the single most important piece of physics in the simulator.

A cabinet exchanges air with the lobby, so its *absolute* humidity (mixing
ratio, g water / kg dry air) tracks the lobby's. Relative humidity is then a
derived quantity: RH = e / e_s(T). When the cabinet heats up, e_s(T) rises
while e stays put, so **RH falls**.

Consequence for the ML problem:

    COOLING_FAILURE  ->  T up,   RH down,  absolute humidity flat
    HIGH_HUMIDITY    ->  T flat, RH up,    absolute humidity up

If we generated RH directly from a Gaussian, both faults would look like
"humidity moved" and Stage-2 would separate them on an artefact that does not
exist in a real cabinet. Getting this right is what makes the confusion matrix
mean something.

Magnus-Tetens coefficients over water, Alduchov & Eskridge (1996) revision:
    e_s(T) = 6.1094 * exp(17.625 T / (243.04 + T))   [hPa, T in degC]
Maximum relative error is under 0.4% over -40..50 degC.
"""

from __future__ import annotations

import numpy as np

MAGNUS_A = 6.1094  # hPa
MAGNUS_B = 17.625
MAGNUS_C = 243.04  # degC

# Ratio of molar masses of water vapour and dry air, x1000 for g/kg.
EPSILON_G_PER_KG = 621.97


def saturation_vapour_pressure(temp_c):
    """Saturation vapour pressure e_s in hPa."""
    t = np.asarray(temp_c, dtype=float)
    return MAGNUS_A * np.exp(MAGNUS_B * t / (MAGNUS_C + t))


def vapour_pressure(temp_c, rh_pct):
    """Actual vapour pressure e in hPa."""
    return np.asarray(rh_pct, dtype=float) / 100.0 * saturation_vapour_pressure(temp_c)


def mixing_ratio(temp_c, rh_pct, pressure_hpa):
    """Absolute humidity as mixing ratio w, in g/kg dry air."""
    e = vapour_pressure(temp_c, rh_pct)
    p = np.asarray(pressure_hpa, dtype=float)
    denom = np.maximum(p - e, 1e-6)
    return EPSILON_G_PER_KG * e / denom


def rh_from_mixing_ratio(temp_c, w_g_per_kg, pressure_hpa):
    """Invert: relative humidity (%) implied by a mixing ratio at temperature T."""
    w = np.asarray(w_g_per_kg, dtype=float)
    p = np.asarray(pressure_hpa, dtype=float)
    e = w * p / (EPSILON_G_PER_KG + w)
    rh = 100.0 * e / saturation_vapour_pressure(temp_c)
    return np.clip(rh, 0.0, 100.0)


def dewpoint(temp_c, rh_pct):
    """Dew point in degC (Magnus inverse). Useful as a derived ML feature."""
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.1, 100.0)
    t = np.asarray(temp_c, dtype=float)
    gamma = np.log(rh / 100.0) + MAGNUS_B * t / (MAGNUS_C + t)
    return MAGNUS_C * gamma / (MAGNUS_B - gamma)
