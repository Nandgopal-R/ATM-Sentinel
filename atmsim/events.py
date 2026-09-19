"""Layer 5: discrete event processes — people, transactions, vibration.

Footfall is a non-homogeneous Poisson process driven by an hourly rate curve.
Everything else hangs off it:

  * HC-SR501 output latches HIGH for a hold time after each detection and
    retriggers while a person stays in front of the fascia. Motion samples are
    therefore strongly autocorrelated by construction. Drawing independent
    Bernoulli 0/1s would make `motion_event_count` trivially separable from
    tamper, which it should not be.
  * The SW-420 is a normally-closed reed/spring module. At rest its contact is
    closed; vibration chatters it open. Firmware counts falling edges with a
    5 ms debounce and reports the per-second count. It is a rate, not an
    amplitude. Background counts come from building traffic and passing
    vehicles and are present in HEALTHY runs, which is what makes TAMPER a
    detection problem rather than a `vibration > 0` check.
  * A withdrawal dumps heat: shutter motor, dispenser, receipt printer. That
    heat enters the thermal model, so busy periods show a small genuine
    temperature rise that the model must learn not to call a cooling fault.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EventStreams:
    presence: np.ndarray        # bool, person at the fascia
    motion: np.ndarray          # int 0/1, PIR output after latching
    vibration: np.ndarray       # int counts per second
    q_transaction_w: np.ndarray # extra heat load, W
    n_transactions: int


def build_events(cfg, n, hour_of_day, rng, dt=1.0) -> EventStreams:
    ev = cfg["events"]
    rates = np.asarray(ev["hourly_rate"], dtype=float)

    # Non-homogeneous Poisson: per-second arrival probability from the hourly
    # rate at that sample's local hour.
    lam = rates[np.floor(hour_of_day).astype(int) % 24] / 3600.0 * dt
    arrivals = np.flatnonzero(rng.random(n) < lam)

    presence = np.zeros(n, dtype=bool)
    vibration = np.zeros(n, dtype=np.int32)
    q_burst = np.zeros(n, dtype=float)

    for idx in arrivals:
        dur = int(rng.uniform(*ev["presence_duration_s"]) / dt)
        end = min(n, idx + dur)
        presence[idx:end] = True

        # Mechanical activity: card insert, keypad, shutter, dispense.
        act_len = int(rng.uniform(*ev["vib_transaction_active_s"]) / dt)
        if end - idx > act_len + 4:
            a0 = idx + int(rng.integers(3, max(4, end - idx - act_len)))
            a1 = min(n, a0 + act_len)
            total = int(rng.integers(*ev["vib_transaction_counts"]))
            if a1 > a0:
                vibration[a0:a1] += rng.poisson(total / (a1 - a0), a1 - a0).astype(np.int32)

        # Heat burst during the dispense phase.
        b0 = idx + int(0.35 * (end - idx))
        b1 = min(n, b0 + int(20 / dt))
        q_burst[b0:b1] += cfg["thermal"]["q_transaction_burst_w"]

    # Ambient mechanical background.
    vibration += rng.poisson(ev["vib_background_rate_per_s"] * dt, n).astype(np.int32)

    motion = latch(presence, int(ev["pir_hold_s"] / dt))
    return EventStreams(
        presence=presence,
        motion=motion.astype(np.int8),
        vibration=vibration,
        q_transaction_w=q_burst,
        n_transactions=len(arrivals),
    )


def latch(trigger: np.ndarray, hold: int) -> np.ndarray:
    """PIR retriggerable one-shot: output stays HIGH for `hold` samples after
    the last trigger."""
    out = np.zeros(len(trigger), dtype=bool)
    remaining = 0
    for i, trig in enumerate(trigger):
        if trig:
            remaining = hold
        if remaining > 0:
            out[i] = True
            remaining -= 1
    return out
