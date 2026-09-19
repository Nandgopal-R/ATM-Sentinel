"""Invariants that must hold or the dataset is not defensible.

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmsim import psychro, thermal                       # noqa: E402
from atmsim.cli import load_config                        # noqa: E402
from atmsim.runner import simulate_run                    # noqa: E402
from atmsim.splits import RunSpec, plan_runs              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg_schema():
    cfg, schema, _ = load_config(ROOT / "configs" / "default.yaml")
    cfg["run"]["hours"] = 1.0
    return cfg, schema


def _run(cfg, schema, subtype, seed=7):
    spec = RunSpec("RUN_T", "ATM_009", "train", "X", subtype, 13.0, seed, 0)
    return simulate_run(spec, cfg, schema)


# ------------------------------------------------------------ psychrometrics
def test_magnus_reference_points():
    # es(0 C) ~ 6.11 hPa, es(20 C) ~ 23.4 hPa, es(100 C) ~ 1013 hPa
    assert psychro.saturation_vapour_pressure(0.0) == pytest.approx(6.11, abs=0.02)
    assert psychro.saturation_vapour_pressure(20.0) == pytest.approx(23.4, abs=0.2)


def test_mixing_ratio_roundtrip():
    t, rh, p = 28.0, 55.0, 1008.0
    w = psychro.mixing_ratio(t, rh, p)
    assert psychro.rh_from_mixing_ratio(t, w, p) == pytest.approx(rh, abs=1e-6)


def test_rh_falls_when_temperature_rises_at_constant_absolute_humidity():
    """The single most important physical constraint in the simulator."""
    p = 1008.0
    w = psychro.mixing_ratio(28.0, 55.0, p)
    warm = psychro.rh_from_mixing_ratio(40.0, w, p)
    assert warm < 30.0, "RH must fall sharply as a sealed cabinet heats up"


# -------------------------------------------------------------------- thermal
def test_steady_state_and_time_constant_bracket(cfg_schema):
    cfg, _ = cfg_schema
    assert 3.5 < thermal.steady_state_rise(cfg, 1.0) < 5.0     # healthy: a few K
    assert 18.0 < thermal.steady_state_rise(cfg, 0.0) < 26.0   # dead fan: tens of K
    assert 400 < thermal.time_constant_s(cfg, 1.0) < 600       # ~8 min
    assert 2000 < thermal.time_constant_s(cfg, 0.0) < 3200     # ~43 min


def test_cooling_failure_is_not_a_linear_ramp(cfg_schema):
    """A degrading cabinet decelerates toward a new equilibrium. If the second
    half of the rise is as steep as the first, the model is wrong."""
    cfg, schema = cfg_schema
    cfg = {**cfg, "run": {**cfg["run"], "hours": 4.0}}
    tel, truth, _ = _run(cfg, schema, "FAN_STALL", seed=11)
    fault = np.flatnonzero(truth["state"].to_numpy() == "FAN_STALL")
    seg = tel["temperature"].to_numpy()[fault]
    seg = np.nan_to_num(seg, nan=np.nanmean(seg))
    assert len(seg) > 3600
    first = np.polyfit(np.arange(900), seg[:900], 1)[0]
    last = np.polyfit(np.arange(900), seg[-900:], 1)[0]
    assert first > last * 2.0, "temperature rise must decelerate, not ramp linearly"


# ------------------------------------------------------------------ scenarios
def test_high_humidity_raises_absolute_humidity_but_not_temperature(cfg_schema):
    cfg, schema = cfg_schema
    tel, truth, _ = _run(cfg, schema, "HIGH_HUMIDITY", seed=3)
    st = truth["state"].to_numpy()
    w = np.asarray(psychro.mixing_ratio(tel.temperature, tel.humidity, tel.pressure))
    t = tel["temperature"].to_numpy()
    pre, post = st == "NORMAL", st == "HIGH_HUMIDITY"
    assert np.nanmean(w[post]) > np.nanmean(w[pre]) * 1.2
    assert abs(np.nanmean(t[post]) - np.nanmean(t[pre])) < 1.0


def test_cooling_degradation_leaves_absolute_humidity_flat(cfg_schema):
    cfg, schema = cfg_schema
    cfg = {**cfg, "run": {**cfg["run"], "hours": 4.0}}
    tel, truth, _ = _run(cfg, schema, "COOLING_DEGRADATION", seed=5)
    st = truth["state"].to_numpy()
    w = np.asarray(psychro.mixing_ratio(tel.temperature, tel.humidity, tel.pressure))
    pre = st == "NORMAL"
    post = st != "NORMAL"
    ratio = np.nanmean(w[post]) / np.nanmean(w[pre])
    assert 0.9 < ratio < 1.1, "a thermal fault must not change the mixing ratio"


def test_fan_current_signatures_are_opposite(cfg_schema):
    cfg, schema = cfg_schema
    cfg = {**cfg, "run": {**cfg["run"], "hours": 4.0}}
    deg_t, deg_s, _ = _run(cfg, schema, "COOLING_DEGRADATION", seed=5)
    stl_t, stl_s, _ = _run(cfg, schema, "FAN_STALL", seed=6)
    base = np.nanmean(deg_t["current"].to_numpy()[deg_s["state"].to_numpy() == "NORMAL"])
    deg = np.nanmean(deg_t["current"].to_numpy()[deg_s["state"].to_numpy() != "NORMAL"])
    stl = np.nanmean(stl_t["current"].to_numpy()[stl_s["state"].to_numpy() == "FAN_STALL"])
    assert deg > base * 1.05, "bearing strain must RAISE fan current"
    assert stl < base * 0.95, "a stopped fan must DROP rail current"


def test_undervoltage_is_invisible_on_the_current_channel(cfg_schema):
    """Above SMPS dropout the rail is regulated. If `current` moved here, the
    power model would be handing the classifier a channel it should not have."""
    cfg, schema = cfg_schema
    tel, truth, _ = _run(cfg, schema, "POWER_UNDERVOLTAGE", seed=4)
    st = truth["state"].to_numpy()
    i = tel["current"].to_numpy()
    v = tel["voltage"].to_numpy()
    pre, post = st == "NORMAL", st == "POWER_UNDERVOLTAGE"
    assert np.nanmean(v[post]) < np.nanmean(v[pre]) - 8.0
    assert abs(np.nanmean(i[post]) - np.nanmean(i[pre])) < 0.02


# -------------------------------------------------------------------- sensors
def test_device_bias_is_stable_per_unit_and_differs_between_units(cfg_schema):
    from atmsim.sensors import device_bias

    cfg, schema = cfg_schema
    a1 = device_bias("ATM_001", cfg, schema)
    a2 = device_bias("ATM_001", cfg, schema)
    b1 = device_bias("ATM_002", cfg, schema)
    assert a1 == a2, "a unit's calibration must not change between runs"
    assert a1["temperature"] != b1["temperature"]


def test_schema_conformance(cfg_schema):
    cfg, schema = cfg_schema
    tel, _, _ = _run(cfg, schema, "NORMAL")
    assert list(tel.columns) == [
        "atm_id", "timestamp", "temperature", "humidity", "pressure",
        "light", "voltage", "current", "vibration", "motion",
    ]
    for ch, spec in schema["fields"].items():
        if "range" not in spec or ch not in tel:
            continue
        lo, hi = spec["range"]
        v = tel[ch].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        assert v.min() >= lo - 1e-6 and v.max() <= hi + 1e-6, ch
    assert set(np.unique(tel["motion"].dropna())) <= {0.0, 1.0}
    assert (tel["timestamp"].diff().dropna() == 1).all()


def test_no_label_column_leaks_into_telemetry(cfg_schema):
    cfg, schema = cfg_schema
    tel, _, _ = _run(cfg, schema, "TAMPER")
    for bad in ("state", "label", "scenario", "severity", "subtype"):
        assert bad not in tel.columns


# --------------------------------------------------------------------- splits
def test_atm_pools_are_disjoint(cfg_schema):
    cfg, _ = cfg_schema
    _, pools = plan_runs(cfg, np.random.default_rng(1))
    tr, va, te = set(pools["train"]), set(pools["validation"]), set(pools["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_every_subtype_reaches_every_split(cfg_schema):
    cfg, _ = cfg_schema
    specs, _ = plan_runs(cfg, np.random.default_rng(2))
    by_sub = {}
    for s in specs:
        by_sub.setdefault(s.subtype, set()).add(s.split)
    for sub, splits in by_sub.items():
        assert splits == {"train", "validation", "test"}, f"{sub} missing from {splits}"


def test_determinism(cfg_schema):
    cfg, schema = cfg_schema
    a, _, _ = _run(cfg, schema, "POWER_INSTABILITY", seed=99)
    b, _, _ = _run(cfg, schema, "POWER_INSTABILITY", seed=99)
    assert a.equals(b)
