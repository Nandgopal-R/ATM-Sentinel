"""Check firmware/atm_sentinel/features.h against atmsim/features.py.

    python firmware/test/test_features.py      (needs g++, numpy, pandas)
"""
import subprocess, sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atmsim.features import extract

order = json.loads((ROOT / "data/manifest.json").read_text())["feature_columns"]
exe = Path("/tmp/features_host")
subprocess.run(["g++", "-O1", "-o", exe, ROOT / "firmware/test/features_host.cpp"], check=True)

rng = np.random.default_rng(1)
n = 300
def window(nulls=True, stuck=None):
    t = 28 + 0.004 * np.arange(n) + rng.normal(0, 0.02, n)
    df = pd.DataFrame({
        "temperature": np.round(t, 2),
        "humidity": np.round(56 - 0.01 * np.arange(n) + rng.normal(0, 0.05, n), 2),
        "pressure": np.round(1008 + rng.normal(0, 0.02, n), 2),
        "light": np.round(240 + rng.normal(0, 0.8, n), 1),
        "voltage": np.round(np.where(rng.random(n) < 0.1, 200.0, 230 + rng.normal(0, 0.4, n)), 1),
        "current": np.round(1.417 + rng.normal(0, 0.002, n), 4),
        "vibration": rng.poisson(0.3, n).astype(float),
        "motion": (np.sin(np.arange(n) / 20) > 0.5).astype(float),
    })
    if nulls:
        for ch, k in (("temperature", 5), ("light", 3), ("humidity", 1)):
            df.loc[rng.choice(n, k, replace=False), ch] = np.nan
        df.loc[:2, "pressure"] = np.nan       # leading edge fill
        df.loc[n - 4:, "current"] = np.nan    # trailing edge fill
    if stuck:
        df[stuck] = 1.4104
    return df

def run(df):
    csv = "\n".join(",".join("" if np.isnan(v) else repr(float(v)) for v in row) for row in df.to_numpy()) + "\n"
    out = subprocess.run([exe], input=csv, capture_output=True, text=True, check=True).stdout.split(",")
    return int(out[0]), np.array(out[1:], dtype=float)

worst = 0.0
for name, df, want_mask in (("plain", window(nulls=False), 0),
                            ("nulls", window(), 0),
                            ("stuck_current", window(stuck="current"), 1 << 5)):
    mask, c = run(df)
    ref = np.array([extract(df, n, 1)[k][0] for k in order])
    err = np.abs(c - ref) / np.maximum(np.abs(ref), 1e-3)
    worst = max(worst, err.max())
    # 1e-3: pressure ~1008 hPa in float32 has 6e-5 spacing vs 0.02 std -> ~4e-4 on pressure_std
    bad = [(order[i], c[i], ref[i]) for i in np.flatnonzero(err > 1e-3)]
    assert mask == want_mask, (name, mask, want_mask)
    assert not bad, (name, bad)
    print(f"{name:14s} ok  max rel err {err.max():.2e}")
print("PASS")
