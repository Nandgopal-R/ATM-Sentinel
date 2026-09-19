"""Windowing, labelling and balancing.

Labelling policy:

* A window's Stage-2 label is the **majority** ground-truth class across its
  300 samples.
* `purity` is the share held by that majority class. Windows below the purity
  threshold straddle a fault onset and are marked `ambiguous` and excluded from
  the delivered sets by default.
* They are not deleted. `--keep-ambiguous` retains them, because those are
  exactly the windows the detection-lead-time analysis needs: they are the
  moment the fault is becoming visible.
* Purity is computed on the **Stage-2** class, not the fine state, so the
  internal COOLING_DEGRADATION -> COOLING_FAILURE transition is not treated as
  an ambiguous boundary. It is one class throughout.

`time_to_failure_s` is negative before the ground-truth failure instant. Feed
it nothing during training; use it afterwards to answer "how many minutes of
warning did Stage-1 actually give us", which is a far stronger claim than
accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import extract


def _stage2(states: np.ndarray, label_map: dict) -> np.ndarray:
    lut = np.vectorize(lambda s: label_map[s])
    return lut(states)


def window_run(telemetry, truth, record, cfg, stride, keep_ambiguous=False):
    wcfg = cfg["windowing"]
    size = int(wcfg["window_s"])
    dt = 1.0 / cfg["run"]["sample_rate_hz"]
    n = len(telemetry)
    if n < size:
        return pd.DataFrame()

    feats = extract(telemetry, size, stride, dt)
    n_win = len(next(iter(feats.values())))

    states = truth["state"].to_numpy()
    stage2 = _stage2(states, cfg["labels"]["map"])

    from numpy.lib.stride_tricks import sliding_window_view

    sw = sliding_window_view(stage2, size)[::stride]
    fw = sliding_window_view(states, size)[::stride]

    labels, purity, fine = [], [], []
    for i in range(n_win):
        vals, counts = np.unique(sw[i], return_counts=True)
        j = int(np.argmax(counts))
        labels.append(str(vals[j]))
        purity.append(counts[j] / size)
        fvals, fcounts = np.unique(fw[i], return_counts=True)
        fine.append(str(fvals[int(np.argmax(fcounts))]))

    labels = np.array(labels)
    purity = np.array(purity)
    ts = telemetry["timestamp"].to_numpy()
    start_ts = sliding_window_view(ts, size)[::stride][:, 0]
    end_ts = sliding_window_view(ts, size)[::stride][:, -1]

    df = pd.DataFrame(feats)
    df.insert(0, "run_id", record["run_id"])
    df.insert(1, "atm_id", record["atm_id"])
    df.insert(2, "window_start_ts", start_ts)
    df.insert(3, "window_end_ts", end_ts)
    df["label"] = labels
    df["stage1"] = np.where(labels == "NORMAL", "HEALTHY", "ABNORMAL")
    df["fine_state"] = fine
    df["purity"] = purity.round(4)
    df["ambiguous"] = purity < wcfg["ambiguous_purity_threshold"]
    df["severity"] = record["severity"]
    df["run_subtype"] = record["subtype"]

    if record["t_failure"] is not None:
        df["time_to_failure_s"] = end_ts - record["t_failure"]
    else:
        df["time_to_failure_s"] = np.nan

    if not keep_ambiguous:
        df = df[~df["ambiguous"]].reset_index(drop=True)
    return df


def balance(df, target_per_class, rng):
    """Cap each class at `target_per_class`. Never upsamples — a duplicated
    window is not new information and inflates every metric it touches."""
    parts = []
    for cls, grp in df.groupby("label", sort=True):
        if len(grp) > target_per_class:
            # Sample across runs rather than taking a contiguous block, so no
            # single run dominates a class.
            idx = rng.choice(len(grp), size=target_per_class, replace=False)
            grp = grp.iloc[np.sort(idx)]
        parts.append(grp)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(
        drop=True
    )
