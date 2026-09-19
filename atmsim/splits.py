"""Run planning and split assignment.

Splitting happens BEFORE generation, at two levels:

1. **ATM identity.** The fleet is partitioned into disjoint train / validation /
   test pools. A unit that appears in training never appears in test, so the
   per-device calibration bias cannot leak. This is the difference between
   "our model works" and "our model recognises ATM_007".
2. **Run.** Runs are allocated to splits per class, so each split sees every
   scenario. Windows are only ever cut inside a run, and runs never straddle a
   split boundary.

Randomly splitting overlapping windows would put near-identical 300-second
neighbours on both sides of the train/test line and inflate every metric.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .scenarios import STAGE2_SUBTYPES, choose_subtype


@dataclass
class RunSpec:
    run_id: str
    atm_id: str
    split: str
    stage2_class: str
    subtype: str
    start_hour: float
    seed: int
    run_index: int

    def to_dict(self):
        return asdict(self)


def partition_fleet(cfg, rng):
    n = cfg["fleet"]["n_atms"]
    fr = cfg["fleet"]["split_fractions"]
    ids = [f"ATM_{i + 1:03d}" for i in range(n)]
    rng.shuffle(ids)
    n_train = int(round(fr["train"] * n))
    n_val = int(round(fr["validation"] * n))
    return {
        "train": ids[:n_train],
        "validation": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


SUBTYPE_TO_CLASS = {
    sub: cls for cls, subs in STAGE2_SUBTYPES.items() for sub in subs
}


def plan_runs(cfg, rng):
    pools = partition_fleet(cfg, rng)
    fr = cfg["fleet"]["split_fractions"]
    counts = cfg["counts"]

    per_subtype = {"NORMAL": int(counts["normal_runs"])}
    per_subtype.update({k: int(v) for k, v in counts["runs_per_subtype"].items()})

    specs = []
    idx = 0
    lo, hi = cfg["run"]["start_hour_range"]
    for subtype, n_runs in per_subtype.items():
        # Every subtype is allocated to every split, with at least one run each,
        # so no failure mode can go missing from validation or test.
        n_train = max(1, int(round(fr["train"] * n_runs)))
        n_val = max(1, int(round(fr["validation"] * n_runs)))
        n_test = max(1, n_runs - n_train - n_val)
        for split, k in (("train", n_train), ("validation", n_val), ("test", n_test)):
            for _ in range(k):
                specs.append(
                    RunSpec(
                        run_id=f"RUN_{idx:05d}",
                        atm_id=str(rng.choice(pools[split])),
                        split=split,
                        stage2_class=SUBTYPE_TO_CLASS[subtype],
                        subtype=subtype,
                        start_hour=float(rng.uniform(lo, hi)),
                        seed=int(rng.integers(0, 2**31 - 1)),
                        run_index=idx,
                    )
                )
                idx += 1
    rng.shuffle(specs)
    return specs, pools
