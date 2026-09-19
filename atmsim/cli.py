"""Command-line entry point.

    python -m atmsim generate --config configs/default.yaml --out .

Reproducibility: every run's RNG is seeded from the master seed through the
plan, and the config hash is written into manifest.json. Re-running the same
config reproduces the dataset bit-for-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .runner import simulate_run
from .splits import plan_runs
from .windowing import balance, window_run

META_COLUMNS = [
    "run_id",
    "atm_id",
    "window_start_ts",
    "window_end_ts",
    "label",
    "stage1",
    "fine_state",
    "purity",
    "ambiguous",
    "severity",
    "run_subtype",
    "time_to_failure_s",
]


def load_config(path: Path):
    text = path.read_text()
    cfg = yaml.safe_load(text)
    schema_path = path.parent.parent / cfg["schema"]
    if not schema_path.exists():
        schema_path = Path(cfg["schema"])
    schema = yaml.safe_load(schema_path.read_text())
    cfg_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return cfg, schema, cfg_hash


def generate(args):
    cfg, schema, cfg_hash = load_config(Path(args.config))
    out = Path(args.out)
    raw_dir = out / cfg["output"]["raw_dir"]
    data_dir = out / cfg["output"]["data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        cfg["run"]["hours"] = 1.5
        cfg["counts"]["normal_runs"] = 6
        cfg["counts"]["runs_per_fault_class"] = 5
        cfg["counts"]["runs_override"] = {"TAMPER": 6}
        cfg["windowing"]["balance_target_per_class"] = 150

    rng = np.random.default_rng(cfg["seed"])
    specs, pools = plan_runs(cfg, rng)

    wcfg = cfg["windowing"]
    strides = {
        "train": int(wcfg["train_stride_s"]),
        "validation": int(wcfg["eval_stride_s"]),
        "test": int(wcfg["eval_stride_s"]),
    }

    buckets = {"train": [], "validation": [], "test": []}
    records = []
    t0 = time.time()
    total_rows = 0

    for k, spec in enumerate(specs, 1):
        telemetry, truth, record = simulate_run(spec, cfg, schema)
        total_rows += len(telemetry)

        if not args.no_raw:
            telemetry.assign(state=truth["state"]).to_parquet(
                raw_dir / f"{spec.run_id}.parquet", index=False, compression="zstd"
            )

        win = window_run(
            telemetry, truth, record, cfg, strides[spec.split], keep_ambiguous=args.keep_ambiguous
        )
        if len(win):
            buckets[spec.split].append(win)
        rec = dict(record)
        rec["params"] = json.dumps(rec["params"], default=float)
        rec["n_windows"] = len(win)
        records.append(rec)

        if k % 10 == 0 or k == len(specs):
            print(
                f"  [{k:>4}/{len(specs)}] {spec.run_id} {spec.subtype:<20} "
                f"{len(telemetry):>6} rows  {time.time() - t0:6.1f}s",
                flush=True,
            )

    events = pd.DataFrame(records)
    events.to_csv(data_dir / "events.csv", index=False)

    # ------------------------------------------------------------ balance ---
    target = int(wcfg["balance_target_per_class"])
    fr = cfg["fleet"]["split_fractions"]
    targets = {
        "train": target,
        "validation": max(1, int(round(target * fr["validation"] / fr["train"]))),
        "test": max(1, int(round(target * fr["test"] / fr["train"]))),
    }

    summary = {}
    frames = {}
    for split, parts in buckets.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        pre = df["label"].value_counts().to_dict() if len(df) else {}
        if len(df):
            df = balance(df, targets[split], rng)
        frames[split] = df
        summary[split] = {
            "before_balance": {k: int(v) for k, v in pre.items()},
            "after_balance": {k: int(v) for k, v in df["label"].value_counts().items()}
            if len(df)
            else {},
            "n_windows": int(len(df)),
            "n_runs": int(sum(1 for s in specs if s.split == split)),
            "n_atms": len(pools[split]),
            "stride_s": strides[split],
        }

    feature_cols = [c for c in frames["train"].columns if c not in META_COLUMNS]

    # --------------------------------------------- scaler fitted on TRAIN ---
    tr = frames["train"]
    scaler = {
        "fitted_on": "train",
        "n_samples": int(len(tr)),
        "features": feature_cols,
        "mean": {c: float(tr[c].mean()) for c in feature_cols},
        "std": {c: float(tr[c].std(ddof=0)) or 1.0 for c in feature_cols},
    }
    (data_dir / "scaler.json").write_text(json.dumps(scaler, indent=2))

    ordered = feature_cols + META_COLUMNS
    for split, fname in (
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ):
        frames[split][ordered].to_csv(data_dir / fname, index=False, float_format="%.6g")

    manifest = {
        "generator_version": __version__,
        "schema_version": schema["schema_version"],
        "config_hash": cfg_hash,
        "seed": cfg["seed"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_runs": len(specs),
        "run_hours": cfg["run"]["hours"],
        "raw_rows": int(total_rows),
        "window_s": wcfg["window_s"],
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "atm_pools": pools,
        "splits": summary,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== summary ===")
    for split in ("train", "validation", "test"):
        s = summary[split]
        print(f"{split:<11} {s['n_windows']:>6} windows  {s['n_runs']:>3} runs  {s['n_atms']:>3} ATMs")
        for cls, cnt in sorted(s["after_balance"].items()):
            print(f"              {cls:<18} {cnt:>5}")
    print(f"\nfeatures: {len(feature_cols)}   raw rows: {total_rows:,}   "
          f"wall clock: {manifest['wall_clock_s']}s")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="atmsim")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config", default="configs/default.yaml")
    g.add_argument("--out", default=".")
    g.add_argument("--quick", action="store_true", help="tiny dataset for smoke tests")
    g.add_argument("--no-raw", action="store_true", help="skip writing raw parquet")
    g.add_argument("--keep-ambiguous", action="store_true")
    g.set_defaults(func=generate)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
