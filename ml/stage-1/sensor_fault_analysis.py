"""Validation-only investigation of SENSOR_FAULT subtypes at the original threshold."""

from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from common import ARTIFACT_DIR, DATA_DIR, load_split, save_json

ORIGINAL_THRESHOLD = 0.7615026
OUTPUT_DIR = ARTIFACT_DIR.parent / "analysis"

SUBTYPE_POLICY = {
    "SENSOR_STUCK": {
        "bucket": "A: deterministic integrity fault",
        "recommendation": "pre-ML rule",
        "reason": "Simulator freezes the post-onset raw value exactly, optionally adding nulls.",
        "rule": "Flag a nullable float channel when all non-null samples in a 300 s inference window equal one schema-quantized value; also flag a sustained missing run that prevents a 60 s publish interval from being formed.",
    },
    "SENSOR_BIAS": {
        "bucket": "B: subtle ML fault",
        "recommendation": "retain for ML",
        "reason": "A fixed offset remains within the declared schema range and can resemble stable per-device calibration bias.",
        "rule": "Only schema-range violations or persistent endpoint saturation are deterministic; otherwise use ML/cross-channel checks.",
    },
    "SENSOR_DRIFT": {
        "bucket": "B: subtle ML fault",
        "recommendation": "retain for ML",
        "reason": "The simulator ramps a bounded offset; it can remain physically plausible and within schema limits.",
        "rule": "Only schema-range violations or persistent endpoint saturation are deterministic; otherwise use ML/cross-channel checks.",
    },
    "SENSOR_SPIKE": {
        "bucket": "B: subtle ML fault",
        "recommendation": "retain for ML",
        "reason": "Spikes are clipped to schema range before telemetry is emitted and may be indistinguishable from a real transient.",
        "rule": "Reject samples outside schema range before clipping; retain in-range transient detection for ML.",
    },
    "SENSOR_NOISE": {
        "bucket": "B: subtle ML fault",
        "recommendation": "retain for ML",
        "reason": "Variance inflation is in-range and must be distinguished from environmental or load variation.",
        "rule": "Reject samples outside schema range before clipping; retain in-range variance changes for ML.",
    },
}


def summarize(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    return {"min": float(values.min()), "p05": float(np.quantile(values, .05)),
            "p25": float(np.quantile(values, .25)), "median": float(np.median(values)),
            "mean": float(values.mean()), "p75": float(np.quantile(values, .75)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def feature_comparison(fault: pd.DataFrame, normal: pd.DataFrame, channel: str) -> dict:
    features = [f"{channel}_{name}" for name in ("null_frac", "mean", "std", "min", "max", "slope", "delta")]
    output = {}
    for feature in features:
        if feature not in fault:
            continue
        output[feature] = {
            "fault_mean": float(fault[feature].mean()),
            "fault_median": float(fault[feature].median()),
            "normal_mean": float(normal[feature].mean()),
            "normal_median": float(normal[feature].median()),
        }
    return output


def save_plot(frame: pd.DataFrame, path) -> None:
    order = ["SENSOR_STUCK", "SENSOR_BIAS", "SENSOR_DRIFT", "SENSOR_SPIKE", "SENSOR_NOISE"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([frame.loc[frame.run_subtype == subtype, "probability"] for subtype in order],
               tick_labels=[x.removeprefix("SENSOR_") for x in order], showfliers=False)
    ax.axhline(ORIGINAL_THRESHOLD, color="tab:red", linestyle="--", label="original threshold")
    ax.set(ylabel="Stage 1 ABNORMAL probability", ylim=(-.02, 1.02),
           title="Validation SENSOR_FAULT subtype probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    features, binary_labels = load_split("validation")
    frame = pd.read_csv(DATA_DIR / "validation.csv")
    if not np.array_equal(binary_labels, (frame.label.to_numpy() != "NORMAL").astype(np.int32)):
        raise ValueError("validation labels are misaligned")
    events = pd.read_csv(DATA_DIR / "events.csv").set_index("run_id")
    model = tf.keras.models.load_model(ARTIFACT_DIR / "stage1.keras")
    frame["probability"] = model.predict(features, verbose=0).reshape(-1)
    sensor = frame[frame.label == "SENSOR_FAULT"].copy()
    sensor["fault_channel"] = sensor.run_id.map(events.fault_channel)
    normal = frame[frame.label == "NORMAL"]

    rows = []
    details = {}
    for subtype, group in sensor.groupby("run_subtype", sort=True):
        channels = group.fault_channel.dropna().unique().tolist()
        if len(channels) != 1:
            raise ValueError(f"expected one fault channel for {subtype}, got {channels}")
        channel = channels[0]
        probabilities = group.probability
        false_negatives = int((probabilities < ORIGINAL_THRESHOLD).sum())
        policy = SUBTYPE_POLICY[subtype]
        details[subtype] = {
            "fault_channel": channel,
            "n_windows": int(len(group)),
            "original_threshold": ORIGINAL_THRESHOLD,
            "probability_distribution": summarize(probabilities),
            "false_negatives": false_negatives,
            "false_negative_rate": float(false_negatives / len(group)),
            "relevant_feature_comparison_to_normal": feature_comparison(group, normal, channel),
            **policy,
        }
        rows.append({
            "subtype": subtype, "fault_channel": channel, "n_windows": len(group),
            "probability_median": probabilities.median(), "probability_mean": probabilities.mean(),
            "false_negatives": false_negatives, "false_negative_rate": false_negatives / len(group),
            "bucket": policy["bucket"], "recommendation": policy["recommendation"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "sensor_fault_subtype_analysis.json", {
        "split": "validation only", "model_changed": False, "threshold_changed": False,
        "original_threshold": ORIGINAL_THRESHOLD, "subtypes": details,
    })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sensor_fault_subtype_analysis.csv", index=False)
    save_plot(sensor, OUTPUT_DIR / "sensor_fault_subtype_probabilities.png")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
