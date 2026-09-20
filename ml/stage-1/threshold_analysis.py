"""Choose a Stage 1 operating threshold using validation data only.

Run without --evaluate-test to write validation analysis. After selecting a
strategy, rerun with --strategy and --evaluate-test for the single final test
evaluation at that already-fixed threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from common import ARTIFACT_DIR, DATA_DIR, load_split, metrics, save_json, select_threshold

OUTPUT_DIR = ARTIFACT_DIR / "threshold_analysis"
STRATEGIES = {
    "balanced_accuracy": "A: maximize validation balanced accuracy",
    "recall_90": "B: recall >= 0.90, then highest specificity",
    "recall_95": "C: recall >= 0.95, then highest specificity",
}


def select_recall_constrained(y: np.ndarray, probabilities: np.ndarray, minimum_recall: float) -> tuple[float, dict]:
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    feasible: list[tuple[float, dict]] = []
    for threshold in candidates:
        result = metrics(y, probabilities, float(threshold))
        if result["recall_abnormal"] >= minimum_recall:
            feasible.append((float(threshold), result))
    if not feasible:
        raise ValueError(f"no threshold reaches validation recall {minimum_recall}")
    # Maximize specificity under the recall constraint. A higher threshold is
    # preferred when specificity ties, since it creates no extra alerts.
    return max(feasible, key=lambda item: (item[1]["specificity_normal"], item[0]))


def probability_summary(frame: pd.DataFrame) -> dict:
    summary = {}
    for label, group in frame.groupby("label", sort=True):
        p = group["probability"].to_numpy()
        summary[label] = {
            "n": int(len(p)), "min": float(p.min()), "p05": float(np.quantile(p, .05)),
            "p25": float(np.quantile(p, .25)), "median": float(np.median(p)),
            "mean": float(p.mean()), "p75": float(np.quantile(p, .75)),
            "p95": float(np.quantile(p, .95)), "max": float(p.max()),
        }
    for label, group in frame.assign(binary=np.where(frame.label == "NORMAL", "NORMAL", "ABNORMAL")).groupby("binary", sort=True):
        p = group["probability"].to_numpy()
        summary[f"binary_{label}"] = {
            "n": int(len(p)), "min": float(p.min()), "p05": float(np.quantile(p, .05)),
            "p25": float(np.quantile(p, .25)), "median": float(np.median(p)),
            "mean": float(p.mean()), "p75": float(np.quantile(p, .75)),
            "p95": float(np.quantile(p, .95)), "max": float(p.max()),
        }
    return summary


def false_negatives_by_class(frame: pd.DataFrame, threshold: float) -> dict:
    abnormal = frame[frame.label != "NORMAL"].copy()
    abnormal["false_negative"] = abnormal.probability < threshold
    rows = []
    for label, group in abnormal.groupby("label", sort=True):
        fn = int(group.false_negative.sum())
        rows.append({"label": label, "n_abnormal": int(len(group)), "false_negatives": fn,
                     "false_negative_rate": float(fn / len(group)),
                     "probability_median": float(group.probability.median())})
    return {"threshold": float(threshold), "rows": rows,
            "total_false_negatives": int(abnormal.false_negative.sum())}


def save_probability_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["NORMAL", "COOLING_FAILURE", "HIGH_HUMIDITY", "POWER_FAILURE", "TAMPER", "SENSOR_FAULT"]
    data = [frame.loc[frame.label == label, "probability"] for label in labels]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set(ylabel="Predicted ABNORMAL probability", ylim=(-.02, 1.02), title="Validation probability distributions")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES, help="fixed validation-selected operating policy")
    parser.add_argument("--evaluate-test", action="store_true", help="evaluate the selected policy once on test")
    args = parser.parse_args()
    if args.evaluate_test and not args.strategy:
        parser.error("--evaluate-test requires --strategy")

    features, labels = load_split("validation")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    if not np.array_equal(labels, (validation.label.to_numpy() != "NORMAL").astype(np.int32)):
        raise ValueError("validation labels are misaligned")
    model = tf.keras.models.load_model(ARTIFACT_DIR / "stage1.keras")
    validation["probability"] = model.predict(features, verbose=0).reshape(-1)

    threshold_a, result_a = select_threshold(labels, validation.probability.to_numpy())
    threshold_b, result_b = select_recall_constrained(labels, validation.probability.to_numpy(), .90)
    threshold_c, result_c = select_recall_constrained(labels, validation.probability.to_numpy(), .95)
    decisions = {
        "balanced_accuracy": (threshold_a, result_a),
        "recall_90": (threshold_b, result_b),
        "recall_95": (threshold_c, result_c),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = []
    fn_analysis = {}
    for name, (threshold, result) in decisions.items():
        comparison.append({"strategy": name, "description": STRATEGIES[name], **result})
        fn_analysis[name] = false_negatives_by_class(validation, threshold)
    save_json(OUTPUT_DIR / "threshold_comparison.json", {"split": "validation", "strategies": comparison})
    pd.DataFrame([{key: value for key, value in row.items() if key not in {"confusion_matrix", "counts"}}
                  for row in comparison]).to_csv(OUTPUT_DIR / "threshold_comparison.csv", index=False)
    save_json(OUTPUT_DIR / "false_negatives_by_class.json", {"split": "validation", "strategies": fn_analysis})
    save_json(OUTPUT_DIR / "probability_distributions.json", probability_summary(validation))
    save_probability_plot(validation, OUTPUT_DIR / "validation_probability_distributions.png")

    if not args.evaluate_test:
        for row in comparison:
            print(f"{row['strategy']}: threshold={row['threshold']:.6f} recall={row['recall_abnormal']:.4f} specificity={row['specificity_normal']:.4f}")
        return

    threshold, validation_result = decisions[args.strategy]
    # Test data is deliberately loaded only here, after the selected strategy
    # and threshold are fully determined from validation.
    test_features, test_labels = load_split("test")
    test_probabilities = model.predict(test_features, verbose=0).reshape(-1)
    test_result = metrics(test_labels, test_probabilities, threshold)
    selection = {
        "strategy": args.strategy,
        "description": STRATEGIES[args.strategy],
        "selected_on": "validation only",
        "threshold": threshold,
        "validation": validation_result,
        "test": test_result,
    }
    save_json(OUTPUT_DIR / "chosen_operating_point.json", selection)
    save_json(OUTPUT_DIR / "final_test_metrics.json", test_result)
    # Make the chosen threshold the default for downstream inference/evaluate.py.
    save_json(ARTIFACT_DIR / "threshold.json", {
        "threshold": threshold, "selected_on": "validation only",
        "selection_strategy": args.strategy, "selection_description": STRATEGIES[args.strategy],
    })
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
