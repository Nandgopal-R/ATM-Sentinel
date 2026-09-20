"""End-to-end evaluation of the frozen ATM Sentinel Stage 1 -> Stage 2 cascade."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atm-sentinel-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = ROOT / "ml" / "stage-1"
STAGE2_DIR = ROOT / "ml" / "stage-2"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FINAL_LABELS = ["NORMAL", "COOLING_FAILURE", "HIGH_HUMIDITY", "POWER_FAILURE", "TAMPER", "SENSOR_FAULT"]


def load_module(name: str, path: Path):
    """Import a stage helper without treating hyphenated directories as packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage1_common = load_module("stage1_common", STAGE1_DIR / "common.py")
stage2_common = load_module("stage2_common", STAGE2_DIR / "common.py")


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(FINAL_LABELS)),
        yticks=range(len(FINAL_LABELS)),
        xticklabels=FINAL_LABELS,
        yticklabels=FINAL_LABELS,
        xlabel="Predicted",
        ylabel="Actual",
        title="ATM Sentinel end-to-end cascade: held-out test split",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(FINAL_LABELS)):
        for column in range(len(FINAL_LABELS)):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def main() -> None:
    # load_split is the existing Stage 1 preprocessing path: manifest order,
    # train-only scaler, and fixed clipping. It returns all test windows.
    features, stage1_truth = stage1_common.load_split("test")
    frame = pd.read_csv(ROOT / "data" / "test.csv")
    truth_labels = frame["label"].to_numpy()
    if len(features) != len(frame):
        raise ValueError("Stage 1 features and test rows have different lengths")

    threshold = float(json.loads((STAGE1_DIR / "artifacts" / "threshold.json").read_text())["threshold"])
    stage1_model = tf.keras.models.load_model(STAGE1_DIR / "artifacts" / "stage1.keras")
    stage2_model = tf.keras.models.load_model(STAGE2_DIR / "artifacts" / "stage2.keras")

    stage1_probabilities = stage1_model.predict(features, verbose=0).reshape(-1)
    forwarded = stage1_probabilities >= threshold
    final_labels = np.full(len(frame), "NORMAL", dtype=object)
    if forwarded.any():
        stage2_probabilities = stage2_model.predict(features[forwarded], verbose=0)
        stage2_ids = np.argmax(stage2_probabilities, axis=1)
        final_labels[forwarded] = np.asarray(stage2_common.LABELS, dtype=object)[stage2_ids]

    matrix = confusion_matrix(truth_labels, final_labels, labels=FINAL_LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth_labels, final_labels, labels=FINAL_LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(FINAL_LABELS)
    }

    abnormal = truth_labels != "NORMAL"
    normal = ~abnormal
    stage1_missed = abnormal & ~forwarded
    normal_forwarded = normal & forwarded
    correctly_routed_faults = abnormal & forwarded
    stage2_error = correctly_routed_faults & (final_labels != truth_labels)
    fault_routing = {}
    for label in stage2_common.LABELS:
        is_label = truth_labels == label
        fault_routing[label] = {
            "total_test_windows": int(is_label.sum()),
            "stage1_missed_as_normal": int((is_label & ~forwarded).sum()),
            "reached_stage2": int((is_label & forwarded).sum()),
            "finally_classified_correctly": int((is_label & (final_labels == label)).sum()),
        }

    result = {
        "evaluation": "frozen end-to-end Stage 1 -> Stage 2 cascade on held-out test split",
        "n_samples": int(len(frame)),
        "final_labels": FINAL_LABELS,
        "frozen_artifacts": {
            "stage1_model": str((STAGE1_DIR / "artifacts" / "stage1.keras").relative_to(ROOT)),
            "stage1_threshold": threshold,
            "stage2_model": str((STAGE2_DIR / "artifacts" / "stage2.keras").relative_to(ROOT)),
            "preprocessing": "data/manifest.json + data/scaler.json; existing normalization and +/-8 clipping",
        },
        "overall": {
            "accuracy": float(accuracy_score(truth_labels, final_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(truth_labels, final_labels)),
            "macro_precision": float(precision_score(truth_labels, final_labels, labels=FINAL_LABELS, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(truth_labels, final_labels, labels=FINAL_LABELS, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(truth_labels, final_labels, labels=FINAL_LABELS, average="macro", zero_division=0)),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": FINAL_LABELS, "values": matrix.tolist()},
        "cascade_error_sources": {
            "abnormal_stopped_by_stage1_as_normal": {"count": int(stage1_missed.sum()), "percentage_of_abnormal": rate(int(stage1_missed.sum()), int(abnormal.sum()))},
            "normal_forwarded_to_stage2": {"count": int(normal_forwarded.sum()), "percentage_of_normal": rate(int(normal_forwarded.sum()), int(normal.sum()))},
            "stage2_errors_after_correct_stage1_routing": {"count": int(stage2_error.sum()), "percentage_of_correctly_routed_faults": rate(int(stage2_error.sum()), int(correctly_routed_faults.sum()))},
            "stage2_invocations": {"count": int(forwarded.sum()), "percentage_of_all_windows": rate(int(forwarded.sum()), len(frame))},
        },
        "fault_class_routing": fault_routing,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    save_confusion_matrix(matrix, ARTIFACT_DIR / "confusion_matrix.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
