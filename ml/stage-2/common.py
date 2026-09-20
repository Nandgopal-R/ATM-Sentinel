"""Shared data and reporting helpers for the Stage 2 fault classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SEED = 20260918
NORMALIZED_CLIP = 8.0
LABELS = ["COOLING_FAILURE", "HIGH_HUMIDITY", "POWER_FAILURE", "TAMPER", "SENSOR_FAULT"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load preserved abnormal-only rows in manifest order and Stage 1 preprocessing."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    scaler = json.loads((data_dir / "scaler.json").read_text())
    features = manifest["feature_columns"]
    if features != scaler["features"] or len(features) != 57:
        raise ValueError("expected identical manifest/scaler 57-feature order")

    frame = pd.read_csv(data_dir / f"{split}.csv")
    frame = frame.loc[frame["label"] != "NORMAL"].reset_index(drop=True)
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"unexpected Stage 2 labels: {unknown}")
    x = frame.loc[:, features].to_numpy(dtype=np.float32)
    mean = np.asarray([scaler["mean"][name] for name in features], dtype=np.float32)
    std = np.asarray([scaler["std"][name] for name in features], dtype=np.float32)
    if np.any(std == 0):
        raise ValueError("scaler contains a zero standard deviation")
    x = np.clip((x - mean) / std, -NORMALIZED_CLIP, NORMALIZED_CLIP)
    if not np.isfinite(x).all():
        raise ValueError(f"{split} contains non-finite normalized features")
    y = frame["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    return x, y, frame


def metrics(y_true: np.ndarray, probabilities: np.ndarray, frame: pd.DataFrame) -> dict:
    """Multiclass metrics and subtype slices; prediction is argmax of softmax output."""
    predicted = np.argmax(probabilities, axis=1).astype(np.int32)
    matrix = confusion_matrix(y_true, predicted, labels=range(len(LABELS)))
    report = classification_report(y_true, predicted, labels=range(len(LABELS)),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in LABELS
    }
    result = {
        "n_samples": int(len(y_true)),
        "labels": LABELS,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_precision": float(precision_score(y_true, predicted, labels=range(len(LABELS)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, labels=range(len(LABELS)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, labels=range(len(LABELS)), average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "values": matrix.tolist()},
    }
    subtype_results = {}
    for subtype in ("SENSOR_BIAS", "SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_SPIKE", "SENSOR_NOISE"):
        indices = np.flatnonzero(frame["run_subtype"].to_numpy() == subtype)
        if len(indices):
            truth = y_true[indices]
            subtype_results[subtype] = {
                "n_samples": int(len(indices)),
                "accuracy": float(accuracy_score(truth, predicted[indices])),
                "predicted_class_counts": {label: int((predicted[indices] == idx).sum()) for idx, label in enumerate(LABELS)},
            }
    result["sensor_fault_subtypes"] = subtype_results
    return result


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
