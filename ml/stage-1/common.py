"""Shared data loading and metric helpers for the Stage 1 classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SEED = 20260918
NORMALIZED_CLIP = 8.0


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load one preserved split in manifest order and apply the train-only scaler."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    scaler = json.loads((data_dir / "scaler.json").read_text())
    features = manifest["feature_columns"]
    if features != scaler["features"]:
        raise ValueError("manifest and scaler feature orders differ")
    if len(features) != 57:
        raise ValueError(f"expected 57 features, found {len(features)}")

    frame = pd.read_csv(data_dir / f"{split}.csv")
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"{split} is missing features: {sorted(missing)}")
    x = frame.loc[:, features].to_numpy(dtype=np.float32)
    mean = np.asarray([scaler["mean"][name] for name in features], dtype=np.float32)
    std = np.asarray([scaler["std"][name] for name in features], dtype=np.float32)
    if np.any(std == 0):
        raise ValueError("scaler contains a zero standard deviation")
    # A shared int8 input tensor has one scale for all 57 values. Clipping the
    # standardized tails bounds that scale while preserving the train-only
    # centering/scaling contract. The bound is deliberately wide (8 sigma).
    x = np.clip((x - mean) / std, -NORMALIZED_CLIP, NORMALIZED_CLIP)
    if not np.isfinite(x).all():
        raise ValueError(f"{split} contains non-finite normalized features")
    y = (frame["label"].to_numpy() != "NORMAL").astype(np.int32)
    return x, y


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Binary metrics where 1 means ABNORMAL and 0 means NORMAL."""
    predicted = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_label": "ABNORMAL",
        "negative_label": "NORMAL",
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_abnormal": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_abnormal": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity_normal": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_abnormal": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {"labels": ["NORMAL", "ABNORMAL"], "values": matrix.tolist()},
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    """Choose validation threshold by balanced accuracy, with deterministic ties."""
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    ranked: list[tuple[tuple[float, float, float], float, dict]] = []
    for threshold in candidates:
        result = metrics(y_true, probabilities, float(threshold))
        # Balanced accuracy is the sole selection objective. Prefer a threshold
        # nearest 0.5 only when that objective is exactly tied.
        ranked.append(((result["balanced_accuracy"], -abs(float(threshold) - 0.5), -float(threshold)), float(threshold), result))
    _, threshold, result = max(ranked, key=lambda item: item[0])
    return threshold, result


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
