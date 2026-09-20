"""Evaluate a saved Stage 2 model; default is the final held-out test split."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from common import ARTIFACT_DIR, LABELS, load_split, metrics, save_json
from train import save_confusion_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    model = tf.keras.models.load_model(ARTIFACT_DIR / "stage2.keras")
    features, labels, frame = load_split(args.split)
    probabilities = model.predict(features, verbose=0)
    result = metrics(labels, probabilities, frame)
    save_json(ARTIFACT_DIR / "metrics" / f"{args.split}.json", result)
    save_confusion_plot(result, ARTIFACT_DIR / "metrics" / f"{args.split}_confusion_matrix.png")
    if args.split == "test":
        results_path = ARTIFACT_DIR / "results.json"
        run = json.loads(results_path.read_text())
        run["test"] = result
        run["final_test_evaluation"] = "independent Stage 2 evaluation conditioned on known ABNORMAL input"
        save_json(results_path, run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
