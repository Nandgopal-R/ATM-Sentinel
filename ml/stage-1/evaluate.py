"""Evaluate a saved Stage 1 Keras model with its fixed validation threshold."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from common import ARTIFACT_DIR, load_split, metrics, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    threshold = json.loads((ARTIFACT_DIR / "threshold.json").read_text())["threshold"]
    model = tf.keras.models.load_model(ARTIFACT_DIR / "stage1.keras")
    features, labels = load_split(args.split)
    probabilities = model.predict(features, verbose=0).reshape(-1)
    result = metrics(labels, probabilities, threshold)
    save_json(ARTIFACT_DIR / "metrics" / f"{args.split}_reevaluated.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
