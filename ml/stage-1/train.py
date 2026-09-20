"""Train the compact Stage 1 NORMAL / ABNORMAL classifier.

The test split is evaluated only after training and validation threshold
selection are complete.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import ARTIFACT_DIR, NORMALIZED_CLIP, SEED, load_split, metrics, save_json, select_threshold


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(57,), name="normalized_features"),
            tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
            tf.keras.layers.Dense(8, activation="relu", name="dense_8"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="abnormal_probability"),
        ],
        name="atm_sentinel_stage1",
    )


def save_confusion_plot(result: dict, path: Path, title: str) -> None:
    values = np.asarray(result["confusion_matrix"]["values"])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NORMAL", "ABNORMAL"],
           yticklabels=["NORMAL", "ABNORMAL"], xlabel="Predicted", ylabel="Actual", title=title)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, tf.errors.UnimplementedError):
        pass

    x_train, y_train = load_split("train")
    x_validation, y_validation = load_split("validation")
    # Load test only after the model architecture, weights, stopping epoch, and
    # validation-derived threshold are all fixed.
    normal_count = int((y_train == 0).sum())
    abnormal_count = int((y_train == 1).sum())
    total = len(y_train)
    class_weight = {0: total / (2 * normal_count), 1: total / (2 * abnormal_count)}

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=25,
                                         restore_best_weights=True, verbose=0),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                        epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weight,
                        shuffle=True, verbose=0, callbacks=callbacks)
    model.save(ARTIFACT_DIR / "stage1.keras")

    validation_probabilities = model.predict(x_validation, verbose=0).reshape(-1)
    threshold, validation_metrics = select_threshold(y_validation, validation_probabilities)
    save_json(ARTIFACT_DIR / "threshold.json", {
        "threshold": threshold,
        "selected_on": "validation",
        "selection_metric": "balanced_accuracy",
        "tie_break": "nearest_to_0.5_then_lower_threshold",
    })
    save_json(ARTIFACT_DIR / "metrics" / "validation.json", validation_metrics)
    save_confusion_plot(validation_metrics, ARTIFACT_DIR / "metrics" / "validation_confusion_matrix.png",
                        "Validation confusion matrix")

    x_test, y_test = load_split("test")
    test_probabilities = model.predict(x_test, verbose=0).reshape(-1)
    test_metrics = metrics(y_test, test_probabilities, threshold)
    save_json(ARTIFACT_DIR / "metrics" / "test.json", test_metrics)
    save_confusion_plot(test_metrics, ARTIFACT_DIR / "metrics" / "test_confusion_matrix.png",
                        "Final test confusion matrix")

    best_epoch = int(np.argmax(history.history["val_auc"]) + 1)
    run = {
        "seed": SEED,
        "architecture": [57, 16, 8, 1],
        "trainable_parameters": int(model.count_params()),
        "normalization": f"data/scaler.json, fitted on train only; standardized values clipped to +/-{NORMALIZED_CLIP:g}",
        "class_weight": {"NORMAL_0": class_weight[0], "ABNORMAL_1": class_weight[1]},
        "class_counts": {"train_normal": normal_count, "train_abnormal": abnormal_count},
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.epoch),
        "best_validation_auc_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_json(ARTIFACT_DIR / "results.json", run)
    save_json(ARTIFACT_DIR / "history.json", {key: [float(x) for x in values] for key, values in history.history.items()})
    print(f"threshold={threshold:.6f}")
    print(f"validation balanced_accuracy={validation_metrics['balanced_accuracy']:.4f}")
    print(f"test balanced_accuracy={test_metrics['balanced_accuracy']:.4f} accuracy={test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
