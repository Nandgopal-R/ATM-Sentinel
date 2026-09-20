"""Train Stage 2 only on non-NORMAL rows; test data is never loaded here."""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import ARTIFACT_DIR, LABELS, NORMALIZED_CLIP, SEED, load_split, metrics, save_json


def build_model() -> tf.keras.Model:
    """Small built-in-op-only MLP: 57 -> 16 ReLU -> 8 ReLU -> 5 softmax."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=(57,), name="normalized_features"),
        tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
        tf.keras.layers.Dense(8, activation="relu", name="dense_8"),
        tf.keras.layers.Dense(len(LABELS), activation="softmax", name="fault_probabilities"),
    ], name="atm_sentinel_stage2")


def save_confusion_plot(result: dict, path) -> None:
    values = np.asarray(result["confusion_matrix"]["values"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS,
           yticklabels=LABELS, xlabel="Predicted", ylabel="Actual", title="Stage 2 validation confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
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

    x_train, y_train, train_frame = load_split("train")
    x_validation, y_validation, validation_frame = load_split("validation")
    counts = np.bincount(y_train, minlength=len(LABELS))
    class_weight = {index: float(len(y_train) / (len(LABELS) * count)) for index, count in enumerate(counts)}

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=25,
                                                   restore_best_weights=True, verbose=0)]
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                        epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weight,
                        shuffle=True, verbose=0, callbacks=callbacks)
    model.save(ARTIFACT_DIR / "stage2.keras")

    validation_probabilities = model.predict(x_validation, verbose=0)
    validation_metrics = metrics(y_validation, validation_probabilities, validation_frame)
    save_json(ARTIFACT_DIR / "metrics" / "validation.json", validation_metrics)
    save_confusion_plot(validation_metrics, ARTIFACT_DIR / "metrics" / "validation_confusion_matrix.png")
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    run = {
        "seed": SEED,
        "architecture": [57, 16, 8, 5],
        "trainable_parameters": int(model.count_params()),
        "labels": {str(index): label for index, label in enumerate(LABELS)},
        "normalization": f"data/scaler.json, fitted on train only; standardized values clipped to +/-{NORMALIZED_CLIP:g}",
        "class_counts": {label: int(counts[index]) for index, label in enumerate(LABELS)},
        "class_weight": {label: class_weight[index] for index, label in enumerate(LABELS)},
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.epoch),
        "best_validation_loss_epoch": best_epoch,
        "validation": validation_metrics,
        "test_data_loaded": False,
    }
    save_json(ARTIFACT_DIR / "results.json", run)
    save_json(ARTIFACT_DIR / "history.json", {key: [float(x) for x in values] for key, values in history.history.items()})
    print(f"validation accuracy={validation_metrics['accuracy']:.4f} macro_f1={validation_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
