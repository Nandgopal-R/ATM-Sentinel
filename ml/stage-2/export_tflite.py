"""Export float32 Stage 2 TFLite and validate all validation outputs vs Keras."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from common import ARTIFACT_DIR, LABELS, NORMALIZED_CLIP, load_split, save_json


def main() -> None:
    model = tf.keras.models.load_model(ARTIFACT_DIR / "stage2.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_path = ARTIFACT_DIR / "stage2.tflite"
    tflite_path.write_bytes(converter.convert())

    validation_features, _, _ = load_split("validation")
    keras_probabilities = model.predict(validation_features, verbose=0)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    converted = np.empty_like(keras_probabilities)
    for index, row in enumerate(validation_features):
        interpreter.set_tensor(input_detail["index"], row[np.newaxis, :])
        interpreter.invoke()
        converted[index] = interpreter.get_tensor(output_detail["index"])[0]
    max_difference = float(np.max(np.abs(keras_probabilities - converted)))
    if not np.allclose(keras_probabilities, converted, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"TFLite output differs from Keras: max abs error {max_difference}")
    metadata = {
        "format": "TFLite / LiteRT FlatBuffer",
        "quantization": "float32; built-in Dense/ReLU/Softmax operations only",
        "input": {"shape": input_detail["shape"].tolist(), "dtype": str(input_detail["dtype"]),
                  "contract": f"57 train-standardized then clipped-to-+/-{NORMALIZED_CLIP:g} features in data/manifest.json order"},
        "output": {"shape": output_detail["shape"].tolist(), "dtype": str(output_detail["dtype"]),
                   "meaning": "softmax probabilities", "label_mapping": {str(i): label for i, label in enumerate(LABELS)}},
        "file_bytes": tflite_path.stat().st_size,
        "validation_max_abs_probability_difference": max_difference,
        "validation_argmax_agreement": float(np.mean(np.argmax(keras_probabilities, axis=1) == np.argmax(converted, axis=1))),
    }
    save_json(ARTIFACT_DIR / "tflite_metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
