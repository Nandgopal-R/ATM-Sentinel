"""Export the trained normalized-feature model as a TFLite/LiteRT FlatBuffer."""

from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from common import ARTIFACT_DIR, NORMALIZED_CLIP, load_split, save_json


def main() -> None:
    model = tf.keras.models.load_model(ARTIFACT_DIR / "stage1.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Keep float32 I/O and arithmetic for this first embedded artifact. A
    # single int8 input scale across the heterogeneous engineered features
    # materially changes probabilities; quantization-aware training or a
    # per-feature fixed-point input transform is the appropriate next step.
    # The resulting model is still only a few KiB and uses TFLite built-ins.
    tflite_path = ARTIFACT_DIR / "stage1.tflite"
    tflite_path.write_bytes(converter.convert())

    # Validate conversion against Keras on validation examples only. Test data
    # remains reserved for the final model metrics written by train.py.
    validation_features, _ = load_split("validation")
    keras_probabilities = model.predict(validation_features, verbose=0).reshape(-1)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    converted = []
    for row in validation_features:
        interpreter.set_tensor(input_detail["index"], row[np.newaxis, :])
        interpreter.invoke()
        converted.append(float(interpreter.get_tensor(output_detail["index"])[0, 0]))
    metadata = {
        "format": "TFLite / LiteRT FlatBuffer",
        "quantization": "float32; quantization-friendly Dense/ReLU/Sigmoid built-in operations",
        "input": {"shape": input_detail["shape"].tolist(), "dtype": str(input_detail["dtype"]),
                  "contract": f"57 train-standardized then clipped-to-+/-{NORMALIZED_CLIP:g} features in data/manifest.json order"},
        "output": {"shape": output_detail["shape"].tolist(), "dtype": str(output_detail["dtype"]),
                   "meaning": "ABNORMAL probability"},
        "file_bytes": tflite_path.stat().st_size,
        "validation_max_abs_probability_difference": float(max(abs(a - b) for a, b in zip(keras_probabilities, converted))),
    }
    save_json(ARTIFACT_DIR / "tflite_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
