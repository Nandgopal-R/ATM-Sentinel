# ATM Sentinel Stage 1: NORMAL / ABNORMAL

Stage 1 collapses all non-`NORMAL` labels into `ABNORMAL`:

| source label | Stage 1 value |
|---|---:|
| `NORMAL` | 0 |
| `COOLING_FAILURE`, `HIGH_HUMIDITY`, `POWER_FAILURE`, `TAMPER`, `SENSOR_FAULT` | 1 |

## Model and preprocessing

The model is a compact MLP: **57 → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, sigmoid)**. It has 1,073 trainable parameters.

Features are read strictly in `data/manifest.json` order. `data/scaler.json` supplies the existing training-only mean and standard deviation; standardized values are clipped to ±8 before inference. The same order, scaler, and clipping must be used on the ESP32 before passing a `[1, 57]` `float32` tensor to the model.

The six-class training CSV is balanced per source class, but binary collapsing produces 1,150 NORMAL and 5,750 ABNORMAL windows. Training uses inverse-frequency class weights: NORMAL `3.0`, ABNORMAL `0.6`. This gives each binary class equal loss mass without duplicating overlapping windows.

The model uses only Dense, ReLU, and sigmoid operations supported by the TFLite Micro built-in resolver. The deployable FlatBuffer is float32 because validation of a straightforward full-int8 conversion introduced material probability error from its one shared input scale across heterogeneous engineered features. The float32 model is 6,668 bytes and matches Keras on validation within `2.98e-7`. Quantization-aware training or a per-feature fixed-point transform can be evaluated later without changing the source dataset or scaler.

## Threshold and results

Stage 1 uses the original validation-selected balanced-accuracy threshold: **0.761503**. A recall-constrained threshold comparison is retained as analysis only; it is not the deployed policy because it routes too many healthy windows to Stage 2.

| validation-only strategy | threshold | recall | specificity | false negatives | false positives |
|---|---:|---:|---:|---:|---:|
| maximize balanced accuracy (previous) | 0.761503 | 0.8290 | 0.9919 | 93 | 2 |
| recall ≥ 0.90 (rejected) | 0.051815 | 0.9007 | 0.5650 | 54 | 107 |
| recall ≥ 0.95 | 0.011955 | 0.9504 | 0.1992 | 27 | 197 |

The 90% and 95% options were rejected: the latter escalates 80% of known-NORMAL validation windows, and even the former lowers specificity to 56.5%. SENSOR_STUCK is now investigated as a deterministic integrity fault before changing the model.

| split at deployed threshold | accuracy | balanced accuracy | precision (ABNORMAL) | recall / sensitivity (ABNORMAL) | specificity (NORMAL) | F1 (ABNORMAL) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.8797 | 0.9105 | 0.9956 | 0.8290 | 0.9919 | 0.9047 | 0.9069 |
| final test | 0.8864 | 0.9100 | 0.9843 | 0.8525 | 0.9675 | 0.9137 | 0.9105 |

Final test confusion matrix at the deployed threshold (rows actual, columns predicted; NORMAL then ABNORMAL): `[[238, 8], [87, 503]]`.

At the previous threshold, 85 of 93 validation false negatives were `SENSOR_FAULT`; cooling, humidity, power, and tamper were already highly separable. `SENSOR_FAULT` probability overlaps NORMAL (medians 0.090 and 0.036). This points primarily to separability in the sensor-fault simulation/features rather than the threshold or binary class weighting. Threshold reduction helps, but it cannot solve that overlap without increasing false alerts.

Machine-readable outputs for the deployed operating point are in [`artifacts/`](artifacts/results.json). The earlier recall-threshold comparison is retained under [`artifacts/threshold_analysis/`](artifacts/threshold_analysis/threshold_comparison.json):

- [`metrics/validation.json`](artifacts/metrics/validation.json), [`metrics/test.json`](artifacts/metrics/test.json), and [`threshold.json`](artifacts/threshold.json)
- [`threshold_comparison.json`](artifacts/threshold_analysis/threshold_comparison.json) and [`false_negatives_by_class.json`](artifacts/threshold_analysis/false_negatives_by_class.json)
- [`sensor-fault handling analysis`](analysis/SENSOR_FAULT.md)
- [`stage1.keras`](artifacts/stage1.keras) and [`stage1.tflite`](artifacts/stage1.tflite)
- [`tflite_metadata.json`](artifacts/tflite_metadata.json)

`artifacts/results.json` and `artifacts/metrics/` retain the earlier balanced-accuracy baseline created during training. The selected threshold and final evaluation above are the current Stage 1 operating point.

## Run

Use uv from this directory. It selects Python 3.13 from `.python-version` and creates an isolated `.venv`.

```bash
cd ml/stage-1
uv sync
uv run train.py
uv run export_tflite.py
uv run evaluate.py --split test
uv run sensor_fault_analysis.py
```

`train.py` writes the balanced-accuracy threshold. `sensor_fault_analysis.py` inspects SENSOR_FAULT subtypes using validation only and does not change the threshold. `evaluate.py` reuses the stored threshold and never selects a new one. `export_tflite.py` validates TFLite versus Keras on validation only.
