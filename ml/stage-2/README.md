# ATM Sentinel Stage 2: abnormal-fault classifier

Stage 2 is evaluated independently: each input is assumed to have already been
identified as `ABNORMAL`. It is **not** the Stage 1 → Stage 2 cascade.

## Model and contract

The model is a compact built-in-op MLP:

`57 normalized features → Dense(16, ReLU) → Dense(8, ReLU) → Dense(5, softmax)`

It has 1,109 trainable parameters. Labels use this fixed output mapping:

| output index | class |
|---:|---|
| 0 | `COOLING_FAILURE` |
| 1 | `HIGH_HUMIDITY` |
| 2 | `POWER_FAILURE` |
| 3 | `TAMPER` |
| 4 | `SENSOR_FAULT` |

Input is a `[1, 57]` float32 tensor in the exact `data/manifest.json`
feature order. Preprocessing uses the existing train-only `data/scaler.json`:
`clip((feature - mean) / std, -8, 8)`. This is the same scaler, feature order,
and clipping contract as Stage 1; preprocessing is external to the model.

Only non-`NORMAL` rows are used. The preserved training split has 1,150
examples in each of the five classes, so inverse-frequency weights are all 1.0;
the code still computes and records the weights to guard against a later split
change. All SENSOR_FAULT subtypes are retained, including `SENSOR_STUCK`,
`SENSOR_BIAS`, and `SENSOR_DRIFT`. The documented future deterministic
`SENSOR_STUCK` pre-check has not been applied to this training dataset.

Fixed seed: `20260918`. Training uses sparse categorical cross-entropy, Adam,
and validation-loss early stopping (25-epoch patience). It stopped after 29
epochs; epoch 4 was restored. The test CSV is not loaded by `train.py`.

## Results

These are direct multiclass results conditioned on known abnormal input.

| split | samples | accuracy | balanced accuracy | macro precision | macro recall | macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 544 | 0.9136 | 0.9243 | 0.9219 | 0.9243 | 0.9182 |
| held-out test | 590 | 0.8847 | 0.8921 | 0.8843 | 0.8921 | 0.8828 |

Held-out test per-class metrics:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| COOLING_FAILURE | 0.9239 | 0.9551 | 0.9392 | 89 |
| HIGH_HUMIDITY | 0.8200 | 0.9609 | 0.8849 | 128 |
| POWER_FAILURE | 0.8148 | 0.9167 | 0.8627 | 96 |
| TAMPER | 0.9929 | 0.9929 | 0.9929 | 140 |
| SENSOR_FAULT | 0.8700 | 0.6350 | 0.7342 | 137 |

Rows are actual and columns predicted, in the fixed label order:

```text
[[85,  2,  0,   0,  2],
 [ 0,123,  0,   0,  5],
 [ 1,  2, 88,   0,  5],
 [ 0,  0,  0, 139,  1],
 [ 6, 23, 20,   1, 87]]
```

The principal weakness is SENSOR_FAULT recall. On held-out test data,
`SENSOR_BIAS` is 28/29, `SENSOR_STUCK` is 29/29, `SENSOR_SPIKE` is 22/25,
but `SENSOR_DRIFT` is 8/30 (mostly `POWER_FAILURE`) and `SENSOR_NOISE` is
0/24 (mostly `HIGH_HUMIDITY`). This is a real subtype generalization gap and
should be addressed later with a validation-only feature/training experiment;
it was not masked with test-set tuning.

Detailed machine-readable metrics and subtype slices are in
[`artifacts/metrics/test.json`](artifacts/metrics/test.json); the visual matrix
is [`artifacts/metrics/test_confusion_matrix.png`](artifacts/metrics/test_confusion_matrix.png).

## Deployment artifact

[`artifacts/stage2.tflite`](artifacts/stage2.tflite) is a 6,844-byte float32
TFLite/LiteRT FlatBuffer using only Dense, ReLU, and Softmax operations. It is
intended for a TFLite Micro built-in resolver. Float32 preserves the shared
Stage-1 preprocessing contract; a later int8 deployment needs a separately
validated per-feature input representation or quantization-aware training.

Conversion was checked across all validation rows against the saved Keras
model: maximum absolute probability difference `2.53e-7` and 100% argmax
agreement. Input/output details are recorded in
[`artifacts/tflite_metadata.json`](artifacts/tflite_metadata.json).

## Run

```bash
cd ml/stage-2
uv sync
uv run train.py              # train + validation only
uv run export_tflite.py      # converts and validates on validation only
uv run evaluate.py --split test
```
