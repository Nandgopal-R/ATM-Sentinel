# SENSOR_FAULT handling: validation-only investigation

This analysis keeps the trained MLP and the original validation-selected Stage 1 threshold (**0.7615026**) unchanged. It uses only `data/validation.csv`; no test data informs these conclusions.

## What the simulator implements

The simulator picks one float channel from temperature, humidity, pressure, light, voltage, or current and applies one subtype after fault onset:

| subtype | simulator behavior | validation windows | median Stage 1 probability | false negatives |
|---|---|---:|---:|---:|
| `SENSOR_STUCK` | freezes the raw channel exactly; may add nulls | 25 | 0.0018 | 25 / 25 |
| `SENSOR_BIAS` | fixed 8–40 noise-sigma offset | 26 | 0.0887 | 26 / 26 |
| `SENSOR_DRIFT` | linear 20–120 noise-sigma ramp | 34 | 0.0458 | 34 / 34 |
| `SENSOR_SPIKE` | 1–6% random signed spikes, then schema clipping | 26 | 0.9999 | 0 / 26 |
| `SENSOR_NOISE` | 12–55× extra channel noise, then schema clipping | 25 | 1.0000 | 0 / 25 |

There is no separate simulator subtype for clipping or out-of-range values. The sensor emulator clips values to the schema range before emitting telemetry. Ordinary background read dropouts occur at `2e-5` per sample; `SENSOR_STUCK` additionally injects a 0–8% null rate.

## Recommended split

`SENSOR_STUCK` belongs in a deterministic pre-ML sensor-validation layer. Its validation fault channel is current: all 25 windows have `current_std = 0`, `current_delta = 0`, and an exact fixed value of 1.4104 A. NORMAL current windows have mean standard deviation 0.0060 A. The MLP treating this as NORMAL is a pipeline-boundary problem, not a threshold problem.

`SENSOR_BIAS` and `SENSOR_DRIFT` should remain ML-detected integrity faults. Their values can be in schema range and overlap stable device calibration or genuine environmental variation. The validation bias run affects temperature and the drift run affects humidity. The drift does carry a signal (`humidity_slope` mean −0.0228/min versus +0.0078/min for NORMAL), but the present MLP assigns both subtypes low probabilities. This is evidence for a later, separately validated training or feature experiment; it does not justify a schema-only rule.

`SENSOR_SPIKE` and `SENSOR_NOISE` also remain in Stage 1 ML. The current features already distinguish their validation examples: current standard deviation is 0.0242 A for spikes versus 0.0060 A in NORMAL, and voltage standard deviation is 21.68 V for noise versus 1.05 V in NORMAL.

## Pre-ML sensor validation design

Apply these checks to raw 1 Hz samples before interpolation and feature extraction:

1. **Telemetry contract.** Reject non-finite values, wrong types, and values outside the relevant `schema.yaml` range. This must happen before any clipping in firmware. A schema violation is an integrity event.
2. **Sustained missing data.** `null` is permitted by the schema, and a single missing sample is not a sensor fault. Flag a channel only when it has no usable reading across a full 60-second publish interval, which prevents the required summary from being formed.
3. **Stuck value.** For a nullable float channel, flag `SENSOR_STUCK` when all non-null samples in the 300-second inference window equal one schema-quantized value. This follows the simulator’s exact freeze behavior and uses each field’s declared resolution instead of a learned threshold.
4. **Endpoint saturation.** Record repeated readings at a schema endpoint as a diagnostic. Do not independently label it as sensor failure: 0 V can be a real power outage and 0 A can be a real stopped load. Escalate only after channel-specific physical consistency checks are added.

The resulting path is:

```text
raw 1 Hz telemetry → sensor validation → SENSOR_FAULT when a deterministic rule fires
                                  ↓ otherwise
                         feature extraction → Stage 1 ML → ABNORMAL → Stage 2 ML
```

This removes obvious integrity failures from the binary gate while retaining subtle, in-range bias, drift, spike, and noise cases for ML. It requires no MLP retraining now.
