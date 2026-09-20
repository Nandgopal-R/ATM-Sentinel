# Sensor validation firmware TODO

Implement this layer in ESP32 C++/Arduino firmware before feature extraction and inference:

```text
raw 1 Hz sensor readings
        ↓
sensor validation
        ├── deterministic sensor fault → SENSOR_FAULT
        ↓ otherwise
300-second feature window
        ↓
feature extraction
        ↓
normalization
        ↓
Stage 1 ML
        ↓ abnormal
Stage 2 ML
```

## Deterministic checks

1. **Telemetry contract**
   - Validate the expected type.
   - Reject non-finite values.
   - Reject readings outside the corresponding `schema.yaml` range.
   - Preserve `null` for missing readings; do not replace it with zero.

2. **Sustained missing data**
   - Occasional missing samples are allowed.
   - Flag a channel when no usable reading exists across a complete 60-second publish interval.

3. **`SENSOR_STUCK`**
   - Evaluate each nullable float channel over the 300-second inference window.
   - Flag `SENSOR_FAULT` when every usable reading has exactly one schema-quantized value.

4. **Endpoint saturation**
   - Record persistent readings at a schema endpoint for diagnostics.
   - Do not classify saturation alone as `SENSOR_FAULT`: valid states can include 0 V or 0 A.

## ML boundary

- `SENSOR_BIAS` and `SENSOR_DRIFT` remain known ML limitations for later work.
- `SENSOR_SPIKE` and `SENSOR_NOISE` remain in the ML path.
- Firmware must reproduce the trained model’s manifest feature order, train-only normalization constants, standardized-value clipping, and all feature preprocessing exactly.
