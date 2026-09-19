# ATM Sentinel — dataset QA report

- generator 1.0.0, schema 1.0.0, config `9ca78540bd20f7c2`
- 197 runs x 4.0 h = 2,836,800 raw 1 Hz observations
- 57 features, 300 s windows

## 1. Healthy-baseline channel statistics

| channel | mean | std | min | max | lag-1 autocorr | 1-sample diff std |
|---|---|---|---|---|---|---|
| temperature | 29.061 | 0.592 | 27.540 | 30.630 | 0.9988 | 0.0293 |
| humidity | 54.396 | 2.693 | 47.920 | 61.150 | 0.9996 | 0.0732 |
| pressure | 1008.150 | 1.725 | 1003.190 | 1011.380 | 0.9998 | 0.0357 |
| light | 317.826 | 100.711 | 157.200 | 475.000 | 0.9996 | 2.7295 |
| voltage | 230.103 | 2.574 | 212.600 | 239.900 | 0.9520 | 0.7978 |
| current | 1.415 | 0.007 | 1.387 | 1.443 | 0.2205 | 0.0085 |

Lag-1 autocorrelation near 1.0 on the environmental channels is the signature of a
physically-lagged process. An i.i.d. row generator produces ~0.0 here. The 1-sample
difference std should sit near the datasheet short-term noise for each sensor.

## 2. Temperature-humidity coupling (the anti-shortcut check)

| class | temp_slope (/min) | humidity_slope (/min) | abs_humidity_slope (/min) | temp_rh_corr |
|---|---|---|---|---|
| COOLING_FAILURE | +0.0853 | -0.1712 | +0.0023 | -0.738 |
| HIGH_HUMIDITY | -0.0015 | +0.2141 | +0.0537 | -0.593 |
| NORMAL | -0.0016 | +0.0088 | +0.0014 | -0.787 |
| POWER_FAILURE | +0.0308 | -0.0677 | +0.0003 | -0.811 |
| SENSOR_FAULT | +0.0002 | -0.0063 | -0.0013 | -0.723 |
| TAMPER | +0.0003 | -0.0061 | -0.0013 | -0.803 |

COOLING_FAILURE: temperature up, RH down, **absolute humidity flat**, correlation strongly negative.
HIGH_HUMIDITY: temperature flat, RH up, **absolute humidity up**. The two faults are separated by
the mixing ratio, not by the RH level — which is the real physics rather than a synthetic artefact.

## 3. Fan-current signature (degradation vs stall)

| subtype | current_mean | current_slope | current_min | temperature_slope |
|---|---|---|---|---|
| COOLING_DEGRADATION | 1.878 | +0.00378 | 1.849 | +0.0666 |
| FAN_STALL | 0.917 | -0.00039 | 0.904 | +0.1117 |
| NORMAL | 1.417 | +0.00002 | 1.399 | -0.0055 |

Degradation raises fan current (bearing strain); a stall drops it. Both raise temperature.

## 4. Detection lead time (COOLING_DEGRADATION, test split)

- runs with a pre-failure detection: 2
- median lead time: **130.2 min** before the ground-truth failure threshold
- range: 115.0 – 145.4 min

This is the number the predictive-maintenance claim should rest on. Accuracy on faults you
injected yourself is cheap; warning time is not.

## 5. Difficulty

A RandomForest baseline (`qa/baseline.py`) reaches ~0.88 Stage-1 and ~0.93 Stage-2 on the
held-out test split, with genuine SENSOR_FAULT / HIGH_HUMIDITY confusion. A dataset that
scored 0.99 would mean the faults were injected too cleanly to be worth modelling.
