# ATM Sentinel — canonical telemetry format + synthetic simulator

Phases 1 and 2 of the ATM Sentinel data pipeline: a single canonical telemetry
schema, and a physically-grounded simulator that produces labelled time-series
runs from it.

```
ATM simulator  ->  raw 1 Hz telemetry  ->  300 s windows  ->  features
                            |                                    |
                            v                                    v
                   data/raw/*.parquet              data/{train,validation,test}.csv
                   (demo / replay / DB)                 (Stage 1 / Stage 2)
```

## Quick start

```bash
pip install -r requirements.txt
python -m atmsim generate --config configs/default.yaml --out .
python -m pytest tests -q
python qa/baseline.py          # sanity classifier
python qa/qa_report.py         # realism + difficulty report, writes qa/QA_REPORT.md
```

Generation takes about 10 seconds and produces ~2.8 M raw observations.

## What comes out

| path | contents |
|---|---|
| `data/raw/RUN_*.parquet` | full 1 Hz telemetry per run + the ground-truth `state` column. Use for the demo replay and the dashboard. |
| `data/events.csv` | one row per run: scenario, subtype, onset timestamp, failure timestamp, severity, parameters. The ground-truth annotation file. |
| `data/train.csv` | 6,900 windows, balanced at 1,150 per class |
| `data/validation.csv` | 790 windows |
| `data/test.csv` | 836 windows |
| `data/scaler.json` | per-feature mean/std **fitted on train only**. These constants get baked into the ESP32 firmware. |
| `data/manifest.json` | config hash, seed, ATM pools per split, feature list, per-split class counts |

57 features per window. Metadata columns (`run_id`, `label`, `purity`,
`severity`, `time_to_failure_s`, …) sit at the end of each CSV and must be
dropped before training — the feature list in `manifest.json` is authoritative.

## Canonical format (Phase 1)

`schema.yaml` is the single source of truth. Ten fields, 1 Hz:

```json
{"atm_id": "ATM_001", "timestamp": 1790793000, "temperature": 28.41,
 "humidity": 54.87, "pressure": 1008.22, "light": 183.4, "voltage": 230.9,
 "current": 1.414, "vibration": 0, "motion": 1}
```

Three rules that the ESP32 firmware must also follow:

1. **Missing is `null`**, never `0` and never `-999`. The SENSOR_FAULT class
   needs a way to say "no reading", and `*_null_frac` is a real feature.
2. **Labels never appear in telemetry.** Ground truth lives in `events.csv`.
   The raw stream is exactly the shape the hardware will publish.
3. **`vibration` is an edge count, not an amplitude.** The SW-420 is a
   normally-closed contact module; firmware counts falling edges with a 5 ms
   debounce and reports the count for that second. **`motion` is a latched
   binary** — the HC-SR501 holds its output HIGH for ~3 s after each detection.

Not in telemetry, deliberately: cash and transaction counts (they come from the
bank's transaction system, not a sensor) and network/RSSI state (observable at
the ESP8266 gateway, so network failure is a gateway rule rather than a
Stage-2 class).

## How the simulator works (Phase 2)

Six layers. A fault is injected into the *physics*, never written directly onto
a channel, which is why the cross-channel signatures come out coherent.

1. **Environment** — outdoor diurnal temperature/RH, the semidiurnal barometric
   tide, synoptic pressure wander.
2. **Lobby** — AC hysteresis cycling across a deadband (not a held setpoint),
   infiltration of the outdoor swing, daylight through the glazing.
3. **Cabinet thermal** — lumped first-order ODE. Losing cooling capacity both
   raises the equilibrium and lengthens the time constant, so a fault is a
   decelerating exponential approach to a new equilibrium, not a linear ramp.
   Healthy: +4.1 K over lobby, τ ≈ 8 min. Dead fan: +22 K, τ ≈ 43 min.
4. **Psychrometrics** — absolute humidity (mixing ratio) is the state variable;
   RH is derived from it via Magnus. See below.
5. **Events** — non-homogeneous Poisson footfall driving PIR latching, SW-420
   bursts, and transaction heat.
6. **Sensor emulation** — per-device calibration bias, datasheet noise,
   quantisation, range clipping, dropouts.

### The three design decisions that make the dataset defensible

**Absolute humidity is the state variable.** When a cabinet heats up at constant
mixing ratio, saturation vapour pressure rises and RH *falls*. So:

| class | temp slope | RH slope | **abs. humidity slope** |
|---|---|---|---|
| NORMAL | −0.002 /min | +0.009 /min | +0.001 |
| COOLING_FAILURE | **+0.085** | **−0.171** | +0.002 (flat) |
| HIGH_HUMIDITY | −0.002 (flat) | **+0.214** | **+0.054** |

Both faults move relative humidity. Only moisture ingress moves the mixing
ratio. If RH were generated from a Gaussian, Stage-2 would separate these two on
an artefact that does not exist in a real cabinet.

**The fan-current channel separates degradation from stall.** A dust-loaded or
worn fan draws more current while moving less air; a seized fan draws none. Both
raise temperature.

| subtype | current mean | temp slope |
|---|---|---|
| NORMAL | 1.417 A | −0.006 /min |
| COOLING_DEGRADATION | **1.878 A** | +0.067 |
| FAN_STALL | **0.917 A** | +0.112 |

Current rises *before* temperature moves, which is the physical basis of the
predictive-maintenance claim.

**Mains and the DC rail are separate circuits.** ZMPT101B reads AC mains;
INA219 reads the 12 V auxiliary rail. Above the SMPS dropout voltage the rail is
regulated, so POWER_UNDERVOLTAGE is genuinely a single-channel fault. Below
dropout the rail droops and, under a roughly constant-power load, current *rises*
as voltage falls. In a full outage the fan stops and the cabinet begins to heat —
a three-channel cascade that a row-wise random generator cannot produce.

## Splitting

Splitting happens **before** generation, at two levels:

* **ATM identity.** 44 virtual units partitioned into disjoint pools —
  31 train / 7 validation / 6 test. Per-device calibration bias cannot leak.
  Without this a model can memorise absolute values and will fall apart on a
  unit it has never seen, which is exactly what a fleet deployment does to it.
* **Run.** Runs are allocated per *subtype*, not per class, with at least one run
  of every failure mode in every split. (Allocating per class left the test split
  with two POWER runs that both happened to be short outages and three windows
  for the whole class.)

Windows are cut only inside a run. Train uses a 60 s stride, so windows overlap
as augmentation; validation and test use a 300 s stride and do not overlap.
Normalisation statistics are fitted on train alone.

## Window labelling

A window's label is the **majority** ground-truth class across its 300 samples,
with `purity` recording the majority share. Windows below 80% purity straddle a
fault onset; they are marked `ambiguous` and excluded by default, and retained
with `--keep-ambiguous`. Purity is computed on the Stage-2 class, not the fine
state, so the internal COOLING_DEGRADATION → COOLING_FAILURE transition is not
treated as a boundary — it is one class throughout.

`time_to_failure_s` is negative before the ground-truth failure instant. Never
feed it to the model; use it afterwards for the lead-time metric.

## Difficulty, and why that matters

A RandomForest baseline on the held-out test split:

* **Stage 1** (healthy / abnormal): 0.878 accuracy
* **Stage 2** (fault type): 0.932 accuracy, with genuine SENSOR_FAULT ↔
  HIGH_HUMIDITY confusion and 0.759 recall on SENSOR_FAULT
* **Detection lead time** on COOLING_DEGRADATION: median 130 min before the
  ground-truth failure threshold

Stage-1 at 0.88 rather than 0.99 is the intended result. A synthetic dataset that
scores 0.99 out of the box means the faults were injected too cleanly to be worth
modelling, and any examiner who has seen this before will know why. The report
promises ≥90% Stage-1 — the honest routes there are temporal persistence
(require N of M consecutive abnormal windows, which is what a real deployment
does anyway) and feature selection, not looser fault injection.

Three things worth doing before the report claims the number means something:

1. Train on one severity band and test on another (`severity` is recorded per
   run for exactly this).
2. Hold out a sensor-fault subtype entirely and check that Stage-1 still flags
   it as abnormal — that tests whether you built an anomaly detector or a
   memoriser.
3. Report per-class confusion, not overall accuracy.

## Tuning

Everything lives in `configs/default.yaml`. The knobs most worth revisiting:

* `windowing.window_s` / `train_stride_s` — window size trades detection lead
  time against latency. 300 s is the floor for thermal faults: at 30 s a
  degradation moves temperature ~0.05 °C, inside the BME280 noise floor.
* `counts.runs_per_subtype` — dataset size and class balance.
* `scenarios.*.develop_s` and severity ranges — fault difficulty.
* `sensors.device_bias_sigma` — how hard cross-unit generalisation is.

## Moving to hardware

When the ESP32 node exists, nothing downstream changes. The firmware emits the
same ten fields at 1 Hz; `features.py` and `scaler.json` apply unchanged. The
simulator then becomes the pre-training and augmentation source, and the report
describes it as documented synthetic data generation with a stated calibration
basis — which is a standard, publishable methodology as long as it is described
as exactly that.

See `qa/CALIBRATION.md` for the justification of every physical constant.
