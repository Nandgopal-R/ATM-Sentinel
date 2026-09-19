# Calibration basis

Every constant in `configs/default.yaml`, and where it comes from. This is the
section that turns "we made up some numbers" into a defensible methodology, so
it is worth carrying into the report more or less directly.

The calibration level used here is **parametric**: constants are taken from
sensor datasheets, supply-quality standards and standard building-physics
values, and the resulting statistics are checked against published
characteristics of indoor sensor networks. Nothing is downloaded at generation
time, so the pipeline is self-contained and reproducible offline.

## Sensor noise and resolution

| constant | value | basis |
|---|---|---|
| BME280 temperature resolution / accuracy | 0.01 °C / ±1.0 °C | BST-BME280 datasheet |
| BME280 temperature short-term noise | 0.02 °C RMS | datasheet normal-mode oversampling figure |
| BME280 humidity resolution / accuracy | 0.01 %RH / ±3 %RH | datasheet |
| BME280 pressure resolution / noise | 0.01 hPa / 0.02 hPa | datasheet (absolute accuracy ±1 hPa) |
| VEML7700 error model | multiplicative, 3.5% 1σ | gain/integration-time calibration error is proportional, not additive |
| INA219 resolution | 0.8 mA | 12-bit, 0.1 Ω shunt, ±3.2 A range |
| ZMPT101B effective accuracy | ±2 V, 0.4 V noise | dominated by RMS-estimation error over a mains cycle, not by ADC LSB |
| SW-420 debounce | 5 ms | contact bounce on a spring/reed module |
| HC-SR501 hold time | 3 s | module's adjustable retriggerable one-shot, set to minimum |

**Verification.** `qa/QA_REPORT.md` §1 reports the 1-sample difference standard
deviation per channel on healthy runs. It should land near the datasheet
short-term noise for each sensor. It also reports lag-1 autocorrelation, which
sits near 1.0 on the environmental channels — the signature of a physically
lagged process. An i.i.d. row generator produces ~0.0 there, so this single
number is the cheapest evidence that the data is a time series rather than a
pile of samples.

## Thermal model

The lumped model is `C dT/dt = (T_lobby − T)/R + Q_elec − k_cool·health·(T − T_lobby)`.

| constant | value | basis |
|---|---|---|
| `heat_capacity_j_per_k` | 6480 J/K | a few kg of sheet steel plus electronics; c_p(steel) ≈ 490 J/kg·K |
| `q_electronics_w` | 55 W | idle draw of an ATM's controller, display and card reader |
| `q_transaction_burst_w` | 40 W for ~20 s | shutter motor, dispenser, receipt printer |
| `passive_conductance_w_per_k` | 2.5 W/K | natural convection plus conduction through a vented cabinet |
| `cooling_conductance_w_per_k` | 11 W/K | forced convection from a running cabinet fan |

Derived behaviour, which is what should actually be checked:

| cooling health | steady-state rise | time constant |
|---|---|---|
| 1.00 (healthy) | +4.1 K | 8.0 min |
| 0.35 | +8.7 K | 17.0 min |
| 0.05 | +18.0 K | 35.4 min |
| 0.00 (dead fan) | +22.0 K | 43.2 min |

A healthy cabinet therefore sits around 28.6 °C at a 24.5 °C lobby setpoint,
and a dead fan takes it to roughly 46 °C over about an hour. Both are in the
range reported for sealed equipment enclosures with a 50–60 W internal load.

The 30–90 minute degradation window matches the shape of NAB's
`machine_temperature_system_failure.csv`, which is real degradation-to-failure
data from an industrial machine. If a stronger citation is wanted later, that
file is the one to overlay against a COOLING_DEGRADATION run.

## Psychrometrics

Magnus-Tetens with the Alduchov & Eskridge (1996) coefficients
(A = 6.1094 hPa, B = 17.625, C = 243.04 °C), maximum relative error under 0.4%
over −40…50 °C. Mixing ratio uses ε = 621.97 g/kg.

`tests/test_simulator.py` checks es(0 °C) ≈ 6.11 hPa and es(20 °C) ≈ 23.4 hPa
against reference values, and asserts the round-trip
RH → mixing ratio → RH is exact.

## Lobby and environment

| constant | value | basis |
|---|---|---|
| outdoor mean / amplitude | 29 °C / ±5 °C | generic Indian metro annual average with a typical diurnal range |
| outdoor RH mean / amplitude | 62% / ±14%, anti-phase with temperature | standard diurnal RH behaviour at roughly constant absolute humidity |
| barometric semidiurnal tide | 1.2 hPa, peaks ~10:00 and 22:00 LT | the S2 atmospheric tide, a real feature of barometric records |
| AC setpoint / deadband | 24.5 °C / ±0.8 °C | typical commercial split-AC thermostat |
| AC cycle period | ~18 min | short-cycling period of a lightly loaded split unit |
| infiltration gain | 0.10 | fraction of the outdoor swing reaching a conditioned vestibule |
| lobby lighting | 185 lux + daylight to 240 lux | office/retail ambient (IS 3646 recommends 150–300 lux for such spaces) |

The AC deadband cycling matters more than it looks: it puts a real periodic
signal into `temperature_std`, which is why a naive "temperature is unstable"
rule produces false positives and why the model has to learn something better.

## Power

| constant | value | basis |
|---|---|---|
| nominal mains | 230 V | IS 12360 / CEA standard LV supply |
| slow drift | 2.2 V 1σ, τ = 15 min | normal distribution-feeder wander within statutory tolerance |
| background sags | 2.5/h, 3–12 V deep, 2–25 s | other loads on a shared distribution transformer |
| sag / swell feature thresholds | 207 V / 253 V | ±10% of nominal |
| SMPS dropout | 165 V | typical universal-input SMPS hold-up limit |
| undervoltage target | 185–207 V | below statutory tolerance, above SMPS dropout |
| overvoltage target | 253–271 V | above statutory tolerance |

Background sags occur in **healthy** runs too. That is deliberate: they are the
nuisance events a naive voltage threshold would flag, and they are what makes
POWER_FAILURE a detection problem rather than a comparison.

If waveform-level power analysis is ever added, the public power-quality
disturbance datasets are sampled around 12.8 kHz and would require FFT on the
ESP32 — they do not transfer to a 1 Hz RMS channel, so they are deliberately
not used here.

## Events

| constant | value | basis |
|---|---|---|
| hourly transaction rate | 0.4–14 /h, twin peaks ~10:00 and ~18:00 | standard retail-banking footfall shape |
| presence duration | 45–130 s | a full withdrawal interaction |
| SW-420 background rate | 0.004 counts/s | building traffic and passing vehicles |
| tamper intensity | 18–65 counts/s | sustained prying or drilling against an enclosure |

TAMPER runs contain 4–8 incidents each. That density is unrealistic for a single
ATM and is recorded in `manifest.json` rather than hidden: it exists because
tamper incidents are short and the class would otherwise be too small to
balance. There is no public dataset of someone prying at a metal enclosure, so
this class is the least externally grounded in the whole simulator, and the
report should say so. It is also the cheapest class to collect for real — an
afternoon with the SW-420 and the enclosure would replace it outright, and doing
that is the single highest-value hardware step available.

## Sensor faults

Subtypes follow the Bruijn et al. (2016) taxonomy — random, malfunction, bias,
drift, polynomial drift — which gives a published reference and a released
fault-injected Intel Lab benchmark to check the implementation against. Fault
magnitudes are expressed in multiples of each channel's own noise σ, so a bias
fault is equally hard to detect on every channel by construction rather than by
accident.

Compound faults (a sensor fault on top of another scenario) are implemented as a
hook but disabled. Enabling them would be the obvious way to make Stage-2
harder in a later iteration.

## What is *not* calibrated

Honest gaps, worth listing in the report rather than glossing:

* Tamper vibration amplitude and rate have no external reference (see above).
* The cabinet heat capacity is an estimate, not a measurement. It sets the time
  constant, so it directly affects detection lead time.
* Lobby occupancy shadowing of the light sensor (−22 lux) is a plausible figure,
  not a measured one.
* No seasonal variation: every run is drawn from the same annual-average
  climate. Monsoon would substantially change the HIGH_HUMIDITY baseline.
