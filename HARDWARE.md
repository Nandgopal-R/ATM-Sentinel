# Hardware reference — ATM Sentinel sensor node

Physical configuration of the ESP32 sensor node. This file is the source of
truth for pin assignments. Code that contradicts it will compile and then fail
silently on hardware.

Board: **DOIT ESP32 DEVKIT V1** (ESP32-D0WD-V3, dual core, 240 MHz)
Serial: 115200 baud

---

## Pin map

| GPIO | Direction | Connected to | Notes |
|---|---|---|---|
| 21 | I2C SDA | BME280, VEML7700, INA219 | shared bus |
| 22 | I2C SCL | BME280, VEML7700, INA219 | shared bus |
| 26 | input | SW-420 vibration, `OUT` | external pull-up on module |
| 27 | input | HC-SR501 PIR, `OUT` | `INPUT_PULLDOWN` |
| 34 | — | **unused** | reserved, see "Not connected" |

All three I2C devices share a single `Wire` bus at **100 kHz**. This is
deliberate — do not split them across `Wire` and `Wire1`.

### I2C addresses

| Device | Address | Measures |
|---|---|---|
| VEML7700 | `0x10` | ambient light (lux) |
| INA219 | `0x40` | bus voltage, current, power |
| BME280 | `0x77` | temperature, humidity, pressure |

> **The BME280 is at `0x77`**, confirmed by I2C scan — this is the address
> with `SDO` left floating. Many library examples hardcode `0x76` (the
> `SDO`-grounded address) and will fail silently on this board. Any code
> touching the BME280 must use `0x77` or probe both.

---

## ESP32 pin constraints

These are properties of the chip, not choices. Any future pin reassignment
must respect them.

- **GPIO 6–11** are wired to the SPI flash. Never usable.
- **GPIO 34–39** are **input only** and have **no internal pull-up or
  pull-down**. A digital sensor on these pins needs an external resistor.
- **ADC2 pins** (GPIO 0, 2, 4, 12–15, 25–27) stop returning valid readings
  once WiFi is active. Any analog input **must** be on ADC1 (GPIO 32–39).
- **GPIO 0, 2, 12, 15** are strapping pins, sampled at boot. Avoid — a sensor
  holding one at the wrong level prevents the board from starting.
- **GPIO 26 and 27** were chosen for the digital sensors specifically because
  they support internal pulls, which GPIO 34–39 do not.

---

## Sensor behaviour

Characteristics that change how the code must be written, not just what it
reads.

### SW-420 vibration (GPIO 26)

A mechanical vibration **switch**, not a measurement sensor. A spring makes
contact with a central pin at rest and breaks contact when disturbed.

- **Normally closed.** Idles **LOW** on this board — verify after any rewiring,
  as it varies between modules.
- **The output chatters.** One physical knock produces a burst of edges as the
  spring rings and settles. Observed: ~8 edges for a light tap, ~500 for a
  firm knock, **0 when undisturbed**.
- **Count edges over a window.** Attach an interrupt on `CHANGE` and count.
  Reading the level in the main loop misses most events, because the flicker
  is shorter than a typical loop interval.
- **No frequency or amplitude information is available.** The switch cannot
  encode it. Only pulse timing and density are real features. Spectral
  features would require an accelerometer (MPU6050 / ADXL345).
- The onboard potentiometer adjusts the comparator threshold, not mechanical
  sensitivity.

### HC-SR501 PIR (GPIO 27)

- **Requires ~60 s to settle after power-on.** It fires randomly during this
  window. Suppress or discard motion events for the first minute after boot.
- Powered from **VIN (~4.7 V over USB)**, not the 3.3 V rail — it needs
  >4.5 V for its onboard regulator.
- Output is 3.3 V logic even on a 5 V supply, so it connects directly.
- Jumper set to **H** (repeat trigger).

### BME280 (I2C)

- The die self-heats, and the module sits near the ESP32 which warms under
  WiFi load. Temperature readings run **1–2 °C high** on the breadboard.
- Pressure is of limited value here. Absolute barometric pressure tracks
  weather and altitude, neither of which indicates machine state. Useful only
  if the enclosure is sealed enough for a door-open event to show as a step.
- **Sanity-check the readings.** A frozen, byte-identical set of values across
  consecutive samples means the chip ACKed but is returning stale registers —
  `begin()` succeeding is not proof the sensor works.

### VEML7700 (I2C)

- Configured at gain 1, 100 ms integration time.
- Indoor ambient reads roughly 25–40 lux under room lighting.

### INA219 (I2C)

- Measures current through `VIN+` / `VIN-`, which must be in **series** with
  the load.
- With nothing in the circuit it still initialises and reports ~0.00 V. A
  failure at `begin()` means it is not answering on the bus at all — a
  different problem from a zero reading.
- Calibrated with `setCalibration_32V_2A()`.

---

## Power

Two rails, common ground:

| Rail | Source | Feeds |
|---|---|---|
| 3.3 V | ESP32 `3V3` | BME280, VEML7700, INA219, SW-420 |
| 5 V | ESP32 `VIN` | HC-SR501 |

- Over USB, `VIN` sits at roughly **4.7 V** (Schottky drop), which is only
  just above the HC-SR501's 4.5 V minimum.
- **A brownout reset has been observed** when the PIR draws current on a
  shared supply. Mitigations in place: 470 µF bulk capacitor across the 3.3 V
  rail near the ESP32, 100 nF at each module. Do not add further load to
  `VIN` without re-testing.
- A brownout appears in the serial log as garbage characters mid-line,
  followed by the boot banner — the ROM bootloader prints at a different baud
  rate.

---

## Not connected

Deliberate omissions. Code should **not** reference these.

- **ZMPT101B (AC voltage sensor)** — not wired. The power channel is
  **simulated**, not measured. GPIO 34 is reserved for it but currently
  unused. If it is ever connected: ADC1 only, powered at 3.3 V (not 5 V, which
  would put ~5 V peaks on the ADC pin), through a 1 kΩ series resistor, and
  fed from a low-voltage AC source — never mains.
- **Microphone / audio channel** — dropped from the design. The report
  specified an INMP441; no microphone is present and there is no audio
  feature path.
- **Bluetooth module** — an external module is in the parts inventory but is
  not wired. The ESP32 has Bluetooth built in, making it redundant.

---

## Gateway

The **ESP8266 gateway is not physically wired to the ESP32.** Communication is
wireless — ESP-NOW from node to gateway, then WiFi/MQTT from gateway to cloud.
There is no serial link between the two boards.

---

## Known-good baseline

The bring-up sketch (`atm_sentinel_bringup.ino`) verifies all five connected
sensors and prints readings at 1 Hz. Footprint with no ML or networking:

- **Flash:** 326 KB (24% of 1.31 MB)
- **RAM:** 24 KB globals (7% of 320 KB)

The remaining RAM is the budget for the TFLite tensor arena, WiFi stack, and
MQTT buffers. Size the arena explicitly and check it against this baseline —
an arena too small fails at interpreter init, too large starves the network
stack.
