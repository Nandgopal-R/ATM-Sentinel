# ATM Sentinel: ESP32 TinyML audio and sensor-fusion module

Offline training on a PC, deployment to an ESP32 (INMP441 + SW-420 + HC-SR501). This module sits
**beside** the existing two-stage malfunction/tamper cascade. It does not retrain or replace that cascade.

```
INMP441 ─► streaming 512-sample frames ─► MFCC/RMS/ZCR/spectral (per frame) ─► 1 s window mean/std (36 features)
                                                                           └► INT8 MLP 36-16-3 ─► P(BACKGROUND, DISPENSE, ABNORMAL)
SW-420 edges/s ─┐
PIR 1 Hz ───────┼─► event segmentation ─► fusion features (13) ─► tamper-priority rule ─► fusion tree ─► UNKNOWN rules
cascade tamper ─┘                                                                    └► EventSummary ─► rolling window (16)
                                                                                         └► compact JSON ─► gateway / backend
```

## Layout

| path | what |
|---|---|
| `train_esp32_model.py` | full offline pipeline: datasets, models A/B/C, INT8 export, C headers, report |
| `data/recordings.csv` | recording manifest: label, recording-level split. Add new recordings here |
| `datasets/atm_full_features.csv` | every audio feature (160 columns incl. metadata) plus synthetic vibration/PIR context |
| `datasets/atm_esp32_features.csv` | only the 36 features the ESP32 computes, plus provenance and split |
| `datasets/atm_fusion_events.csv` | event-level fusion dataset (synthetic sensors, real audio windows) |
| `models/atm_audio_model.tflite` | deployed INT8 audio model (3,144 B) |
| `models/atm_audio_cnn.tflite` | log-Mel CNN, kept for comparison only, not deployed |
| `model_metadata.json` | feature order, DSP parameters, normalisation, quantisation, rules |
| `../../firmware/atm_sentinel/` | **where the module runs**: `esp32_preprocessing.*`, `esp32_inference.*` and the generated headers, beside the main firmware |
| `../../firmware/tests/atm_tinyml_demo/` | stand-alone sketch (module only) with copies of those 6 files; `tools/check_demo_sync.sh` checks they match |
| `tools/host_parity_test.cpp`, `tools/check_parity.py` | build the firmware's module files on a PC and compare them with Python |
| `reports/` | training report, metrics, parity results |
| `esp32_benchmark.md` | flash/RAM/latency: compiled vs estimated vs still to be measured |

## Reproduce

Run from this folder (`ml/audio_tinyml/`):

```bash
pip install tensorflow scikit-learn pandas scipy        # ffmpeg must be on PATH
python train_esp32_model.py --manifest data/recordings.csv --audio-dir data/src --out .
#   writes the two headers straight into ../../firmware/atm_sentinel/ (FW_DIR)
FW=../../firmware/atm_sentinel
g++ -O2 -std=c++17 -iquote $FW tools/host_parity_test.cpp \
    $FW/esp32_preprocessing.cpp $FW/esp32_inference.cpp -o build/host_parity_test
python tools/check_parity.py          # C++ vs Python feature and prediction parity
./build/host_parity_test scenario     # scripted event run, prints the JSON the backend receives
tools/check_demo_sync.sh              # demo sketch copies == firmware/atm_sentinel
```

`-iquote`, not `-I`: the firmware folder also holds the cascade's `features.h`, which would
shadow glibc's `<features.h>` on a PC and break every libc header. (The ESP32 toolchain has no
such header, so the firmware itself is unaffected.) After regenerating the headers, copy the
6 module files into `../../firmware/tests/atm_tinyml_demo/`; the sync check fails until you do.

Both sketches cross-compile for "ESP32 Dev Module" with esp32 core 3.3.12 (sizes in
`esp32_benchmark.md`). The stand-alone demo (`firmware/tests/atm_tinyml_demo/`, pins 26/27 as in
HARDWARE.md) prints the `self-test`, `bench` and `[stats]` lines to copy into `esp32_benchmark.md`.

## How it is integrated (firmware/atm_sentinel/atm_sentinel.ino, `ENABLE_AUDIO`)

1. The module files sit beside the main sketch; the cascade files (`features.h`, `models.h`) are untouched.
2. `audioTask()` runs pinned to core 0 (8 KB stack, priority 3); the demo's 2 KB benchmark buffer is not included.
3. The existing 1 Hz loop calls `interp.tick(audioSec, s[CH_VIB], s[CH_MOT] > 0.5, freshAudio, cascadeTamper)`
   after the sample and any inference for that second. `s[CH_VIB]` is the firmware's own SW-420 count
   (falling edges, 5 ms debounce, per 1 s, max 200; one ISR on the pin). `cascadeTamper` is the latest
   cascade class == TAMPER, held until the next inference (60 s), and always wins.
4. Events go out as `formatEventJson()` on `atm/<id>/event` over the firmware's direct WiFi MQTT link
   (no ESP-NOW gateway exists yet; 338-370 B payloads fit the 2 KB MQTT buffer). Never raw audio.
5. `atm::selfTest()` runs at boot and logs PASS/CHECK; `[stats]` prints every 30 s behind `DEBUG_AUDIO_STATS`.
6. Backend: `tools/mqtt_to_sqlite.py` stores events in table `event`; the admin dashboard shows them in
   "Activity" (cash dispensed, last dispense).

## Adding real recordings (the most valuable next step)

Add rows to `data/recordings.csv` with `split` = `train`, `test` or `block`, then rerun the script. With
≥2 recordings per class, each class gets a recording-level test hold-out. Recordings made through the
INMP441 in place remove the biggest domain gap: the current clips are internet MP3s.
