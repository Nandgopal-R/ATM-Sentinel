# ESP32 benchmark: ATM Sentinel TinyML module

Model version `atm-tinyml-20261020-2dad03`. Target: classic ESP32 (Xtensa LX6, 240 MHz, 520 KB SRAM,
single-precision FPU, 4 MB flash), Arduino-ESP32 core 3.x.

> **Status: NOT verified on hardware.** Every number below is one of three kinds:
> **compiled** (exact, from the real ESP32 toolchain/linker or from the artifacts), **host-measured** (on a PC,
> not ESP32 timing), or **estimate**. The last table must be filled in from the demo sketch's serial
> output on the real board before anyone calls this deployment verified.

## 1. Model artifacts (exact)

| item | size | notes |
|---|---|---|
| `atm_audio_model.tflite` (INT8 MLP 36→16→3) | 3,144 B | only linked when `ATM_USE_TFLM` is defined |
| native INT8 weights + biases + requant params | 576 + 64 + 64 + 16 (L0), 48 + 12 + 12 + 3 (L1) = **795 B** | default backend |
| feature normalisation (mean/std, 36×2 float) | 288 B | |
| fusion decision tree (31 nodes) | 465 B | feature, threshold, children, 6 leaf probs per node |
| self-test vectors (9 windows) | ~1.4 KB | can be compiled out after bring-up |
| log-Mel CNN (comparison only, not deployed) | 8,752 B | also needs a 31×26 float patch buffer (3.2 KB) |

## 2. Cross-compiled firmware size (exact, ESP32 toolchain)

Compiled with arduino-cli, `esp32:esp32@3.3.12`, board `esp32:esp32:esp32` (ESP32 Dev Module),
default partition scheme (1.25 MB app). The baseline sketch has the same I2S, interrupt, FreeRTOS
task and serial setup without the TinyML module, so the difference is this module's own cost.
The build has **zero compiler warnings**.

| build | flash (program storage) | static RAM (globals) |
|---|---|---|
| `atm_tinyml_demo` (native INT8 backend) | 325,364 B (24 % of 1.25 MB) | 44,724 B (13 %), 282,956 B left for stack/heap |
| baseline (I2S + ISR + task, no TinyML) | 298,008 B | 24,524 B |
| **module cost** | **+27,356 B** | **+20,200 B** |

Integrated into the main firmware (`firmware/atm_sentinel`, same toolchain and board, `ENABLE_AUDIO 1`,
compiled 2026-10-20). The baseline is the same sketch before integration (cascade + sensors + WiFi/MQTT):

| build | flash (program storage) | static RAM (globals) |
|---|---|---|
| main firmware before integration | 1,024,994 B (78 % of 1.25 MB) | 80,596 B (24 %) |
| main firmware with the module | 1,072,850 B (81 %) | 99,852 B (30 %), 227,828 B left |
| **cost inside the main firmware** | **+47,856 B** | **+19,256 B** |
| stand-alone demo (`firmware/tests/atm_tinyml_demo`, pins 26/27, debounced SW-420) | 325,328 B | 44,732 B |

The flash cost is higher than the +27 KB above because the main firmware did not use I2S before; the
`ESP_I2S` driver is now linked in. No new compiler warnings (the one warning in the sketch,
`vibEdges++` on a volatile, predates the module).

Breakdown from the ELF symbol table:

| symbol | section | bytes |
|---|---|---|
| `AudioFeatureExtractor fx` (all DSP tables + work buffers) | .dram0.bss | 16,296 |
| demo self-benchmark frame buffer (`setup()::frame`) | .dram0.bss | 2,048 (removable) |
| I2S block buffers `raw` + `pcm` | .dram0.bss | 2,048 |
| `EventInterpreter interp` | .dram0.bss | 552 |
| all model constants `ATM_*` (weights, requant, normalisation, fusion tree) | .flash.rodata | 1,548 |
| self-test vectors `ATM_TV_*` | .flash.rodata | 1,404 |
| module code (extractor, FFT, INT8 engine, tree, rules, JSON) | .flash.text | ~8 KB |

The TFLite Micro backend (`-DATM_USE_TFLM`) was **not** cross-compiled for ESP32: the Arduino
TFLM library could not be installed in this environment. It was compiled and run on a PC against
tflite-micro built from source (see section 6). Expect roughly +40–60 KB flash for the TFLM runtime
with two kernels, plus the 3,144 B flatbuffer and the 6 KB arena.

## 3. RAM budget (exact sizes from the code + estimates)

| consumer | bytes | kind |
|---|---|---|
| `AudioFeatureExtractor` (window, FFT twiddles, bit-reverse, sparse mel, DCT, 1 frame of work buffers, accumulators) | 16,296 | exact (ESP32 ELF) |
| `EventInterpreter` (+ 16-event rolling window, 4-summary queue) | 552 | exact (ESP32 ELF) |
| I2S read buffers (256 × int32 + 256 × float) | 2,048 | exact (ESP32 ELF) |
| demo bench buffer | 2,048 | exact; delete when integrating |
| audio task stack | 8,192 | configured; the sketch prints the unused high-water mark |
| classifier scratch (stack) | ~300 | estimate |
| TFLM tensor arena (only with `ATM_USE_TFLM`) | 1,360 used / 6,144 reserved | host-measured with TFLM `arena_used_bytes()` |
| I2S DMA descriptors/buffers (core default) | ~4–8 K | estimate, owned by the core driver |
| **total for this module** | **≈ 27 KB native, ≈ 33 KB with TFLM** | out of ~300 KB typically free on an ESP32 Arduino build |

No raw audio history is kept: one 512-sample frame plus running mean/M2 accumulators per window.
No spectrogram is stored.

## 4. Latency and CPU (host-measured + estimate)

| stage | host x86-64 (measured) | ESP32 @240 MHz (estimate) | budget |
|---|---|---|---|
| per-frame features (DC removal, Hann, 512-pt FFT, power, centroid/rolloff/flatness, 26 mel, 13 MFCC) | 11.6 µs | ~0.6–1.5 ms | 32 ms per frame |
| INT8 MLP + float softmax (native) | 0.7 µs | ~15–40 µs | once per ~1 s window |
| INT8 MLP via TFLite Micro | 1.0 µs | ~40–100 µs | once per ~1 s window |
| fusion tree + rules | <1 µs | <10 µs | once per event |
| **audio CPU load** | | **~2–5 % of one core** | |

How the estimate was made: the per-frame work is ~25 k float ops (FFT) plus 256 `logf` (flatness) and
~1 k MACs (mel + DCT). The hot loops were written in single precision on purpose, because the
LX6 FPU has no double support. A 50–130× slowdown versus the host is assumed. Treat these as
order-of-magnitude numbers only.

## 5. To be measured on the real board (fill in)

Flash `firmware/atm_tinyml_demo` and copy the lines it prints:

| measurement | sketch line | value |
|---|---|---|
| self-test vs Python reference | `self-test: 9/9 … -> PASS` | |
| per-frame feature time | `bench: frame features … us/frame` | |
| classify time | `bench: … classify … us` / `[stats] classify … avg (… max)` | |
| audio CPU load | `[stats] audio CPU ~…%` | |
| free heap / min free heap | `[stats] heap free … (min …)` | |
| audio task stack headroom | `[stats] audio stack free … B` | |
| I2S read errors | `[stats] i2s errors …` | |

## 6. Verification done so far

| check | result |
|---|---|
| C++ features vs Python float64 reference (36 windows, 5 recordings) | max abs diff 7.9e-6 (worst: `flatness_std`) |
| native INT8 engine vs TFLite interpreter (same inputs) | max prob diff 0.0039 (= 1 step of TFLite's int8 softmax), argmax 36/36 |
| TFLite Micro backend (host build of tflite-micro) vs TFLite interpreter | max prob diff 5e-7, argmax 36/36, arena 1,360 B |
| embedded self-test vectors, both backends | 9/9 |
| fusion `decide()` in C++ on 750 held-out synthetic events | accuracy 0.853, UNKNOWN rate 0.125, tamper recall 1.000 |
| scripted event scenario (`host_parity_test scenario`) | short visits, repeated-visit flag, abnormal event, tamper early alert + priority behave as specified; the scripted normal withdrawal returned UNKNOWN (tree confidence < 0.6) |
| **re-run after integration (2026-10-20), host built from `firmware/atm_sentinel/`** | self-test 9/9 (max prob diff 0.0039); audio argmax 36/36; max feature diff 7.9e-6; fusion accuracy 0.853, tamper recall 1.000; scenario output byte-identical to the original build; demo copies in sync |
| existing cascade after integration | `ml/cascade/evaluate.py` reproduces `artifacts/results.json` exactly; `firmware/test/test_features.py` PASS; `tests/` 15/15 |
| backend path (PC, test broker) | 16 scenario events published on `atm/ATM_001/event` -> stored by `mqtt_to_sqlite.py` -> shown on the admin page |

## 7. Limitations (read before presenting results)

1. **Tiny, mismatched audio data.** About 37 s of audio across 5 internet MP3 clips. Only one clip
   is ABNORMAL, so the abnormal test set is 5 windows from the last 2 s of that clip (temporal block
   split, not a recording-level split). Two of the four DISPENSE clips are banknote counters, used as
   stand-ins for dispenser note transport. Nothing was recorded through the INMP441, so microphone
   response, enclosure acoustics and level will shift the features.
2. **Accuracy numbers are inflated by an easy class.** Test accuracy ~0.97 is dominated by synthetic
   BACKGROUND windows. The honest numbers are macro-F1 0.73 and per-class recall: DISPENSE 0.97,
   ABNORMAL 1/5. Leave-one-recording-out DISPENSE accuracy ranges from 0.56 to 0.97.
3. **ABNORMAL training data is mostly synthetic** (jam and grind variants built from dispense
   clips). The model can learn "synthetic artifact = abnormal". A start-stop variant was removed
   because real dispensers also start and stop.
4. **Fusion data is synthetic.** Vibration and PIR streams are simulated from assumptions (dispense
   vibration 4–20 edges/s, short visits ~22 s, tamper ≥ 75 edges/s). The fusion tree's thresholds
   reflect those assumptions, not measured ATM behaviour. Audio inside each event comes from real
   windows. Training events use out-of-fold audio probabilities, so the tree is not overconfident in audio.
5. **Vibration units.** The tamper threshold (75) assumes SW-420 edges per 1 s bin. If the existing
   tamper rule uses another unit or window, change `TAMPER_VIB_THRESHOLD` and retrain.
6. **Cash dispensed is heard, not measured.** It counts confirmed NORMAL_DISPENSE events (dispense
   sound + presence, never tamper-like vibration). The project does not estimate how much cash an ATM
   holds; the earlier experimental cash-availability risk was removed.
7. **Not hardware-verified.** Section 5 is empty until the sketch runs on a real ESP32.
