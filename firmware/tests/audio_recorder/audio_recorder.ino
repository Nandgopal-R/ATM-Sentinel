/*
  INMP441 recorder for audio-model training data.

  Streams raw 16 kHz mono int16 PCM over serial on request. Pair it with
  ml/audio_tinyml/tools/record_inmp441.py, which sends "R <seconds>" and saves a WAV.
  Close the Arduino Serial Monitor first: only one program can hold the port.

  Wiring as in HARDWARE.md: SCK -> 32, WS -> 25, SD -> 33, L/R -> GND, VDD -> 3.3 V.

  Scaling: int16 = raw >> 16, which the training script reads back as /32768. That equals the
  firmware's inmp441ToFloat() (raw >> 8) / 2^23, so dB levels match the [audio] lines.
*/

#include <ESP_I2S.h>

static const int8_t   PIN_I2S_SCK = 32, PIN_I2S_WS = 25, PIN_I2S_SD = 33;
static const uint32_t SR          = 16000;
static const uint32_t BAUD        = 921600;  // 32 KB/s of audio needs > 320 kbit/s; must match the PC script
static const size_t   BLOCK       = 256;
static const uint32_t MAX_SECONDS = 600;

I2SClass i2s;
static int32_t raw[BLOCK];
static int16_t pcm[BLOCK];

static void readBlock() {
  size_t got = 0;
  while (got < sizeof(raw)) got += i2s.readBytes((char*)raw + got, sizeof(raw) - got);
}

void setup() {
  Serial.begin(BAUD);
  i2s.setPins(PIN_I2S_SCK, PIN_I2S_WS, -1, PIN_I2S_SD, -1);
  if (!i2s.begin(I2S_MODE_STD, SR, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    for (;;) { Serial.println("ERR I2S begin failed"); delay(1000); }
  }
  for (int i = 0; i < 32; ++i) readBlock();  // ~0.5 s: INMP441 start-up
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) { readBlock(); return; }  // keep the DMA ring fresh while idle
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (!cmd.startsWith("R ")) { Serial.println("READY"); return; }
  uint32_t sec = cmd.substring(2).toInt();
  if (sec == 0 || sec > MAX_SECONDS) { Serial.println("ERR seconds 1..600"); return; }

  const uint32_t blocks = sec * SR / BLOCK;
  Serial.printf("REC %lu\n", (unsigned long)(blocks * BLOCK * sizeof(int16_t)));
  for (int i = 0; i < 8; ++i) readBlock();  // drop audio buffered before the request
  for (uint32_t b = 0; b < blocks; ++b) {
    readBlock();
    for (size_t i = 0; i < BLOCK; ++i) pcm[i] = (int16_t)(raw[i] >> 16);
    Serial.write((const uint8_t*)pcm, sizeof(pcm));
  }
  Serial.println("\nEND");
}
