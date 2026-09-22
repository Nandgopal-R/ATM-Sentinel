/*
  ATM Sentinel — ESP32 sensor node firmware

  raw 1 Hz readings -> schema validation -> 300 s ring buffer -> 57 features
  -> scaler -> Stage 1 (NORMAL/ABNORMAL) -> Stage 2 (fault type) -> serial + MQTT

  Board:   DOIT ESP32 DEVKIT V1, 115200 baud. Pin map is HARDWARE.md.
  Sketch:  atm_sentinel.ino  (this file: sensors, models, network, main loop)
           features.h        (feature extraction + scaler; mirrors atmsim/features.py)
           models.h          (generated: xxd -i of the two .tflite files)

  Libraries (Arduino Library Manager):
    Adafruit BME280, Adafruit VEML7700, Adafruit INA219, Adafruit Unified Sensor,
    Chirale_TensorFlowLite (TFLite Micro for ESP32 Arduino core 3.x),
    PubSubClient (Nick O'Leary).

  MQTT topics (all JSON):
    atm/<id>/telemetry   every second, the schema.yaml record (null = missing)
    atm/<id>/inference   every INFER_STRIDE_S, model output + the 57 raw features
    atm/<id>/status      once at boot, retained: sensor health + provisional constants

  Demo topology: node -> WiFi -> Mosquitto on the laptop -> tools/mqtt_to_sqlite.py.
  The ESP-NOW -> ESP8266 gateway from the design is bypassed; say so in the report.
*/

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_VEML7700.h>
#include <Adafruit_INA219.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <time.h>
#include <esp_system.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "features.h"
#include "models.h"


#define DEBUG_FEATURES 1        // 1: print the raw 57-feature vector + top-|z| at every inference
#define ENABLE_MQTT    1        // 0: serial only

static const char ATM_ID[]    = "ATM_001";        // PROVISIONAL: schema ^ATM_[0-9]{3}$
static const char WIFI_SSID[] = "FILL HERE (Wifi Network Name)";      // PROVISIONAL
static const char WIFI_PASS[] = "FILL HERE (Wifi Password)";      // PROVISIONAL
static const char MQTT_HOST[] = "FILL HERE";   // PROVISIONAL: laptop IP running Mosquitto (ip a / ipconfig)
static const uint16_t MQTT_PORT = 1883;
static const char NTP_SERVER[] = "pool.ntp.org";  // timestamp = UTC epoch once synced; seconds-since-boot before that


static const float SIM_VOLTAGE_NOMINAL_V = 230.0f;   // PROVISIONAL
static const float SIM_VOLTAGE_NOISE_V   = 0.4f;     // PROVISIONAL


#define CURRENT_SIMULATED 1                           // PROVISIONAL: set 0 once a real load is in series with VIN+/VIN-
static const float SIM_CURRENT_NOMINAL_A = 1.417f;   // PROVISIONAL: NORMAL fan-rail mean (README)
static const float SIM_CURRENT_NOISE_A   = 0.002f;   // PROVISIONAL: schema noise_sigma


static const float PRESSURE_OFFSET_HPA = 37.0f;      // PROVISIONAL: site calibration; bench read 971 hPa. 0 = raw.

// Bench reads 25-40 lux; training light_mean is 246 lux (-2.2 sigma).
static const float LIGHT_SCALE = 1.0f;               // PROVISIONAL: 1.0 = true lux; ~6.0 maps the bench into the training band

// SW-420: schema.yaml says FALLING edges, 5 ms debounce, clip at 200.
// The bring-up sketch used CHANGE with no debounce (~2x the count).
#define VIB_EDGE_MODE FALLING                         // PROVISIONAL: schema wins over bring-up
static const uint32_t VIB_DEBOUNCE_US = 5000;        // PROVISIONAL
static const uint16_t VIB_MAX_COUNT   = 200;         // schema range

// HC-SR501 fires randomly for ~60 s after power-on. Hold time is the module
static const uint32_t PIR_WARMUP_MS = 60000;

static const uint16_t INFER_STRIDE_S           = 60;   // PROVISIONAL: schema window_stride_s
static const uint8_t  ABNORMAL_PERSIST_WINDOWS = 2;    // PROVISIONAL: consecutive abnormal windows before ALERT (README "N of M")
static const uint16_t MISSING_RUN_S            = 60;   // SENSOR_VALIDATION.md: no usable reading across a publish interval
static const int      TENSOR_ARENA_BYTES       = 2048; // measured: ~1.1 KB used per model

// ============================ FIXED CONTRACT ===============================
static const float STAGE1_THRESHOLD = 0.7615026235580444f;  // ml/stage-1/artifacts/threshold.json
enum FinalClass { NORMAL, COOLING_FAILURE, HIGH_HUMIDITY, POWER_FAILURE, TAMPER, SENSOR_FAULT };
static const char* CLASS_NAMES[] = {"NORMAL", "COOLING_FAILURE", "HIGH_HUMIDITY", "POWER_FAILURE", "TAMPER", "SENSOR_FAULT"};

// ================================ PINS ====================================
static const uint8_t PIN_SDA = 21, PIN_SCL = 22, PIN_PIR = 27, PIN_VIB = 26;

// ================================ STATE ===================================
Adafruit_BME280   bme;
Adafruit_VEML7700 veml;
Adafruit_INA219   ina219;
bool okBme = false, okVeml = false, okIna = false;
uint8_t bmeAddr = 0;

volatile uint32_t vibEdges = 0, vibLastUs = 0;

static float    ring[N_CHAN][WINDOW_S];   // 9.6 KB. NaN = null.
static float    lin[N_CHAN][WINDOW_S];    // 9.6 KB. Window in time order for the extractor.
static uint16_t ringHead = 0, ringCount = 0, sinceInfer = 0;
static float    lastSample[N_CHAN];
static float    feat[N_FEATURES], normed[N_FEATURES];

static uint8_t  consecutiveAbnormal = 0;
static uint32_t lastSampleMs = 0;

static tflite::MicroMutableOpResolver<3> resolver;
alignas(16) static uint8_t arena1[TENSOR_ARENA_BYTES], arena2[TENSOR_ARENA_BYTES];
static tflite::MicroInterpreter *stage1 = nullptr, *stage2 = nullptr;

WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);
static char  topicTelemetry[40], topicInference[40], topicStatus[40];
static char  msg[2048];   // largest payload: inference with 57 features (~1.2 KB)
static uint32_t lastMqttAttemptMs = 0;

// ============================== SENSORS ===================================

// Vibration Sensor Setup
void IRAM_ATTR vibISR() {
  uint32_t now = micros();
  if (now - vibLastUs >= VIB_DEBOUNCE_US) { vibEdges++; vibLastUs = now; }
}

// Box-Muller on the hardware RNG, for the simulated channels.
static float gaussian() {
  float u1 = ((float)esp_random() + 1.0f) / 4294967296.0f;
  float u2 = (float)esp_random() / 4294967296.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// Telemetry contract (SENSOR_VALIDATION.md #1): non-finite or out of schema
// range -> null. In-range values are quantised to the schema resolution, as
// the simulator does, so the stuck rule compares like with like.
static float contract(float v, uint8_t ch) {
  if (!isfinite(v) || v < CH_LO[ch] || v > CH_HI[ch]) return NAN;
  return roundf(v / CH_RES[ch]) * CH_RES[ch];
}

// Starting the sensors

static void initSensors() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000);  // HARDWARE.md: single bus at 100 kHz

  // BME280(Temp) is at 0x77 on this board (SDO floating); probe 0x76 too.
  const uint8_t candidates[] = {0x77, 0x76};
  for (uint8_t a : candidates) if (bme.begin(a, &Wire)) { okBme = true; bmeAddr = a; Serial.printf("  BME280    OK   (0x%02X)\n", a); break; }
  if (okBme) {
    bme.setSampling(Adafruit_BME280::MODE_FORCED, Adafruit_BME280::SAMPLING_X1, Adafruit_BME280::SAMPLING_X1,
                    Adafruit_BME280::SAMPLING_X1, Adafruit_BME280::FILTER_OFF);
  } else Serial.println(F("  BME280    FAIL -> temperature/humidity/pressure will be null"));

  // VEML(Light sensor)
  okVeml = veml.begin();
  if (okVeml) { veml.setGain(VEML7700_GAIN_1); veml.setIntegrationTime(VEML7700_IT_100MS); Serial.println(F("  VEML7700  OK   (0x10)")); }
  else Serial.println(F("  VEML7700  FAIL -> light will be null"));

  // INA219 (Current sensor)
  okIna = ina219.begin();
  if (okIna) { ina219.setCalibration_32V_2A(); Serial.println(F("  INA219    OK   (0x40)")); }
  else Serial.println(F("  INA219    FAIL"));

  // HC-SR501 (Motion Sensor)
  pinMode(PIN_PIR, INPUT_PULLDOWN);
  pinMode(PIN_VIB, INPUT);  // module has its own pull-up
  delay(50);
  Serial.printf("  SW-420    idle level = %s\n", digitalRead(PIN_VIB) ? "HIGH" : "LOW");
  attachInterrupt(digitalPinToInterrupt(PIN_VIB), vibISR, VIB_EDGE_MODE);
  Serial.println(F("  HC-SR501  warming up; motion forced 0 for the first minute"));
}

static void readSample(float* s) {
  s[CH_T] = s[CH_RH] = s[CH_P] = NAN;
  if (okBme) {
    bme.takeForcedMeasurement();
    s[CH_T]  = contract(bme.readTemperature(), CH_T);
    s[CH_RH] = contract(bme.readHumidity(), CH_RH);
    s[CH_P]  = contract(bme.readPressure() / 100.0f + PRESSURE_OFFSET_HPA, CH_P);
  }
  s[CH_LUX] = okVeml ? contract(veml.readLux() * LIGHT_SCALE, CH_LUX) : NAN;
  s[CH_V]   = contract(SIM_VOLTAGE_NOMINAL_V + SIM_VOLTAGE_NOISE_V * gaussian(), CH_V);
#if CURRENT_SIMULATED
  s[CH_I]   = contract(SIM_CURRENT_NOMINAL_A + SIM_CURRENT_NOISE_A * gaussian(), CH_I);
#else
  s[CH_I]   = okIna ? contract(ina219.getCurrent_mA() / 1000.0f, CH_I) : NAN;
#endif

  noInterrupts();
  uint32_t edges = vibEdges; vibEdges = 0;
  interrupts();
  s[CH_VIB] = (float)min<uint32_t>(edges, VIB_MAX_COUNT);
  s[CH_MOT] = (millis() < PIR_WARMUP_MS) ? 0.0f : (float)digitalRead(PIN_PIR);
}

static void pushSample(const float* s) {
  for (uint8_t c = 0; c < N_CHAN; c++) ring[c][ringHead] = s[c];
  ringHead = (ringHead + 1) % WINDOW_S;
  if (ringCount < WINDOW_S) ringCount++;
  sinceInfer++;
}

// ============================== NETWORK ===================================
// UTC epoch once NTP has synced, seconds since boot before that.
static uint32_t nowTs() {
  time_t t = time(nullptr);
  return t > 1600000000 ? (uint32_t)t : millis() / 1000;
}

// Appends to msg; returns the new length. All JSON is built with this.
static int put(int len, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  len += vsnprintf(msg + len, sizeof(msg) - len, fmt, ap);
  va_end(ap);
  return len;
}
static int putNum(int len, const char* key, float v, int dec) {
  return isnan(v) ? put(len, "\"%s\":null,", key) : put(len, "\"%s\":%.*f,", key, dec, v);
}

// Print to serial and, when connected, publish.
static void emit(const char* topic, bool retain = false) {
  Serial.println(msg);
#if ENABLE_MQTT
  if (mqtt.connected()) mqtt.publish(topic, msg, retain);
#endif
}

// One schema.yaml record per second — the exact shape the gateway would forward.
static void emitSample(const float* s) {
  int n = put(0, "{\"atm_id\":\"%s\",\"timestamp\":%lu,", ATM_ID, (unsigned long)nowTs());
  n = putNum(n, "temperature", s[CH_T], 2);  n = putNum(n, "humidity", s[CH_RH], 2);  n = putNum(n, "pressure", s[CH_P], 2);
  n = putNum(n, "light", s[CH_LUX], 1);      n = putNum(n, "voltage", s[CH_V], 1);    n = putNum(n, "current", s[CH_I], 4);
  put(n, "\"vibration\":%d,\"motion\":%d}", (int)s[CH_VIB], (int)s[CH_MOT]);
  emit(topicTelemetry);
}

// Boot record: what this run was configured as. Retained so a late subscriber sees it.
static void emitStatus() {
  int n = put(0, "{\"atm_id\":\"%s\",\"timestamp\":%lu,\"uptime_s\":%lu,\"ip\":\"%s\",",
              ATM_ID, (unsigned long)nowTs(), (unsigned long)(millis() / 1000), WiFi.localIP().toString().c_str());
  n = put(n, "\"sensors\":{\"bme280\":%s,\"bme280_addr\":\"0x%02X\",\"veml7700\":%s,\"ina219\":%s},",
          okBme ? "true" : "false", bmeAddr, okVeml ? "true" : "false", okIna ? "true" : "false");
  n = put(n, "\"models\":{\"stage1\":%s,\"stage2\":%s,\"arena1_used\":%u,\"arena2_used\":%u,\"stage1_threshold\":%.7f},",
          stage1 ? "true" : "false", stage2 ? "true" : "false",
          stage1 ? (unsigned)stage1->arena_used_bytes() : 0, stage2 ? (unsigned)stage2->arena_used_bytes() : 0, STAGE1_THRESHOLD);
  n = put(n, "\"provisional\":{\"voltage_simulated\":true,\"sim_voltage_v\":%.1f,\"current_simulated\":%s,\"sim_current_a\":%.3f,"
             "\"pressure_offset_hpa\":%.1f,\"light_scale\":%.2f,\"vib_debounce_us\":%lu,\"pir_warmup_ms\":%lu,"
             "\"infer_stride_s\":%u,\"persist_windows\":%u,\"missing_run_s\":%u}}",
          SIM_VOLTAGE_NOMINAL_V, CURRENT_SIMULATED ? "true" : "false", SIM_CURRENT_NOMINAL_A,
          PRESSURE_OFFSET_HPA, LIGHT_SCALE, (unsigned long)VIB_DEBOUNCE_US, (unsigned long)PIR_WARMUP_MS,
          INFER_STRIDE_S, ABNORMAL_PERSIST_WINDOWS, MISSING_RUN_S);
  emit(topicStatus, true);
}

#if ENABLE_MQTT
static void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_START:        Serial.println(F("WIFI sta start")); break;
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:    Serial.println(F("WIFI associated, waiting for DHCP")); break;
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:       Serial.printf("WIFI got IP %s  rssi=%d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI()); break;
    // reason codes: 2=auth expire 15=4way handshake timeout (wrong password) 201=no AP found 202=auth fail
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED: Serial.printf("WIFI disconnected, reason=%d\n", info.wifi_sta_disconnected.reason); break;
    default: break;
  }
}

// Non-blocking: called every loop, never stalls the 1 Hz sampler.
static void serviceNetwork() {
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastMqttAttemptMs < 5000) return;
    lastMqttAttemptMs = millis();
    Serial.printf("WIFI status=%d (0=idle 1=no ssid 4=fail 6=disconnected) heap=%u\n", WiFi.status(), (unsigned)ESP.getFreeHeap());
    return;
  }
  if (!mqtt.connected()) {
    if (millis() - lastMqttAttemptMs < 5000) return;
    lastMqttAttemptMs = millis();
    Serial.printf("MQTT connecting to %s:%u ...\n", MQTT_HOST, MQTT_PORT);
    if (mqtt.connect(ATM_ID)) { Serial.printf("MQTT connected to %s\n", MQTT_HOST); emitStatus(); }
    else Serial.printf("MQTT connect failed rc=%d (is Mosquitto listening on 0.0.0.0:%u?)\n", mqtt.state(), MQTT_PORT);
    return;
  }
  mqtt.loop();
}
#endif

// =============================== MODELS ===================================
static tflite::MicroInterpreter* loadModel(const unsigned char* data, uint8_t* arena, const char* name) {
  const tflite::Model* model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) { Serial.printf("  %s  schema version mismatch\n", name); return nullptr; }
  auto* it = new tflite::MicroInterpreter(model, resolver, arena, TENSOR_ARENA_BYTES);
  if (it->AllocateTensors() != kTfLiteOk) { Serial.printf("  %s  AllocateTensors FAILED - raise TENSOR_ARENA_BYTES\n", name); return nullptr; }
  Serial.printf("  %s  OK   arena used %u / %d bytes\n", name, (unsigned)it->arena_used_bytes(), TENSOR_ARENA_BYTES);
  return it;
}

static const float* runModel(tflite::MicroInterpreter* it) {
  memcpy(it->input(0)->data.f, normed, sizeof(normed));
  return it->Invoke() == kTfLiteOk ? it->output(0)->data.f : nullptr;
}

// ============================= INFERENCE ==================================
static void infer() {
  uint16_t start = (ringHead + WINDOW_S - ringCount) % WINDOW_S;
  for (uint8_t c = 0; c < N_CHAN; c++)
    for (uint16_t i = 0; i < WINDOW_S; i++) lin[c][i] = ring[c][(start + i) % WINDOW_S];
  const float* chan[N_CHAN];
  for (uint8_t c = 0; c < N_CHAN; c++) chan[c] = lin[c];

  uint8_t faultMask = extractFeatures(chan, feat, MISSING_RUN_S);
  normaliseFeatures(feat, normed); // Converts into Z-score

  // Top-5 |z|: which inputs are furthest from the training data.
  uint8_t topIdx[5];
  { float z[N_FEATURES]; memcpy(z, normed, sizeof(z));
    for (uint8_t k = 0; k < 5; k++) {
      uint8_t best = 0;
      for (uint8_t i = 1; i < N_FEATURES; i++) if (fabsf(z[i]) > fabsf(z[best])) best = i;
      topIdx[k] = best; z[best] = 0;
    } }

#if DEBUG_FEATURES
  Serial.print(F("FEAT"));
  for (uint8_t i = 0; i < N_FEATURES; i++) Serial.printf(",%.6g", feat[i]);
  Serial.print(F("\nTOPZ"));
  for (uint8_t k = 0; k < 5; k++) Serial.printf(" %s=%+.1f", FEATURE_NAMES[topIdx[k]], normed[topIdx[k]]);
  Serial.println();
#endif

  float pAbnormal = NAN, stage2Out[5] = {NAN, NAN, NAN, NAN, NAN};
  uint8_t finalClass = NORMAL;

  if (faultMask) {
    // Deterministic integrity fault: bypass the models (SENSOR_VALIDATION.md).
    finalClass = SENSOR_FAULT;
    Serial.print(F("SENSOR_FAULT deterministic:"));
    for (uint8_t ch = 0; ch < N_FLOAT_CHAN; ch++) if (faultMask & (1 << ch)) Serial.printf(" %s", CHAN_NAMES[ch]);
    Serial.println();
  } else if (stage1) {
    const float* out1 = runModel(stage1);
    if (!out1) { Serial.println(F("Stage 1 Invoke failed")); return; }
    pAbnormal = out1[0];
    if (pAbnormal >= STAGE1_THRESHOLD && stage2) {
      const float* out2 = runModel(stage2);
      if (!out2) { Serial.println(F("Stage 2 Invoke failed")); return; }
      memcpy(stage2Out, out2, sizeof(stage2Out));
      uint8_t best = 0;
      for (uint8_t k = 1; k < 5; k++) if (out2[k] > out2[best]) best = k;
      finalClass = best + 1;  // stage-2 index 0..4 -> COOLING_FAILURE..SENSOR_FAULT
    }
  }

  consecutiveAbnormal = (finalClass == NORMAL) ? 0 : min<int>(consecutiveAbnormal + 1, 255);
  bool alert = consecutiveAbnormal >= ABNORMAL_PERSIST_WINDOWS;

  Serial.printf("INFER uptime=%lus p_abnormal=%.4f class=%s persist=%u/%u%s\n",
                (unsigned long)(millis() / 1000), pAbnormal, CLASS_NAMES[finalClass],
                consecutiveAbnormal, ABNORMAL_PERSIST_WINDOWS, alert ? " ALERT" : "");

  int n = put(0, "{\"atm_id\":\"%s\",\"timestamp\":%lu,\"window_s\":%u,\"fault_mask\":%u,",
              ATM_ID, (unsigned long)nowTs(), WINDOW_S, faultMask);
  n = putNum(n, "p_abnormal", pAbnormal, 4);
  n = put(n, "\"stage2\":[");
  for (uint8_t k = 0; k < 5; k++) n = isnan(stage2Out[k]) ? put(n, "null%s", k < 4 ? "," : "") : put(n, "%.4f%s", stage2Out[k], k < 4 ? "," : "");
  n = put(n, "],\"class\":\"%s\",\"persist\":%u,\"alert\":%s,\"topz\":{",
          CLASS_NAMES[finalClass], consecutiveAbnormal, alert ? "true" : "false");
  for (uint8_t k = 0; k < 5; k++) n = put(n, "\"%s\":%.2f%s", FEATURE_NAMES[topIdx[k]], normed[topIdx[k]], k < 4 ? "," : "");
  n = put(n, "},\"features\":[");
  for (uint8_t i = 0; i < N_FEATURES; i++) n = put(n, "%.6g%s", feat[i], i < N_FEATURES - 1 ? "," : "");
  put(n, "]}");
  emit(topicInference);
}

// ================================ MAIN ====================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\n\n=== ATM Sentinel node ==="));
  // 1=power-on 3=software 4/5/7/8/9=watchdog/panic 15=BROWNOUT (radio current spike on a weak USB supply)
  Serial.printf("reset reason=%d  heap=%u\n", (int)esp_reset_reason(), (unsigned)ESP.getFreeHeap());
  Serial.println(F("--- sensors ---"));
  initSensors();

  Serial.println(F("--- models ---"));
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddSoftmax();
  stage1 = loadModel(STAGE1_TFLITE, arena1, "stage1");
  stage2 = loadModel(STAGE2_TFLITE, arena2, "stage2");

  snprintf(topicTelemetry, sizeof(topicTelemetry), "atm/%s/telemetry", ATM_ID);
  snprintf(topicInference, sizeof(topicInference), "atm/%s/inference", ATM_ID);
  snprintf(topicStatus,    sizeof(topicStatus),    "atm/%s/status",    ATM_ID);
#if ENABLE_MQTT
  Serial.println(F("--- network ---"));
  WiFi.onEvent(onWiFiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);   // non-blocking; serviceNetwork() picks it up
  configTime(0, 0, NTP_SERVER);       // UTC
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setBufferSize(sizeof(msg));    // PubSubClient default is 256 B; inference is ~1.2 KB
  Serial.printf("  WiFi joining %s, MQTT -> %s:%u\n", WIFI_SSID, MQTT_HOST, MQTT_PORT);
#endif

#if DEBUG_FEATURES
  Serial.print(F("FEAT_HEADER"));
  for (uint8_t i = 0; i < N_FEATURES; i++) { Serial.print(','); Serial.print(FEATURE_NAMES[i]); }
  Serial.println();
#endif
  Serial.printf("--- sampling at 1 Hz; first inference after %u s, then every %u s ---\n", WINDOW_S, INFER_STRIDE_S);
}

void loop() {
#if ENABLE_MQTT
  serviceNetwork();
#endif

  uint32_t now = millis();
  if (now - lastSampleMs < 1000) return;
  lastSampleMs += 1000;                              // fixed cadence, no drift from loop latency
  if (now - lastSampleMs > 1000) lastSampleMs = now; // resync after a long stall

  readSample(lastSample);
  emitSample(lastSample);
  pushSample(lastSample);

  if (ringCount == WINDOW_S && sinceInfer >= INFER_STRIDE_S) {
    sinceInfer = 0;
    infer();
  }
}
