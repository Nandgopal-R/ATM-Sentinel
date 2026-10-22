"""Subscribe to atm/+/{telemetry,inference,status,event} and store every message in SQLite.

    pip install paho-mqtt
    python tools/mqtt_to_sqlite.py                 # broker localhost, db atm_sentinel.db
    python tools/mqtt_to_sqlite.py --db demo.db --host 127.0.0.1

Tables mirror the topics. telemetry is the schema.yaml record (null preserved);
inference keeps the 57 raw features as JSON so a window can be re-scored offline.
event is the audio + fusion module's interaction summary (schema.yaml `event`); its device time `t` is
seconds since the node booted, so `received` (logger clock) is the timestamp to sort and display by.
"""
import argparse, json, sqlite3, sys, time
import paho.mqtt.client as mqtt

SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
  atm_id TEXT, timestamp INTEGER, temperature REAL, humidity REAL, pressure REAL,
  light REAL, voltage REAL, current REAL, vibration INTEGER, motion INTEGER,
  PRIMARY KEY (atm_id, timestamp));
CREATE TABLE IF NOT EXISTS inference (
  atm_id TEXT, timestamp INTEGER, window_s INTEGER, fault_mask INTEGER, p_abnormal REAL,
  stage2 TEXT, class TEXT, persist INTEGER, alert INTEGER, topz TEXT, features TEXT,
  PRIMARY KEY (atm_id, timestamp));
CREATE TABLE IF NOT EXISTS status (atm_id TEXT, timestamp INTEGER, json TEXT);
CREATE TABLE IF NOT EXISTS event (
  atm_id TEXT, received INTEGER, device_t INTEGER, duration_s INTEGER, class TEXT, final INTEGER,
  rule TEXT, confidence REAL, tamper INTEGER, dispensed INTEGER, json TEXT);
"""
TELEMETRY = ["atm_id", "timestamp", "temperature", "humidity", "pressure", "light", "voltage", "current", "vibration", "motion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--db", default="atm_sentinel.db")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    db.executescript(SCHEMA)

    def on_message(_client, _userdata, m):
        kind = m.topic.rsplit("/", 1)[-1]
        try:
            d = json.loads(m.payload)
        except json.JSONDecodeError:
            print("bad json on", m.topic, m.payload[:80], file=sys.stderr)
            return
        if kind == "telemetry":
            db.execute("INSERT OR REPLACE INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?)", [d.get(k) for k in TELEMETRY])
        elif kind == "inference":
            db.execute("INSERT OR REPLACE INTO inference VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (d["atm_id"], d["timestamp"], d["window_s"], d["fault_mask"], d["p_abnormal"],
                        json.dumps(d["stage2"]), d["class"], d["persist"], int(d["alert"]),
                        json.dumps(d["topz"]), json.dumps(d["features"])))
            print(f"INFER {d['atm_id']} t={d['timestamp']} p={d['p_abnormal']} class={d['class']}"
                  f"{' ALERT' if d['alert'] else ''}  topz={d['topz']}")
        elif kind == "status":
            db.execute("INSERT INTO status VALUES (?,?,?)", (d["atm_id"], d["timestamp"], json.dumps(d)))
            print("STATUS", json.dumps(d))
        elif kind == "event":
            atm_id = m.topic.split("/")[1]
            # named columns: a db created before cash_risk was dropped still accepts the row
            db.execute("INSERT INTO event (atm_id, received, device_t, duration_s, class, final, rule, confidence,"
                       " tamper, dispensed, json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (atm_id, int(time.time()), d["t"], d["dur"], d["ev"], d["final"], d["rule"], d["conf"],
                        d["tamper"], d["dispensed"], json.dumps(d)))
            print(f"EVENT {atm_id} {d['ev']}{'' if d['final'] else ' (early alert)'} rule={d['rule']} "
                  f"conf={d['conf']} dur={d['dur']}s")
        db.commit()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.on_connect = lambda c, *_: (c.subscribe("atm/+/#"), print(f"subscribed, writing {args.db}"))
    client.connect(args.host, args.port)
    client.loop_forever()


if __name__ == "__main__":
    main()
