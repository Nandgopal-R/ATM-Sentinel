"""ATM Sentinel web UI. Reads the SQLite file that mqtt_to_sqlite.py writes.

    pip install flask paho-mqtt
    ATM_ADMIN_PASSWORD=... python tools/dashboard.py          # http://localhost:5000
    python tools/dashboard.py --db demo.db --port 8080

    /        public map: which ATMs in Coimbatore are working (no login)
    /admin   operations console: per-ATM faults, model output, raw telemetry (login)
    /demo    two buttons that send txn:1 / txn:0 to the live ATM on atm/<id>/cmd (demo or admin login)

Admin credentials come from ATM_ADMIN_USER (default "admin") and ATM_ADMIN_PASSWORD;
without the latter a random password is printed at startup. Same for the demo login
(user "demo", ATM_DEMO_PASSWORD). Pages live in tools/web/.
Only ATM_001 is real hardware; the rest of the fleet is DEMO fixtures so the map has a network.
"""
import argparse, hmac, json, math, os, random, secrets, sqlite3, time
import paho.mqtt.publish as mqtt_publish
from flask import Flask, jsonify, redirect, render_template, request, session

app = Flask(__name__, static_folder="web", static_url_path="/static", template_folder="web")
app.secret_key = os.environ.get("ATM_SECRET_KEY") or secrets.token_hex(32)  # unset: sessions end on restart
app.config.update(SESSION_COOKIE_SAMESITE="Lax")  # cross-site POSTs carry no session
ADMIN_USER = os.environ.get("ATM_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ATM_ADMIN_PASSWORD") or secrets.token_urlsafe(9)
DEMO_USER = "demo"
DEMO_PASSWORD = os.environ.get("ATM_DEMO_PASSWORD") or secrets.token_urlsafe(9)
MQTT = ("localhost", 1883)

DB = "atm_sentinel.db"
STAGE1_THRESHOLD = 0.7615026
STAGE2 = ["COOLING_FAILURE", "HIGH_HUMIDITY", "POWER_FAILURE", "TAMPER", "SENSOR_FAULT"]
OFFLINE_S = 30  # telemetry is 1 Hz; this long without a sample = not reporting

# Where the real nodes are installed. PROVISIONAL: edit when the node moves.
SITES = {"ATM_001": ("Amrita", "Amrita Nagar, Ettimadai", 10.9036, 76.8997)}

BASE = dict(temperature=27.6, humidity=57.0, pressure=1009.2, light=182.0, voltage=230.4,
            current=1.42, vibration=0, motion=0)


def demo(atm_id, name, street, lat, lng, cls="NORMAL", p=0.07, persist=0, seen=2, **readings):
    return dict(atm_id=atm_id, name=name, street=street, lat=lat, lng=lng, demo=True, cls=cls,
                p_abnormal=p, persist=persist, alert=persist >= 2, seen_ago=seen, readings={**BASE, **readings})


DEMO = [
    demo("ATM_002", "Gandhipuram", "Cross Cut Road", 11.0176, 76.9674),
    demo("ATM_003", "R.S. Puram", "D.B. Road", 11.0089, 76.9496, p=0.11),
    demo("ATM_004", "Race Course", "West Club Road", 10.9997, 76.9779, "COOLING_FAILURE", 0.93, 4, temperature=41.6),
    demo("ATM_005", "Town Hall", "Big Bazaar Street", 10.9937, 76.9616, "TAMPER", 0.99, 2, vibration=23, motion=1),
    demo("ATM_006", "Saibaba Colony", "N.S.R. Road", 11.0253, 76.9419, "SENSOR_FAULT", None, 3, humidity=None),
    demo("ATM_007", "Singanallur", "Trichy Road", 10.9985, 77.0322, seen=7300),
    demo("ATM_008", "Saravanampatti", "Sathy Road", 11.0797, 76.9997, "HIGH_HUMIDITY", 0.81, 1, humidity=86.4),
    demo("ATM_009", "Ukkadam", "Palakkad Road", 10.9853, 76.9590, "POWER_FAILURE", 0.97, 5, voltage=0.0, current=0.0),
    demo("ATM_010", "Ganapathy", "Mettupalayam Road", 11.0410, 76.9790, p=0.05),
]
DEMO_BY_ID = {d["atm_id"]: d for d in DEMO}

# How each demo fault plays out in its 300 s window: where it starts (fraction of the window), how it
# arrives, and the features it pushes furthest from training data (z, the model's "most unusual" list)
FAULT_SHAPE = {
    "COOLING_FAILURE": (0.15, "ramp", {"temperature_slope": 6.8, "temperature_delta": 6.1, "temperature_max": 5.2, "temperature_std": 3.1, "temp_rh_corr": -2.4}),
    "HIGH_HUMIDITY": (0.5, "ramp", {"humidity_slope": 5.4, "humidity_delta": 4.9, "humidity_max": 4.2, "abs_humidity_slope": 3.6, "temp_rh_corr": 1.8}),
    "POWER_FAILURE": (0.6, "step", {"voltage_dropout_frac": 8.0, "voltage_min": -8.0, "current_min": -7.5, "voltage_delta": -6.9, "current_mean": -3.2}),
    "TAMPER": (0.55, "step", {"vibration_max": 7.9, "vibration_active_frac": 6.2, "motion_frac": 4.8, "motion_transitions": 3.9, "vibration_sum": 3.5}),
    "SENSOR_FAULT": (0.3, "step", {"humidity_null_frac": 8.0, "temp_rh_corr": -1.2, "humidity_std": -0.9}),
}
NOISE = dict(temperature=0.05, humidity=0.2, pressure=0.05, light=2.0, voltage=0.3, current=0.003)
CALM = ["temperature_std", "humidity_std", "pressure_slope", "light_mean", "light_std", "current_mean", "voltage_std", "motion_frac"]


def demo_detail(a):
    """Simulated history for a demo unit, in the same shape as a real node's. Seeded by (unit, second),
    so every poll returns the same past and the window scrolls like live telemetry."""
    end = int(time.time()) - a["seen_ago"]
    onset, how, topz = FAULT_SHAPE.get(a["cls"], (2, "step", {}))  # onset 2: never, the unit is healthy
    tel = []
    for t in range(end - 299, end + 1):
        rnd, f = random.Random(f"{a['atm_id']}{t}"), (t - end + 299) / 299
        k = 0 if f < onset else (f - onset) / (1 - onset) if how == "ramp" else 1
        r = dict(atm_id=a["atm_id"], timestamp=t)
        for key, sd in NOISE.items():
            target = a["readings"][key]
            if target is None:  # the sensor that dropped out
                r[key] = None if k else round(BASE[key] + sd * (2 * math.sin(t / (23 + 7 * len(key)) + len(key)) + rnd.gauss(0, 0.5)), 3)
                continue
            v = BASE[key] + (target - BASE[key]) * k
            r[key] = round(v + sd * (2 * math.sin(t / (23 + 7 * len(key)) + len(key)) + rnd.gauss(0, 0.5)), 3) if v else 0.0  # slow drift + jitter, like a real sensor
        r["vibration"] = rnd.randint(8, a["readings"]["vibration"]) if k and a["readings"]["vibration"] else int(rnd.random() < 0.05)
        r["motion"] = int(rnd.random() < 0.7) if k and a["readings"]["motion"] else int(t % 97 < 18)  # a customer every ~1.5 min
        tel.append(r)
    inf = []
    for j in range(29, -1, -1):  # one inference a minute, the firmware's stride
        rnd, run = random.Random(f"{a['atm_id']}inf{end - 60 * j}"), a["persist"] - j
        bad = run > 0 and a["cls"] != "NORMAL"
        cls = a["cls"] if bad else "NORMAL"
        p = None if cls == "SENSOR_FAULT" else round(min(0.999, a["p_abnormal"] - 0.02 * j + rnd.gauss(0, 0.01)), 4) if bad else round(rnd.uniform(0.04, 0.18), 4)
        stage2 = [None] * 5
        if bad and cls != "SENSOR_FAULT":
            hit, rest = rnd.uniform(0.82, 0.94), [rnd.random() for _ in STAGE2]
            stage2 = [round(hit if s == cls else (1 - hit) * w / (sum(rest) - rest[STAGE2.index(cls)]), 4) for s, w in zip(STAGE2, rest)]
        z = {n: round(v * rnd.uniform(0.85, 1.05), 2) for n, v in topz.items()} if bad else \
            {n: round(rnd.gauss(0, 0.7), 2) for n in rnd.sample(CALM, 5)}
        z = dict(sorted(z.items(), key=lambda kv: -abs(kv[1])))  # the firmware sends them largest first
        inf.append(dict(atm_id=a["atm_id"], timestamp=end - 60 * j, window_s=300, fault_mask=2 if cls == "SENSOR_FAULT" else 0,
                        p_abnormal=p, stage2=stage2, persist=max(run, 0) if bad else 0, alert=bad and run >= 2, topz=z, features=[], **{"class": cls}))
    return dict(telemetry=tel, inference=inf, status=None, threshold=STAGE1_THRESHOLD, stage2_labels=STAGE2,
                events=demo_events(a, end))


DEMO_VISITS = ["NORMAL_DISPENSE"] * 5 + ["SHORT_VISIT"] * 2 + ["VISIT_NO_DISPENSE", "UNKNOWN"]


def demo_events(a, end):
    """Simulated audio + fusion events for a demo unit, newest first, in the shape the node publishes
    (firmware formatEventJson) plus `received`. Ages are fixed per unit, so the list doesn't reshuffle."""
    rnd = random.Random(a["atm_id"] + "events")
    kinds = [rnd.choice(DEMO_VISITS) for _ in range(8)]
    if a["cls"] == "TAMPER":
        kinds[-1] = "SUSPECTED_TAMPER"
    ages = sorted((rnd.randint(120, 7200) for _ in kinds), reverse=True)
    out, disp = [], 0
    for k, age in zip(kinds, ages):
        disp += k == "NORMAL_DISPENSE"
        out.append(dict(
            received=end - age, ev=k, final=1, dur=rnd.randint(8, 28) if k == "SHORT_VISIT" else rnd.randint(30, 95),
            conf=round(rnd.uniform(0.45, 0.59) if k == "UNKNOWN" else rnd.uniform(0.66, 0.98), 2),
            rule={"SUSPECTED_TAMPER": "TAMPER_PRIORITY", "UNKNOWN": "LOW_CONFIDENCE"}.get(k, "MODEL"),
            audio=dict(pdmax=round(rnd.uniform(0.82, 0.96) if k == "NORMAL_DISPENSE" else rnd.uniform(0.02, 0.3), 2),
                       pamax=round(rnd.uniform(0.01, 0.08), 2)),
            vib=dict(pk=rnd.randint(110, 190) if k == "SUSPECTED_TAMPER" else rnd.randint(0, 14)),
            pir=dict(frac=round(rnd.uniform(0.6, 0.95), 2)), tamper=int(k == "SUSPECTED_TAMPER"),
            dispensed=int(k == "NORMAL_DISPENSE")))
    # rolling stats exactly as RollingWindow::stats() computes them, attached to the newest event
    score, recent = 0.0, sum(e["ev"] == "SHORT_VISIT" and end - e["received"] <= 600 for e in out)
    for e in reversed(out):
        if e["ev"] == "NORMAL_DISPENSE":
            break
        score += {"VISIT_NO_DISPENSE": 1.0, "SHORT_VISIT": 0.5}.get(e["ev"], 0.0)
    last = next((e["received"] for e in reversed(out) if e["dispensed"]), None)
    out[-1]["roll"] = dict(disp=disp, sv10m=recent, rsv=int(recent >= 3), nd=score,
                           lastDisp=-1 if last is None else end - last)
    return out[::-1]


def rows(sql, *args):
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args)]
    except sqlite3.OperationalError:  # db or tables not created yet
        return []
    finally:
        db.close()


def public_status(a):
    if a["seen_ago"] is None or a["seen_ago"] > OFFLINE_S:
        return "unknown"
    # a sensor fault means we can't see the machine well, not that it can't dispense
    fault = a["alert"] and not a["overridden"] and a["cls"] != "SENSOR_FAULT"
    return "down" if fault or a["cash_ok"] is False else "ok"


def add_cash_and_override(out):
    """Latest withdrawal outcome per ATM, and the admin's "mark as working": it clears a cascade alert
    until the cascade has scored a NORMAL window again, and a failed withdrawal from before the click."""
    now = int(time.time())
    ov = {r["atm_id"]: r["at"] for r in rows("SELECT atm_id, at FROM override")}
    normal = {r["atm_id"]: r["t"] for r in rows("SELECT atm_id, max(timestamp) AS t FROM inference WHERE class='NORMAL' GROUP BY atm_id")}
    cash = {r["atm_id"]: (r["ok"], r["fail"]) for r in rows(
        "SELECT atm_id, max(CASE WHEN dispensed=1 THEN received END) AS ok, "
        "max(CASE WHEN class='ABNORMAL_MECHANICAL' THEN received END) AS fail FROM event GROUP BY atm_id")}
    for a in out:
        o = ov.get(a["atm_id"])
        if a["demo"]:
            ev = demo_events(a, now - a["seen_ago"])
            ok_t = max((e["received"] for e in ev if e["dispensed"]), default=None)
            fail_t = max((e["received"] for e in ev if e["ev"] == "ABNORMAL_MECHANICAL"), default=None)
        else:
            ok_t, fail_t = cash.get(a["atm_id"], (None, None))
        if fail_t is not None and o is not None and fail_t <= o:
            fail_t = None
        failed = fail_t is not None and (ok_t is None or fail_t > ok_t)
        t = fail_t if failed else ok_t
        a.update(override_at=o, cash_ok=None if t is None else not failed, cash_ago=None if t is None else max(0, now - t),
                 overridden=bool(a["alert"] and o is not None and (normal.get(a["atm_id"]) or 0) < o))


def fleet(with_demo=True):
    now = time.time()
    tel = {r["atm_id"]: r for r in rows("SELECT * FROM telemetry WHERE rowid IN (SELECT max(rowid) FROM telemetry GROUP BY atm_id)")}
    inf = {r["atm_id"]: r for r in rows("SELECT * FROM inference WHERE rowid IN (SELECT max(rowid) FROM inference GROUP BY atm_id)")}
    out = []
    for atm_id, t in tel.items():
        i = inf.get(atm_id, {})
        name, street, lat, lng = SITES.get(atm_id, (atm_id, "Location not set", None, None))
        # pre-NTP timestamps are seconds since boot; fall back to when the db was last written
        ts = t["timestamp"] if t["timestamp"] > 1e9 else os.path.getmtime(DB)
        out.append(dict(atm_id=atm_id, name=name, street=street, lat=lat, lng=lng, demo=False,
                        cls=i.get("class"), p_abnormal=i.get("p_abnormal"), persist=i.get("persist", 0),
                        alert=bool(i.get("alert")), seen_ago=max(0, round(now - ts)),
                        readings={k: t[k] for k in BASE}))
    for atm_id in SITES.keys() - tel.keys():  # installed but never reported
        name, street, lat, lng = SITES[atm_id]
        out.append(dict(atm_id=atm_id, name=name, street=street, lat=lat, lng=lng, demo=False, cls=None, p_abnormal=None,
                        persist=0, alert=False, seen_ago=None, readings=dict.fromkeys(BASE)))
    if with_demo:
        out += [dict(d) for d in DEMO if d["atm_id"] not in tel and d["atm_id"] not in SITES]
    add_cash_and_override(out)
    for a in out:
        a["status"] = public_status(a)
    return out


@app.before_request
def require_admin():
    p = request.path
    if p in ("/admin/login", "/admin/logout"):
        return None
    if p.startswith(("/admin", "/api/admin", "/static/admin")) and not session.get("admin") or \
            p.startswith(("/demo", "/api/demo")) and not (session.get("admin") or session.get("demo")):
        return (jsonify(error="sign in required"), 401) if p.startswith("/api/") else redirect("/admin/login")


@app.get("/")
def customer_page():
    return app.send_static_file("map.html")


@app.get("/api/atms")
def public_atms():
    # customers get location, a yes/no and the latest withdrawal outcome, never the fault class or event details
    keep = ("atm_id", "name", "street", "lat", "lng", "status", "seen_ago", "cash_ago", "cash_ok")
    return jsonify(atms=[{k: a[k] for k in keep} for a in fleet(request.args.get("demo") != "0") if a["lat"] is not None])


@app.route("/admin/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("admin") or session.get("demo"):
            return redirect("/admin" if session.get("admin") else "/demo")
        return render_template("login.html")
    user, pw = request.form.get("username", ""), request.form.get("password", "")
    # ponytail: no rate limit on attempts; put it behind a reverse proxy limit before exposing it publicly
    for role, u, p, home in (("admin", ADMIN_USER, ADMIN_PASSWORD, "/admin"), ("demo", DEMO_USER, DEMO_PASSWORD, "/demo")):
        if hmac.compare_digest(user.encode(), u.encode()) & hmac.compare_digest(pw.encode(), p.encode()):
            session.clear()
            session[role] = True
            return redirect(home)
    return render_template("login.html", error=True, username=user), 401


@app.post("/admin/logout")
def logout():
    session.clear()
    return redirect("/admin/login")


@app.get("/demo")
def demo_page():
    return app.send_static_file("demo.html")


@app.get("/api/demo/status")
def demo_status():
    atm_id = next(iter(SITES))
    a = next((a for a in fleet(False) if a["atm_id"] == atm_id), None)
    return jsonify(atm_id=atm_id, name=SITES[atm_id][0],
                   online=bool(a and a["seen_ago"] is not None and a["seen_ago"] <= OFFLINE_S))


last_txn = 0.0


@app.post("/api/demo/txn")
def demo_txn():
    global last_txn
    if time.time() - last_txn < 3:
        return jsonify(error="wait a moment between presses"), 429
    dispensed = bool((request.get_json(silent=True) or {}).get("dispensed"))
    try:
        mqtt_publish.single(f"atm/{next(iter(SITES))}/cmd", f"txn:{int(dispensed)}", hostname=MQTT[0], port=MQTT[1])
    except OSError as e:
        return jsonify(error=f"MQTT broker not reachable: {e}"), 502
    last_txn = time.time()
    return jsonify(ok=True, dispensed=dispensed, at=int(last_txn))


@app.get("/admin")
def admin_page():
    return app.send_static_file("admin.html")


@app.post("/api/admin/atm/<atm_id>/override")
def admin_override(atm_id):
    if not any(a["atm_id"] == atm_id for a in fleet()):
        return jsonify(error="unknown ATM"), 404
    db = sqlite3.connect(DB)
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS override (atm_id TEXT PRIMARY KEY, at INTEGER)")
            db.execute("INSERT OR REPLACE INTO override VALUES (?, ?)", (atm_id, int(time.time())))
    finally:
        db.close()
    return jsonify(ok=True)


@app.get("/api/admin/fleet")
def admin_fleet():
    return jsonify(atms=fleet(request.args.get("demo") != "0"), user=ADMIN_USER, offline_s=OFFLINE_S, threshold=STAGE1_THRESHOLD)


@app.get("/api/admin/atm/<atm_id>")
def admin_atm(atm_id):
    # rowid order, not timestamp: pre-NTP rows carry seconds-since-boot and would sort wrong
    tel = rows("SELECT * FROM telemetry WHERE atm_id=? ORDER BY rowid DESC LIMIT 300", atm_id)[::-1]
    if not tel and atm_id in DEMO_BY_ID:
        return jsonify(demo_detail(DEMO_BY_ID[atm_id]))
    inf = rows("SELECT * FROM inference WHERE atm_id=? ORDER BY rowid DESC LIMIT 30", atm_id)[::-1]
    for r in inf:
        for k in ("stage2", "topz", "features"):
            r[k] = json.loads(r[k])
    st = rows("SELECT json FROM status WHERE atm_id=? ORDER BY rowid DESC LIMIT 1", atm_id)
    # audio + fusion events (ENABLE_AUDIO firmware); IDLE heartbeats arrive every minute, so fetch enough
    events = [json.loads(r["json"]) | {"received": r["received"]}
              for r in rows("SELECT received, json FROM event WHERE atm_id=? ORDER BY rowid DESC LIMIT 60", atm_id)]
    return jsonify(telemetry=tel, inference=inf, status=json.loads(st[0]["json"]) if st else None,
                   threshold=STAGE1_THRESHOLD, stage2_labels=STAGE2, events=events)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    args = ap.parse_args()
    DB = args.db
    MQTT = (args.mqtt_host, args.mqtt_port)
    if not os.environ.get("ATM_ADMIN_PASSWORD"):
        print(f"admin login: {ADMIN_USER} / {ADMIN_PASSWORD}   (set ATM_ADMIN_PASSWORD to fix it)")
    if not os.environ.get("ATM_DEMO_PASSWORD"):
        print(f"demo login: {DEMO_USER} / {DEMO_PASSWORD}   (set ATM_DEMO_PASSWORD to fix it)")
    app.run(host="0.0.0.0", port=args.port, debug=False)
