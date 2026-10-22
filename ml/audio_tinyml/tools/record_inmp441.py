"""Record training audio through the INMP441 (firmware/tests/audio_recorder) and add it to the manifest.

  python3 tools/record_inmp441.py <name> <seconds> <BACKGROUND|DISPENSE|ABNORMAL> [--port /dev/ttyUSB0]

Writes data/src/mic_<name>.wav (16 kHz mono int16) and adds/replaces its row in data/recordings.csv
with split "block" (last 40 % held out for testing, see train_esp32_model.py). Stdlib only.
Close the Arduino Serial Monitor first.
"""
import argparse
import array
import csv
import math
import os
import termios
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SR, BAUD = 16000, termios.B921600  # must match audio_recorder.ino


def open_port(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    a = termios.tcgetattr(fd)
    a[0] = a[1] = a[3] = 0                                       # raw: no input/output/local processing
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    a[4] = a[5] = BAUD
    a[6][termios.VMIN], a[6][termios.VTIME] = 0, 10                # reads time out after 1 s
    termios.tcsetattr(fd, termios.TCSANOW, a)
    return fd


def read_line(fd, deadline):
    buf = b""
    while time.time() < deadline:
        c = os.read(fd, 1)
        if c == b"\n":
            return buf.decode(errors="replace").strip()
        buf += c
    raise TimeoutError("no reply from the board: is audio_recorder flashed and the Serial Monitor closed?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("seconds", type=int)
    ap.add_argument("label", choices=["BACKGROUND", "DISPENSE", "ABNORMAL"])
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--note", default="", help="what was recorded, for the manifest")
    a = ap.parse_args()

    fd = open_port(a.port)
    time.sleep(2.5)                      # opening the port resets the ESP32; let it boot
    termios.tcflush(fd, termios.TCIFLUSH)
    os.write(fd, f"R {a.seconds}\n".encode())
    deadline = time.time() + 10
    while not (line := read_line(fd, deadline)).startswith("REC "):
        if line.startswith("ERR"):
            raise SystemExit(line)
    n = int(line.split()[1])
    print(f"recording {a.seconds} s ... make the sound now")
    data, t0 = bytearray(), time.time()
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk and time.time() - t0 > a.seconds + 10:
            raise SystemExit(f"stream stalled after {len(data)} of {n} bytes")
        data += chunk
    os.close(fd)

    os.makedirs(os.path.join(DATA, "src"), exist_ok=True)
    fname = f"mic_{a.name}.wav"
    with wave.open(os.path.join(DATA, "src", fname), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(data)

    # per-second level, same dBFS scale as the firmware's [audio] lines
    s = array.array("h", bytes(data))
    db = [20 * math.log10(math.sqrt(sum(v * v for v in s[i:i + SR]) / SR) / 32768 + 1e-9)
          for i in range(0, len(s) - SR + 1, SR)]
    print(f"saved data/src/{fname}: {len(s) / SR:.0f} s, level min {min(db):.0f} / median "
          f"{sorted(db)[len(db) // 2]:.0f} / max {max(db):.0f} dBFS")
    if max(db) < -75:
        print("WARNING: almost silent. Check the mic wiring (L/R to GND).")

    path = os.path.join(DATA, "recordings.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    rows = [r for r in rows if r["recording_id"] != f"mic_{a.name}"]
    rows.append(dict(recording_id=f"mic_{a.name}", file=fname, label=a.label,
                     source_desc=f"INMP441 recording: {a.note or a.name}", split="block",
                     notes="recorded through the node's own mic; temporal block split"))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"manifest: mic_{a.name} -> {a.label}")


if __name__ == "__main__":
    main()
