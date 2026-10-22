#!/usr/bin/env python3
"""Compare the host-compiled firmware (tools/host_parity_test) against the Python reference.

  python tools/check_parity.py --audio-dir data/src --manifest data/recordings.csv

For each recording: decodes to 16 kHz mono float32, runs the C++ extractor + classifier,
recomputes the same non-overlapping 31-frame windows with FrameDSP, runs the deployed
.tflite with the TFLite interpreter, and reports feature and probability differences.
Also checks the C++ fusion decision against the sklearn tree on atm_fusion_events.csv.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_esp32_model as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--audio-dir", default="data/src")
ap.add_argument("--manifest", default="data/recordings.csv")
ap.add_argument("--bin", default="build/host_parity_test")
ap.add_argument("--root", default=".")
ap.add_argument("--json-out", default="reports/parity.json")
a = ap.parse_args()

meta = json.load(open(os.path.join(a.root, "model_metadata.json")))
mu, sd = np.array(meta["audio_features"]["mean"]), np.array(meta["audio_features"]["std"])
rt = T.TFLiteRunner(open(os.path.join(a.root, meta["audio_model"]["tflite_file"]), "rb").read())

man = pd.read_csv(a.manifest)
rows, all_rel, all_abs, all_pd, agree, nwin = [], [], [], [], 0, 0
for _, r in man.iterrows():
    x = T.load_audio(os.path.join(a.audio_dir, r["file"]))
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        x.astype("<f4").tofile(f.name)
        out = subprocess.run([a.bin, "features", f.name], capture_output=True, text=True, check=True).stdout
    os.unlink(f.name)
    C = np.array([[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()])
    c_feat, c_prob = C[:, 1:1 + 36], C[:, 37:40]
    fr = T.DSP.frames(x)
    base, _, _ = T.DSP.per_frame(fr)
    nw = len(fr) // T.FRAMES_PER_WINDOW
    P = np.stack([T.window_compact(base[i * 31:(i + 1) * 31]) for i in range(nw)])
    assert len(P) == len(C), (len(P), len(C))
    absd = np.abs(c_feat - P)
    reld = absd / (np.abs(P) + 1e-3)
    p_ref = rt.predict(((P - mu) / sd).astype(np.float32))           # Python features -> TFLite
    p_cpp_feat = rt.predict(((c_feat - mu) / sd).astype(np.float32))  # C features -> TFLite
    pd_ = np.abs(c_prob - p_cpp_feat).max()                           # C engine vs TFLite, same input
    ag = (c_prob.argmax(1) == p_ref.argmax(1)).sum()
    agree += ag; nwin += len(P)
    all_abs.append(absd.max(0)); all_rel.append(reld.max(0)); all_pd.append(pd_)
    rows.append(dict(recording=r["recording_id"], windows=len(P), max_abs_feat_diff=float(absd.max()),
                     worst_feature=T.COMPACT_FEATURES[int(absd.max(0).argmax())],
                     max_rel_feat_diff=float(reld.max()), max_prob_diff_engine=float(pd_),
                     argmax_agree_vs_python=int(ag)))
    print(rows[-1])

absmax = np.max(np.stack(all_abs), 0)
print("\nper-feature max abs diff (C++ vs Python):")
for n, v in sorted(zip(T.COMPACT_FEATURES, absmax), key=lambda t: -t[1])[:8]:
    print(f"  {n:16s} {v:.3g}")
print(f"\naudio argmax agreement C++ (features+INT8 engine) vs Python+TFLite: {agree}/{nwin}")

# fusion tree parity
ev = pd.read_csv(os.path.join(a.root, "datasets", "atm_fusion_events.csv"))
out = subprocess.run([a.bin, "fusion", os.path.join(a.root, "datasets", "atm_fusion_events.csv")],
                     capture_output=True, text=True, check=True).stdout
cpp = pd.DataFrame([ln.split(",") for ln in out.strip().splitlines()],
                   columns=["event_id", "split", "label", "pred", "conf"])
te = cpp[cpp.split == "test"]
acc = float((te.label == te.pred).mean())
unk = float((te.pred == "UNKNOWN").mean())
print(f"fusion (C++ decide() incl. rules) on test events: accuracy {acc:.3f}, UNKNOWN rate {unk:.3f}")
tam = te[te.label == "SUSPECTED_TAMPER"]
print(f"  tamper events classified SUSPECTED_TAMPER: {(tam.pred == 'SUSPECTED_TAMPER').mean():.3f}")
conf = pd.crosstab(te.label, te.pred)
print(conf.to_string())

json.dump(dict(recordings=rows, audio_argmax_agree=int(agree), audio_windows=int(nwin),
               per_feature_max_abs_diff=dict(zip(T.COMPACT_FEATURES, map(float, absmax))),
               fusion_cpp_test_accuracy=acc, fusion_cpp_unknown_rate=unk,
               fusion_confusion=conf.to_dict()),
          open(os.path.join(a.root, a.json_out), "w"), indent=2)
