"""Collect the evaluation results of every ML model into ml/evaluation_metrics.json.

    cd ml/cascade && uv run python ../build_evaluation_metrics.py

Stage 1, Stage 2 and the cascade come from their saved results.json; the humidity-disabled cascade
(the configuration the firmware runs with ENABLE_HUMIDITY 0) is re-evaluated here on the test split.
Audio and fusion numbers come from ml/audio_tinyml/reports/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
ML = Path(__file__).resolve().parent; sys.path.insert(0, str(ML / "cascade")); import evaluate as ev
J = lambda p: json.loads((ML / p).read_text())
r = lambda x: round(float(x), 4)

def rnd(x):
    if isinstance(x, float): return float(f"{x:.4g}")
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, list): return [rnd(v) for v in x]
    return x

# deployed cascade: humidity sensor disabled, its 12 inputs at z = 0
feats, _ = ev.stage1_common.load_split("test")
truth = pd.read_csv(ev.ROOT / "data" / "test.csv")["label"].to_numpy()
names = J("../data/manifest.json")["feature_columns"]
hum = [i for i, n in enumerate(names) if "humid" in n or n in ("dewpoint_mean", "temp_rh_corr")]
thr = J("stage-1/artifacts/threshold.json")["threshold"]
m1 = tf.keras.models.load_model(ML / "stage-1/artifacts/stage1.keras"); m2 = tf.keras.models.load_model(ML / "stage-2/artifacts/stage2.keras")
x = feats.copy(); x[:, hum] = 0.0
fw = m1.predict(x, verbose=0).reshape(-1) >= thr
pred = np.full(len(x), "NORMAL", dtype=object)
pred[fw] = np.asarray(ev.stage2_common.LABELS, dtype=object)[np.argmax(m2.predict(x[fw], verbose=0), 1)]
L = ev.FINAL_LABELS
p, rc, f, s = precision_recall_fscore_support(truth, pred, labels=L, zero_division=0)
no_hum = dict(
    description="deployed configuration: BME280 humidity disabled, its 12 derived inputs fed z = 0 (training mean); same frozen models and test split",
    humidity_features_neutralised=[names[i] for i in hum],
    overall=dict(accuracy=r(accuracy_score(truth, pred)), balanced_accuracy=r(balanced_accuracy_score(truth, pred)),
                 macro_precision=r(p.mean()), macro_recall=r(rc.mean()), macro_f1=r(f.mean())),
    per_class={l: dict(precision=r(p[i]), recall=r(rc[i]), f1=r(f[i]), support=int(s[i])) for i, l in enumerate(L)},
    confusion_matrix=dict(labels=L, rows="true", columns="predicted", values=confusion_matrix(truth, pred, labels=L).tolist()))

s1, s2, cas = J("stage-1/artifacts/results.json"), J("stage-2/artifacts/results.json"), J("cascade/artifacts/results.json")
am, par = J("audio_tinyml/reports/metrics.json"), J("audio_tinyml/reports/parity.json")
s1keys = ["n_samples", "threshold", "accuracy", "balanced_accuracy", "precision_abnormal", "recall_abnormal", "f1_abnormal",
          "specificity_normal", "roc_auc", "confusion_matrix"]
s2keys = ["n_samples", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class", "confusion_matrix", "sensor_fault_subtypes"]
out = {
  "about": {
    "project": "ATM Sentinel",
    "data": "synthetic, physically modelled telemetry (atmsim); test split = held-out ATM units and runs, never used for training or threshold selection",
    "audio_data": "~37 s of internet audio clips plus synthetic background/abnormal variants; DISPENSE test windows come from a held-out recording",
    "hardware": "all numbers are offline (PC) evaluations; none are measured on the deployed ESP32",
    "confusion_matrix_layout": "rows = true class, columns = predicted class",
  },
  "health_cascade": {
    "stage1_normal_vs_abnormal": {
      "model": "MLP 57-16-8-1, sigmoid; INT8 TFLite on ESP32", "trainable_parameters": s1["trainable_parameters"],
      "validation": {k: s1["validation"][k] for k in s1keys}, "test": {k: s1["test"][k] for k in s1keys}},
    "stage2_fault_type": {
      "model": "MLP 57-16-8-5, softmax; INT8 TFLite on ESP32", "trainable_parameters": s2["trainable_parameters"],
      "note": s2["final_test_evaluation"],
      "validation": {k: s2["validation"][k] for k in s2keys}, "test": {k: s2["test"][k] for k in s2keys}},
    "end_to_end": {k: cas[k] for k in ["n_samples", "overall", "per_class", "confusion_matrix", "cascade_error_sources", "fault_class_routing"]},
    "end_to_end_humidity_disabled": no_hum,
  },
  "audio_tinyml": {
    "classifier_deployed": {
      "model": "B1_mlp_int8: MLP 36-16-3 full-integer INT8", "classes": ["BACKGROUND", "DISPENSE", "ABNORMAL"],
      "flash_bytes": am["results"]["B1_mlp_int8"]["flash_bytes"], "ram_bytes": am["results"]["B1_mlp_int8"]["ram_bytes"],
      "test": am["results"]["B1_mlp_int8"]["metrics"],
      "leave_one_recording_out_mean_accuracy": am["results"]["B1_mlp_int8"]["loro_mean_acc"],
      "float_vs_int8_agreement": am["mlp_float_vs_int8_agreement"],
      "test_window_counts": am["counts"],
      "caveat": "accuracy is dominated by synthetic BACKGROUND windows; ABNORMAL test set is 5 windows from one recording"},
    "classifier_candidates": {k: {"accuracy": v["metrics"]["accuracy"], "macro_f1": v["metrics"]["macro_f1"],
                                  "loro_mean_accuracy": v.get("loro_mean_acc"), "flash_bytes": v["flash_bytes"], "ram_bytes": v["ram_bytes"]}
                              for k, v in am["results"].items()},
    "fusion_event_model": {
      "model": "C1_fusion_decision_tree (31 nodes) over audio + SW-420 + PIR + cascade tamper",
      "data": "synthetic visit events; audio drawn from out-of-fold real windows",
      "tree_only_test": am["fusion"]["C1_fusion_decision_tree"]["metrics"],
      "deployed_cpp_with_rules_test": {"accuracy": par["fusion_cpp_test_accuracy"], "unknown_rate": par["fusion_cpp_unknown_rate"],
                                       "confusion": par["fusion_confusion"]},
      "alternative_logreg_test": am["fusion"]["C2_fusion_logreg"]["metrics"]},
    "firmware_parity_vs_python": {
      "audio_argmax_agreement": f'{par["audio_argmax_agree"]}/{par["audio_windows"]}',
      "max_abs_feature_diff": max(par["per_feature_max_abs_diff"].values())},
  },
}
(ML / "evaluation_metrics.json").write_text(json.dumps(rnd(out), indent=2) + "\n")
print("written", ML / "evaluation_metrics.json")
