"""Sanity baseline. NOT the deliverable model — a check that the dataset is
learnable, that it is not trivially learnable, and that nothing leaks."""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

META = {"run_id","atm_id","window_start_ts","window_end_ts","label","stage1",
        "fine_state","purity","ambiguous","severity","run_subtype","time_to_failure_s"}

d = sys.argv[1] if len(sys.argv) > 1 else "data"
tr = pd.read_csv(f"{d}/train.csv"); va = pd.read_csv(f"{d}/validation.csv"); te = pd.read_csv(f"{d}/test.csv")
feats = [c for c in tr.columns if c not in META]
sc = json.load(open(f"{d}/scaler.json"))
def X(df): return ((df[feats] - pd.Series(sc["mean"])[feats]) / pd.Series(sc["std"])[feats]).values

print(f"features={len(feats)}  train={len(tr)} val={len(va)} test={len(te)}")
print("ATM overlap train/test:", set(tr.atm_id) & set(te.atm_id))
print("run overlap  train/test:", set(tr.run_id) & set(te.run_id))

for name, ycol in (("STAGE 1 (healthy/abnormal)","stage1"), ("STAGE 2 (fault type)","label")):
    sub_tr = tr if ycol == "stage1" else tr[tr.label != "NORMAL"]
    sub_te = te if ycol == "stage1" else te[te.label != "NORMAL"]
    m = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=0)
    m.fit(X(sub_tr), sub_tr[ycol]); p = m.predict(X(sub_te))
    print(f"\n===== {name} =====  test acc = {accuracy_score(sub_te[ycol], p):.4f}")
    print(classification_report(sub_te[ycol], p, digits=3, zero_division=0))
    labs = sorted(sub_te[ycol].unique())
    print(pd.DataFrame(confusion_matrix(sub_te[ycol], p, labels=labs), index=labs, columns=labs))
    if ycol == "label":
        imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
        print("\ntop features:", ", ".join(f"{k}={v:.3f}" for k,v in imp.head(10).items()))
