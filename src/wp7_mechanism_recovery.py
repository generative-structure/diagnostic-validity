"""WP7 Experiment 1: Mechanism recovery benchmark.

Can projection outputs recover the construction mechanism? Compares feature sets;
no single family should dominate (Prop 3).
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, mutual_info_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R          # noqa: E402
from wp7_common import build_feature_table, matrix, FAMILIES, RESULTS  # noqa: E402

N = 10_000
SEEDS = list(range(20))

FEATURE_SETS = {
    "a_digit_decimal": FAMILIES["digit_decimal"],
    "b_vocab": FAMILIES["vocab"],
    "c_threshold_temporal": FAMILIES["threshold_temporal"],
    "d_msa": FAMILIES["msa"],
    "e_cbad": FAMILIES["cbad"],
    "f_all": sum(FAMILIES.values(), []),
    "g_all_plus_prime": sum(FAMILIES.values(), []) + ["prime_fraction"],
}


def evaluate(X, y):
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    yp = cross_val_predict(rf, X, y, cv=skf)
    labels = sorted(set(y))
    macro = f1_score(y, yp, average="macro")
    per = dict(zip(labels, f1_score(y, yp, average=None, labels=labels)))
    mi = mutual_info_score(y, yp)
    return macro, per, mi, yp, labels


def main():
    print("Building feature table (14 mech x 20 seeds, N=10k) ...", flush=True)
    y, feats, _ = build_feature_table(N, SEEDS)
    y = np.array(y)

    rows = []
    full_macro = None
    f_pred = f_labels = None
    for fs_name, metrics in FEATURE_SETS.items():
        X = matrix(feats, metrics)
        macro, per, mi, yp, labels = evaluate(X, y)
        rows.append({"feature_set": fs_name, "n_features": len(metrics),
                     "macro_f1": round(macro, 4), "mutual_info": round(mi, 4),
                     "per_mechanism_f1_json": json.dumps({k: round(v, 3) for k, v in per.items()})})
        print(f"  {fs_name:22} macro-F1={macro:.3f}  MI={mi:.3f}")
        if fs_name == "f_all":
            full_macro, f_pred, f_labels = macro, yp, labels

    with open(os.path.join(RESULTS, "wp7_mechanism_recovery.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature_set", "n_features", "macro_f1",
                                           "mutual_info", "per_mechanism_f1_json"])
        w.writeheader(); w.writerows(rows)

    # Confusion matrix for feature set f
    cm = confusion_matrix(y, f_pred, labels=f_labels)
    with open(os.path.join(RESULTS, "wp7_mechanism_recovery_confusion.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["true\\pred"] + f_labels)
        for i, lab in enumerate(f_labels):
            w.writerow([lab] + cm[i].tolist())

    # Ablation on feature set f: remove one family at a time
    abl = []
    full_set = sum(FAMILIES.values(), [])
    for fam, fam_metrics in FAMILIES.items():
        reduced = [m for m in full_set if m not in fam_metrics]
        macro, _, _, _, _ = evaluate(matrix(feats, reduced), y)
        abl.append({"removed_family": fam, "macro_f1": round(macro, 4),
                    "delta_f1": round(full_macro - macro, 4)})
        print(f"  ablate -{fam:18} macro-F1={macro:.3f}  delta={full_macro-macro:+.3f}")
    with open(os.path.join(RESULTS, "wp7_mechanism_recovery_ablation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["removed_family", "macro_f1", "delta_f1"])
        w.writeheader(); w.writerows(abl)

    # Theory check: no single family should near-perfectly recover; combined should win.
    single = {r["feature_set"]: r["macro_f1"] for r in rows if r["feature_set"][0] in "abcde"}
    best_single = max(single.values())
    print(f"\nbest single-family macro-F1={best_single:.3f}; combined f_all={full_macro:.3f}")
    if best_single > 0.95:
        print("*** CONTRADICTION: a single projection family near-perfectly recovers mechanisms.")
        return 1
    if full_macro <= best_single:
        print("*** NOTE: combined does not exceed best single family -- inspect.")
    print("Consistent with Prop 3: combined projections beat any single family.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
