"""WP7 Experiment 5: sample-size power, contamination, and null false-positive curves.

Per-seed signatures are computed against a null distribution (20 magnitude-matched
or pure-null draws): z_seed = (Phi_mech_seed - null_mean)/null_std;
d_seed = (Phi_mech_seed - null_mean)/pooled_sd. detection = |z|>3; material =
|z|>3 and |d|>0.2.
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
import generators as G             # noqa: E402
from projections import compute_all_projections   # noqa: E402
from baselines import magnitude_matched_null      # noqa: E402

RESULTS = R.RESULTS
SEEDS = list(range(20))
N_GRID = [100, 500, 1000, 5000, 10000, 50000, 100000]

# (mechanism -> [(column_label, representative_metric)], ordered_flag)
FOCUS_5A = {
    "round": ([("TerminalDigit", "terminal_digit_chisq"), ("CentsEntropy", "cents_entropy"),
               ("DecimalResidue", "mod100_tv"), ("cp_spectrum", "c10")], False),
    "divisibility": ([("cp_spectrum", "c7"), ("L_profile", "L1"), ("TailMass", "tail_mass")], False),
    "product": ([("cp_spectrum", "c2"), ("L_profile", "L1"), ("TailMass", "tail_mass"),
                 ("LeadingDigit", "leading_digit_mad")], False),
    "temporal_concat": ([("CBAD_R2", "cbad_r2"), ("CBAD_c", "cbad_c"),
                         ("VocabGini", "vocab_gini"), ("cp_spectrum", "c2")], True),
}
FOCUS_5B = {"round": ("cents_entropy", False), "divisibility": ("c7", False),
            "temporal_concat": ("cbad_r2", True)}
EPS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
ALL_METRICS = ["leading_digit_mad", "terminal_digit_chisq", "cents_entropy",
               "mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv", "vocab_gini",
               "c2", "c3", "c5", "c7", "c10", "c5sq", "L1", "L2", "L3", "L4",
               "tail_mass", "cbad_a", "cbad_c", "cbad_r2"]


def _gen(mech, n, seed):
    params, threshold = R.MECHANISMS[mech]
    res = R.GEN_FUNCS[mech](n=n, seed=seed, **params)
    months = res[1] if isinstance(res, tuple) else None
    values = res[0] if isinstance(res, tuple) else res
    return values, months, threshold


def w_5a(task):
    mech, n, seed, ordered, kind = task
    values, months, threshold = _gen(mech, n, seed)
    if kind == "null":
        values = magnitude_matched_null(values, np.random.default_rng(seed + R.NULL_SEED_OFFSET))
        if months is not None:
            months = np.random.default_rng(seed + R.NULL_SEED_OFFSET + 1).integers(1, 13, size=values.size)
    d = compute_all_projections(values, ordered=ordered, months=months, threshold=threshold)
    return mech, n, seed, kind, {k: float(d.get(k, np.nan)) for k in ALL_METRICS}


def w_5b(task):
    mech, eps, seed, ordered = task
    n = 100_000
    nm = int(round(eps * n)); nn = n - nm
    vm = _gen(mech, nm, seed)[0] if nm > 0 else np.array([], dtype=np.int64)
    vn = G.gen_null(nn, seed + 5_000_000) if nn > 0 else np.array([], dtype=np.int64)
    values = np.concatenate([vm, vn])
    d = compute_all_projections(values, ordered=ordered)
    return mech, eps, seed, {k: float(d.get(k, np.nan)) for k in ALL_METRICS}


def w_5c(task):
    n, seed, kind = task
    if kind == "mech":
        values = G.gen_null(n, seed)
    else:
        values = magnitude_matched_null(G.gen_null(n, seed),
                                        np.random.default_rng(seed + R.NULL_SEED_OFFSET))
    d = compute_all_projections(values, ordered=True)
    return n, seed, kind, {k: float(d.get(k, np.nan)) for k in ALL_METRICS}


def perseed_stats(mech_vals, null_vals):
    mech_vals = np.array(mech_vals, float); null_vals = np.array(null_vals, float)
    mv = mech_vals[np.isfinite(mech_vals)]; nv = null_vals[np.isfinite(null_vals)]
    if mv.size < 2 or nv.size < 2:
        return None
    nm, ns = nv.mean(), nv.std(ddof=1)
    ms = mv.std(ddof=1)
    pooled = np.sqrt((ms ** 2 + ns ** 2) / 2) or 1e-12
    if ns == 0:
        ns = 1e-12
    z = (mech_vals - nm) / ns
    d = (mech_vals - nm) / pooled
    z = z[np.isfinite(z)]; d = d[np.isfinite(d)]
    det = np.abs(z) > 3
    mat = det & (np.abs(d) > 0.2)
    return dict(detection_rate=float(det.mean()), material_rate=float(mat.mean()),
                mean_z=float(np.mean(z)), mean_d=float(np.mean(d)))


def main():
    P.ensure_spf_cache(R.SPF_CACHE); P.set_spf_cache(R.SPF_CACHE)

    # ---------- 5A power curves ----------
    tasks = []
    for mech, (focus, ordered) in FOCUS_5A.items():
        for n in N_GRID:
            for s in SEEDS:
                tasks.append((mech, n, s, ordered, "mech"))
                tasks.append((mech, n, s, ordered, "null"))
    store = {}
    with ProcessPoolExecutor(max_workers=R.N_WORKERS, initializer=R._init_worker,
                             initargs=(R.SPF_CACHE,)) as ex:
        for mech, n, seed, kind, md in ex.map(w_5a, tasks):
            store.setdefault((mech, n, kind), []).append(md)
    rows = []
    for mech, (focus, ordered) in FOCUS_5A.items():
        for n in N_GRID:
            mlist = store[(mech, n, "mech")]; nlist = store[(mech, n, "null")]
            for col, metric in focus:
                st = perseed_stats([d[metric] for d in mlist], [d[metric] for d in nlist])
                if st is None:
                    continue
                rows.append({"mechanism": mech, "projection": col, "N": n, **{k: round(v, 4) for k, v in st.items()}})
    with open(os.path.join(RESULTS, "wp7_power_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mechanism", "projection", "N",
                                           "detection_rate", "material_rate", "mean_z", "mean_d"])
        w.writeheader(); w.writerows(rows)
    def det_at(mech, col, n):
        for r in rows:
            if r["mechanism"] == mech and r["projection"] == col and r["N"] == n:
                return f"{r['detection_rate']:.2f}"
        return "na"
    print("5A power curves (detection_rate at N=100,1000,100000):")
    for mech, (focus, _) in FOCUS_5A.items():
        col, metric = focus[0]
        print(f"  {mech:16}/{col:14} det: N=100->{det_at(mech, col, 100)} "
              f"N=1k->{det_at(mech, col, 1000)} N=100k->{det_at(mech, col, 100000)}")

    # ---------- 5B contamination curves ----------
    tasksb = [(mech, eps, s, ordered) for mech, (metric, ordered) in FOCUS_5B.items()
              for eps in EPS for s in SEEDS]
    refb = [(mech, 0.0, s, ordered) for mech, (metric, ordered) in FOCUS_5B.items() for s in SEEDS]
    storeb = {}
    with ProcessPoolExecutor(max_workers=R.N_WORKERS, initializer=R._init_worker,
                             initargs=(R.SPF_CACHE,)) as ex:
        for mech, eps, seed, md in ex.map(w_5b, tasksb + refb):
            storeb.setdefault((mech, eps), []).append(md)
    rowsb = []
    for mech, (metric, ordered) in FOCUS_5B.items():
        ref = storeb[(mech, 0.0)]
        for eps in EPS:
            cur = storeb[(mech, eps)]
            st = perseed_stats([d[metric] for d in cur], [d[metric] for d in ref])
            if st is None:
                continue
            rowsb.append({"mechanism": mech, "projection": metric, "epsilon": eps,
                          "detection_rate": round(st["detection_rate"], 4),
                          "material_rate": round(st["material_rate"], 4),
                          "mean_z": round(st["mean_z"], 3)})
    with open(os.path.join(RESULTS, "wp7_contamination_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mechanism", "projection", "epsilon",
                                           "detection_rate", "material_rate", "mean_z"])
        w.writeheader(); w.writerows(rowsb)
    print("\n5B contamination (detection_rate by epsilon):")
    for mech, (metric, _) in FOCUS_5B.items():
        seq = "  ".join(f"{r['epsilon']}:{r['detection_rate']:.2f}"
                        for r in rowsb if r["mechanism"] == mech)
        print(f"  {mech:16}/{metric:12} {seq}")

    # ---------- 5C null FPR ----------
    seeds_c = list(range(200))
    tasksc = [(n, s, k) for n in N_GRID for s in seeds_c for k in ("mech", "null")]
    storec = {}
    with ProcessPoolExecutor(max_workers=R.N_WORKERS, initializer=R._init_worker,
                             initargs=(R.SPF_CACHE,)) as ex:
        for n, seed, kind, md in ex.map(w_5c, tasksc):
            storec.setdefault((n, kind), []).append(md)
    # Use the engine's across-seed resolve_columns (where the Bonferroni+materiality
    # gate operates): split the 50 seeds into groups of 5; a group is a "trial".
    GROUP = 20   # the engine's design seed count; materiality gate is tuned for this
    rowsc = []
    for n in N_GRID:
        mlist = storec[(n, "mech")]; nlist = storec[(n, "null")]
        ng = min(len(mlist), len(nlist)) // GROUP
        fp, mat, maxz = [], [], []
        for g in range(ng):
            md = mlist[g * GROUP:(g + 1) * GROUP]; nd = nlist[g * GROUP:(g + 1) * GROUP]
            cols = R.resolve_columns(md, nd, R.COLUMN_METRICS)
            fp.append(any(cols[c]["verdict"] == "+" for c in R.MATRIX_COLUMNS))
            mat.append(any(cols[c]["verdict_material"] == "+" for c in R.MATRIX_COLUMNS))
            zs = [abs(cols[c]["z"]) for c in R.MATRIX_COLUMNS if np.isfinite(cols[c]["z"])]
            maxz.append(max(zs) if zs else 0.0)
        rowsc.append({"N": n, "n_seeds": len(mlist),
                      "false_positive_rate": round(float(np.mean(fp)), 4),
                      "false_material_rate": round(float(np.mean(mat)), 4),
                      "max_z_mean": round(float(np.mean(maxz)), 3)})
    with open(os.path.join(RESULTS, "wp7_null_fpr.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["N", "n_seeds", "false_positive_rate",
                                           "false_material_rate", "max_z_mean"])
        w.writeheader(); w.writerows(rowsc)
    print("\n5C null false-positive rates:")
    for r in rowsc:
        print(f"  N={r['N']:>6}  FPR(any +)={r['false_positive_rate']:.3f}  "
              f"false_material={r['false_material_rate']:.3f}  max|z|mean={r['max_z_mean']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
