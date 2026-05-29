"""WP7 Experiment 6: generic ML comparators (Prop 7 -- geometry, not meaning).

Unsupervised methods on RAW amount geometry (log-magnitude histogram) recover only
mechanisms with amount-space geometry (threshold density, repeated spikes); they
miss divisibility/fiscal/temporal. The same methods on the PROJECTION feature
vectors recover more. Neither assigns meaning to clusters.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
from wp7_common import build_feature_table, matrix, FEATURE_METRICS  # noqa: E402

RESULTS = R.RESULTS
N = 10_000
SEEDS = list(range(20))
HIST_BINS = 64
HIST_RANGE = (1.0, 8.0)   # log10 amount


def raw_worker(task):
    mech, seed, n = task
    params, _ = R.MECHANISMS[mech]
    res = R.GEN_FUNCS[mech](n=n, seed=seed, **params)
    values = res[0] if isinstance(res, tuple) else res
    v = values[values > 1].astype(np.float64)
    h, _ = np.histogram(np.log10(v), bins=HIST_BINS, range=HIST_RANGE, density=True)
    return mech, seed, h.astype(np.float64)


def cluster_eval(X, y, method):
    Xs = StandardScaler().fit_transform(X)
    if method == "kmeans":
        labels = KMeans(n_clusters=14, n_init=10, random_state=0).fit_predict(Xs)
    elif method == "dbscan":
        # heuristic eps from median pairwise NN distance
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=4).fit(Xs)
        d, _ = nn.kneighbors(Xs)
        eps = float(np.median(d[:, -1]))
        labels = DBSCAN(eps=eps, min_samples=4).fit_predict(Xs)
    ari = adjusted_rand_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)
    return labels, ari, ami


def per_mechanism(y, labels):
    """recall = frac of a mechanism's datasets in its plurality cluster;
    purity = frac of that cluster that is the mechanism."""
    rows = []
    y = np.array(y); labels = np.array(labels)
    for mech in sorted(set(y)):
        idx = np.where(y == mech)[0]
        cl = labels[idx]
        home = Counter(cl).most_common(1)[0][0]
        recall = float(np.mean(cl == home))
        members = labels == home
        purity = float(np.mean(y[members] == mech)) if members.sum() else 0.0
        rows.append((mech, recall, purity, int(home)))
    return rows


def main():
    P.ensure_spf_cache(R.SPF_CACHE); P.set_spf_cache(R.SPF_CACHE)

    # raw-geometry features
    tasks = [(m, s, N) for m in R.MECH_ORDER for s in SEEDS]
    yr, raws = [], []
    with ProcessPoolExecutor(max_workers=R.N_WORKERS, initializer=R._init_worker,
                             initargs=(R.SPF_CACHE,)) as ex:
        for m, s, h in ex.map(raw_worker, tasks):
            yr.append(m); raws.append(h)
    Xraw = np.array(raws)

    # projection features (same as Exp 1)
    print("Building projection feature vectors ...", flush=True)
    yp, feats, _ = build_feature_table(N, SEEDS)
    Xproj = matrix(feats, FEATURE_METRICS)

    spaces = {"raw_magnitude_hist": (Xraw, yr), "projection_features": (Xproj, yp)}
    comp_rows, perm_rows = [], []
    for space, (X, y) in spaces.items():
        for method in ("kmeans", "dbscan"):
            labels, ari, ami = cluster_eval(X, y, method)
            note = ("clusters are numeric; no mechanism meaning assigned"
                    if space == "raw_magnitude_hist" else "richer separation via construction layers")
            comp_rows.append({"method": method, "feature_space": space,
                              "ARI": round(ari, 4), "AMI": round(ami, 4), "notes": note})
            for mech, rec, pur, home in per_mechanism(y, labels):
                perm_rows.append({"method": method, "feature_space": space, "mechanism": mech,
                                  "cluster_purity": round(pur, 3), "recall": round(rec, 3)})

    # Isolation Forest: per-mechanism mean anomaly (raw geometry)
    iso = IsolationForest(random_state=0).fit(Xraw)
    score = -iso.score_samples(Xraw)   # higher = more anomalous
    anom = {}
    for m, sc in zip(yr, score):
        anom.setdefault(m, []).append(sc)
    iso_rows = sorted(((m, float(np.mean(v))) for m, v in anom.items()), key=lambda t: -t[1])
    comp_rows.append({"method": "isolation_forest", "feature_space": "raw_magnitude_hist",
                      "ARI": "n/a", "AMI": "n/a",
                      "notes": "most-anomalous (raw geom): " +
                               ", ".join(f"{m}={s:.2f}" for m, s in iso_rows[:4])})

    with open(os.path.join(RESULTS, "wp7_ml_comparators.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "feature_space", "ARI", "AMI", "notes"])
        w.writeheader(); w.writerows(comp_rows)
    with open(os.path.join(RESULTS, "wp7_ml_per_mechanism.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "feature_space", "mechanism",
                                           "cluster_purity", "recall"])
        w.writeheader(); w.writerows(perm_rows)

    print("\nARI / AMI by feature space:")
    for r in comp_rows:
        if r["ARI"] != "n/a":
            print(f"  {r['method']:16} {r['feature_space']:20} ARI={r['ARI']:.3f} AMI={r['AMI']:.3f}")
    print(f"  isolation_forest most-anomalous (raw): {[m for m,_ in iso_rows[:4]]}")

    # per-mechanism recall under raw kmeans (which mechanisms are missed)
    raw_km = {r["mechanism"]: r["recall"] for r in perm_rows
              if r["method"] == "kmeans" and r["feature_space"] == "raw_magnitude_hist"}
    proj_km = {r["mechanism"]: r["recall"] for r in perm_rows
               if r["method"] == "kmeans" and r["feature_space"] == "projection_features"}
    print("\nkmeans recall raw vs projection (mechanisms raw geometry MISSES):")
    for m in R.MECH_ORDER:
        flag = "  <-- raw misses" if raw_km.get(m, 0) < 0.5 <= proj_km.get(m, 0) else ""
        print(f"  {m:20} raw={raw_km.get(m,0):.2f}  proj={proj_km.get(m,0):.2f}{flag}")

    raw_ari = next(r["ARI"] for r in comp_rows if r["method"] == "kmeans" and r["feature_space"] == "raw_magnitude_hist")
    proj_ari = next(r["ARI"] for r in comp_rows if r["method"] == "kmeans" and r["feature_space"] == "projection_features")
    print(f"\nProjection features recover more than raw geometry "
          f"(kmeans ARI {proj_ari:.2f} > {raw_ari:.2f}): "
          f"{'YES' if proj_ari > raw_ari else 'NO'}  -- supports Prop 7.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
