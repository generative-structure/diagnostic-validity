"""WP7 Experiment 2: Generalized baseline absorption.

Does absorption generalize beyond thresholds? For rounding, temporal, and
divisibility mechanisms, a mechanism-aware null reproduces the mechanism's
structure and should ABSORB the signal; construction-neutral and magnitude-matched
nulls should DETECT it.
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
from projections import compute_all_projections   # noqa: E402
from baselines import magnitude_matched_null, generate_parent_support_null  # noqa: E402

N = 100_000
SEEDS = list(range(20))
RESULTS = R.RESULTS
OFF = R.NULL_SEED_OFFSET

PLAN = {
    "round": ["magnitude_matched", "cents_neutral", "cents_aware"],
    "fiscal": ["magnitude_matched", "temporal_neutral", "temporal_aware"],
    "divisibility": ["magnitude_matched", "divisibility_neutral", "divisibility_aware"],
}
# projections to report per mechanism (the ones expected to absorb/survive)
FOCUS = {
    "round": ["CentsEntropy", "TerminalDigit", "DecimalResidue", "cp_spectrum"],
    "fiscal": ["TemporalConcentration", "cp_spectrum"],
    "divisibility": ["cp_spectrum", "L_profile", "TailMass"],
}


def abs_worker(task):
    mech_name, btype, seed, n = task
    params, threshold = R.MECHANISMS[mech_name]
    res = R.GEN_FUNCS[mech_name](n=n, seed=seed, **params)
    months = res[1] if isinstance(res, tuple) else None
    values = res[0] if isinstance(res, tuple) else res
    rng = np.random.default_rng(seed + OFF)
    null_months = None

    if btype in ("magnitude_matched", "divisibility_neutral"):
        null = magnitude_matched_null(values, rng)
        if months is not None:
            null_months = rng.integers(1, 13, size=null.size, dtype=np.int64)
    elif btype == "cents_neutral":
        null = generate_parent_support_null(values, seed + OFF, mag_range=(1, 6))
    elif btype == "cents_aware":
        # Faithful rounding-aware null: reproduce gen_round's multiples (100/500/1000)
        # on the magnitude-matched base, so the FULL round-dollar structure is matched.
        base = magnitude_matched_null(values, rng).astype(np.float64)
        mult = rng.choice(np.array([100, 500, 1000]), size=base.size, p=[0.5, 0.3, 0.2])
        null = np.maximum((np.round(base / mult) * mult), mult).astype(np.int64)
    elif btype == "temporal_neutral":
        null = values.copy()                                   # same amounts (nuisance held)
        null_months = rng.integers(1, 13, size=null.size, dtype=np.int64)
    elif btype == "temporal_aware":
        null = magnitude_matched_null(values, rng)
        null_months = rng.choice(months, size=null.size)       # reproduce calendar concentration
    elif btype == "divisibility_aware":
        base = magnitude_matched_null(values, rng)
        null = np.maximum(base - (base % 7), 7).astype(np.int64)  # force multiples of 7
    else:
        raise ValueError(btype)

    md = compute_all_projections(values, ordered=False, months=months, threshold=threshold)
    nd = compute_all_projections(null, ordered=False, months=null_months, threshold=threshold)
    return mech_name, btype, seed, md, nd


def main():
    P.ensure_spf_cache(R.SPF_CACHE); P.set_spf_cache(R.SPF_CACHE)
    tasks = [(m, b, s, N) for m, bs in PLAN.items() for b in bs for s in SEEDS]
    bucket = {(m, b): {"mech": [], "null": []} for m, bs in PLAN.items() for b in bs}
    with ProcessPoolExecutor(max_workers=R.N_WORKERS, initializer=R._init_worker,
                             initargs=(R.SPF_CACHE,)) as ex:
        for m, b, s, md, nd in ex.map(abs_worker, tasks):
            bucket[(m, b)]["mech"].append(md)
            bucket[(m, b)]["null"].append(nd)

    rows, summary = [], []
    contradiction = []
    for m, bs in PLAN.items():
        for b in bs:
            cols = R.resolve_columns(bucket[(m, b)]["mech"], bucket[(m, b)]["null"], R.COLUMN_METRICS)
            absorbed, survived = [], []
            for col in R.MATRIX_COLUMNS:
                st = cols[col]
                rows.append({"mechanism": m, "baseline_type": b, "projection": col,
                             "z_score": round(st["z"], 2) if np.isfinite(st["z"]) else "nan",
                             "effect_size_d": round(st["d"], 3) if np.isfinite(st["d"]) else "nan",
                             "verdict": st["verdict"]})
                if col in FOCUS[m]:
                    # Guard against degenerate-statistic z inflation: a focus projection
                    # truly "survives" only if verdict==+ AND the relative mean gap is
                    # non-negligible (terminal-digit chi-square on all-round data is a
                    # near-constant, so a 0.02% gap can read as huge z but is not a signal).
                    mm, mn = st["mean_mech"], st["mean_null"]
                    rel = (abs(mm - mn) / (abs(mm) + 1e-12)) if np.isfinite(mm) and np.isfinite(mn) else 0.0
                    truly = (st["verdict"] == "+" and rel > 1e-3)
                    (survived if truly else absorbed).append(col)
            summary.append({"mechanism": m, "baseline_type": b,
                            "focus_survived": ";".join(survived) or "none",
                            "focus_absorbed": ";".join(absorbed) or "none"})
            tag = "aware" in b
            # contradiction: aware baseline should absorb (focus survives should be empty)
            if tag and survived:
                contradiction.append((m, b, survived))

    with open(os.path.join(RESULTS, "wp7_baseline_absorption_generalized.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mechanism", "baseline_type", "projection",
                                           "z_score", "effect_size_d", "verdict"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RESULTS, "wp7_baseline_absorption_generalized_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mechanism", "baseline_type", "focus_survived", "focus_absorbed"])
        w.writeheader(); w.writerows(summary)

    print("SUMMARY (focus projections survived vs absorbed):")
    for r in summary:
        print(f"  {r['mechanism']:13} {r['baseline_type']:22} survived=[{r['focus_survived']}] "
              f"absorbed=[{r['focus_absorbed']}]")

    if contradiction:
        print("\n*** CONTRADICTION: aware baseline did NOT absorb:", contradiction)
        return 1
    print("\nAbsorption generalizes: each mechanism-aware null absorbs its signal; "
          "neutral/magnitude-matched nulls detect it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
