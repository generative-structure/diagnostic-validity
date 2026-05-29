"""End-to-end broker retrospective on real procurement data (Interior FY2023).

Runs the full broker over a mixed real dataset and reports the decision-state
distribution, contrasted with a naive flat-detection system. Demonstrates broker
performance rather than component validation.

Method. For amount projections, mechanism variation is obtained by bootstrap
resampling (B=20) of the real data; the null is B magnitude-matched draws; z and
the verdict use the engine's cell_stats/verdict rule. Because Cohen's d is
ill-defined on a single observed sample, real-data materiality uses a transparent
relative-effect gate: |mean_obs - mean_null| / |mean_null| > 0.05. Threshold
density is evaluated at known FAR thresholds under both a magnitude-matched and a
construction-neutral null. Dimension-dependent projections whose dimension is not
recorded (temporal concentration without timestamps; CBAD without a meaningful
order) are routed to Uninterpretable.
"""
from __future__ import annotations

import csv
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
from projections import compute_all_projections, threshold_density_ratio   # noqa: E402
from baselines import magnitude_matched_null, generate_parent_support_null  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = R.RESULTS
ZIP = os.path.join(ROOT, "data", "wp6", "interior_fy2023.zip")
B = 20
REL_MATERIAL = 0.05
csv.field_size_limit(10 ** 7)

AMOUNT_COLS = {
    "LeadingDigit": ["leading_digit_mad"], "TerminalDigit": ["terminal_digit_chisq"],
    "CentsEntropy": ["cents_entropy"],
    "DecimalResidue": ["mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv"],
    "VocabGini": ["vocab_gini"],
    "cp_spectrum": ["c2", "c3", "c5", "c7", "c10", "c5sq"],
    "L_profile": ["L1", "L2", "L3", "L4"], "TailMass": ["tail_mass"],
}


def load():
    with zipfile.ZipFile(ZIP) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as fh:
            rd = csv.reader((l.decode("utf-8", "replace") for l in fh))
            hdr = next(rd); ai = hdr.index("federal_action_obligation")
            amts = []
            for row in rd:
                try:
                    amts.append(float(row[ai]))
                except (ValueError, IndexError):
                    pass
    a = np.round(np.array(amts) * 100).astype(np.int64)
    return a[(a >= 10_000) & (a <= 50_000_000)]


def reduce_z(mdicts, ndicts, metrics):
    best = None
    for m in metrics:
        st = R.cell_stats(np.array([d.get(m, np.nan) for d in mdicts], float),
                          np.array([d.get(m, np.nan) for d in ndicts], float))
        if not (st["z"] is None or np.isnan(st["z"])):
            if best is None or abs(st["z"]) > abs(best["z"]):
                best = st
    return best


def main():
    P.ensure_spf_cache(R.SPF_CACHE); P.set_spf_cache(R.SPF_CACHE)
    cents = load()
    n = cents.size
    print(f"Interior FY2023 amounts in $100..$500k: n={n}")
    rng = np.random.default_rng(0)

    # bootstrap mech projection dicts + magnitude-matched null dicts
    mdicts, ndicts = [], []
    for b in range(B):
        boot = cents[rng.integers(0, n, size=n)]
        mdicts.append(compute_all_projections(boot, ordered=False))
        ndicts.append(compute_all_projections(
            magnitude_matched_null(cents, np.random.default_rng(b + 7_000_000)), ordered=False))

    rows = []
    naive = discover = suppress = 0
    for col, mets in AMOUNT_COLS.items():
        st = reduce_z(mdicts, ndicts, mets)
        z = st["z"]; mm, mn = st["mean_mech"], st["mean_null"]
        verdict = "+" if abs(z) > 3 else ("0" if abs(z) < 1.5 else "?")
        rel = abs(mm - mn) / (abs(mn) + 1e-12)
        material = rel > REL_MATERIAL
        if verdict == "+":
            naive += 1
            state = "Discover" if material else "Suppress"
            discover += int(material); suppress += int(not material)
        else:
            state = "0 (no alert)"
        rows.append({"diagnostic": col, "z": round(z, 1), "rel_effect": round(rel, 3),
                     "naive_alert": verdict == "+", "broker_state": state})

    # threshold density at known FAR thresholds, two baselines
    def td_z(T, baseline):
        obs = threshold_density_ratio(cents, T, 0.05)
        nulls = []
        for b in range(B):
            r2 = np.random.default_rng(b + 9_000_000)
            nl = (magnitude_matched_null(cents, r2) if baseline == "magnitude_matched"
                  else generate_parent_support_null(cents, b + 9_000_000, mag_range=(4, 7.7)))
            nulls.append(threshold_density_ratio(nl, T, 0.05))
        nv = np.array(nulls, float); nv = nv[np.isfinite(nv)]
        if not np.isfinite(obs) or nv.size < 2 or nv.std(ddof=1) == 0:
            return float("nan"), obs
        return (obs - nv.mean()) / nv.std(ddof=1), obs

    for T, lab in [(1_000_000, "$10k"), (2_500_000, "$25k")]:
        zmm, obs = td_z(T, "magnitude_matched")
        zcn, _ = td_z(T, "construction_neutral")
        # broker: discovery uses construction-neutral; naive uses magnitude-matched flat
        naive_alert = np.isfinite(zmm) and abs(zmm) > 3
        if np.isfinite(zcn) and abs(zcn) > 3:
            state = "Discover (neutral)" if not naive_alert else "Discover"
            if not naive_alert:
                pass  # broker discovers what naive (magnitude-matched) absorbs
        else:
            state = "0 (no alert)"
        if naive_alert:
            naive += 1
        rows.append({"diagnostic": f"ThresholdDensity@{lab}", "z": round(zmm, 1) if np.isfinite(zmm) else "nan",
                     "rel_effect": "n/a", "naive_alert": bool(naive_alert),
                     "broker_state": ("Absorbed (mag-matched); " + state) if (np.isfinite(zmm) and abs(zmm) < 3 and "Discover" in state) else state})
        if "Discover" in state:
            discover += 1

    # dimension-dependent projections without recorded dimension
    uninterpretable = 0
    for col in ["TemporalConcentration", "CBAD_a/c/R2"]:
        rows.append({"diagnostic": col, "z": "n/a", "rel_effect": "n/a",
                     "naive_alert": False, "broker_state": "Uninterpretable (dimension absent)"})
        uninterpretable += 1

    with open(os.path.join(RESULTS, "wp10_broker_retrospective.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["diagnostic", "z", "rel_effect", "naive_alert", "broker_state"])
        w.writeheader(); w.writerows(rows)

    reduction = (naive - discover) / naive if naive else float("nan")
    print("\nDIAGNOSTIC -> STATE:")
    for r in rows:
        print(f"  {r['diagnostic']:24} z={str(r['z']):>7} rel={str(r['rel_effect']):>6} "
              f"naive={'Y' if r['naive_alert'] else '.'}  -> {r['broker_state']}")
    print(f"\nNaive flat-detection alerts: {naive}")
    print(f"Broker Discover: {discover}  Suppress: {suppress}  "
          f"Uninterpretable: {uninterpretable}")
    print(f"Alert-volume reduction (naive -> Discover): {reduction*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
