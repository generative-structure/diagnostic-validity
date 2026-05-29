"""WP7 Experiment 3: Missing-dimension identifiability.

A mechanism present in the data is only identifiable if the construction dimension
it acts on is recorded. Projections requiring an unrecorded dimension return null
or degrade.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
import generators as G             # noqa: E402
from baselines import magnitude_matched_null   # noqa: E402

N = 100_000
SEEDS = list(range(20))
RESULTS = R.RESULTS
OFF = R.NULL_SEED_OFFSET


def zrow(mech_vals, null_vals):
    return R.cell_stats(np.array(mech_vals, float), np.array(null_vals, float))


def main():
    P.ensure_spf_cache(R.SPF_CACHE); P.set_spf_cache(R.SPF_CACHE)
    rows = []

    def add(mech, dim, proj, value, z, verdict):
        rows.append({"mechanism": mech, "dimension_present": dim, "projection": proj,
                     "value": ("nan" if not np.isfinite(value) else round(value, 5)),
                     "z_score": ("nan" if not np.isfinite(z) else round(z, 3)),
                     "verdict": verdict})

    # ---- 3A: fiscal +/- timestamps ----
    tc_mech, tc_null, ld_mech, ld_null = [], [], [], []
    for s in SEEDS:
        amounts, months = G.gen_fiscal(N, s)
        rng = np.random.default_rng(s + OFF)
        tc_mech.append(P.temporal_concentration(months))
        tc_null.append(P.temporal_concentration(rng.integers(1, 13, size=months.size)))
        ld_mech.append(P.leading_digit_mad(amounts[amounts > 1]))
        nl = magnitude_matched_null(amounts, rng)
        ld_null.append(P.leading_digit_mad(nl[nl > 1]))
    st = zrow(tc_mech, tc_null)
    add("fiscal", "timestamps_present", "TemporalConcentration", st["mean_mech"], st["z"],
        R.verdict_from(st["z"], st["sc"]))
    add("fiscal", "timestamps_withheld", "TemporalConcentration", float("nan"), float("nan"),
        "uncomputable (dimension not recorded)")
    st = zrow(ld_mech, ld_null)
    add("fiscal", "timestamps_present", "LeadingDigit(amount)", st["mean_mech"], st["z"],
        R.verdict_from(st["z"], st["sc"]))

    # ---- 3B: threshold +/- known T ----
    TRUE_T = 25_000
    cfgs = [("T=25000_known", 25_000), ("T=10000_wrong", 10_000),
            ("T=50000_wrong", 50_000), ("T=24000_wrong_inrange", 24_000)]
    data = [G.gen_threshold(N, s, threshold=TRUE_T, bandwidth=0.10) for s in SEEDS]
    nulls = [magnitude_matched_null(data[i], np.random.default_rng(s + OFF)) for i, s in enumerate(SEEDS)]
    for label, T in cfgs:
        m = [P.threshold_density_ratio(data[i], T) for i in range(len(SEEDS))]
        nl = [P.threshold_density_ratio(nulls[i], T) for i in range(len(SEEDS))]
        st = zrow(m, nl)
        v = R.verdict_from(st["z"], st["sc"]) if np.isfinite(st["z"]) else "uncomputable (no data at T)"
        add("threshold", label, "ThresholdDensity", st["mean_mech"], st["z"], v)
    # no threshold at all
    add("threshold", "T=None", "ThresholdDensity", float("nan"), float("nan"),
        "uncomputable (threshold unknown)")

    # ---- 3C: repeated +/- ordering ----
    a_ord, a_shuf, r2_ord, r2_shuf = [], [], [], []
    for s in SEEDS:
        vals = G.gen_repeated(N, s)
        ordered = np.sort(vals)                       # identical values consecutive
        shuffled = vals.copy()
        np.random.default_rng(s).shuffle(shuffled)
        co = P.compute_cbad(ordered); cs = P.compute_cbad(shuffled)
        a_ord.append(co["cbad_a"]); a_shuf.append(cs["cbad_a"])
        r2_ord.append(co["cbad_r2"]); r2_shuf.append(cs["cbad_r2"])
    st = zrow(a_ord, a_shuf)   # ordered vs shuffled (lower a = more compressible)
    add("repeated", "ordering_present", "cbad_a", float(np.nanmean(a_ord)), st["z"],
        "+ (compressible: a lower than shuffled)" if np.nanmean(a_ord) < np.nanmean(a_shuf) else "0")
    add("repeated", "ordering_withheld(shuffled)", "cbad_a", float(np.nanmean(a_shuf)),
        float("nan"), "baseline (higher a)")
    add("repeated", "ordering_present", "cbad_r2", float(np.nanmean(r2_ord)), float("nan"), "info")
    add("repeated", "ordering_withheld(shuffled)", "cbad_r2", float(np.nanmean(r2_shuf)),
        float("nan"), "info")

    with open(os.path.join(RESULTS, "wp7_missing_dimension.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mechanism", "dimension_present", "projection",
                                           "value", "z_score", "verdict"])
        w.writeheader(); w.writerows(rows)

    print("MISSING-DIMENSION RESULTS:")
    for r in rows:
        print(f"  {r['mechanism']:10} {r['dimension_present']:30} {r['projection']:22} "
              f"val={r['value']:>10} z={r['z_score']:>8} -> {r['verdict']}")

    # consistency checks
    def get(dim, proj):
        return next(r for r in rows if r["dimension_present"] == dim and r["projection"] == proj)
    ok = (get("T=25000_known", "ThresholdDensity")["verdict"] == "+"
          and get("timestamps_present", "TemporalConcentration")["verdict"] == "+"
          and float(np.nanmean(a_ord)) < float(np.nanmean(a_shuf)))
    print("\nIdentifiability requires the recorded dimension:",
          "CONFIRMED" if ok else "CHECK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
