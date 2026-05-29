"""Smoke test for the WP3 engine (Step 4 only, N=1000, seeds=[0,1]).

Runs the main experiment at small scale, then performs the 10 checks from the
WP3 corrections prompt. Re-runnable: ``python src/smoke_test.py``.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generators as G          # noqa: E402
import projections as P          # noqa: E402
from baselines import magnitude_matched_null   # noqa: E402
import run_experiments as R      # noqa: E402

N = 1_000
SEED = 0
RESULTS = R.RESULTS
RESOLVED = os.path.join(RESULTS, "signature_matrix_resolved.csv")
CLEAN = os.path.join(RESULTS, "signature_matrix_clean.csv")

_checks = []


def check(name, ok, detail=""):
    _checks.append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def load_resolved():
    with open(RESOLVED) as f:
        return list(csv.DictReader(f)), f


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    print("=" * 70)
    print("WP3 SMOKE RUN  (Step 4 only, N=1000, seeds=[0,1])")
    print("=" * 70)

    t0 = time.time()
    R.run_step4(seeds=[0, 1], n=N)
    elapsed = time.time() - t0
    print(f"\nStep 4 wall-clock: {elapsed:.1f} s\n")

    with open(RESOLVED) as f:
        rows = list(csv.DictReader(f))
        header = rows[0].keys() if rows else []

    print("Running checks:")

    # 1. No NaNs in non-exogenous matrix cells.
    def exogenous(mech, proj):
        return ((proj == "TemporalConcentration" and mech != "fiscal") or
                (proj == "ThresholdDensity" and mech != "threshold"))
    bad = [(r["mechanism"], r["projection"]) for r in rows
           if r["projection"] in R.MATRIX_COLUMNS
           and not exogenous(r["mechanism"], r["projection"])
           and math.isnan(_f(r["z_score"]))]
    check("1. no NaNs in non-exogenous cells", len(bad) == 0,
          "all finite" if not bad else f"NaN cells: {bad[:8]}")

    # 2. divisibility-by-7 + c3 ~ 0, c7 elevated.
    dv = G.gen_divisibility(N, SEED, divisor=7)
    all7 = bool(np.all(dv % 7 == 0))
    fp = P.factor_projections(dv)
    c3, c7 = fp["c3"], fp["c7"]
    check("2. gen_divisibility all %7==0", all7)
    check("2. divisibility c7 elevated vs c3", c7 > 0.15 and c7 > 2 * c3,
          f"c3={c3:.4f}, c7={c7:.4f}")

    # 3. prime fractions.
    pf_null = P.factor_projections(G.gen_null(N, SEED))["prime_fraction"]
    pf_div = P.factor_projections(dv)["prime_fraction"]
    pf_fix = P.factor_projections(G.gen_fixed_factor(N, SEED, bit_length=12))["prime_fraction"]
    check("3. null prime_fraction ~ 1/ln(N)", 0.02 < pf_null < 0.18, f"{pf_null:.4f}")
    check("3. divisibility prime_fraction ~ 0", pf_div < 0.01, f"{pf_div:.4f}")
    check("3. fixed_factor prime_fraction == 0", pf_fix == 0.0, f"{pf_fix:.4f}")

    # 4. fixed_factor L-profile.
    fpf = P.factor_projections(G.gen_fixed_factor(N, SEED, bit_length=12))
    L1, L2, tl = fpf["L1"], fpf["L2"], fpf["tail_mass"]
    check("4. fixed_factor L1~0.5, L2~0.5, tail~0",
          0.45 < L1 < 0.60 and 0.40 < L2 < 0.55 and tl < 0.05,
          f"L1={L1:.3f}, L2={L2:.3f}, tail={tl:.3f}")

    # 5. cents entropy: round low vs null near-max.
    ce_round = P.cents_entropy(G.gen_round(N, SEED))
    ce_null = P.cents_entropy(G.gen_null(N, SEED))
    check("5. cents entropy: round << null(~6.6)",
          ce_round < 3.5 and ce_null > 6.0,
          f"round={ce_round:.2f} bits, null={ce_null:.2f} bits "
          f"(round not ~0 because x10 multiples retain 10 cent values)")

    # 6. vocab Gini: repeated high, null low.
    g_rep = P.vocab_gini(G.gen_repeated(N, SEED))
    g_null = P.vocab_gini(G.gen_null(N, SEED))
    check("6. vocab Gini: repeated >> null", g_rep > 0.25 and g_null < 0.10,
          f"repeated={g_rep:.3f}, null={g_null:.3f}")

    # 7. CBAD correctness: responds to compressibility (repeated vs null).
    cb_rep = P.compute_cbad(G.gen_repeated(N, SEED))
    cb_nl = P.compute_cbad(G.gen_null(N, SEED))
    check("7. CBAD detects compressibility (repeated vs null)",
          cb_rep["cbad_r2"] > 0.9 and cb_rep["cbad_a"] < cb_nl["cbad_a"],
          f"repeated R2={cb_rep['cbad_r2']:.3f} a={cb_rep['cbad_a']:.3f}; "
          f"null a={cb_nl['cbad_a']:.3f}")
    # 7a (informational, not scored): temporal_concat regime change.
    r2_tc = P.compute_cbad(G.gen_temporal_concat(N, SEED))["cbad_r2"]
    print(f"  [INFO] 7a. temporal_concat R2={r2_tc:.4f} vs null R2={cb_nl['cbad_r2']:.4f} "
          f"-- null->div7 is NOT compression-separable (raw R(N) == null even at "
          f"N=100k); temporal_concat CBAD_R2 will likely resolve to 0, not '+'.")

    # 8. resolved CSV has all required columns.
    missing = [c for c in R.RESOLVED_FIELDS if c not in header]
    check("8. resolved CSV has all audit columns", not missing,
          "all present" if not missing else f"missing: {missing}")

    # 9. c10 and c5sq present; c10 == c2 + c5.
    f9 = P.factor_projections(G.gen_round(N, SEED))
    have = ("c10" in f9 and "c5sq" in f9 and "c5sq_composite" in f9)
    ident = abs(f9["c10"] - (f9["c2"] + f9["c5"])) < 1e-9
    check("9. c10 & c5sq present, c10==c2+c5", have and ident,
          f"c10={f9['c10']:.4f}, c2+c5={f9['c2'] + f9['c5']:.4f}")

    # 10. magnitude-matched null digit distribution matches mechanism.
    vals = G.gen_round(N, SEED)
    null = magnitude_matched_null(vals, np.random.default_rng(123))
    def dighist(a):
        a = a[a > 1]
        return np.bincount(np.floor(np.log10(a)).astype(int) + 1, minlength=10)
    diff = int(np.abs(dighist(vals) - dighist(null)).max())
    check("10. null digit-length dist matches mechanism", diff == 0,
          f"max stratum count diff = {diff}")

    # ---- Diagnostics: max integer / sieve fallback ----
    maxima = {}
    for name in R.MECH_ORDER:
        params = R.MECHANISMS[name][0]
        v, _ = R._generate(name, N, SEED, params)
        maxima[name] = int(np.max(v))
    overall_max = max(maxima.values())
    fallback = overall_max >= P.MAX_SIEVE
    argmax = max(maxima, key=maxima.get)
    print(f"\nMax integer encountered: {overall_max:,} (from '{argmax}'); "
          f"sieve limit={P.MAX_SIEVE:,}; fallback hit: {fallback}")

    # ---- Preview rows with audit columns ----
    print("\nPreview (resolved CSV, cp_spectrum rows):")
    cols = ["mechanism", "projection", "z_score", "selected_submetric",
            "num_submetrics", "passes_bonferroni", "effect_size_d",
            "material", "verdict", "verdict_material"]
    print("  " + " | ".join(cols))
    shown = 0
    for r in rows:
        if r["projection"] == "cp_spectrum" and r["mechanism"] in (
                "round", "quarter", "divisibility", "fixed_factor"):
            print("  " + " | ".join(str(r[c]) for c in cols))
            shown += 1
        if shown >= 4:
            break

    # ---- Summary ----
    n_fail = sum(1 for _, ok, _ in _checks if not ok)
    print("\n" + "=" * 70)
    print(f"SMOKE SUMMARY: {len(_checks) - n_fail}/{len(_checks)} checks passed, "
          f"{n_fail} failed.  ({elapsed:.1f}s)")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
