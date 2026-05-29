"""Full run + audit artifacts + pass/fail checklist (Work Package 3).

Runs Steps 4, 5a, 5b, 6 at N=100,000 / 20 seeds (the same orchestration as
run_experiments.main()), then writes the five additional audit artifacts and
evaluates the post-run checklist.

Usage:  python src/full_run.py
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generators as G          # noqa: E402
import projections as P          # noqa: E402
import baselines                 # noqa: E402
import run_experiments as R      # noqa: E402
from projections import NAN      # noqa: E402

N = 100_000
SEEDS = list(range(20))
RESULTS = R.RESULTS
REDUCED = {k: R.COLUMN_METRICS[k] for k in ("DecimalResidue", "cp_spectrum", "L_profile")}

_checks = []


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _write(path, fields, rows):
    with open(os.path.join(RESULTS, path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def check(name, ok, detail=""):
    _checks.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #
# Additional artifacts (6-10)
# --------------------------------------------------------------------------- #
def write_null_row_audit(resolved):
    fields = ["projection", "z_score", "effect_size_d", "verdict",
              "verdict_material", "selected_submetric", "passes_bonferroni"]
    rows = [{k: r[k] for k in fields} for r in resolved if r["mechanism"] == "null"]
    _write("null_row_audit.csv", fields, rows)


def write_prime_fractions(collected):
    rows = []
    for name in R.MECH_ORDER:
        pf = np.array([d.get("prime_fraction", NAN) for d in collected[name]["mech"]], float)
        pf = pf[np.isfinite(pf)]
        rows.append({"mechanism": name,
                     "prime_fraction_mean": round(float(pf.mean()), 6) if pf.size else "nan",
                     "prime_fraction_std": round(float(pf.std(ddof=1)), 6) if pf.size > 1 else 0.0})
    _write("prime_fractions.csv", ["mechanism", "prime_fraction_mean", "prime_fraction_std"], rows)


def write_magnitude_diagnostic(n, seeds=(0, 1, 2)):
    rows = []
    for name in R.MECH_ORDER:
        params = R.MECHANISMS[name][0]
        mm, nm = [], []
        for s in seeds:
            v, _ = R._generate(name, n, s, params)
            v = v[v > 1]
            mm.append(float(np.mean(np.log10(v))))
            nl = baselines.magnitude_matched_null(v, np.random.default_rng(s + R.NULL_SEED_OFFSET))
            nl = nl[nl > 1]
            nm.append(float(np.mean(np.log10(nl))))
        a, b = float(np.mean(mm)), float(np.mean(nm))
        rows.append({"mechanism": name, "mean_log10_mech": round(a, 4),
                     "mean_log10_null": round(b, 4), "abs_diff": round(abs(a - b), 4),
                     "flag": abs(a - b) > 0.1})
    _write("magnitude_match_diagnostic.csv",
           ["mechanism", "mean_log10_mech", "mean_log10_null", "abs_diff", "flag"], rows)
    return rows


def write_fallback_rate(n):
    lim = P.MAX_SIEVE
    rows = []

    def count(values):
        return int(values.size), int(np.sum(values >= lim))

    # Step 4 (N=n, 20 seeds) and Step 5b (N=n, 5 seeds) share the generators.
    for step, seeds in (("4", SEEDS), ("5b", list(range(5)))):
        for name in R.MECH_ORDER:
            params = R.MECHANISMS[name][0]
            tot = fb = 0
            for s in seeds:
                v, _ = R._generate(name, n, s, params)
                t, f = count(v); tot += t; fb += f
            rows.append({"step": step, "mechanism": name, "n_values": tot,
                         "n_fallback": fb, "fraction": round(fb / tot, 8)})
    # Step 5a (N in {1k,10k,100k}, 5 seeds), aggregated per mechanism.
    for name in R.MECH_ORDER:
        params = R.MECHANISMS[name][0]
        tot = fb = 0
        for nn in (1_000, 10_000, 100_000):
            for s in range(5):
                v, _ = R._generate(name, nn, s, params)
                t, f = count(v); tot += t; fb += f
        rows.append({"step": "5a", "mechanism": name, "n_values": tot,
                     "n_fallback": fb, "fraction": round(fb / tot, 8)})
    # Step 6 (aggregation), per base x m (10 seeds) -- m=50 reported as its own row.
    for base in ("round", "divisibility"):
        for m in (1, 2, 5, 10, 20, 50):
            tot = fb = 0
            for s in range(10):
                v = R._agg_values(base, n, s, m)
                t, f = count(v); tot += t; fb += f
            rows.append({"step": "6", "mechanism": f"{base}_m{m}", "n_values": tot,
                         "n_fallback": fb, "fraction": round(fb / tot, 8)})
    _write("factorization_fallback_rate.csv",
           ["step", "mechanism", "n_values", "n_fallback", "fraction"], rows)


def write_submetric_summary(collected):
    rows = []
    for name in R.MECH_ORDER:
        md, nd = collected[name]["mech"], collected[name]["null"]
        for col, mets in REDUCED.items():
            zs = {}
            for mt in mets:
                st = R.cell_stats(np.array([d.get(mt, NAN) for d in md], float),
                                  np.array([d.get(mt, NAN) for d in nd], float))
                zs[mt] = st["z"]
            valid = {k: v for k, v in zs.items() if not _isnan(v)}
            sel = max(valid, key=lambda k: abs(valid[k])) if valid else "n/a"
            selz = zs[sel] if sel != "n/a" else NAN
            allz = ";".join(f"{k}={'nan' if _isnan(zs[k]) else round(zs[k], 3)}" for k in mets)
            rows.append({"mechanism": name, "column": col, "selected_submetric": sel,
                         "selected_z": "nan" if _isnan(selz) else round(selz, 3),
                         "all_submetric_z": allz})
    _write("submetric_summary.csv",
           ["mechanism", "column", "selected_submetric", "selected_z", "all_submetric_z"], rows)


# --------------------------------------------------------------------------- #
# Checklist
# --------------------------------------------------------------------------- #
def run_checklist(resolved, mag_rows):
    res = {(r["mechanism"], r["projection"]): r for r in resolved}
    agg = list(csv.DictReader(open(os.path.join(RESULTS, "aggregation_decay.csv"))))
    aggd = {(r["base_mechanism"], int(_f(r["m"])), r["projection"]): r for r in agg}

    def V(m, p):
        return res.get((m, p), {}).get("verdict", "MISSING")

    def VM(m, p):
        return res.get((m, p), {}).get("verdict_material", "MISSING")

    def SUB(m, p):
        return res.get((m, p), {}).get("selected_submetric", "MISSING")

    print("\nPOST-RUN CHECKLIST (1-7 are hard gates)")
    # 1
    nm = [c for c in R.MATRIX_COLUMNS if VM("null", c) == "+"]
    check("1. null row: no material positives", not nm, "none" if not nm else str(nm))
    # 2
    agg_round_ok = (aggd.get(("round", 50, "CentsEntropy"), {}).get("verdict") == "+"
                    and aggd.get(("round", 50, "cp_spectrum"), {}).get("verdict") == "+")
    check("2. ROUND: Cents+/Term+/cp+ and aggregated m=50 Cents+/cp+",
          V("round", "CentsEntropy") == "+" and V("round", "TerminalDigit") == "+"
          and V("round", "cp_spectrum") == "+" and agg_round_ok,
          f"cp submetric={SUB('round','cp_spectrum')}; m50 Cents="
          f"{aggd.get(('round',50,'CentsEntropy'),{}).get('verdict')} "
          f"cp={aggd.get(('round',50,'cp_spectrum'),{}).get('verdict')}")
    # 3
    check("3. QUARTER cp submetric in {c5,c5sq}", SUB("quarter", "cp_spectrum") in ("c5", "c5sq"),
          f"submetric={SUB('quarter','cp_spectrum')}")
    # 4
    lp1 = aggd.get(("divisibility", 1, "L_profile"), {})
    lp50 = aggd.get(("divisibility", 50, "L_profile"), {})
    attenuates = (lp1 and lp50 and abs(_f(lp1["z_score"])) > abs(_f(lp50["z_score"])))
    check("4. DIVISIBILITY cp=c7; L_profile + at m=1; attenuates to m=50",
          SUB("divisibility", "cp_spectrum") == "c7" and lp1.get("verdict") == "+" and attenuates,
          f"cp={SUB('divisibility','cp_spectrum')}; L_profile |z| m1={abs(_f(lp1.get('z_score'))):.1f} "
          f"-> m50={abs(_f(lp50.get('z_score'))):.1f}")
    # 5
    check("5. temporal_concat CBAD_R2 = + (material)",
          V("temporal_concat", "CBAD_R2") == "+" and VM("temporal_concat", "CBAD_R2") == "+",
          f"verdict={V('temporal_concat','CBAD_R2')} material={VM('temporal_concat','CBAD_R2')}")
    # 6
    check("6. temporal_concat_arith CBAD_R2 in {0,?}",
          V("temporal_concat_arith", "CBAD_R2") in ("0", "?"),
          f"verdict={V('temporal_concat_arith','CBAD_R2')}")
    # 7
    exo_bad = []
    for m in R.MECH_ORDER:
        if m != "fiscal" and V(m, "TemporalConcentration") != "?":
            exo_bad.append((m, "TemporalConcentration", V(m, "TemporalConcentration")))
        if m != "threshold" and V(m, "ThresholdDensity") != "?":
            exo_bad.append((m, "ThresholdDensity", V(m, "ThresholdDensity")))
    check("7. exogenous cells remain ?", not exo_bad, "all ?" if not exo_bad else str(exo_bad[:6]))
    # 8
    pf = {r["mechanism"]: _f(r["prime_fraction_mean"])
          for r in csv.DictReader(open(os.path.join(RESULTS, "prime_fractions.csv")))}
    zero_mechs = ["divisibility", "product", "fixed_factor", "sum_of_rounded", "sum_of_divisibility"]
    check("8. prime fractions plausible",
          0.05 < pf.get("null", 0) < 0.10 and all(pf.get(m, 1) < 1e-6 for m in zero_mechs),
          f"null={pf.get('null'):.4f}, zero-mechs max={max(pf.get(m,0) for m in zero_mechs):.2e}")
    # 9
    flagged = [r["mechanism"] for r in mag_rows if r["flag"]]
    others_ok = all(not r["flag"] for r in mag_rows if r["mechanism"] != "product")
    check("9. magnitude match: only product flagged",
          ("product" in flagged) and others_ok, f"flagged={flagged}")
    # 10
    preserved = (aggd.get(("divisibility", 50, "cp_spectrum"), {}).get("verdict") == "+"
                 or aggd.get(("round", 50, "CentsEntropy"), {}).get("verdict") == "+")
    tm1 = abs(_f(aggd.get(("divisibility", 1, "TailMass"), {}).get("z_score")))
    tm50 = abs(_f(aggd.get(("divisibility", 50, "TailMass"), {}).get("z_score")))
    attenuated = (tm1 > tm50) or attenuates
    check("10. Prop 6 pattern (>=1 preserved & >=1 attenuated)", preserved and attenuated,
          f"preserved={preserved}; TailMass |z| m1={tm1:.1f}->m50={tm50:.1f}")

    hard_fail = [n for n, ok in _checks[:7] if not ok]
    return hard_fail


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    print("Building SPF sieve cache ...", flush=True)
    P.ensure_spf_cache(R.SPF_CACHE)
    P.set_spf_cache(R.SPF_CACHE)
    with ProcessPoolExecutor(max_workers=R.N_WORKERS,
                             initializer=R._init_worker, initargs=(R.SPF_CACHE,)) as ex:
        print("Step 4 ...", flush=True)
        out = R.run_main(ex, seeds=SEEDS, n=N)                 # outputs 1,2
        print("Step 5a ...", flush=True)
        R.run_sample_size(ex, out["resolved"])                  # output 3
    print("Step 5b ...", flush=True)
    R.run_magnitude()                                           # output 4
    print("Step 6 ...", flush=True)
    R.run_aggregation()                                         # output 5
    elapsed = time.time() - t0
    print(f"\nCore steps done in {elapsed:.1f}s. Writing audit artifacts ...", flush=True)

    write_null_row_audit(out["resolved"])                       # 6
    write_prime_fractions(out["collected"])                     # 7
    mag_rows = write_magnitude_diagnostic(N)                    # 8
    write_fallback_rate(N)                                       # 9
    write_submetric_summary(out["collected"])                   # 10

    hard_fail = run_checklist(out["resolved"], mag_rows)
    total = time.time() - t0
    print(f"\nTotal wall-clock: {total:.1f}s")

    print("\nCLEAN MATRIX:")
    with open(os.path.join(RESULTS, "signature_matrix_clean.csv")) as fh:
        for line in fh:
            print("  " + line.rstrip())

    if hard_fail:
        print("\n*** HARD-GATE FAILURE(S):", hard_fail, "-- stopping for diagnosis.")
        return 1
    print("\nAll hard gates (1-7) PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
