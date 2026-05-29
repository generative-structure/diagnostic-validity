"""Pilot run + evaluation (N=10,000, seeds=0..4).

Runs Step 4 + Step 5b + Step 6 (m<=10), then reports the six pass/fail criteria
and the requested diagnostics. Re-runnable: ``python src/pilot_report.py``.
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

N = 10_000
SEEDS = list(range(5))
RESULTS = R.RESULTS

_crit = []


def crit(n, ok, detail=""):
    _crit.append((n, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f" -- {detail}" if detail else ""), flush=True)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_resolved():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "signature_matrix_resolved.csv"))))
    return {(r["mechanism"], r["projection"]): r for r in rows}, rows


def main():
    print("=" * 72)
    print("WP3 PILOT  (N=10,000, seeds=0..4, Steps 4 + 5b + 6[m<=10])")
    print("=" * 72)
    t0 = time.time()
    R.run_pilot(n=N, seeds=SEEDS, agg_ms=(1, 2, 5, 10))
    elapsed = time.time() - t0
    print(f"\nPilot wall-clock: {elapsed:.1f} s\n")

    res, rows = load_resolved()

    def V(mech, proj):
        return res.get((mech, proj), {}).get("verdict", "MISSING")

    def VM(mech, proj):
        return res.get((mech, proj), {}).get("verdict_material", "MISSING")

    def SUB(mech, proj):
        return res.get((mech, proj), {}).get("selected_submetric", "MISSING")

    print("SIX PASS/FAIL CRITERIA")
    # 1. Null row non-material.
    null_mat = [c for c in R.MATRIX_COLUMNS if VM("null", c) == "+"]
    crit("1. null row has no material positives", not null_mat,
         "none" if not null_mat else f"material+ cells: {null_mat}")

    # 2. ROUND behaves as ROUND.
    rvals = G.gen_round(N, 0)
    ce_round = P.cents_entropy(rvals)
    fr = P.factor_projections(rvals)
    td_chisq = P.terminal_digit_chisq(rvals)
    round_ok = (ce_round < 0.01
                and V("round", "TerminalDigit") == "+"
                and V("round", "CentsEntropy") == "+"
                and V("round", "DecimalResidue") == "+"
                and V("round", "cp_spectrum") == "+")
    crit("2. ROUND behaves as ROUND", round_ok,
         f"cents_entropy={ce_round:.4f} (~0), term_chisq={td_chisq:.0f}, c10={fr['c10']:.3f}; "
         f"verdicts Term={V('round','TerminalDigit')} Cents={V('round','CentsEntropy')} "
         f"Mod={V('round','DecimalResidue')} cp={V('round','cp_spectrum')}")

    # 3. QUARTER selects c5 / c5sq.
    q_sub = SUB("quarter", "cp_spectrum")
    crit("3. QUARTER cp_spectrum submetric in {c5,c5sq} + cents signal",
         q_sub in ("c5", "c5sq") and V("quarter", "CentsEntropy") == "+",
         f"cp submetric={q_sub}, CentsEntropy verdict={V('quarter','CentsEntropy')}")

    # 4. DIVISIBILITY selects c7; L_profile & TailMass +.
    d_sub = SUB("divisibility", "cp_spectrum")
    crit("4. DIVISIBILITY cp submetric=c7, L_profile & TailMass +",
         d_sub == "c7" and V("divisibility", "L_profile") == "+"
         and V("divisibility", "TailMass") == "+",
         f"cp submetric={d_sub}, L_profile={V('divisibility','L_profile')}, "
         f"TailMass={V('divisibility','TailMass')}")

    # 5. temporal_concat degrades CBAD R2.
    crit("5. temporal_concat CBAD_R2 verdict = +", V("temporal_concat", "CBAD_R2") == "+",
         f"verdict={V('temporal_concat','CBAD_R2')}, "
         f"z={_f(res.get(('temporal_concat','CBAD_R2'),{}).get('z_score')):.2f}")

    # 6. temporal_concat_arith does NOT degrade CBAD R2.
    arith_v = V("temporal_concat_arith", "CBAD_R2")
    crit("6. temporal_concat_arith CBAD_R2 verdict in {0,?}", arith_v in ("0", "?"),
         f"verdict={arith_v}, "
         f"z={_f(res.get(('temporal_concat_arith','CBAD_R2'),{}).get('z_score')):.2f}")

    # ------------------------------------------------------------------ #
    print("\nDIAGNOSTICS")

    # Mean log10(n): mechanism vs matched null.
    print("\n  Mean log10(value): mechanism vs matched null (flag diff > 0.1):")
    for name in R.MECH_ORDER:
        params = R.MECHANISMS[name][0]
        v, _ = R._generate(name, N, 0, params)
        v = v[v > 1]
        mech_m = float(np.mean(np.log10(v)))
        null = magnitude_matched_null(v, np.random.default_rng(7))
        null_m = float(np.mean(np.log10(null[null > 1])))
        d = abs(mech_m - null_m)
        flag = "  <-- FLAG" if d > 0.1 else ""
        print(f"    {name:22} mech={mech_m:.3f}  null={null_m:.3f}  diff={d:.3f}{flag}")

    # Null row |z| > 2 cells.
    print("\n  Null row cells with |z| > 2:")
    any_null = False
    for c in R.MATRIX_COLUMNS:
        r = res.get(("null", c), {})
        z = _f(r.get("z_score"))
        if not math.isnan(z) and abs(z) > 2:
            any_null = True
            k = int(_f(r.get("num_submetrics")))
            why = (f"max-|z| over {k} sub-metrics (selection inflation)" if k > 1
                   else "seed variance at 5 seeds")
            print(f"    {c:22} z={z:.2f} sub={r.get('selected_submetric')} "
                  f"material={r.get('material')} verdict_material={r.get('verdict_material')} -- {why}")
    if not any_null:
        print("    (none)")

    # Prime fraction table.
    print("\n  Prime fraction by mechanism:")
    for name in R.MECH_ORDER:
        r = res.get((name, "prime_fraction"), {})
        print(f"    {name:22} {_f(r.get('mean_mechanism')):.4f}")

    # Aggregation preview.
    print("\n  Aggregation decay (Step 6), m in [1,2,5,10]:")
    agg = list(csv.DictReader(open(os.path.join(RESULTS, "aggregation_decay.csv"))))
    def show(base, proj, label):
        cells = {int(_f(r["m"])): r for r in agg
                 if r["base_mechanism"] == base and r["projection"] == proj}
        seq = "  ".join(f"m={m}:{cells[m]['verdict']}(z={_f(cells[m]['z_score']):.1f})"
                        for m in (1, 2, 5, 10) if m in cells)
        print(f"    {label:42} {seq}")
    show("divisibility", "DecimalResidue", "divisibility DecimalResidue (preserved?)")
    show("divisibility", "cp_spectrum", "divisibility cp_spectrum (c7 preserved?)")
    show("divisibility", "L_profile", "divisibility L_profile (attenuate?)")
    show("divisibility", "TailMass", "divisibility TailMass (attenuate?)")
    show("round", "CentsEntropy", "round CentsEntropy (preserved?)")
    show("round", "cp_spectrum", "round cp_spectrum (preserved?)")

    n_fail = sum(1 for _, ok, _ in _crit if not ok)
    print("\n" + "=" * 72)
    print(f"PILOT SUMMARY: {len(_crit) - n_fail}/{len(_crit)} criteria passed, "
          f"{n_fail} failed.  ({elapsed:.1f}s)")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
