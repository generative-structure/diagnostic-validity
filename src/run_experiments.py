"""Main experiment runner (Work Package 3).

Generates each mechanism, computes projections vs. a magnitude-matched null over
many seeds, applies the verdict rule from CANONICAL_DEFINITIONS.md, and writes:

  results/signature_matrix_resolved.csv   per-cell stats + audit/materiality columns
  results/signature_matrix_clean.csv      wide matrix of +/0/? verdicts
  results/sensitivity_sample_size.csv      Step 5a: borderline cells vs N
  results/sensitivity_magnitude.csv        Step 5b: signatures per digit stratum
  results/aggregation_decay.csv            Step 6: Prop-6 aggregation transformation

Usage:
  python src/run_experiments.py            full run (Steps 4, 5a, 5b, 6)
  python src/run_experiments.py smoke      Step 4 only, N=1000, seeds=[0,1]

Nothing runs on import.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

# Make sibling modules importable both on import and in spawned workers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generators as G                       # noqa: E402
import projections as P                       # noqa: E402
from baselines import baseline_projections   # noqa: E402
from projections import compute_all_projections, NAN  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEEDS = list(range(20))
N = 100_000
MAG_RANGE = (1, 6)
N_WORKERS = 13
NULL_SEED_OFFSET = 1_000_000
ALPHA = 0.002                                 # two-sided; |z|>3 ~ p<0.003

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SPF_CACHE = os.path.join(RESULTS, "_spf_cache.npy")

GEN_FUNCS = {
    "null": G.gen_null,
    "repeated": G.gen_repeated,
    "round": G.gen_round,
    "quarter": G.gen_quarter,
    "psychological": G.gen_psychological,
    "threshold": G.gen_threshold,
    "fiscal": G.gen_fiscal,
    "divisibility": G.gen_divisibility,
    "product": G.gen_product,
    "fixed_factor": G.gen_fixed_factor,
    "temporal_concat": G.gen_temporal_concat,
    "temporal_concat_arith": G.gen_temporal_concat_arith,
    "sum_of_rounded": G.gen_sum_of_rounded,
    "sum_of_divisibility": G.gen_sum_of_divisibility,
}

# name -> (params, threshold-for-projection or None)
MECHANISMS = {
    "null": ({}, None),
    "repeated": ({}, None),
    "round": ({}, None),
    "quarter": ({}, None),
    "psychological": ({}, None),
    "threshold": ({}, 10_000),
    "fiscal": ({}, None),
    "divisibility": ({"divisor": 7}, None),
    "product": ({}, None),
    "fixed_factor": ({"bit_length": 12}, None),
    "temporal_concat": ({}, None),
    "temporal_concat_arith": ({}, None),
    "sum_of_rounded": ({"m": 5}, None),
    "sum_of_divisibility": ({"m": 5, "divisor": 7}, None),
}
MECH_ORDER = list(MECHANISMS.keys())

# Signature-matrix columns -> engine metric(s). Multi-metric columns are reduced
# by max |z| (CANONICAL_DEFINITIONS.md section 0).
MATRIX_COLUMNS = [
    "LeadingDigit", "TerminalDigit", "CentsEntropy", "DecimalResidue",
    "VocabGini", "ThresholdDensity", "TemporalConcentration",
    "cp_spectrum", "L_profile", "TailMass", "CBAD_a", "CBAD_c", "CBAD_R2",
]
COLUMN_METRICS = {
    "LeadingDigit": ["leading_digit_mad"],
    "TerminalDigit": ["terminal_digit_chisq"],
    "CentsEntropy": ["cents_entropy"],
    "DecimalResidue": ["mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv"],
    "VocabGini": ["vocab_gini"],
    "ThresholdDensity": ["threshold_density"],
    "TemporalConcentration": ["temporal_concentration"],
    "cp_spectrum": ["c2", "c3", "c5", "c7", "c10", "c5sq"],
    "L_profile": ["L1", "L2", "L3", "L4"],
    "TailMass": ["tail_mass"],
    "CBAD_a": ["cbad_a"],
    "CBAD_c": ["cbad_c"],
    "CBAD_R2": ["cbad_r2"],
}
# Sensitivity columns: resolved CSV only, never affect verdicts / clean matrix.
SENSITIVITY_COLUMNS = {
    "cp_spectrum_composite": ["c2_composite", "c3_composite", "c5_composite",
                              "c7_composite", "c10_composite", "c5sq_composite"],
    "L_profile_composite": ["L1_composite", "L2_composite", "L3_composite", "L4_composite"],
    "TailMass_composite": ["tail_mass_composite"],
    "prime_fraction": ["prime_fraction"],
}

RESOLVED_FIELDS = [
    "mechanism", "projection", "mean_mechanism", "mean_null", "z_score",
    "sign_consistency", "verdict", "effect_size",
    "selected_submetric", "num_submetrics", "raw_z", "bonferroni_z_threshold",
    "passes_bonferroni", "effect_size_d", "material", "verdict_material",
]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def cell_stats(mech: np.ndarray, null: np.ndarray) -> dict:
    """z, sign consistency and Cohen's d from per-seed arrays (NaNs dropped)."""
    mech = np.asarray(mech, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    mv = mech[np.isfinite(mech)]
    nv = null[np.isfinite(null)]
    if mv.size < 2 or nv.size < 2:
        return {"mean_mech": NAN, "mean_null": NAN, "z": NAN, "sc": NAN, "d": NAN}
    m1, m2 = mv.mean(), nv.mean()
    s1, s2 = mv.var(ddof=1), nv.var(ddof=1)
    denom = math.sqrt(s1 / mv.size + s2 / nv.size)
    if denom > 0:
        z = (m1 - m2) / denom
    elif m1 != m2:
        z = math.inf if m1 > m2 else -math.inf
    else:
        z = 0.0
    pair = np.isfinite(mech) & np.isfinite(null)
    diffs = mech[pair] - null[pair]
    if diffs.size > 0 and np.mean(diffs) != 0:
        sigma = np.sign(np.mean(diffs))
        sc = float(np.mean(np.sign(diffs) == sigma))
    else:
        sc = NAN
    pooled = math.sqrt((s1 + s2) / 2.0)
    d = (m1 - m2) / pooled if pooled > 0 else (math.inf if m1 != m2 else 0.0)
    return {"mean_mech": float(m1), "mean_null": float(m2),
            "z": float(z), "sc": sc, "d": float(d)}


def verdict_from(z, sc) -> str:
    if _isnan(z):
        return "?"
    if abs(z) > 3 and not _isnan(sc) and sc > 0.8:
        return "+"
    if abs(z) < 1.5:
        return "0"
    return "?"


def _resolve_one(metrics: list, mech_dicts: list, null_dicts: list) -> dict:
    """Reduce a column's sub-metrics by max |z| and attach audit/materiality fields."""
    num = len(metrics)
    thr = float(norm.ppf(1.0 - ALPHA / (2.0 * num)))
    best, best_metric = None, None
    for mt in metrics:
        mech_arr = np.array([d.get(mt, NAN) for d in mech_dicts], dtype=np.float64)
        null_arr = np.array([d.get(mt, NAN) for d in null_dicts], dtype=np.float64)
        st = cell_stats(mech_arr, null_arr)
        if _isnan(st["z"]):
            continue
        if best is None or abs(st["z"]) > abs(best["z"]):
            best, best_metric = st, mt
    if best is None:
        return {"mean_mech": NAN, "mean_null": NAN, "z": NAN, "sc": NAN, "d": NAN,
                "verdict": "?", "selected_submetric": "n/a", "num_submetrics": num,
                "raw_z": NAN, "bonferroni_z_threshold": thr, "passes_bonferroni": False,
                "effect_size_d": NAN, "material": False, "verdict_material": "?",
                "note": "not resolvable by engine (exogenous dimension)"}
    raw_z = best["z"]
    if num == 1:
        passes = (not _isnan(raw_z)) and abs(raw_z) > 3
    else:
        passes = (not _isnan(raw_z)) and abs(raw_z) > thr
    v = verdict_from(best["z"], best["sc"])
    material = (not _isnan(best["d"])) and abs(best["d"]) > 0.2
    # verdict_material requires a material effect AND, for max-|z|-reduced columns,
    # survival of the multiple-comparison (Bonferroni) correction -- otherwise a
    # selection-inflated null cell can read as a material positive. The detection
    # `verdict` column itself is left unchanged.
    vm = "?" if (v == "+" and not (material and passes)) else v
    return {"mean_mech": best["mean_mech"], "mean_null": best["mean_null"],
            "z": best["z"], "sc": best["sc"], "d": best["d"], "verdict": v,
            "selected_submetric": best_metric, "num_submetrics": num,
            "raw_z": raw_z, "bonferroni_z_threshold": thr,
            "passes_bonferroni": bool(passes), "effect_size_d": best["d"],
            "material": bool(material), "verdict_material": vm, "note": ""}


def resolve_columns(mech_dicts: list, null_dicts: list, columns: dict) -> dict:
    return {col: _resolve_one(metrics, mech_dicts, null_dicts)
            for col, metrics in columns.items()}


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
def _init_worker(spf_path: str) -> None:
    P.set_spf_cache(spf_path)


def _generate(name: str, n: int, seed: int, params: dict):
    res = GEN_FUNCS[name](n=n, seed=seed, **params)
    if isinstance(res, tuple):
        return res[0], res[1]            # (values, months)
    return res, None


def run_one(task: tuple) -> tuple:
    """One (mechanism, seed): mechanism projections + magnitude-matched-null projections."""
    name, params, threshold, seed, n, ordered = task
    values, months = _generate(name, n, seed, params)
    mech = compute_all_projections(values, ordered=ordered, months=months, threshold=threshold)
    rng = np.random.default_rng(seed + NULL_SEED_OFFSET)
    null = baseline_projections(values, rng, ordered=ordered, months=months, threshold=threshold)
    return name, seed, mech, null


def _resolved_row(name: str, col: str, st: dict) -> dict:
    return {
        "mechanism": name, "projection": col,
        "mean_mechanism": st["mean_mech"], "mean_null": st["mean_null"],
        "z_score": st["z"], "sign_consistency": st["sc"],
        "verdict": st["verdict"], "effect_size": st["d"],
        "selected_submetric": st["selected_submetric"],
        "num_submetrics": st["num_submetrics"], "raw_z": st["raw_z"],
        "bonferroni_z_threshold": st["bonferroni_z_threshold"],
        "passes_bonferroni": st["passes_bonferroni"],
        "effect_size_d": st["effect_size_d"], "material": st["material"],
        "verdict_material": st["verdict_material"],
    }


# --------------------------------------------------------------------------- #
# Step 4: main matrix
# --------------------------------------------------------------------------- #
def run_main(executor, seeds=SEEDS, n=N) -> dict:
    tasks = [(name, MECHANISMS[name][0], MECHANISMS[name][1], seed, n, True)
             for name in MECH_ORDER for seed in seeds]
    collected = {name: {"mech": [None] * len(seeds), "null": [None] * len(seeds)}
                 for name in MECH_ORDER}
    seed_index = {s: i for i, s in enumerate(seeds)}
    for name, seed, mech, null in executor.map(run_one, tasks):
        i = seed_index[seed]
        collected[name]["mech"][i] = mech
        collected[name]["null"][i] = null

    resolved_rows = []
    clean = {name: {} for name in MECH_ORDER}
    for name in MECH_ORDER:
        mech_dicts = collected[name]["mech"]
        null_dicts = collected[name]["null"]
        matrix = resolve_columns(mech_dicts, null_dicts, COLUMN_METRICS)
        for col in MATRIX_COLUMNS:
            resolved_rows.append(_resolved_row(name, col, matrix[col]))
            clean[name][col] = matrix[col]["verdict"]
        sens = resolve_columns(mech_dicts, null_dicts, SENSITIVITY_COLUMNS)
        for col in SENSITIVITY_COLUMNS:
            resolved_rows.append(_resolved_row(name, col, sens[col]))

    _write_dicts(os.path.join(RESULTS, "signature_matrix_resolved.csv"),
                 RESOLVED_FIELDS, resolved_rows)
    with open(os.path.join(RESULTS, "signature_matrix_clean.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mechanism"] + MATRIX_COLUMNS)
        for name in MECH_ORDER:
            w.writerow([name] + [clean[name][c] for c in MATRIX_COLUMNS])

    return {"resolved": resolved_rows, "collected": collected}


def run_step4(seeds=SEEDS, n=N) -> dict:
    """Build the sieve cache, open the pool, run Step 4 only. Used by the smoke test."""
    os.makedirs(RESULTS, exist_ok=True)
    P.ensure_spf_cache(SPF_CACHE)
    P.set_spf_cache(SPF_CACHE)
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker,
                             initargs=(SPF_CACHE,)) as ex:
        return run_main(ex, seeds=seeds, n=n)


def run_pilot(n=10_000, seeds=None, agg_ms=(1, 2, 5, 10)) -> dict:
    """Pilot: Step 4 + Step 5b (magnitude) + Step 6 (aggregation, m<=10). Skips
    Step 5a (sample-size sensitivity)."""
    if seeds is None:
        seeds = list(range(5))
    os.makedirs(RESULTS, exist_ok=True)
    P.ensure_spf_cache(SPF_CACHE)
    P.set_spf_cache(SPF_CACHE)
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker,
                             initargs=(SPF_CACHE,)) as ex:
        print("Pilot Step 4: main signature matrix ...", flush=True)
        out = run_main(ex, seeds=seeds, n=n)
    print("Pilot Step 5b: magnitude stratification ...", flush=True)
    run_magnitude(n=n, sub_seeds=seeds)
    print("Pilot Step 6: aggregation decay (m<=10) ...", flush=True)
    run_aggregation(ms=agg_ms, n=n, sub_seeds=seeds)
    return out


# --------------------------------------------------------------------------- #
# Step 5a: sample-size sensitivity (borderline cells only)
# --------------------------------------------------------------------------- #
def run_sample_size(executor, resolved_rows: list) -> None:
    borderline = sorted({
        r["mechanism"] for r in resolved_rows
        if r["projection"] in MATRIX_COLUMNS and r["verdict"] == "?"
        and not _isnan(r["z_score"]) and 1.0 < abs(r["z_score"]) < 3.5
    })
    ns = [1_000, 10_000, 100_000]
    sub_seeds = list(range(5))
    rows = []
    for n_obs in ns:
        tasks = [(name, MECHANISMS[name][0], MECHANISMS[name][1], s, n_obs, True)
                 for name in borderline for s in sub_seeds]
        bucket = {name: {"mech": [], "null": []} for name in borderline}
        for name, seed, mech, null in executor.map(run_one, tasks):
            bucket[name]["mech"].append(mech)
            bucket[name]["null"].append(null)
        for name in borderline:
            cols = resolve_columns(bucket[name]["mech"], bucket[name]["null"], COLUMN_METRICS)
            for col in MATRIX_COLUMNS:
                base = next(r for r in resolved_rows
                            if r["mechanism"] == name and r["projection"] == col)
                if base["verdict"] != "?":
                    continue
                st = cols[col]
                rows.append({"mechanism": name, "projection": col, "N": n_obs,
                             "z_score": st["z"], "verdict": st["verdict"]})
    _write_dicts(os.path.join(RESULTS, "sensitivity_sample_size.csv"),
                 ["mechanism", "projection", "N", "z_score", "verdict"], rows)


# --------------------------------------------------------------------------- #
# Step 5b: magnitude stratification
# --------------------------------------------------------------------------- #
def _stratum_null(values, rng):
    lo = max(2, int(values.min()))
    hi = int(values.max())
    return rng.integers(lo, hi + 1, size=values.size, dtype=np.int64)


def run_magnitude(n=N, sub_seeds=None) -> None:
    if sub_seeds is None:
        sub_seeds = list(range(5))
    rows = []
    for name in MECH_ORDER:
        params, threshold = MECHANISMS[name]
        per_stratum = {}
        for s in sub_seeds:
            values, _ = _generate(name, n, s, params)
            values = values[values > 1]
            digits = np.floor(np.log10(values)).astype(int) + 1
            rng = np.random.default_rng(s + NULL_SEED_OFFSET)
            for d in np.unique(digits):
                mask = digits == d
                if int(mask.sum()) < 200:
                    continue
                vstr = values[mask]
                mdict = compute_all_projections(vstr, ordered=False, threshold=threshold)
                ndict = compute_all_projections(_stratum_null(vstr, rng),
                                                ordered=False, threshold=threshold)
                per_stratum.setdefault(int(d), {"mech": [], "null": []})
                per_stratum[int(d)]["mech"].append(mdict)
                per_stratum[int(d)]["null"].append(ndict)
        for d in sorted(per_stratum):
            cols = resolve_columns(per_stratum[d]["mech"], per_stratum[d]["null"], COLUMN_METRICS)
            for col in MATRIX_COLUMNS:
                if col.startswith("CBAD"):
                    continue
                st = cols[col]
                rows.append({"mechanism": name, "projection": col,
                             "digit_stratum": d, "z_score": st["z"],
                             "verdict": st["verdict"]})
    _write_dicts(os.path.join(RESULTS, "sensitivity_magnitude.csv"),
                 ["mechanism", "projection", "digit_stratum", "z_score", "verdict"], rows)


# --------------------------------------------------------------------------- #
# Step 6: aggregation decay (Proposition 6)
# --------------------------------------------------------------------------- #
def _agg_values(base: str, n: int, seed: int, m: int):
    if base == "round":
        vals = G.gen_round(n * m, seed, MAG_RANGE)
    elif base == "divisibility":
        vals = G.gen_divisibility(n * m, seed, MAG_RANGE, divisor=7)
    else:
        raise ValueError(base)
    if m == 1:
        return vals[:n]
    return vals.reshape(n, m).sum(axis=1).astype(np.int64)


def run_aggregation(ms=(1, 2, 5, 10, 20, 50), n=N, sub_seeds=None) -> None:
    bases = ["round", "divisibility"]
    if sub_seeds is None:
        sub_seeds = list(range(10))
    rows = []
    for base in bases:
        for m in ms:
            bucket = {"mech": [], "null": []}
            for s in sub_seeds:
                vals = _agg_values(base, n, s, m)
                rng = np.random.default_rng(s + NULL_SEED_OFFSET)
                bucket["mech"].append(compute_all_projections(vals, ordered=True))
                bucket["null"].append(baseline_projections(vals, rng, ordered=True))
            cols = resolve_columns(bucket["mech"], bucket["null"], COLUMN_METRICS)
            for col in MATRIX_COLUMNS:
                st = cols[col]
                rows.append({"base_mechanism": base, "m": m, "projection": col,
                             "z_score": st["z"], "verdict": st["verdict"]})
    _write_dicts(os.path.join(RESULTS, "aggregation_decay.csv"),
                 ["base_mechanism", "m", "projection", "z_score", "verdict"], rows)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _write_dicts(path: str, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    print("Building SPF sieve cache (once) ...", flush=True)
    P.ensure_spf_cache(SPF_CACHE)
    P.set_spf_cache(SPF_CACHE)
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker,
                             initargs=(SPF_CACHE,)) as ex:
        print("Step 4: main signature matrix ...", flush=True)
        main_out = run_main(ex)
        print("Step 5a: sample-size sensitivity ...", flush=True)
        run_sample_size(ex, main_out["resolved"])
    print("Step 5b: magnitude stratification ...", flush=True)
    run_magnitude()
    print("Step 6: aggregation decay ...", flush=True)
    run_aggregation()
    print("Done. Wrote results to", RESULTS, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        run_step4(seeds=[0, 1], n=1_000)
        print("Smoke Step 4 complete. Wrote results to", RESULTS, flush=True)
    elif mode == "pilot":
        run_pilot()
        print("Pilot complete. Wrote results to", RESULTS, flush=True)
    else:
        main()
