"""Shared helpers for Work Package 7 validation-closure experiments.

Reuses the existing engine (generators, projections, baselines, run_experiments).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
from projections import compute_all_projections   # noqa: E402

RESULTS = R.RESULTS
SPF_CACHE = R.SPF_CACHE

FEATURE_METRICS = [
    "leading_digit_mad", "terminal_digit_chisq", "cents_entropy",
    "mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv", "vocab_gini",
    "threshold_density", "temporal_concentration",
    "c2", "c3", "c5", "c7", "c10", "c5sq", "L1", "L2", "L3", "L4", "tail_mass",
    "cbad_a", "cbad_c", "cbad_r2", "prime_fraction",
]

FAMILIES = {
    "digit_decimal": ["leading_digit_mad", "terminal_digit_chisq", "cents_entropy",
                      "mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv"],
    "vocab": ["vocab_gini"],
    "threshold_temporal": ["threshold_density", "temporal_concentration"],
    "msa": ["c2", "c3", "c5", "c7", "c10", "c5sq", "L1", "L2", "L3", "L4", "tail_mass"],
    "cbad": ["cbad_a", "cbad_c", "cbad_r2"],
}


def feat_worker(task):
    """(mechanism, seed, n) -> (mechanism, seed, feature dict). Includes raw values
    optionally for ML comparators."""
    name, seed, n, want_values = task
    params, threshold = R.MECHANISMS[name]
    res = R.GEN_FUNCS[name](n=n, seed=seed, **params)
    months = res[1] if isinstance(res, tuple) else None
    values = res[0] if isinstance(res, tuple) else res
    d = compute_all_projections(values, ordered=True, months=months, threshold=threshold)
    feat = {k: float(d.get(k, np.nan)) for k in FEATURE_METRICS}
    raw = None
    if want_values:
        # summary stats of the raw magnitude distribution (ML comparators see "the data")
        v = values[values > 1].astype(np.float64)
        lg = np.log10(v)
        raw = [float(np.mean(lg)), float(np.std(lg)),
               float(np.percentile(lg, 25)), float(np.percentile(lg, 50)),
               float(np.percentile(lg, 75)),
               float(np.mean(v % 100 == 0)), float(np.mean(v % 1000 == 0)),
               float(np.mean(v % 7 == 0))]
    return name, seed, feat, raw


def build_feature_table(n, seeds, want_values=False):
    """Returns (mech_order_labels, X_dicts, raw_list). One row per (mechanism, seed)."""
    P.ensure_spf_cache(SPF_CACHE)
    P.set_spf_cache(SPF_CACHE)
    tasks = [(name, s, n, want_values) for name in R.MECH_ORDER for s in seeds]
    labels, feats, raws = [], [], []
    with ProcessPoolExecutor(max_workers=R.N_WORKERS, initializer=R._init_worker,
                             initargs=(SPF_CACHE,)) as ex:
        for name, seed, feat, raw in ex.map(feat_worker, tasks):
            labels.append(name); feats.append(feat); raws.append(raw)
    return labels, feats, raws


def matrix(feats, metrics):
    X = np.array([[row[m] for m in metrics] for row in feats], dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
