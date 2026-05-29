"""Magnitude-matched null baselines (Work Package 3).

Implements B_j(P_k) from the theory kernel: a null sample whose digit-length
(magnitude) distribution matches the mechanism, but is otherwise uniform. The
signature is the deviation of the mechanism projection from THIS baseline, not
from a global null. See CANONICAL_DEFINITIONS.md section 3.
"""
from __future__ import annotations

import numpy as np

from projections import compute_all_projections


def magnitude_matched_null(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform integers over the mechanism's ``[min, max]`` range, rejection-matched
    to its digit-length histogram.

    Sampling from a single range (as the generators do via ``_uniform_int``) keeps
    numpy's range-dependent terminal-digit sampling artifact identical between the
    mechanism and its null, so the null control has ~zero spurious signature.
    Per-decade sampling (the earlier approach) sat systematically below single-range
    sampling on terminal-digit chi-square and produced a false positive on the null
    row. See CANONICAL_DEFINITIONS.md section 3.
    """
    values = np.asarray(values)
    values = values[values > 1]
    if values.size == 0:
        return np.array([], dtype=np.int64)
    target = np.bincount(np.floor(np.log10(values)).astype(int) + 1)
    lo = max(2, int(values.min()))
    hi = max(lo + 1, int(values.max()))
    need = target.copy()
    out = []
    guard = 0
    while need.sum() > 0 and guard < 1000:
        guard += 1
        cand = rng.integers(lo, hi + 1, size=int(need.sum()) * 2 + 1000, dtype=np.int64)
        dl = np.floor(np.log10(cand)).astype(int) + 1
        for d in np.nonzero(need)[0]:
            sel = cand[dl == d]
            take = min(sel.size, int(need[d]))
            if take:
                out.append(sel[:take])
                need[d] -= take
    null = np.concatenate(out) if out else np.array([], dtype=np.int64)
    rng.shuffle(null)
    return null


def baseline_projections(values: np.ndarray, rng: np.random.Generator,
                         ordered: bool = False, months=None, threshold=None) -> dict:
    """Projections of the magnitude-matched null for ``values``.

    Temporal baseline uses uniform months (the null for calendar concentration);
    threshold baseline reuses the mechanism's threshold T.
    """
    null = magnitude_matched_null(values, rng)
    null_months = None
    if months is not None:
        null_months = rng.integers(1, 13, size=null.size, dtype=np.int64)
    return compute_all_projections(null, ordered=ordered,
                                   months=null_months, threshold=threshold)


def generate_parent_support_null(mechanism_values, seed, mag_range=(1, 6)) -> np.ndarray:
    """Construction-neutral (parent-support) null: uniform integers over the full
    ``mag_range``, NOT matched to the mechanism's digit-length histogram. Asks
    whether the signature would appear if the mechanism were absent. Use for
    mechanism discovery. (Proposition: baseline absorption.)"""
    rng = np.random.default_rng(seed)
    n = int(np.asarray(mechanism_values).size)
    lo = max(2, int(10 ** mag_range[0]))
    hi = max(lo + 1, int(10 ** mag_range[1]))
    return rng.integers(lo, hi + 1, size=n, dtype=np.int64)


def generate_threshold_aware_null(mechanism_values, seed, threshold,
                                  bandwidth=0.10) -> np.ndarray:
    """Mechanism-aware null: uniform integers over the mechanism's observed support
    ``[T(1-bandwidth), T(1+bandwidth)]``, matched to the mechanism's observed
    below/above-threshold split. Conditions on the threshold structure and asks
    whether observations are unusual WITHIN the mechanism. Use for operational
    triage after identification, not for discovery."""
    rng = np.random.default_rng(seed)
    mv = np.asarray(mechanism_values)
    n = int(mv.size)
    frac_below = float(np.mean(mv < threshold)) if n else 0.7
    n_below = int(round(frac_below * n))
    n_above = n - n_below
    lo = threshold * (1.0 - bandwidth)
    hi = threshold * (1.0 + bandwidth)
    below = rng.uniform(lo, threshold, size=n_below)
    above = rng.uniform(threshold, hi, size=n_above)
    vals = np.concatenate([below, above])
    rng.shuffle(vals)
    return np.maximum(np.round(vals), 2).astype(np.int64)
