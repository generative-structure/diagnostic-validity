"""Shared helpers for Work Package 6 real-data demonstrations.

Uses the IDENTICAL projection battery (compute_all_projections) as the synthetic
engine. For a single observed dataset the signature is computed against a
distribution of magnitude-matched nulls (resampled n_null times):

    z = (Phi_mechanism - mean_null) / std_null

(the synthetic engine's cross-seed z does not apply because the real mechanism is
observed once). Verdict rule matches CANONICAL_DEFINITIONS.md: |z|>3 -> +,
|z|<1.5 -> 0, else ?.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from projections import compute_all_projections, modular_residue_tv
from baselines import magnitude_matched_null

# Standard metrics reported for real data (CBAD omitted: cross-sectional / unordered).
METRICS = [
    "leading_digit_mad", "terminal_digit_chisq", "cents_entropy",
    "mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv", "vocab_gini",
    "c2", "c3", "c5", "c7", "c10", "c5sq",
    "L1", "L2", "L3", "L4", "tail_mass", "prime_fraction",
]


def _verdict(z):
    if z is None or not np.isfinite(z):
        return "?"
    if abs(z) > 3:
        return "+"
    if abs(z) < 1.5:
        return "0"
    return "?"


def real_signatures(values, n_null=20, threshold=None, ordered=False,
                    extra_mods=(), null_seed=1_000_000):
    """Return {metric: dict(value, null_mean, null_std, z, verdict)} for one dataset."""
    values = np.asarray(values)
    values = values[values > 1]
    metrics = list(METRICS) + [f"mod{m}_tv" for m in extra_mods]
    mech = compute_all_projections(values, ordered=ordered, threshold=threshold)
    for m in extra_mods:
        mech[f"mod{m}_tv"] = modular_residue_tv(values, m)
    if threshold is not None:
        metrics = metrics + ["threshold_density"]

    null_acc = {k: [] for k in metrics}
    for b in range(n_null):
        nl = magnitude_matched_null(values, np.random.default_rng(null_seed + b))
        nd = compute_all_projections(nl, ordered=ordered, threshold=threshold)
        for m in extra_mods:
            nd[f"mod{m}_tv"] = modular_residue_tv(nl[nl > 1], m)
        for k in metrics:
            null_acc[k].append(nd.get(k, np.nan))

    out = {}
    for k in metrics:
        mv = mech.get(k, np.nan)
        arr = np.array(null_acc[k], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2 or not np.isfinite(mv):
            out[k] = dict(value=float(mv) if np.isfinite(mv) else np.nan,
                          null_mean=np.nan, null_std=np.nan, z=np.nan, verdict="?")
            continue
        mu, sd = float(arr.mean()), float(arr.std(ddof=1))
        if sd > 0:
            z = (mv - mu) / sd
        elif mv != mu:
            z = float("inf") if mv > mu else float("-inf")
        else:
            z = 0.0
        out[k] = dict(value=float(mv), null_mean=mu, null_std=sd,
                      z=float(z), verdict=_verdict(z))
    return out


def reduce_column(sig, submetrics):
    """Max-|z| reduction over a column's sub-metrics. Returns (submetric, z, verdict)."""
    cands = [(k, sig[k]["z"]) for k in submetrics
             if k in sig and sig[k]["z"] is not None and np.isfinite(sig[k]["z"])]
    if not cands:
        return ("n/a", np.nan, "?")
    k, z = max(cands, key=lambda t: abs(t[1]))
    return (k, z, _verdict(z))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def provenance_append(path, text):
    with open(path, "a") as f:
        f.write(text.rstrip() + "\n")
