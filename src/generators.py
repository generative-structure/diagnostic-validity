"""Synthetic construction generators (Work Package 3).

Each generator returns a 1-D ``numpy.int64`` array of positive integers (> 1),
deterministic given ``seed``. ``gen_fiscal`` additionally returns month labels.

Monetary generators operate in INTEGER CENTS: ``mag_range = (low, high)`` bounds
the final recorded integer in [10**low, 10**high]. See CANONICAL_DEFINITIONS.md
(section 0, "Unit convention").

All randomness flows through ``numpy.random.Generator``.
"""
from __future__ import annotations

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _uniform_int(rng: np.random.Generator, low_exp: float, high_exp: float, n: int) -> np.ndarray:
    lo = max(2, int(10 ** low_exp))
    hi = max(lo + 1, int(10 ** high_exp))
    return rng.integers(lo, hi + 1, size=n, dtype=np.int64)


def gen_null(n: int = 100_000, seed: int = 0, mag_range=(1, 6)) -> np.ndarray:
    """Magnitude-generic null: uniform random integers in [10**low, 10**high]."""
    rng = _rng(seed)
    return _uniform_int(rng, mag_range[0], mag_range[1], n)


def gen_repeated(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
                 n_types: int = 50, concentration: float = 2.0) -> np.ndarray:
    """Repeated exact values: draw ``n_types`` templates, then sample from them
    with Dirichlet(concentration) weights. High Gini, low vocabulary diversity."""
    rng = _rng(seed)
    templates = _uniform_int(rng, mag_range[0], mag_range[1], n_types)
    weights = rng.dirichlet(np.full(n_types, concentration))
    idx = rng.choice(n_types, size=n, p=weights)
    return templates[idx].astype(np.int64)


def _round_to_multiples(rng, mag_range, n, multiples, weights) -> np.ndarray:
    base = _uniform_int(rng, mag_range[0], mag_range[1], n).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mult = rng.choice(np.asarray(multiples, dtype=np.int64), size=n, p=w)
    vals = np.round(base / mult) * mult
    vals = np.maximum(vals, mult)  # never round down to 0
    return vals.astype(np.int64)


def gen_round(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
              multiples=(100, 500, 1000), weights=(0.5, 0.3, 0.2)) -> np.ndarray:
    """Whole-dollar ROUND values (integer cents): rounded to the nearest $1, $5,
    or $10. Every value satisfies ``value % 100 == 0`` (cents = 00), matching the
    waterfall papers' ROUND definition."""
    return _round_to_multiples(_rng(seed), mag_range, n, multiples, weights)


def gen_dime_focal(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
                   multiples=(10, 100, 1000), weights=(0.5, 0.3, 0.2)) -> np.ndarray:
    """Dime-focal rounding (the former ``gen_round``): cents in {00, 10, ..., 90}.
    Kept for reference; NOT in the canonical MECHANISMS dict."""
    return _round_to_multiples(_rng(seed), mag_range, n, multiples, weights)


def gen_quarter(n: int = 100_000, seed: int = 0, mag_range=(1, 6)) -> np.ndarray:
    """Quarter-dollar focal values: cents drawn from {0, 25, 50, 75} uniformly."""
    rng = _rng(seed)
    base = _uniform_int(rng, mag_range[0], mag_range[1], n)
    dollars = base // 100
    cents = rng.choice(np.array([0, 25, 50, 75], dtype=np.int64), size=n)
    return np.maximum(dollars * 100 + cents, 2).astype(np.int64)


def gen_psychological(n: int = 100_000, seed: int = 0, mag_range=(1, 6)) -> np.ndarray:
    """Psychological pricing endings: cents from {95, 99, 49, 79} with weights
    {0.4, 0.3, 0.15, 0.15}; dollar part uniform in ``mag_range``."""
    rng = _rng(seed)
    base = _uniform_int(rng, mag_range[0], mag_range[1], n)
    dollars = base // 100
    cents = rng.choice(np.array([95, 99, 49, 79], dtype=np.int64),
                       size=n, p=[0.4, 0.3, 0.15, 0.15])
    return np.maximum(dollars * 100 + cents, 2).astype(np.int64)


def gen_threshold(n: int = 100_000, seed: int = 0,
                  threshold: int = 10_000, bandwidth: float = 0.10) -> np.ndarray:
    """Threshold-adjacent values: 70% just below the threshold, 30% just above."""
    rng = _rng(seed)
    n_below = int(round(0.7 * n))
    n_above = n - n_below
    below = rng.uniform(threshold * (1 - bandwidth), threshold, size=n_below)
    above = rng.uniform(threshold, threshold * (1 + bandwidth), size=n_above)
    vals = np.concatenate([below, above])
    rng.shuffle(vals)
    return np.maximum(np.round(vals), 2).astype(np.int64)


def gen_fiscal(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
               peak_months=(9, 10), peak_weight: float = 0.4):
    """Calendar-concentrated values. Amounts drawn from the null; timestamps
    concentrated in ``peak_months``. Returns ``(amounts, months)``."""
    rng = _rng(seed)
    amounts = _uniform_int(rng, mag_range[0], mag_range[1], n)
    is_peak = rng.random(n) < peak_weight
    peak = rng.choice(np.asarray(peak_months, dtype=np.int64), size=n)
    uniform_m = rng.integers(1, 13, size=n, dtype=np.int64)
    months = np.where(is_peak, peak, uniform_m).astype(np.int64)
    return amounts.astype(np.int64), months


def gen_divisibility(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
                     divisor: int = 7) -> np.ndarray:
    """Divisibility-constrained values: X = divisor * Q, Q uniform so that X
    falls in ``mag_range``."""
    rng = _rng(seed)
    lo = max(2, int(10 ** mag_range[0]) // divisor)
    hi = max(lo + 1, int(10 ** mag_range[1]) // divisor)
    q = rng.integers(lo, hi + 1, size=n, dtype=np.int64)
    return (divisor * q).astype(np.int64)


def gen_product(n: int = 100_000, seed: int = 0, mag_range=(1, 6)) -> np.ndarray:
    """Product-composed values: X = A * B with A, B independent integers chosen
    so X falls in ``mag_range``."""
    rng = _rng(seed)
    half_lo = mag_range[0] / 2.0
    half_hi = mag_range[1] / 2.0
    a = _uniform_int(rng, half_lo, half_hi, n)
    b = _uniform_int(rng, half_lo, half_hi, n)
    return (a * b).astype(np.int64)


def _primes_in_range(lo: int, hi: int) -> np.ndarray:
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    p = np.nonzero(sieve)[0]
    return p[(p >= lo) & (p <= hi)]


def gen_fixed_factor(n: int = 100_000, seed: int = 0, bit_length: int = 12) -> np.ndarray:
    """Fixed two-factor algorithmic values: X = p * q with p, q random primes of
    equal bit length. Use small ``bit_length`` (10-16) for tractability. These
    have L1 ~= L2 ~= 0.5 and minimal tail."""
    rng = _rng(seed)
    lo = 1 << (bit_length - 1)
    hi = (1 << bit_length) - 1
    primes = _primes_in_range(lo, hi)
    if primes.size < 2:
        raise ValueError(f"too few primes for bit_length={bit_length}")
    p = rng.choice(primes, size=n).astype(np.int64)
    q = rng.choice(primes, size=n).astype(np.int64)
    return p * q


def gen_temporal_concat(n: int = 100_000, seed: int = 0, mag_range=(1, 6)) -> np.ndarray:
    """Temporal concatenation with a COMPRESSIBILITY shift: first half from
    ``gen_null`` (high diversity), second half from ``gen_repeated`` (low
    diversity). Ordered, regime change at the midpoint (do not shuffle). The two
    regimes compress very differently, so CBAD R^2 should degrade."""
    h1 = n // 2
    h2 = n - h1
    a = gen_null(h1, seed, mag_range)
    b = gen_repeated(h2, seed + 1, mag_range, n_types=20, concentration=5.0)
    return np.concatenate([a, b]).astype(np.int64)


def gen_temporal_concat_arith(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
                              divisor: int = 7) -> np.ndarray:
    """Negative control: arithmetic-only regime shift (null -> divisibility-by-7).
    The two regimes share identical decimal-string compressibility, so CBAD R^2
    should NOT degrade. Tests projection partiality -- CBAD sees compression
    structure, not arithmetic residue."""
    h1 = n // 2
    h2 = n - h1
    a = gen_null(h1, seed, mag_range)
    b = gen_divisibility(h2, seed + 1, mag_range, divisor=divisor)
    return np.concatenate([a, b]).astype(np.int64)


def gen_sum_of_rounded(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
                       m: int = 5) -> np.ndarray:
    """Aggregated sums of rounded values: each observation is the sum of ``m``
    values from ``gen_round``."""
    vals = gen_round(n * m, seed, mag_range)
    return vals.reshape(n, m).sum(axis=1).astype(np.int64)


def gen_sum_of_divisibility(n: int = 100_000, seed: int = 0, mag_range=(1, 6),
                            m: int = 5, divisor: int = 7) -> np.ndarray:
    """Aggregated sums of divisibility-constrained values: each observation is the
    sum of ``m`` values from ``gen_divisibility`` (divisibility is preserved by
    summation -- Proposition 6)."""
    vals = gen_divisibility(n * m, seed, mag_range, divisor=divisor)
    return vals.reshape(n, m).sum(axis=1).astype(np.int64)
