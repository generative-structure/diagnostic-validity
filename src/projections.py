"""Projection metrics (Work Package 3).

Implements every formula in CANONICAL_DEFINITIONS.md exactly. Each public metric
takes a 1-D array of positive integers and returns a float; ``compute_all_projections``
returns the full metric dict.

Factorization uses a smallest-prime-factor (SPF) sieve (sympy is not required).
The sieve can be memory-mapped from disk so the 13 worker processes share one
copy -- see ``ensure_spf_cache`` / ``set_spf_cache``.
"""
from __future__ import annotations

import math
import os
import zlib

import numpy as np
from scipy.optimize import curve_fit

NAN = float("nan")

# Benford leading-digit proportions p_d = log10(1 + 1/d), d = 1..9
BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))
_LOGP = {2: math.log(2), 3: math.log(3), 5: math.log(5), 7: math.log(7)}
_LOG25 = math.log(25)

# Covers every default mechanism value: fixed_factor (12-bit) -> <= 4093^2 ~ 1.68e7,
# sums (m=5) -> <= 5e6, products -> <= 1e6. Raise if generators are reconfigured;
# values above the limit fall back to trial division automatically.
MAX_SIEVE = 17_000_000

_SPF = None              # in-memory or memmapped SPF array
_PRIMES = None           # primes up to MAX_SIEVE (for trial-division fallback)
_SPF_CACHE_PATH = None    # if set, _get_spf() mmaps this file


def set_spf_cache(path: str) -> None:
    """Point workers at a prebuilt SPF cache file (called from the pool initializer)."""
    global _SPF_CACHE_PATH
    _SPF_CACHE_PATH = path


def _build_spf(limit: int) -> np.ndarray:
    spf = np.zeros(limit + 1, dtype=np.int32)
    i = 2
    while i * i <= limit:
        if spf[i] == 0:                      # i is prime
            seg = spf[i * i::i]              # view into spf
            seg[seg == 0] = i               # writes through to spf
        i += 1
    rem = spf == 0
    rem[0] = rem[1] = False
    idx = np.nonzero(rem)[0]                 # remaining zeros are primes
    spf[idx] = idx.astype(np.int32)
    return spf


def ensure_spf_cache(path: str, limit: int = MAX_SIEVE) -> str:
    """Build the SPF sieve to ``path`` if missing; return ``path``. Run once in the
    main process before spawning workers."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, _build_spf(limit))
    return path


def _get_spf() -> np.ndarray:
    global _SPF
    if _SPF is None:
        if _SPF_CACHE_PATH and os.path.exists(_SPF_CACHE_PATH):
            _SPF = np.load(_SPF_CACHE_PATH, mmap_mode="r")
        else:
            _SPF = _build_spf(MAX_SIEVE)
    return _SPF


def _get_primes() -> np.ndarray:
    """Small primes (<= 100000) for the trial-division fallback. Built lazily and
    only when a value exceeds the sieve limit; 100000 > sqrt(n) for any n < 1e10."""
    global _PRIMES
    if _PRIMES is None:
        limit = 100_000
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = False
        _PRIMES = np.nonzero(sieve)[0]
    return _PRIMES


def factorize(n: int) -> dict:
    """Return {prime: exponent} for n > 1 (exact p-adic valuations)."""
    spf = _get_spf()
    f: dict = {}
    if n < spf.shape[0]:
        while n > 1:
            p = int(spf[n])
            while n % p == 0:
                f[p] = f.get(p, 0) + 1
                n //= p
        return f
    # Fallback for values above the sieve limit.
    for p in _get_primes():
        p = int(p)
        if p * p > n:
            break
        if n % p == 0:
            c = 0
            while n % p == 0:
                n //= p
                c += 1
            f[p] = c
    if n > 1:
        f[int(n)] = f.get(int(n), 0) + 1
    return f


# --------------------------------------------------------------------------- #
# Distributional projections (no factorization)
# --------------------------------------------------------------------------- #
def leading_digit_mad(values: np.ndarray) -> float:
    v = values[values >= 1]
    exp = np.floor(np.log10(v) + 1e-9).astype(np.int64)
    fd = (v // (10 ** exp)).astype(np.int64)
    fd = fd[(fd >= 1) & (fd <= 9)]
    if fd.size == 0:
        return NAN
    counts = np.bincount(fd, minlength=10)[1:10].astype(np.float64)
    props = counts / counts.sum()
    return float(np.mean(np.abs(props - BENFORD)))


def terminal_digit_chisq(values: np.ndarray) -> float:
    counts = np.bincount(values % 10, minlength=10)[:10].astype(np.float64)
    exp = counts.sum() / 10.0
    if exp == 0:
        return NAN
    return float(np.sum((counts - exp) ** 2 / exp))


def cents_entropy(values: np.ndarray) -> float:
    counts = np.bincount(values % 100, minlength=100).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def modular_residue_tv(values: np.ndarray, m: int) -> float:
    counts = np.bincount(values % m, minlength=m).astype(np.float64)
    f = counts / counts.sum()
    return float(0.5 * np.sum(np.abs(f - 1.0 / m)))


def vocab_gini(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    counts = np.sort(counts).astype(np.float64)
    k = counts.size
    if k <= 1:
        return 0.0
    idx = np.arange(1, k + 1)
    return float((2.0 * np.sum(idx * counts) / (k * counts.sum())) - (k + 1) / k)


def temporal_concentration(months: np.ndarray, periods: int = 12) -> float:
    counts = np.bincount(months, minlength=periods + 1)[1:periods + 1].astype(np.float64)
    exp = counts.sum() / periods
    if exp == 0:
        return NAN
    return float(np.sum((counts - exp) ** 2 / exp))


def threshold_density_ratio(values: np.ndarray, threshold: float,
                            bandwidth: float = 0.05) -> float:
    d = bandwidth * threshold
    below = int(np.sum((values >= threshold - d) & (values < threshold)))
    above = int(np.sum((values >= threshold) & (values <= threshold + d)))
    if above == 0:
        return float("inf") if below > 0 else NAN
    return float(below / above)


# --------------------------------------------------------------------------- #
# Factorization projections (composite n > 1 only)
# --------------------------------------------------------------------------- #
# Factorization metrics summed per observation. c10 is added afterward (= c2 + c5).
_FKEYS = ["c2", "c3", "c5", "c7", "c5sq", "L1", "L2", "L3", "L4", "tail_mass"]


def factor_projections(values: np.ndarray) -> dict:
    """Primary metrics over ALL n > 1 (primes contribute L1=1, c_p=0 except the
    prime itself); composite-only variants (suffix ``_composite``) as a sensitivity
    analysis; plus ``prime_fraction``. See CANONICAL_DEFINITIONS.md sections 0-1."""
    acc_all = {k: 0.0 for k in _FKEYS}
    acc_comp = {k: 0.0 for k in _FKEYS}
    cnt_all = cnt_comp = n_prime = n_total = 0
    for n in values:
        n = int(n)
        if n < 2:
            continue
        n_total += 1
        f = factorize(n)
        is_prime = len(f) == 1 and next(iter(f.values())) == 1
        logn = math.log(n)
        if logn <= 0:
            continue
        per = {
            "c2": f.get(2, 0) * _LOGP[2] / logn,
            "c3": f.get(3, 0) * _LOGP[3] / logn,
            "c5": f.get(5, 0) * _LOGP[5] / logn,
            "c7": f.get(7, 0) * _LOGP[7] / logn,
            "c5sq": (f.get(5, 0) // 2) * _LOG25 / logn,
        }
        shares = sorted((a * math.log(p) / logn for p, a in f.items()), reverse=True)
        per["L1"] = shares[0] if shares else 0.0
        per["L2"] = shares[1] if len(shares) > 1 else 0.0
        per["L3"] = shares[2] if len(shares) > 2 else 0.0
        per["L4"] = shares[3] if len(shares) > 3 else 0.0
        per["tail_mass"] = 1.0 - per["L1"] - per["L2"]
        cnt_all += 1
        for k in _FKEYS:
            acc_all[k] += per[k]
        if is_prime:
            n_prime += 1
        else:
            cnt_comp += 1
            for k in _FKEYS:
                acc_comp[k] += per[k]

    out = {}
    if cnt_all > 0:
        for k in _FKEYS:
            out[k] = acc_all[k] / cnt_all
        out["c10"] = out["c2"] + out["c5"]
    else:
        for k in _FKEYS:
            out[k] = NAN
        out["c10"] = NAN
    if cnt_comp > 0:
        for k in _FKEYS:
            out[f"{k}_composite"] = acc_comp[k] / cnt_comp
        out["c10_composite"] = out["c2_composite"] + out["c5_composite"]
    else:
        for k in _FKEYS:
            out[f"{k}_composite"] = NAN
        out["c10_composite"] = NAN
    out["prime_fraction"] = (n_prime / n_total) if n_total > 0 else NAN
    return out


# --------------------------------------------------------------------------- #
# CBAD (compression-convergence) projection
# --------------------------------------------------------------------------- #
def _cbad_model(N, a, b, c):
    return a + b * np.power(N, -c)


def compute_cbad(values: np.ndarray, n_points: int = 12, n_min: int = 100) -> dict:
    n = len(values)
    if n < n_min:
        return {"cbad_a": NAN, "cbad_c": NAN, "cbad_r2": NAN}
    Ns = np.unique(np.geomspace(n_min, n, num=n_points).astype(int))
    Ns = Ns[Ns >= n_min]
    if Ns.size < 4:
        return {"cbad_a": NAN, "cbad_c": NAN, "cbad_r2": NAN}
    strs = [str(int(x)) for x in values]
    R = np.empty(Ns.size, dtype=np.float64)
    for k, Nk in enumerate(Ns):
        raw = " ".join(strs[:int(Nk)]).encode()
        R[k] = len(zlib.compress(raw, 9)) / len(raw)
    Nf = Ns.astype(np.float64)
    try:
        p0 = [R[-1], max(R[0] - R[-1], 1e-3), 0.5]
        popt, _ = curve_fit(_cbad_model, Nf, R, p0=p0, maxfev=10_000,
                            bounds=([0.0, -2.0, 0.0], [1.0, 2.0, 5.0]))
        pred = _cbad_model(Nf, *popt)
        ss_res = float(np.sum((R - pred) ** 2))
        ss_tot = float(np.sum((R - R.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NAN
        return {"cbad_a": float(popt[0]), "cbad_c": float(popt[2]), "cbad_r2": float(r2)}
    except Exception:
        return {"cbad_a": NAN, "cbad_c": NAN, "cbad_r2": NAN}


# --------------------------------------------------------------------------- #
# Aggregate entry point
# --------------------------------------------------------------------------- #
def compute_all_projections(values, ordered: bool = False,
                            months=None, threshold=None) -> dict:
    """Compute all projection metrics. Keys match CANONICAL_DEFINITIONS.md.

    Args:
        values: 1-D array of positive integers (filtered to > 1 internally).
        ordered: if True, compute CBAD metrics (needs a meaningful sequence order).
        months: optional array of month labels (1-12) for the temporal projection.
        threshold: optional threshold T for the threshold-density projection.
    """
    values = np.asarray(values)
    values = values[values > 1]
    out = {
        "leading_digit_mad": leading_digit_mad(values),
        "terminal_digit_chisq": terminal_digit_chisq(values),
        "cents_entropy": cents_entropy(values),
        "vocab_gini": vocab_gini(values),
    }
    for m in (5, 10, 25, 100):
        out[f"mod{m}_tv"] = modular_residue_tv(values, m)
    out["temporal_concentration"] = (
        temporal_concentration(np.asarray(months)) if months is not None else NAN)
    out["threshold_density"] = (
        threshold_density_ratio(values, threshold) if threshold is not None else NAN)
    out.update(factor_projections(values))
    if ordered:
        out.update(compute_cbad(values))
    else:
        out.update({"cbad_a": NAN, "cbad_c": NAN, "cbad_r2": NAN})
    return out
