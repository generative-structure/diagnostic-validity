"""WP7 Experiment 7: real-data procurement (bulk download).

DoE FY2023 contracts (transaction-level federal_action_obligation), USAspending
award_data_archive. Replicates the threshold baseline-absorption result (Prop 8)
on real data and tests aggregation (Prop 6) via vendor totals.
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
from wp6_common import real_signatures, reduce_column, METRICS   # noqa: E402
from baselines import magnitude_matched_null, generate_threshold_aware_null  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = R.RESULTS
ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "wp6", "doe_fy2023.zip")
AGENCY = sys.argv[2] if len(sys.argv) > 2 else "DoE"
PROV = os.path.join(RESULTS, "wp6_data_provenance.md")
csv.field_size_limit(10 ** 7)


def load():
    with zipfile.ZipFile(ZIP) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as fh:
            rd = csv.reader((line.decode("utf-8", "replace") for line in fh))
            header = next(rd)
            ai = header.index("federal_action_obligation")
            vi = header.index("recipient_uei")
            amts, ueis = [], []
            for row in rd:
                try:
                    a = float(row[ai])
                except (ValueError, IndexError):
                    continue
                amts.append(a)
                ueis.append(row[vi] if vi < len(row) else "")
    return np.array(amts, dtype=np.float64), np.array(ueis, dtype=object)


def verdict(z):
    if not np.isfinite(z):
        return "?"
    return "+" if abs(z) > 3 else ("0" if abs(z) < 1.5 else "?")


def band_absorption(band, T_cents, n_null=20):
    obs = P.threshold_density_ratio(band, T_cents, 0.05)
    out = {}
    for bt in ("magnitude_matched", "parent_support", "threshold_aware"):
        tds = []
        for b in range(n_null):
            rng = np.random.default_rng(1000 + b)
            if bt == "magnitude_matched":
                nl = magnitude_matched_null(band, rng)
            elif bt == "parent_support":
                nl = rng.integers(int(band.min()), int(band.max()) + 1, size=band.size, dtype=np.int64)
            else:
                nl = generate_threshold_aware_null(band, 1000 + b, T_cents, 0.10)
            tds.append(P.threshold_density_ratio(nl, T_cents, 0.05))
        tds = np.array(tds, float); tds = tds[np.isfinite(tds)]
        if tds.size < 2 or not np.isfinite(obs):
            out[bt] = (obs, np.nan, np.nan, "?")
            continue
        mu, sd = tds.mean(), (tds.std(ddof=1) or 1e-12)
        z = (obs - mu) / sd
        out[bt] = (obs, mu, z, verdict(z))
    return out


def main():
    P.set_spf_cache(R.SPF_CACHE); P.ensure_spf_cache(R.SPF_CACHE)
    amts, ueis = load()
    print(f"Loaded {amts.size} DoE FY2023 transactions; positive frac={np.mean(amts>0):.3f}")

    pos = amts > 0
    cents_all = np.round(amts * 100).astype(np.int64)
    mask = (cents_all >= 10_000) & (cents_all <= 50_000_000)   # $100..$500k
    cents = cents_all[mask]
    print(f"Transaction-level in $100..$500k: n={cents.size}; whole-dollar frac="
          f"{np.mean(cents % 100 == 0):.3f}")

    # ---- A: full battery (transaction level) ----
    s = real_signatures(cents, n_null=20)
    proj_rows = [{"level": "transaction", "n": int(cents.size),
                  **{m: round(s[m]["value"], 6) if (m in s and np.isfinite(s[m]["value"])) else "nan"
                     for m in METRICS}}]

    # ---- B/C: threshold band absorption at 3 thresholds ----
    thr_rows, abs_rows = [], []
    for Tdollar in (10_000, 25_000, 250_000):
        Tc = Tdollar * 100
        lo, hi = int(Tc * 0.9), int(Tc * 1.1)
        band = cents[(cents >= lo) & (cents <= hi)]
        powerof10 = abs(np.log10(Tc) - round(np.log10(Tc))) < 1e-9
        if band.size < 200:
            abs_rows.append({"threshold_usd": Tdollar, "band_n": int(band.size),
                             "power_of_10_cents": powerof10, "baseline": "n/a",
                             "obs_ratio": "nan", "null_ratio": "nan", "z": "nan",
                             "verdict": "insufficient band data"})
            continue
        res = band_absorption(band, Tc)
        thr_rows.append({"threshold_usd": Tdollar, "band_n": int(band.size),
                         "obs_density_ratio": round(res["magnitude_matched"][0], 3)})
        for bt, (obs, mu, z, v) in res.items():
            abs_rows.append({"threshold_usd": Tdollar, "band_n": int(band.size),
                             "power_of_10_cents": powerof10, "baseline": bt,
                             "obs_ratio": round(obs, 3) if np.isfinite(obs) else "nan",
                             "null_ratio": round(mu, 3) if np.isfinite(mu) else "nan",
                             "z": round(z, 2) if np.isfinite(z) else "nan", "verdict": v})

    # ---- D: vendor aggregation (Prop 6) ----
    order = np.argsort(ueis)
    ueis_s = ueis[order]; cents_full = cents_all[order]
    vendor_tot = {}
    cur = None; acc = 0
    for u, c in zip(ueis_s, cents_full):
        if u != cur:
            if cur is not None and cur != "":
                vendor_tot[cur] = acc
            cur = u; acc = 0
        acc += int(c)
    if cur not in (None, ""):
        vendor_tot[cur] = acc
    vtot = np.array([v for v in vendor_tot.values()], dtype=np.int64)
    vtot = vtot[(vtot >= 10_000) & (vtot <= 50_000_000)]
    agg_rows = []
    if vtot.size >= 200:
        sv = real_signatures(vtot, n_null=20)
        proj_rows.append({"level": "vendor_total", "n": int(vtot.size),
                          **{m: round(sv[m]["value"], 6) if (m in sv and np.isfinite(sv[m]["value"])) else "nan"
                             for m in METRICS}})
        for col, mets in [("CentsEntropy", ["cents_entropy"]), ("TerminalDigit", ["terminal_digit_chisq"]),
                          ("DecimalResidue", ["mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv"]),
                          ("cp_spectrum", ["c2", "c3", "c5", "c7", "c10", "c5sq"]),
                          ("VocabGini", ["vocab_gini"])]:
            _, zt, vt = reduce_column(s, mets)
            _, zv, vv = reduce_column(sv, mets)
            agg_rows.append({"projection": col, "txn_verdict": vt,
                             "txn_z": round(zt, 1) if np.isfinite(zt) else "nan",
                             "vendor_verdict": vv,
                             "vendor_z": round(zv, 1) if np.isfinite(zv) else "nan"})

    # ---- write ----
    def W(name, fields, rows):
        with open(os.path.join(RESULTS, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    W("wp7_procurement_bulk_projections.csv", ["level", "n"] + METRICS, proj_rows)
    W("wp7_procurement_bulk_threshold.csv", ["threshold_usd", "band_n", "obs_density_ratio"], thr_rows)
    W("wp7_procurement_bulk_baseline_absorption.csv",
      ["threshold_usd", "band_n", "power_of_10_cents", "baseline", "obs_ratio", "null_ratio", "z", "verdict"], abs_rows)
    if agg_rows:
        W("wp7_procurement_bulk_aggregation.csv",
          ["projection", "txn_verdict", "txn_z", "vendor_verdict", "vendor_z"], agg_rows)

    print("\nThreshold baseline absorption (real DoE data):")
    for r in abs_rows:
        if r["baseline"] != "n/a":
            print(f"  T=${r['threshold_usd']:>7} pow10={r['power_of_10_cents']!s:5} "
                  f"{r['baseline']:18} obs={r['obs_ratio']} null={r['null_ratio']} "
                  f"z={r['z']} -> {r['verdict']}")
    print("\nVendor aggregation (Prop 6): transaction vs vendor-total verdicts:")
    for r in agg_rows:
        print(f"  {r['projection']:14} txn={r['txn_verdict']}(z={r['txn_z']}) -> "
              f"vendor={r['vendor_verdict']}(z={r['vendor_z']})")

    with open(PROV, "a") as f:
        f.write("\n## WP7 Demonstration: Procurement BULK (real threshold absorption)\n"
                "- Source: USAspending award_data_archive, "
                "FY2023_089_Contracts_Full_20260506.zip (DoE), 5.87 MB; accessed 2026-05-24.\n"
                f"- Transaction-level federal_action_obligation; n_in_range={cents.size}.\n"
                "- Replicates Prop 8 baseline absorption at $10k (=10^6 cents, power of 10) vs "
                "$25k/$250k on real data; vendor aggregation (recipient_uei) tests Prop 6.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
