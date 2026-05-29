"""WP7 Experiment 4: whole-genome replication of the codon (divisibility-by-3) result.

Runs the WP6 genome pipeline across multiple chromosomes (varied size/gene density)
to rule out chr21 idiosyncrasy.
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
import run_experiments as R        # noqa: E402
from wp6_common import real_signatures, reduce_column   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = R.RESULTS
DATA = os.path.join(ROOT, "data", "wp6")
CHROMS = ["1", "7", "19", "21", "22"]
MINLEN, MAXLEN = 2, 100_000


def parse_attr(s, key):
    for kv in s.split(";"):
        if kv.startswith(key + "="):
            return kv[len(key) + 1:]
    return None


def parse_gff(path):
    cds_by_tx = defaultdict(int)
    exons_by_tx = defaultdict(list)
    utr = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            ft, start, end, attrs = c[2], int(c[3]), int(c[4]), c[8]
            ln = end - start + 1
            if ft == "CDS":
                tx = parse_attr(attrs, "Parent")
                if tx:
                    cds_by_tx[tx] += ln
            elif ft == "exon":
                tx = parse_attr(attrs, "Parent")
                if tx:
                    exons_by_tx[tx].append((start, end))
            elif ft in ("five_prime_UTR", "three_prime_UTR"):
                utr.append(ln)
    introns = []
    for ex in exons_by_tx.values():
        ex.sort()
        for i in range(len(ex) - 1):
            g = ex[i + 1][0] - ex[i][1] - 1
            if g > 0:
                introns.append(g)
    def filt(a):
        a = np.array(a, dtype=np.int64)
        return a[(a > MINLEN) & (a < MAXLEN)]
    return (filt(list(cds_by_tx.values())), filt(introns), filt(utr))


def main():
    P.set_spf_cache(R.SPF_CACHE); P.ensure_spf_cache(R.SPF_CACHE)
    rows = []
    agg = {"CDS": [], "intron": [], "UTR": []}
    for ch in CHROMS:
        path = os.path.join(DATA, f"chr{ch}.gff3.gz")
        if not os.path.exists(path):
            print(f"  [skip] chr{ch}: file missing"); continue
        cds, intron, utr = parse_gff(path)
        feats = {"CDS": cds, "intron": intron, "UTR": utr}
        for ftype, vals in feats.items():
            if vals.size < 50:
                continue
            agg[ftype].append(vals)
            s = real_signatures(vals, n_null=15, extra_mods=(3,))
            cp_sub, cp_z, _ = reduce_column(s, ["c2", "c3", "c5", "c7", "c10", "c5sq"])
            frac3 = float(np.mean(vals % 3 == 0))
            rows.append({"chromosome": ch, "feature_type": ftype, "n": int(vals.size),
                         "frac_div3": round(frac3, 4),
                         "c3_mean": round(s["c3"]["value"], 4),
                         "c3_z": round(s["c3"]["z"], 1) if np.isfinite(s["c3"]["z"]) else "nan",
                         "mod3_tv": round(s["mod3_tv"]["value"], 5),
                         "mod3_z": round(s["mod3_tv"]["z"], 1) if np.isfinite(s["mod3_tv"]["z"]) else "nan",
                         "cp_selected": cp_sub})
        print(f"  chr{ch}: CDS n={feats['CDS'].size} frac_div3={np.mean(feats['CDS']%3==0):.3f}")

    with open(os.path.join(RESULTS, "wp7_genome_by_chromosome.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["chromosome", "feature_type", "n", "frac_div3",
                                           "c3_mean", "c3_z", "mod3_tv", "mod3_z", "cp_selected"])
        w.writeheader(); w.writerows(rows)

    # whole-genome aggregate
    summ = []
    for ftype, lists in agg.items():
        if not lists:
            continue
        allv = np.concatenate(lists)
        s = real_signatures(allv, n_null=15, extra_mods=(3,))
        cp_sub, _, _ = reduce_column(s, ["c2", "c3", "c5", "c7", "c10", "c5sq"])
        summ.append({"feature_type": ftype, "n_total": int(allv.size),
                     "frac_div3": round(float(np.mean(allv % 3 == 0)), 4),
                     "c3_mean": round(s["c3"]["value"], 4),
                     "c3_z": round(s["c3"]["z"], 1) if np.isfinite(s["c3"]["z"]) else "nan",
                     "mod3_z": round(s["mod3_tv"]["z"], 1) if np.isfinite(s["mod3_tv"]["z"]) else "nan",
                     "cp_selected": cp_sub})
    with open(os.path.join(RESULTS, "wp7_genome_whole_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature_type", "n_total", "frac_div3",
                                           "c3_mean", "c3_z", "mod3_z", "cp_selected"])
        w.writeheader(); w.writerows(summ)

    print("\nPer-chromosome CDS divisibility-by-3:")
    for r in rows:
        if r["feature_type"] == "CDS":
            print(f"  chr{r['chromosome']:>2}: frac_div3={r['frac_div3']:.3f} c3_z={r['c3_z']} "
                  f"mod3_z={r['mod3_z']} cp_selected={r['cp_selected']}")
    print("\nWhole-genome (5 chromosomes) aggregate:")
    for r in summ:
        print(f"  {r['feature_type']:7} n={r['n_total']:7d} frac_div3={r['frac_div3']:.4f} "
              f"c3_z={r['c3_z']} cp_selected={r['cp_selected']}")

    cds_rows = [r for r in rows if r["feature_type"] == "CDS"]
    all_high = all(r["frac_div3"] >= 0.90 and r["cp_selected"] == "c3" for r in cds_rows)
    print(f"\nReplication: every chromosome CDS frac_div3>=0.90 and cp->c3: "
          f"{'CONFIRMED' if all_high else 'CHECK'} (chr21 not idiosyncratic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
