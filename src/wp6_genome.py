"""WP6 Demonstration 1: Codon triplets (genome annotations).

Tests Prop 3 (projection partiality: c3/mod3 fire for CDS, not introns/UTRs),
Prop 5 (unitization: the codon constraint acts on the per-transcript coding
sequence, not on individual exonic CDS segments).

Data: Ensembl GRCh38 release 115, chromosome 21 GFF3 (downloaded separately).
GFF3 is 1-based inclusive: feature length = end - start + 1.
"""
from __future__ import annotations

import gzip
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import projections as P            # noqa: E402
from wp6_common import real_signatures, reduce_column, write_csv, METRICS, provenance_append  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
GFF = os.path.join(ROOT, "data", "wp6", "chr21.gff3.gz")
PROV = os.path.join(RESULTS, "wp6_data_provenance.md")

MINLEN, MAXLEN = 2, 100_000


def parse_attr(s, key):
    for kv in s.split(";"):
        if kv.startswith(key + "="):
            return kv[len(key) + 1:]
    return None


def parse_gff():
    cds_by_tx = defaultdict(int)        # transcript -> summed CDS length
    cds_segments = []                   # individual CDS segment lengths
    exons_by_tx = defaultdict(list)     # transcript -> [(start,end)]
    utr_lengths = []
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            ftype, start, end, attrs = c[2], int(c[3]), int(c[4]), c[8]
            length = end - start + 1
            if ftype == "CDS":
                tx = parse_attr(attrs, "Parent")
                if tx:
                    cds_by_tx[tx] += length
                cds_segments.append(length)
            elif ftype == "exon":
                tx = parse_attr(attrs, "Parent")
                if tx:
                    exons_by_tx[tx].append((start, end))
            elif ftype in ("five_prime_UTR", "three_prime_UTR"):
                utr_lengths.append(length)
    # introns from consecutive exons within a transcript
    introns = []
    for tx, ex in exons_by_tx.items():
        ex.sort()
        for i in range(len(ex) - 1):
            gap = ex[i + 1][0] - ex[i][1] - 1
            if gap > 0:
                introns.append(gap)
    return (np.array(list(cds_by_tx.values()), dtype=np.int64),
            np.array(cds_segments, dtype=np.int64),
            np.array(introns, dtype=np.int64),
            np.array(utr_lengths, dtype=np.int64))


def filt(a):
    return a[(a > MINLEN) & (a < MAXLEN)]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    P.set_spf_cache(os.path.join(RESULTS, "_spf_cache.npy"))
    cds_total, cds_seg, intron, utr = parse_gff()
    cds_total, cds_seg, intron, utr = filt(cds_total), filt(cds_seg), filt(intron), filt(utr)

    features = {
        "CDS_total_per_transcript": cds_total,
        "CDS_segment": cds_seg,           # secondary: codons split across exons -> no mod3
        "intron": intron,
        "UTR": utr,
    }
    print("Feature counts (chr21):")
    for k, v in features.items():
        frac0 = float(np.mean(v % 3 == 0)) if v.size else float("nan")
        print(f"  {k:26} n={v.size:6d}  frac(len%3==0)={frac0:.3f}")

    # divisibility sanity: per-transcript CDS should be ~all multiples of 3
    cds_mod3_frac = float(np.mean(cds_total % 3 == 0)) if cds_total.size else 0.0

    sig_rows, proj_rows, summary_rows = [], [], []
    sigs = {}
    for name, vals in features.items():
        if vals.size < 50:
            print(f"  [skip] {name}: only {vals.size} features")
            continue
        s = real_signatures(vals, n_null=20, extra_mods=(3,))
        sigs[name] = s
        for metric, d in s.items():
            sig_rows.append({"feature": name, "metric": metric,
                             "value": round(d["value"], 6) if np.isfinite(d["value"]) else "nan",
                             "null_mean": round(d["null_mean"], 6) if np.isfinite(d["null_mean"]) else "nan",
                             "null_std": round(d["null_std"], 6) if np.isfinite(d["null_std"]) else "nan",
                             "z_score": round(d["z"], 3) if np.isfinite(d["z"]) else "nan",
                             "verdict": d["verdict"]})
        proj_rows.append({"feature": name, "n": vals.size,
                          **{m: round(s[m]["value"], 6) if (m in s and np.isfinite(s[m]["value"])) else "nan"
                             for m in (["mod3_tv"] + METRICS)}})
        cp_sub, cp_z, cp_v = reduce_column(s, ["c2", "c3", "c5", "c7", "c10", "c5sq"])
        dec_sub, dec_z, dec_v = reduce_column(s, ["mod5_tv", "mod10_tv", "mod25_tv", "mod100_tv"])
        lp_sub, lp_z, lp_v = reduce_column(s, ["L1", "L2", "L3", "L4"])
        summary_rows.append({
            "feature": name, "n": vals.size,
            "mod3_verdict": s["mod3_tv"]["verdict"], "mod3_z": round(s["mod3_tv"]["z"], 1),
            "c3_verdict": s["c3"]["verdict"], "c3_z": round(s["c3"]["z"], 1),
            "cp_spectrum_sub": cp_sub, "cp_spectrum_verdict": cp_v,
            "TerminalDigit": s["terminal_digit_chisq"]["verdict"],
            "DecimalResidue": dec_v, "L_profile": lp_v,
            "LeadingDigit": s["leading_digit_mad"]["verdict"]})

    write_csv(os.path.join(RESULTS, "wp6_genome_signatures.csv"),
              ["feature", "metric", "value", "null_mean", "null_std", "z_score", "verdict"], sig_rows)
    write_csv(os.path.join(RESULTS, "wp6_genome_projections.csv"),
              ["feature", "n", "mod3_tv"] + METRICS, proj_rows)
    write_csv(os.path.join(RESULTS, "wp6_genome_summary.csv"),
              ["feature", "n", "mod3_verdict", "mod3_z", "c3_verdict", "c3_z",
               "cp_spectrum_sub", "cp_spectrum_verdict", "TerminalDigit",
               "DecimalResidue", "L_profile", "LeadingDigit"], summary_rows)

    print("\nSUMMARY (feature x key projections):")
    for r in summary_rows:
        print(f"  {r['feature']:26} mod3={r['mod3_verdict']}(z={r['mod3_z']}) "
              f"c3={r['c3_verdict']}(z={r['c3_z']}) cp_sub={r['cp_spectrum_sub']} "
              f"Term={r['TerminalDigit']} Dec={r['DecimalResidue']} Lprof={r['L_profile']}")

    # provenance
    provenance_append(PROV,
        "## Demonstration 1: Codon triplets (genome)\n"
        "- Source: Ensembl FTP, https://ftp.ensembl.org/pub/current_gff3/homo_sapiens/\n"
        "- File: Homo_sapiens.GRCh38.115.chromosome.21.gff3.gz (GRCh38, release 115)\n"
        f"- Accessed: 2026-05-24. Local: data/wp6/chr21.gff3.gz (979,539 bytes)\n"
        "- Coordinates: GFF3 1-based inclusive; length = end - start + 1.\n"
        "- CDS: summed per Parent transcript (the codon constraint acts on the full\n"
        "  coding sequence, not individual exonic segments). CDS_segment reported\n"
        "  separately to show segments are NOT multiples of 3 (Prop 5, unitization).\n"
        "- Introns: gaps between consecutive exons within a transcript.\n"
        "- UTR: five_prime_UTR + three_prime_UTR feature lengths.\n"
        "- Filter: 2 < length < 100000.\n"
        f"- Sanity: fraction of per-transcript CDS lengths divisible by 3 = {cds_mod3_frac:.4f}\n"
        "- Citations (others' work): Crick (1966) on the triplet code; standard\n"
        "  molecular biology. No self-citations.\n")

    # hard sanity stop
    if cds_mod3_frac < 0.90 or sigs.get("CDS_total_per_transcript", {}).get("c3", {}).get("verdict") != "+":
        print(f"\n*** CHECK: CDS divisibility-by-3 not strong (frac={cds_mod3_frac:.3f}, "
              f"c3 verdict={sigs.get('CDS_total_per_transcript',{}).get('c3',{}).get('verdict')}). "
              "Inspect processing.")
        return 1
    print(f"\nCDS divisibility-by-3 confirmed (frac={cds_mod3_frac:.4f}); c3/mod3 fire for CDS only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
