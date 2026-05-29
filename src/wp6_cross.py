"""WP6 Demonstration 4: cross-demonstration projection complementarity.

Builds the complementarity table from the REAL results of the completed demos
(genome, procurement). Age heaping is listed as pending (registration-gated;
predictions stated from theory, not fabricated as data).
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wp6_common import write_csv   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    gen = {r["feature"]: r for r in read_rows(os.path.join(RESULTS, "wp6_genome_summary.csv"))}
    proc = read_rows(os.path.join(RESULTS, "wp6_procurement_summary.csv"))[0]
    cds = gen.get("CDS_total_per_transcript", {})

    rows = [
        {"dataset": "genome_CDS", "mechanism_type": "codon triplet (divisibility by 3)",
         "status": "REAL (chr21)",
         "distinguishing_active": f"mod3 ({cds.get('mod3_verdict')}), c3 ({cds.get('c3_verdict')})",
         "cp_spectrum_submetric": cds.get("cp_spectrum_sub"),
         "detected": "mod3; c3; L_profile",
         "not_detected": "cents/round-dollar (not cent-denominated); c5/c7-specific"},
        {"dataset": "procurement_awards", "mechanism_type": "round-dollar / threshold",
         "status": "REAL (DoD FY2023, round-number battery)",
         "distinguishing_active": f"cents_entropy ({proc.get('CentsEntropy')}), "
                                  f"DecimalResidue ({proc.get('DecimalResidue')})",
         "cp_spectrum_submetric": proc.get("cp_spectrum_sub"),
         "detected": "terminal digit; cents entropy; mod100; c2/c5/c10",
         "not_detected": "c3 / mod3 (no codon constraint); CBAD (unordered)"},
        {"dataset": "age_heaping", "mechanism_type": "digit heaping on 0/5",
         "status": "PENDING (registration-gated microdata; see provenance)",
         "distinguishing_active": "terminal_digit, mod5/mod10 (predicted)",
         "cp_spectrum_submetric": "c5/c10 (predicted)",
         "detected": "terminal digit; decimal residue (mod5/mod10); c2/c5 (predicted)",
         "not_detected": "c3, c7; threshold density (predicted)"},
    ]
    write_csv(os.path.join(RESULTS, "wp6_cross_demonstration_complementarity.csv"),
              list(rows[0].keys()), rows)
    print("Cross-demonstration complementarity (different mechanisms -> different projections):")
    for r in rows:
        print(f"  {r['dataset']:18} [{r['status']}]")
        print(f"      cp_spectrum submetric = {r['cp_spectrum_submetric']:12} "
              f"detected: {r['detected']}")
    print("\nKey point (Prop 3): the cp_spectrum family selects a DIFFERENT prime per "
          "mechanism --\n  genome->c3 (codon), procurement->decimal primes (c2/c5/c10), "
          "age->c5 (predicted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
