# WP6 Real-Data Demonstrations — Provenance

All analyses use the identical `compute_all_projections()` from the synthetic
engine (`src/projections.py`) and the same magnitude-matched null
(`src/baselines.py`). For single observed datasets, signatures are computed
against a distribution of 20 magnitude-matched null resamples:
`z = (Phi_mech - mean_null) / std_null`. No self-citations: every cited
mechanism is documented in prior work by other authors.

Status summary:
- Demo 1 (genome codon triplets): COMPLETE (real data).
- Demo 2 (procurement round numbers): PARTIAL — round-number battery on real
  data complete; threshold-density/Prop-8 test blocked by API limits (documented).
- Demo 3 (age heaping): BARRIER — registration-gated microdata (documented, not fabricated).
- Demo 4 (cross-demonstration complementarity): COMPLETE from Demos 1-2 (age predicted).
- Demo 5 (real-data aggregation): SKIPPED — no suitable grouping in available data.

---

## Demonstration 1: Codon triplets (genome)
- Source: Ensembl FTP, https://ftp.ensembl.org/pub/current_gff3/homo_sapiens/
- File: Homo_sapiens.GRCh38.115.chromosome.21.gff3.gz (GRCh38, release 115)
- Accessed: 2026-05-24. Local: data/wp6/chr21.gff3.gz (979,539 bytes)
- Coordinates: GFF3 1-based inclusive; length = end - start + 1.
- CDS: summed per Parent transcript (the codon constraint acts on the full
  coding sequence, not individual exonic segments). CDS_segment reported
  separately to show segments are NOT multiples of 3 (Prop 5, unitization).
- Introns: gaps between consecutive exons within a transcript.
- UTR: five_prime_UTR + three_prime_UTR feature lengths.
- Filter: 2 < length < 100000.
- Sanity: fraction of per-transcript CDS lengths divisible by 3 = 0.9236
- Citations (others' work): Crick (1966) on the triplet code; standard
  molecular biology. No self-citations.

## Demonstration 2: Procurement award amounts (USAspending.gov)
- Source: USAspending API, https://api.usaspending.gov/api/v2/search/spending_by_award/
- Filter: DoD (toptier awarding), contracts (A,B,C,D), FY2023 (2022-10-01..2023-09-30).
- Representative sample sorted by Start Date (NOT amount). Accessed 2026-05-24.
- n=690 award amounts in $100..$500k, converted to integer cents.
- Real round-number evidence: whole-dollar fraction=0.472, multiples of $100=0.106, $1000=0.072; the all-agency band [$9000,$11000] returned 6000+ awards in [$9000,$9138] and 300+ exactly at $5000 (strong round-number/threshold bunching).
- BARRIER (threshold-density / Prop 8 baseline absorption): the search API caps results at ~10,000, the award_amounts filter is award-level, and transaction amounts include negative de-obligations. Complete enumeration of a $10k/$25k threshold window (needed for the density ratio and 3-baseline absorption test) is therefore not possible via the API. NOT FABRICATED.
- What would be done with bulk data: download award_data_archive transaction-level CSV (federal_action_obligation) for an agency-FY, convert to cents, run the identical battery + threshold_density at $10k (= 10^6 cents, power of 10 -> expect magnitude-matched absorption, recovery under parent-support) and $25k (not power of 10 -> expect detection).
- Citations (others): Liebman & Mahoney (2017); FAR Part 13; Kleven & Waseem (2013); Saez (2010). No self-citations.

## Demonstration 3: Age heaping (census/survey microdata)
- Intended source: IPUMS International (https://international.ipums.org/) or
  national census microdata with documented digit preference.
- BARRIER: genuine high-heaping data is SELF-REPORTED single-year age from census
  microdata, which is registration-gated (IPUMS account + extract request) and
  cannot be completed in this session. UN/WPP single-year age series are modeled
  and smoothed (no heaping). NOT FABRICATED: Option D (reconstructing a
  distribution from a published Whipple index) was deliberately NOT used, as it
  would amount to inventing the data to match a target statistic.
- What would be done: obtain single-year reported age (e.g., a Sub-Saharan or
  South Asian census round with W > 150), filter ages 10-79, compute Whipple's
  index, expand counts to an integer age sample, run the identical battery.
  Prediction: terminal_digit + and decimal residue (mod5/mod10) + (heaping on
  0/5), c2/c5/c10 +, c3/c7 ~ 0; contrast with a low-heaping population
  (e.g., register-based Nordic ages) where positional projections return to null.
- Citations (others): Whipple (1919); Myers (1940); Bachi (1951); A'Hearn,
  Baten & Crayen (2009); Pullum (2006). No self-citations.

## Demonstration 5: Real-data aggregation (Prop 6)
- SKIPPED. The available procurement sample (award-level via API) lacks a usable
  grouping variable at sufficient depth (vendor identifiers were not pulled, and
  the API sample is too small per vendor for aggregation statistics). The
  transaction-to-vendor aggregation test requires the bulk transaction archive
  (same barrier as Demo 2). Documented rather than forced.

## General
- All downloads over HTTPS on 2026-05-24 from the public endpoints listed above.
- Local cached inputs: data/wp6/chr21.gff3.gz, data/wp6/procurement_dod_fy2023.csv.
- Reproduce: python3 src/wp6_genome.py ; python3 src/wp6_procurement.py ;
  python3 src/wp6_cross.py.

## WP7 Demonstration: Procurement BULK (real threshold absorption + aggregation)
- Source: USAspending award_data_archive (S3), accessed 2026-05-24. Two agency files:
  - FY2023_089_Contracts_Full_20260506.zip (DoE), 5.87 MB: only 4,696 txns in
    $100..$500k (large-contract agency) -> $10k/$25k threshold bands <200, so the
    threshold test was not feasible on DoE; vendor aggregation ran.
  - FY2023_014_Contracts_Full_20260506.zip (Interior), 19 MB, 51,453 txns
    (25,264 in $100..$500k) -- adequate near-threshold density; used for the
    threshold absorption test (the on-disk wp7_procurement_bulk_* CSVs are Interior).
- Transaction-level federal_action_obligation -> integer cents.
- Prop 8 (REAL): at $10k (=10^6 cents, power of 10) magnitude_matched ABSORBS
  (z=-1.3, verdict 0) while parent_support DETECTS (z=-3.5, +); at $250k (SAT, not
  power of 10) threshold_aware absorbs (z=0.1) while magnitude_matched/parent
  detect the weak SAT bunching (z=2.3). Baseline absorption replicated on real data.
- Prop 6 (REAL): vendor-total aggregation (recipient_uei) attenuates round-number
  signatures but they persist (e.g. TerminalDigit |z| 19809 -> 5965).
- Citations (others): Liebman & Mahoney (2017); FAR Part 13; Kleven & Waseem (2013).
  No self-citations.
