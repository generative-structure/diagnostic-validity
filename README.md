# Diagnostic Validity for Constructed Data

Replication package for "Diagnostic Validity for Constructed Data:
Mechanisms, Projections, and Baselines."

## Structure

- `CANONICAL_DEFINITIONS.md` — Single source of truth for all metric formulas
- `src/` — Synthetic construction engine, projection battery, and analysis scripts
- `results/` — All output CSVs from the synthetic engine and real-data analyses
- `figures/` — Python scripts to generate publication figures
- `signature_matrix_rationale.csv` — Pre-registration of signature matrix predictions
- `wp6_data_provenance.md` — Real-data sources, access dates, and processing steps

## Requirements

- Python 3.8+
- numpy, scipy, matplotlib, scikit-learn
- zlib (standard library)

## Reproduction

```bash
# Full synthetic engine (14 mechanisms x 13 projections x 20 seeds)
python3 src/run_experiments.py

# Threshold baseline absorption experiment
python3 src/threshold_experiment.py

# Real-data demonstrations
python3 src/wp6_genome.py
python3 src/wp6_procurement.py

# Validation closure experiments
python3 src/wp7_mechanism_recovery.py
python3 src/wp7_baseline_generalized.py
python3 src/wp7_missing_dimension.py
python3 src/wp7_power_materiality.py
python3 src/wp7_ml_comparators.py
python3 src/wp7_genome_replication.py
python3 src/wp7_procurement_bulk.py

# End-to-end broker retrospective (decision-state outcomes on real procurement data)
python3 src/wp10_broker_retrospective.py

# Figures
python3 figures/fig0_theory_schematic.py
python3 figures/fig1_recovery.py
python3 figures/fig2_temporal_pair.py
python3 figures/fig3_baseline_absorption.py
python3 figures/fig4_aggregation.py
python3 figures/fig5_realdata.py
```

## Data

Genome annotations: Ensembl GRCh38 release 115
(https://ftp.ensembl.org/pub/current_gff3/homo_sapiens/)

Federal procurement: USAspending.gov award data archive
(https://files.usaspending.gov/award_data_archive/)

See `wp6_data_provenance.md` for complete provenance.

## License

MIT
