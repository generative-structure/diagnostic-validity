"""Figure 6: real-data triangulation.
(A) genome divisibility-by-3 by chromosome/feature; (B) procurement baseline absorption.
Run: python figures/fig5_realdata.py  ->  figures/fig5_realdata.pdf
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.4))

# Panel A: genome
grows = list(csv.DictReader(open(os.path.join(RES, "wp7_genome_by_chromosome.csv"))))
chroms = ["1", "7", "19", "21", "22"]
feats = ["CDS", "intron", "UTR"]
fcol = {"CDS": "#2e7d32", "intron": "#9e9e9e", "UTR": "#bdbdbd"}
g = {(r["chromosome"], r["feature_type"]): float(r["frac_div3"]) for r in grows}
x = np.arange(len(chroms)); w = 0.26
for i, f in enumerate(feats):
    vals = [g.get((c, f), np.nan) for c in chroms]
    axA.bar(x + (i - 1) * w, vals, w, label=f, color=fcol[f], edgecolor="#333", lw=0.4)
axA.axhline(1 / 3, color="#888", linestyle=":", linewidth=1)
axA.text(len(chroms) - 1, 0.36, "uniform $1/3$", ha="right", fontsize=7.5, color="#666")
axA.set_xticks(x); axA.set_xticklabels([f"chr{c}" for c in chroms])
axA.set_ylabel("fraction length divisible by 3"); axA.set_ylim(0, 1.0)
axA.set_title("A. Genome: codon constraint", loc="left", fontweight="bold")
axA.legend(frameon=False, fontsize=7.5)

# Panel B: procurement baseline absorption
prows = list(csv.DictReader(open(os.path.join(RES, "wp7_procurement_bulk_baseline_absorption.csv"))))
groups = [("10000", "magnitude_matched", "$10$k\nmagnitude-matched"),
          ("10000", "parent_support", "$10$k\nparent-support"),
          ("250000", "magnitude_matched", "$250$k\nmagnitude-matched")]
labels, zs, vcol = [], [], []
vmap = {"+": "#2e7d32", "0": "#c62828", "?": "#f9a825"}
for T, bt, lab in groups:
    r = next((x for x in prows if x["threshold_usd"] == T and x["baseline"] == bt), None)
    labels.append(lab)
    zs.append(abs(float(r["z"])) if r and r["z"] not in ("nan", "") else 0.0)
    vcol.append(vmap.get(r["verdict"], "#ccc") if r else "#ccc")
axB.bar(range(len(labels)), zs, color=vcol, edgecolor="#333", lw=0.5)
axB.axhline(3, color="#888", linestyle=":", linewidth=1)
axB.text(len(labels) - 1, 3.2, "$|z|=3$", ha="right", fontsize=7.5, color="#666")
axB.set_xticks(range(len(labels))); axB.set_xticklabels(labels, fontsize=8)
axB.set_ylabel("$|z|$ threshold density")
axB.set_title("B. Procurement: real baseline absorption", loc="left", fontweight="bold")
for i, (z, r) in enumerate(zip(zs, [next((x for x in prows if x["threshold_usd"] == T and x["baseline"] == bt), {}) for T, bt, _ in groups])):
    axB.text(i, z + 0.1, r.get("verdict", ""), ha="center", fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig5_realdata.pdf"))
print("wrote fig5_realdata.pdf")
