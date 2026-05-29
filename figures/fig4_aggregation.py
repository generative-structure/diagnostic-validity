"""Figure 5: aggregation transformation (preserved vs attenuated signatures).
Run: python figures/fig4_aggregation.py  ->  figures/fig4_aggregation.pdf
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

rows = list(csv.DictReader(open(os.path.join(RES, "aggregation_decay.csv"))))


def series(base, proj):
    pts = sorted(((int(r["m"]), abs(float(r["z_score"]))) for r in rows
                  if r["base_mechanism"] == base and r["projection"] == proj),
                 key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts]

curves = [
    ("divisibility", "cp_spectrum", "Divisibility $c_p$ (preserved)", "#2e7d32", "-", "o"),
    ("round", "CentsEntropy", "Rounding cents (preserved)", "#1565c0", "-", "s"),
    ("divisibility", "L_profile", "Divisibility L-profile (attenuated)", "#c62828", "--", "^"),
    ("divisibility", "TailMass", "Divisibility tail mass (attenuated)", "#ef6c00", "--", "v"),
]

fig, ax = plt.subplots(figsize=(6.6, 3.6))
for base, proj, lab, col, ls, mk in curves:
    m, z = series(base, proj)
    if m:
        ax.plot(m, z, ls, color=col, marker=mk, label=lab, markersize=4)
ax.axhline(3, color="#888", linestyle=":", linewidth=1)
ax.text(50, 3.6, "detection threshold $|z|=3$", ha="right", fontsize=7.5, color="#666")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("aggregation depth $m$"); ax.set_ylabel("$|z|$ (log scale)")
ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels([1, 2, 5, 10, 20, 50])
ax.set_title("Aggregation transforms signatures: closure-preserved vs multiplicative",
             loc="left", fontweight="bold", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig4_aggregation.pdf"))
print("wrote fig4_aggregation.pdf")
