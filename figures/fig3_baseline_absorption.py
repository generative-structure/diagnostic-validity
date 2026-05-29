"""Figure 4: baseline absorption matrix (the paper's signature figure).
Run: python figures/fig3_baseline_absorption.py  ->  figures/fig3_baseline_absorption.pdf
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight"})

# columns: construction-neutral, magnitude-matched, mechanism-aware
COLS = ["Construction-\nneutral", "Magnitude-\nmatched", "Mechanism-\naware"]
ROWS = ["Threshold $10$k\n(boundary)", "Threshold $25$k", "Rounding", "Fiscal", "Divisibility"]

# verdict grid (+ detected / 0 absorbed / ? uncertain) and optional z annotations
grid = [["?"] * 3 for _ in ROWS]
ann = [[""] * 3 for _ in ROWS]

# threshold rows from threshold_summary.csv
ts = list(csv.DictReader(open(os.path.join(RES, "threshold_summary.csv"))))
bmap = {"parent_support": 0, "magnitude_matched": 1, "threshold_aware": 2}
for r in ts:
    T = r["threshold"]
    ri = 0 if T == "10000" else 1 if T == "25000" else None
    if ri is None:
        continue
    ci = bmap[r["baseline_type"]]
    grid[ri][ci] = r["ThresholdDensity_verdict"]
    try:
        ann[ri][ci] = f"z={float(r['ThresholdDensity_z']):.1f}"
    except ValueError:
        ann[ri][ci] = ""

# generalized rows from summary (focus_survived empty => absorbed)
gs = list(csv.DictReader(open(os.path.join(RES, "wp7_baseline_absorption_generalized_summary.csv"))))
rowmap = {"round": 2, "fiscal": 3, "divisibility": 4}
neutralmap = {"cents_neutral", "temporal_neutral", "divisibility_neutral"}
for r in gs:
    ri = rowmap.get(r["mechanism"])
    if ri is None:
        continue
    bt = r["baseline_type"]
    ci = 0 if bt in neutralmap else 1 if bt == "magnitude_matched" else 2
    survived = r["focus_survived"] != "none"
    grid[ri][ci] = "+" if survived else "0"

cmap = {"+": "#2e7d32", "0": "#c62828", "?": "#f9a825"}
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for i in range(len(ROWS)):
    for j in range(3):
        v = grid[i][j]
        ax.add_patch(plt.Rectangle((j, len(ROWS) - 1 - i), 1, 1,
                                    facecolor=cmap.get(v, "#cccccc"), edgecolor="white", lw=2))
        label = {"+": "detected", "0": "absorbed", "?": "uncertain"}[v]
        ax.text(j + 0.5, len(ROWS) - 1 - i + 0.60, label, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        if ann[i][j]:
            ax.text(j + 0.5, len(ROWS) - 1 - i + 0.32, ann[i][j], ha="center", va="center",
                    color="white", fontsize=7.5)
ax.set_xlim(0, 3); ax.set_ylim(0, len(ROWS))
ax.set_xticks(np.arange(3) + 0.5); ax.set_xticklabels(COLS)
ax.set_yticks(np.arange(len(ROWS)) + 0.5); ax.set_yticklabels(ROWS[::-1])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Baseline absorption: the same mechanism, read differently by different nulls",
             loc="left", fontweight="bold", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_baseline_absorption.pdf"))
print("wrote fig3_baseline_absorption.pdf")
