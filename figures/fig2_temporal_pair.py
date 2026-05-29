"""Figure 3: projection specificity (temporal pair).
Run: python figures/fig2_temporal_pair.py  ->  figures/fig2_temporal_pair.pdf
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

COLS = ["CBAD_R2", "cp_spectrum", "L_profile", "TailMass", "TerminalDigit", "CentsEntropy"]
MECHS = ["temporal_concat", "temporal_concat_arith"]
NICE = {"temporal_concat": "Temporal concat.\n(compressibility shift)",
        "temporal_concat_arith": "Arithmetic concat.\n(arithmetic shift)"}

rows = list(csv.DictReader(open(os.path.join(RES, "signature_matrix_resolved.csv"))))
data = {(r["mechanism"], r["projection"]): r for r in rows}


def zval(m, c):
    try:
        return float(data[(m, c)]["z_score"])
    except (KeyError, ValueError):
        return float("nan")


def verd(m, c):
    return data.get((m, c), {}).get("verdict", "?")


fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(len(COLS)); w = 0.38
for i, m in enumerate(MECHS):
    zs = [zval(m, c) for c in COLS]
    # clip for display; annotate verdict
    disp = [np.sign(z) * min(abs(z), 30) if np.isfinite(z) else 0 for z in zs]
    cols = ["#2e7d32" if verd(m, c) == "+" else
            "#bdbdbd" if verd(m, c) == "0" else "#f9a825" for c in COLS]
    bars = ax.bar(x + (i - 0.5) * w, disp, w, color=cols,
                  edgecolor="#333", linewidth=0.5,
                  label=NICE[m].replace("\n", " "))
    for xi, (b, c) in enumerate(zip(bars, COLS)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (1 if b.get_height() >= 0 else -2.5),
                verd(m, c), ha="center", fontsize=8)
ax.axhline(0, color="#333", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(COLS, rotation=20, ha="right")
ax.set_ylabel("signed $z$ (clipped at $\\pm$30)")
ax.set_title("Projection specificity: the same regime change is seen by one projection, missed by another",
             loc="left", fontweight="bold", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig2_temporal_pair.pdf"))
print("wrote fig2_temporal_pair.pdf")
