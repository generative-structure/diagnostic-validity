"""Figure 2: mechanism recovery and ablation.
Run: python figures/fig1_recovery.py  ->  figures/fig1_recovery.pdf
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

LABELS = {"a_digit_decimal": "Digit/decimal", "b_vocab": "Vocabulary",
          "c_threshold_temporal": "Threshold/temporal", "d_msa": "Arithmetic",
          "e_cbad": "Compression", "f_all": "All projections",
          "g_all_plus_prime": "All + prime frac."}

rec = list(csv.DictReader(open(os.path.join(RES, "wp7_mechanism_recovery.csv"))))
abl = list(csv.DictReader(open(os.path.join(RES, "wp7_mechanism_recovery_ablation.csv"))))

rows = sorted(((LABELS.get(r["feature_set"], r["feature_set"]), float(r["macro_f1"]))
               for r in rec), key=lambda t: t[1])
names = [r[0] for r in rows]; vals = [r[1] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
colors = ["#3b6ea5" if v < 0.999 else "#2e7d32" for v in vals]
ax1.barh(names, vals, color=colors)
ax1.set_xlim(0, 1.05); ax1.set_xlabel("macro-$F_1$")
ax1.set_title("A. Recovery by feature set", loc="left", fontweight="bold")
for i, v in enumerate(vals):
    ax1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

afam = [("Digit/decimal" if a["removed_family"] == "digit_decimal" else
         "Vocabulary" if a["removed_family"] == "vocab" else
         "Threshold/temporal" if a["removed_family"] == "threshold_temporal" else
         "Arithmetic" if a["removed_family"] == "msa" else "Compression",
         float(a["delta_f1"])) for a in abl]
afam.sort(key=lambda t: t[1])
an = [a[0] for a in afam]; av = [a[1] for a in afam]
acol = ["#c62828" if v == max(av) and v > 0 else "#9e9e9e" for v in av]
ax2.barh(an, av, color=acol)
ax2.set_xlabel(r"$\Delta$ macro-$F_1$ when family removed")
ax2.set_title("B. Ablation loss", loc="left", fontweight="bold")
for i, v in enumerate(av):
    ax2.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_recovery.pdf"))
print("wrote fig1_recovery.pdf")
