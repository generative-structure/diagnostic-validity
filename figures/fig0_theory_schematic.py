"""Figure 1 (theory schematic): construction pipeline (A) and diagnostic framework (B).
Run: python figures/fig0_theory_schematic.py  ->  figures/fig0_theory_schematic.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight"})

BOX = dict(boxstyle="round,pad=0.3", linewidth=1.0, edgecolor="#444444")
FILL_A = "#e8eef5"
FILL_B = "#eef3ea"
WHITE = "#ffffff"


def box(ax, x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, **BOX, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, color="#666666", linewidth=1.0))


fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.4, 6.6))

# ---- Panel A: construction pipeline ----
axA.set_title("A. Construction pipeline", loc="left", fontweight="bold")
labels = ["Latent\nevent", "Unitization", "Integer-\nization", "Constraint",
          "Composition", "Aggregation", "Recorded\ndata"]
w, h, gap = 1.18, 0.95, 0.34
x = 0.1
for i, lab in enumerate(labels):
    box(axA, x, 0.7, w, h, lab, FILL_A)
    if i < len(labels) - 1:
        arrow(axA, x + w, 0.7 + h / 2, x + w + gap, 0.7 + h / 2)
    x += w + gap
axA.set_xlim(0, x); axA.set_ylim(0, 2.5); axA.axis("off")

# ---- Panel B: diagnostic framework ----
axB.set_title("B. Diagnostic framework", loc="left", fontweight="bold")
flow = ["Recorded\ndata", "Projection\nbattery", "Baseline\nselection",
        "Signature", "Diagnostic\nclaim"]
w2, h2, gap2 = 1.35, 0.95, 0.45
x = 0.1
xpos = {}
for i, lab in enumerate(flow):
    xpos[lab] = x
    box(axB, x, 1.7, w2, h2, lab, FILL_B)
    if i < len(flow) - 1:
        arrow(axB, x + w2, 1.7 + h2 / 2, x + w2 + gap2, 1.7 + h2 / 2)
    x += w2 + gap2

# sub-branches from Projection battery
families = ["Digits", "Residues", "Arithmetic", "Compression",
            "Temporal", "Threshold", "Frequency"]
pbx = xpos["Projection\nbattery"] + w2 / 2
y = 3.9
for fam in families:
    box(axB, pbx - 0.7, y, 1.4, 0.40, fam, WHITE, fs=7.5)
    arrow(axB, pbx, 1.7 + h2, pbx - 0.0, y)
    y += 0.52
axB.set_xlim(0, x); axB.set_ylim(1.4, 7.8); axB.axis("off")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig0_theory_schematic.pdf"))
print("wrote fig0_theory_schematic.pdf")
