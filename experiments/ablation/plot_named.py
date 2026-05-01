"""Tier 2 — AARV figure (paper Eq. 13 extended to single neurons / coalitions)."""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
art = os.path.join(config["result_path"], "ablation")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

C_M1E9   = "#d62728"
C_M2E30  = "#2ca02c"
C_M4E14  = "#1f77b4"
C_M14E60 = "#9467bd"

data = torch.load(os.path.join(art, "aarv.pt"))
aarv = data["aarv"].numpy()           # [n_targets, L]
labels = data["labels"]
ablations = data["ablations"]
N_LAYERS = aarv.shape[1]

idx = {lab: i for i, lab in enumerate(labels)}

# ============================================================
fig, axA = plt.subplots(figsize=(6.0, 4.2))

def aarv_curve(label):
    a = aarv[idx[label]]
    c = ablations[idx[label]]["c"]
    ls = np.arange(c + 1, N_LAYERS)
    return ls, a[c + 1:]

# Whole-expert ablations: solid lines, filled markers.
# Dominant-neuron ablations: dashed lines, hollow markers.
# Coalition: dash-dot.

PLOTS = [
    # (label,                  name_in_legend,         color,     marker, linestyle, fillstyle)
    ("A1_M1E9_whole",          "M1E9 (whole expert)",       C_M1E9,    "^",    "-",       "full"),
    ("A2_M1E9_z915",           "M1E9 (z=915)",              C_M1E9,    "^",    (0, (5,2)),"none"),
    ("A3_M2E30_whole",         "M2E30 (whole expert)",      C_M2E30,   "s",    "-",       "full"),
    ("A4_M2E30_z742",          "M2E30 (z=742)",             C_M2E30,   "s",    (0, (5,2)),"none"),
    ("A5_M4E14_whole",         "M4E14 (whole expert)",      C_M4E14,   "p",    "-",       "full"),
    ("A6_M4E14_z391",          "M4E14 (z=391)",             C_M4E14,   "p",    (0, (5,2)),"none"),
    ("A7_M4E14_coalition",     "M4E14 (z=391, 336, 956)",   C_M4E14,   "p",  (0,(3,1.5,1,1.5)), "left"),
    ("C2_M14E60_whole",        "M14E60 (whole expert)",     C_M14E60,  "D",    "-",       "full"),
    ("C3_M14E60_z357",         "M14E60 (z=357)",            C_M14E60,  "D",    (0, (5,2)),"none"),
]

for (lab, leg, col, mk, ls_style, fs) in PLOTS:
    xs, ys = aarv_curve(lab)
    axA.plot(xs, ys, color=col, marker=mk, linestyle=ls_style,
             markersize=6, lw=1.6, label=leg, fillstyle=fs,
             markeredgewidth=1.0)

axA.set_xlabel("Receiving Layer")
axA.set_ylabel("AARV")
axA.set_xticks([0, 5, 10, 15])
axA.set_xlim(-0.5, N_LAYERS - 0.5)
axA.set_ylim(top=12)
axA.grid(True, alpha=0.3, lw=0.4, zorder=0)
axA.set_axisbelow(True)
axA.legend(loc="upper right", framealpha=0.95, ncol=2, fontsize=8)

fig.tight_layout()

out_pdf = os.path.join(art, "tier2_aarv.pdf")
out_png = os.path.join(art, "tier2_aarv.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
fig.savefig(out_png, format="png", bbox_inches="tight", dpi=180)
plt.close(fig)

print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
