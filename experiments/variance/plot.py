"""Tier 1 figure — three-panel narrative:
  (a) per-expert routing influence by receiving layer  (paper-faithful)
  (b) cumulative fraction of within-expert per-neuron variance by top-K
  (c) per-neuron routing influence by receiving layer  (the dominant neurons)
"""
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
art = os.path.join(config["result_path"], "variance")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

C_M1E9   = "#d62728"  # red    (paper: M1E9   ~ red circle)
C_M2E30  = "#2ca02c"  # green  (paper: M2E30  ~ green triangle)
C_M4E14  = "#1f77b4"  # blue   (paper: M4E14  ~ blue square)
C_M14E60 = "#9467bd"  # purple (paper: M11E33 ~ purple diamond -- our 4th slot is M14E60)
C_RAND   = "#cccccc"

# Paper-faithful per-expert variance: score_variance[S, J, R]
score_variance = torch.load(os.path.join(art, "score_variance_per_expert.pt")).numpy()
# Per-neuron variance: T_dyn[S, J, z, R]
T_dyn = torch.load(os.path.join(art, "T_dyn.pt"))
N_LAYERS, N_EXPERTS, D_FFN, _ = T_dyn.shape
neuron_score = T_dyn.sum(dim=3).numpy()  # [S, J, z]  (sum over R, used for ranking neurons within expert)

NAMED_EXPERTS = {
    "M1E9":   (1,  9,  C_M1E9,   "^"),  # red    triangle
    "M2E30":  (2, 30,  C_M2E30,  "s"),  # green  square
    "M4E14":  (4, 14,  C_M4E14,  "p"),  # blue   pentagon
    "M14E60": (14, 60, C_M14E60, "D"),  # purple diamond
}

# Three blue shades for M4E14's three top neurons (dark → light by descending rank).
M4E14_SHADES = ["#1f77b4", "#5e9bcf", "#9bbedb"]

# Top neuron(s) of each named expert. M4E14 → top 3 (super-coalition).
NAMED_NEURONS = []
for name, (c, j, col, marker) in NAMED_EXPERTS.items():
    sorted_idx = np.argsort(neuron_score[c, j])[::-1]
    if name == "M4E14":
        for z, shade in zip(sorted_idx[:3], M4E14_SHADES):
            z_int = int(z)
            NAMED_NEURONS.append((f"{name}, z={z_int}", (c, j, z_int), shade, marker))
    else:
        z_top = int(sorted_idx[0])
        NAMED_NEURONS.append((f"{name}, z={z_top}", (c, j, z_top), col, marker))
named_neuron_set = {coord for _, coord, _, _ in NAMED_NEURONS}

# Shared y-axis range for panels (a) and (c) — fixed 0..6 in steps of 1.0.
Y_MIN = -0.25
Y_MAX = 6.0

# ============================================================
fig, (axA, axB, axC) = plt.subplots(
    1, 3,
    figsize=(15.5, 4.2),
    gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.42},
)

# ---------- Panel A: per-expert variance by receiving layer ----------
named_set = {(c, j) for _, (c, j, _, _) in NAMED_EXPERTS.items()}
for c in range(N_LAYERS):
    for j in range(N_EXPERTS):
        if (c, j) in named_set:
            continue
        for l in range(c + 1, N_LAYERS):
            v = score_variance[c, j, l]
            if v > 0:
                axA.scatter(l, v, color=C_RAND, alpha=0.35, s=6,
                            edgecolor="none", zorder=1)

for name, (c, j, col, marker) in NAMED_EXPERTS.items():
    ls = list(range(c + 1, N_LAYERS))
    vals = [score_variance[c, j, l] for l in ls]
    axA.scatter(ls, vals, color=col, marker=marker,
                s=42, label=name, zorder=3, edgecolor=col, linewidth=0)

axA.scatter([], [], color=C_RAND, alpha=0.35, s=6, label="Other Experts")
axA.set_xlabel("Receiving Layer")
axA.set_ylabel("Variance")
axA.set_title("(a)  Per-expert routing influence by receiving layer")
axA.set_xticks([0, 5, 10, 15])
axA.set_xlim(-0.5, N_LAYERS - 0.5)
axA.set_yticks(np.arange(0, 7, 1))
axA.set_ylim(Y_MIN, Y_MAX)
axA.grid(True, alpha=0.3, lw=0.4, zorder=0)
axA.set_axisbelow(True)
axA.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

# ---------- Panel B: cumulative top-K within-expert variance fraction ----------
named_set = {(c, j) for _, (c, j, _, _) in NAMED_EXPERTS.items()}
KMAX = 20

def cum_frac(c, j, kmax=KMAX):
    arr = np.sort(neuron_score[c, j])[::-1]
    total = arr.sum()
    cum = np.cumsum(arr)
    return np.arange(1, kmax + 1), cum[:kmax] / max(total, 1e-30)

# Top-1 (or top-3 for M4E14) within each named expert.
TOP_NEURONS_FOR = {}
for name, (c, j, _, _) in NAMED_EXPERTS.items():
    sorted_idx = np.argsort(neuron_score[c, j])[::-1]
    k = 3 if name == "M4E14" else 1
    TOP_NEURONS_FOR[name] = [int(z) for z in sorted_idx[:k]]

for name, (c, j, col, marker) in NAMED_EXPERTS.items():
    k, frac = cum_frac(c, j)
    axB.plot(k, frac, marker="o", color=col, lw=2.0, ms=4.5,
             label=name, zorder=3)

# Random expert baseline (mean of 4).
rng = np.random.default_rng(42)
rand_experts = []
while len(rand_experts) < 4:
    cc = int(rng.integers(0, N_LAYERS - 1))
    jj = int(rng.integers(0, N_EXPERTS))
    if (cc, jj) in named_set or (cc, jj) in rand_experts:
        continue
    if neuron_score[cc, jj].sum() <= 0:
        continue
    rand_experts.append((cc, jj))

rand_curves = np.stack([cum_frac(c, j)[1] for c, j in rand_experts])
ks = np.arange(1, KMAX + 1)
axB.plot(ks, rand_curves.mean(axis=0), marker="o", color=C_RAND, lw=1.0, ms=3,
         label="random expert (avg)", zorder=2)
axB.fill_between(ks, rand_curves.min(axis=0), rand_curves.max(axis=0),
                 color=C_RAND, alpha=0.35, lw=0)

# Annotate the dominant neuron(s) on each curve. Just z indices, no variance text.
ANNOTATE_AT = {
    "M1E9":   {"xytext": (3.0, 0.93)},
    "M2E30":  {"xytext": (3.0, 0.83)},
    "M4E14":  {"xytext": (5.5, 0.73)},
    "M14E60": {"xytext": (3.0, 0.34)},
}
for name, (c, j, col, _) in NAMED_EXPERTS.items():
    z_list = TOP_NEURONS_FOR[name]
    K_lbl = len(z_list)
    cum_at_K = sum(neuron_score[c, j, z] for z in z_list) / neuron_score[c, j].sum()
    label = ", ".join(f"z={z}" for z in z_list)
    axB.annotate(label,
                 xy=(K_lbl, cum_at_K),
                 xytext=ANNOTATE_AT[name]["xytext"],
                 fontsize=8.5, color=col, fontweight="bold",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

axB.set_xlabel("Number of top neurons K")
axB.set_ylabel("Fraction of expert's variance")
axB.set_title("(b)  Within-expert variance, captured by top-$K$ neurons")
axB.set_xlim(0.5, KMAX + 0.5)
axB.set_ylim(0, 1.04)
axB.set_xticks([1, 5, 10, 15, 20])
axB.legend(loc="center right", framealpha=0.95, bbox_to_anchor=(1.0, 0.42), fontsize=8.5)
axB.grid(True, axis="y", alpha=0.3, lw=0.4)
axB.set_axisbelow(True)

# ---------- Panel C: per-neuron routing influence by receiving layer ----------
# 500 random non-named neurons for the gray cloud.
N_RAND_NEURONS = 500
rng2 = np.random.default_rng(123)
rand_neurons = []
while len(rand_neurons) < N_RAND_NEURONS:
    c_n = int(rng2.integers(0, N_LAYERS - 1))
    j_n = int(rng2.integers(0, N_EXPERTS))
    z_n = int(rng2.integers(0, D_FFN))
    if (c_n, j_n, z_n) in named_neuron_set:
        continue
    if T_dyn[c_n, j_n, z_n, :].sum().item() <= 0:
        continue
    rand_neurons.append((c_n, j_n, z_n))

for (c_n, j_n, z_n) in rand_neurons:
    for l in range(c_n + 1, N_LAYERS):
        v = T_dyn[c_n, j_n, z_n, l].item()
        if v > 0:
            axC.scatter(l, v, color=C_RAND, alpha=0.25, s=5,
                        edgecolor="none", zorder=1)

for name, (c, j, z), col, marker in NAMED_NEURONS:
    ls = list(range(c + 1, N_LAYERS))
    vals = [T_dyn[c, j, z, l].item() for l in ls]
    axC.scatter(ls, vals, color=col, marker=marker,
                s=44, label=name, zorder=3, edgecolor=col, linewidth=0)

axC.scatter([], [], color=C_RAND, alpha=0.45, s=8, label="Other Neurons")
axC.set_xlabel("Receiving Layer")
axC.set_ylabel("Variance")
axC.set_title("(c)  Per-neuron routing influence by receiving layer")
axC.set_xticks([0, 5, 10, 15])
axC.set_xlim(-0.5, N_LAYERS - 0.5)
axC.set_yticks(np.arange(0, 7, 1))
axC.set_ylim(Y_MIN, Y_MAX)
axC.grid(True, alpha=0.3, lw=0.4, zorder=0)
axC.set_axisbelow(True)
axC.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

fig.tight_layout()

out_pdf = os.path.join(art, "tier1_super_neurons.pdf")
out_png = os.path.join(art, "tier1_super_neurons.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
fig.savefig(out_png, format="png", bbox_inches="tight", dpi=180)
plt.close(fig)

print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
print()
print(f"Y-axis shared between (a) and (c): [{Y_MIN:.3f}, {Y_MAX:.3f}]")
print(f"Top neurons identified per expert: {TOP_NEURONS_FOR}")
