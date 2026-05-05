"""Paper Fig 6 analog at the per-neuron level.

For each named expert (c, j): plot AARV vs neuron index z, with a few
representative receiving layers as different markers.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D

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

# Strong, distinct layer colors (red, blue, green, plus purple if a fourth is needed).
LAYER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
LAYER_MARKERS = ["o", "s", "^", "D"]

D_FFN = 1024
N_LAYERS = 16
TOP_K = 8

results = torch.load(os.path.join(art, "per_neuron_aarv.pt"))

# Layout: 2x2 with one panel per expert; shared y-axis for direct comparison.
fig, axs = plt.subplots(
    2, 2,
    figsize=(13, 7.2),
    sharey=True,
    gridspec_kw={"wspace": 0.10, "hspace": 0.40},
)

PANEL_LAYOUT = [
    ("M1E9",   axs[0, 0], C_M1E9,   [5, 10, 15]),
    ("M2E30",  axs[0, 1], C_M2E30,  [5, 10, 15]),
    ("M4E14",  axs[1, 0], C_M4E14,  [8, 12, 15]),
    ("M14E60", axs[1, 1], C_M14E60, [15]),
]

# Neurons we have informally identified as super-neurons.
SUPER_NEURONS = {
    "M1E9":   [915],
    "M2E30":  [742],
    "M4E14":  [391, 336, 956],
    "M14E60": [],
}

# Global y-range so all panels share the same scale.
global_max = max(results[name]["aarv"].numpy().max() for name, *_ in PANEL_LAYOUT)

for name, ax, expert_color, recv_layers in PANEL_LAYOUT:
    info = results[name]
    aarv = info["aarv"].numpy()        # [d_ffn, L]
    c, j = info["c"], info["j"]
    zs = np.arange(D_FFN)

    valid_ls = [l for l in recv_layers if l > c]
    panel_max = max([aarv[:, l].max() for l in valid_ls] + [0.1])
    # Hide near-zero baseline so the peaks stand out without a dense band of dots near y = 0.
    thr = max(0.1, 0.03 * panel_max)

    legend_handles = []
    for k, l in enumerate(recv_layers):
        if l <= c:
            continue
        col = LAYER_COLORS[k]
        mk = LAYER_MARKERS[k]
        mask = aarv[:, l] > thr
        ax.scatter(zs[mask], aarv[mask, l], color=col, marker=mk, s=24, alpha=0.95,
                   edgecolor="none", zorder=3)
        legend_handles.append(
            Line2D([0], [0], color=col, marker=mk, markersize=6,
                   linestyle="None", label=f"M{l}")
        )

    # Annotate only the informally-identified super-neurons.
    for z in SUPER_NEURONS[name]:
        best_l = max(valid_ls, key=lambda l: aarv[z, l])
        v = aarv[z, best_l]
        ax.annotate(f"$z={z}$",
                    xy=(z, v),
                    xytext=(z + 60, v),
                    fontsize=8.5, color="black",
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    ax.set_xlabel(f"Neurons in {name}")
    ax.set_ylabel("AARV")
    # All four named experts are high-variance experts; the first three concentrate the
    # variance in a small set of neurons, M14E60 distributes it across many.
    if name == "M14E60":
        suffix = " (high-variance, distributed)"
    else:
        suffix = " (high-variance, concentrated)"
    ax.set_title(f"{name}{suffix}")
    ax.set_xlim(-10, D_FFN + 10)
    ax.set_ylim(bottom=-0.05 * global_max, top=1.05 * global_max)
    ax.grid(True, alpha=0.3, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.95, title="Receiving Layer")

out_pdf = os.path.join(art, "tier2_per_neuron_aarv.pdf")
out_png = os.path.join(art, "tier2_per_neuron_aarv.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
fig.savefig(out_png, format="png", bbox_inches="tight", dpi=180)
plt.close(fig)

print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
