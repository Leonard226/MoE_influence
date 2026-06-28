"""Generate the LOO feature-ablation figure for the paper.

Reads:  results/circuits/feature_ablation_logact_logload/loo_summary.json
Writes: 6a154a47401c9f4881c67a3f/figures/feature_ablation.pdf

Two-panel design: within-family similarity on the LEFT, cross-family
similarity on the RIGHT, same y-axis on both. Bars are coloured by Q;
the "full" baseline mean is drawn as a dashed horizontal reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "circuits" / "feature_ablation_logact_logload" / "loo_summary.json"
FIG_DIR = ROOT / "6a154a47401c9f4881c67a3f" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG = FIG_DIR / "feature_ablation.pdf"

ABLATIONS = ["full", "no_depth", "no_out", "no_in", "no_load", "no_act", "no_class"]
DISPLAY = {
    "full":     "full",
    "no_depth": r"$-$depth",
    "no_out":   r"$-$out",
    "no_in":    r"$-$in",
    "no_load":  r"$-$load",
    "no_act":   r"$-$act",
    "no_class": r"$-$class",
}
QUANTILES = [0.9, 0.99, 0.999]
Q_COLORS = ["#4C72B0", "#DD8452", "#55A467"]   # blue / orange / green

with open(JSON) as f:
    summary = json.load(f)

within = np.array([
    [summary[a][f"Q={q}"]["within_mean"] for q in QUANTILES] for a in ABLATIONS
])
cross = np.array([
    [summary[a][f"Q={q}"]["cross_mean"] for q in QUANTILES] for a in ABLATIONS
])

full_within = within[0]   # (3,) for Q axis
full_cross  = cross[0]

x = np.arange(len(ABLATIONS))
width = 0.27
y_max = 1.0

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

for panel_idx, (ax, vals, panel_title, full_ref) in enumerate(zip(
    axes,
    [within, cross],
    [r"Within-family $\mathcal{S}_{\mathrm{within}}$",
     r"Cross-family $\mathcal{S}_{\mathrm{cross}}$"],
    [full_within, full_cross],
)):
    for q_i, (q, col) in enumerate(zip(QUANTILES, Q_COLORS)):
        offset = (q_i - 1) * width
        ax.bar(
            x + offset, vals[:, q_i], width,
            label=f"$Q = {q}$" if panel_idx == 0 else None,
            color=col, edgecolor="black", linewidth=0.5,
        )
    # Reference baseline = mean of full bars (across Q) for this panel.
    ref = float(np.mean(full_ref))
    ax.axhline(y=ref, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="full baseline (mean)" if panel_idx == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[a] for a in ABLATIONS], fontsize=11,
                       fontweight="bold")
    ax.set_title(panel_title, fontsize=13, pad=8, fontweight="bold")
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_ylim(0, y_max)

axes[0].set_ylabel(r"Similarity $\mathcal{S}$", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right", framealpha=0.95, fontsize=10)

plt.tight_layout()
plt.savefig(FIG, format="pdf", bbox_inches="tight")
print(f"Saved: {FIG}")

# Diagnostic dump for the paper.
print("\nDeltas vs full per ablation:")
print(f"{'ablation':<10s}  {'Q':>6s}  {'within Δ':>10s}  {'cross Δ':>10s}  "
      f"{'cross/within':>14s}")
for a in ABLATIONS:
    for qi, q in enumerate(QUANTILES):
        dw = summary[a][f"Q={q}"]["within_mean"] - summary["full"][f"Q={q}"]["within_mean"]
        dc = summary[a][f"Q={q}"]["cross_mean"]  - summary["full"][f"Q={q}"]["cross_mean"]
        ratio = (dc / dw) if abs(dw) > 1e-6 else float("nan")
        print(f"{a:<10s}  {q:>6.3f}  {dw:>+10.4f}  {dc:>+10.4f}  {ratio:>14.2f}")
    print()
