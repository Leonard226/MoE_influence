"""Generate the LOO feature-ablation figure for the paper.

Reads:  results/circuits/feature_ablation/loo_summary.json
Writes: 6a154a47401c9f4881c67a3f/figures/feature_ablation.pdf

Two-panel design: within-family similarity on the LEFT, cross-family
similarity on the RIGHT, same y-axis on both so the asymmetry of the
no_load effect is immediately visible. The cross panel shows the dramatic
jump for -load (the feature that made cross-family pairs distinguishable);
the within panel shows the much smaller jump (within-family pairs were
never strongly differentiated by load).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "circuits" / "feature_ablation" / "loo_summary.json"
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
HIGHLIGHT_KEY = "no_load"

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
hl_idx = ABLATIONS.index(HIGHLIGHT_KEY)

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
        bars = ax.bar(
            x + offset, vals[:, q_i], width,
            label=f"$Q = {q}$" if panel_idx == 0 else None,
            color=col, edgecolor="black", linewidth=0.5,
        )
        bars[hl_idx].set_edgecolor("#B22222")
        bars[hl_idx].set_linewidth(2.0)
    # Reference baseline = mean of full bars (across Q) for this panel.
    ref = float(np.mean(full_ref))
    ax.axhline(y=ref, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="full baseline (mean)" if panel_idx == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[a] for a in ABLATIONS], fontsize=11)
    ax.set_title(panel_title, fontsize=12.5, pad=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_ylim(0, y_max)

axes[0].set_ylabel(r"Similarity $\mathcal{S}$", fontsize=12)
axes[0].legend(loc="upper right", framealpha=0.95, fontsize=10)

# Annotation on the cross-family panel: this is where no_load explodes.
mid_q = 1
ann_x = hl_idx + (mid_q - 1) * width
ann_y_full = cross[0, mid_q]
ann_y_abl  = cross[hl_idx, mid_q]
axes[1].annotate(
    "",
    xy=(ann_x, ann_y_abl),
    xytext=(ann_x, ann_y_full),
    arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.6),
)
axes[1].text(
    hl_idx - 0.45, (ann_y_full + ann_y_abl) / 2 + 0.05,
    r"$-$load: cross jumps" + "\n" + r"$\sim 4\times$ more than within",
    fontsize=11, color="#B22222", fontweight="bold",
    ha="right", va="center",
)

fig.suptitle(
    r"Leave-one-out feature ablation at $\alpha = 0$, $\beta = 0.5$",
    fontsize=13, y=0.99,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
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
