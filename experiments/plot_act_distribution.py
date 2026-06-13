"""Plot per-model raw-activation distribution as a Zipf-style curve.

Reads:  results/circuits/feature_ablation/act_curves_c4.json
Writes: 6a154a47401c9f4881c67a3f/figures/act_distribution.pdf

Single-panel log-log:
  x-axis: vertex rank (descending, so rank 1 = the super-expert)
  y-axis: raw activation value
  8 lines, one per model, colour-coded by family.

The figure replaces the table in §3 of main.tex: it shows BOTH
  (a) the dynamic range (y-axis span) — orders of magnitude
  (b) the tail heaviness (slope at low ranks) — super-expert dominance
in a single glance.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "circuits" / "feature_ablation" / "act_curves_c4.json"
FIG_DIR = ROOT / "6a154a47401c9f4881c67a3f" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG = FIG_DIR / "act_distribution.pdf"


# Family-coloured palette. Within a family, the larger model is dashed.
MODEL_STYLE = {
    "mixtral-8x7b":     {"color": "#5B8BC6", "ls": "-",  "lw": 1.6, "label": "Mixtral-8x7B"},
    "mixtral-8x22b":    {"color": "#1F4E79", "ls": "--", "lw": 1.6, "label": "Mixtral-8x22B"},
    "deepseek-v2-lite": {"color": "#DC6B6B", "ls": "-",  "lw": 1.6, "label": "DeepSeek-V2-Lite"},
    "deepseek-v2":      {"color": "#8E1F1F", "ls": "--", "lw": 1.8, "label": "DeepSeek-V2"},
    "qwen3-30b-a3b":    {"color": "#4FB058", "ls": "-",  "lw": 1.6, "label": "Qwen3-30B-A3B"},
    "qwen3-235b-a22b":  {"color": "#1F5C2A", "ls": "--", "lw": 1.6, "label": "Qwen3-235B-A22B"},
    "olmoe":            {"color": "#E89425", "ls": "-",  "lw": 1.6, "label": "OLMoE"},
    "phi-3.5-moe":      {"color": "#7B2F8B", "ls": "-",  "lw": 1.6, "label": "Phi-3.5-MoE"},
}


def main() -> None:
    with open(JSON) as f:
        curves: dict[str, list[float]] = json.load(f)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model])
        ranks = np.arange(1, len(values) + 1)
        ax.plot(ranks, values, **style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Vertex rank (sorted by $\mathrm{act}$, descending)", fontsize=11)
    ax.set_ylabel(r"Activation magnitude $\mathrm{act}^{m, \mathrm{c4}}(v)$", fontsize=11)
    ax.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9.5, ncol=1)
    ax.set_title("Raw per-vertex activation on c4 (log-log)", fontsize=12, pad=8)

    plt.tight_layout()
    plt.savefig(FIG, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG}")

    # Quick diagnostic printout for the LaTeX text.
    print("\nFor reference, the rank-1 (super-expert) and rank-N values per model:")
    for model in MODEL_STYLE:
        values = np.array(curves[model])
        print(f"  {model:<20s}  N={len(values):>5d}  rank1={values[0]:>10.2e}  "
              f"rankN={values[-1]:>10.2e}  dyn_range={values[0]/max(values[-1], 1e-30):>10.2e}")


if __name__ == "__main__":
    main()
