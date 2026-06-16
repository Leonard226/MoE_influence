"""Plot the per-vertex activation distribution as a 2-panel Zipf-style figure.

Reads:  results/circuits/feature_ablation/act_curves_c4.json
Writes: 6a154a47401c9f4881c67a3f/figures/act_distribution.pdf

Two-panel design (mirrors plot_load_distribution.py):
  Left  (Panel A): raw act magnitude from dag["act"].
                   Both axes log-scale: shows the 3-5 orders of dynamic
                   range within a graph and the additional ~10000x scale
                   asymmetry across models.
  Right (Panel B): log-max normalised act = log(1+act)/log(1+max(act)),
                   matching fgw.py's "log_max" mode.
                   x-axis log (vertex rank), y-axis linear in [0, 1]:
                   every model's super-expert maps to 1 by construction,
                   the tail shape is preserved, and the curves now occupy
                   a common y-range so cross-model comparison is meaningful.

Each panel: sorted activation values plotted descending on a log x-axis
(vertex rank).  8 lines, family-coloured.
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


# Family-coloured palette; within a family, the larger model is dashed.
# Identical to plot_load_distribution.py.
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
        curves = json.load(f)

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # ---- Panel A: raw act --------------------------------------------------
    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model]["act_raw_sorted"])
        ranks = np.arange(1, len(values) + 1)
        # Replace 0 with a tiny floor for log-scale plotting.
        values = np.clip(values, 1e-30, None)
        ax_raw.plot(ranks, values, **style)

    ax_raw.set_xscale("log")
    ax_raw.set_yscale("log")
    ax_raw.set_xlabel(r"Vertex rank (sorted by $\mathrm{act}$, descending)", fontsize=14)
    ax_raw.set_ylabel(r"$\mathrm{act}(v)$", fontsize=14)
    ax_raw.set_title("Panel A: $\\mathrm{act}$ (raw)", fontsize=14, pad=8)
    ax_raw.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax_raw.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax_raw.set_axisbelow(True)
    ax_raw.legend(loc="lower left", framealpha=0.95, fontsize=10, ncol=1)

    # ---- Panel B: log-max normalised act -----------------------------------
    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model]["act_lognorm_sorted"])
        ranks = np.arange(1, len(values) + 1)
        ax_norm.plot(ranks, values, **style)

    ax_norm.set_xscale("log")
    ax_norm.set_xlabel(r"Vertex rank (sorted by $\hat{\mathrm{act}}$, descending)", fontsize=14)
    ax_norm.set_ylabel(r"$\hat{\mathrm{act}}(v)$", fontsize=14)
    ax_norm.set_title("Panel B: $\\hat{\\mathrm{act}}$ (log-max normalised)", fontsize=14, pad=8)
    ax_norm.set_ylim(0, 1.05)
    ax_norm.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax_norm.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax_norm.set_axisbelow(True)
    ax_norm.legend(loc="lower left", framealpha=0.95, fontsize=10, ncol=1)

    plt.tight_layout()
    plt.savefig(FIG, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG}")

    # Diagnostic: peak / median per model (useful for the LaTeX text).
    print("\nDiagnostic (rank-1 / median act, raw and log-max-normalised):")
    for model in MODEL_STYLE:
        raw  = np.array(curves[model]["act_raw_sorted"])
        norm = np.array(curves[model]["act_lognorm_sorted"])
        print(f"  {model:<20s}  N={len(raw):>5d}  "
              f"raw_peak={raw[0]:>10.2e}  raw_min={raw[-1]:>10.2e}  "
              f"norm_peak={norm[0]:.4f}  norm_median={float(np.median(norm)):.4f}")


if __name__ == "__main__":
    main()
