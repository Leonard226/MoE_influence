"""Plot the per-vertex load distribution as a 2-panel Zipf-style figure.

Reads:  results/circuits/feature_ablation/load_curves_c4.json
Writes: 6a154a47401c9f4881c67a3f/figures/load_distribution.pdf

Two-panel design:
  Left  (Panel A): raw load = n_tok / mean_in_layer, range up to N.
                   y-axis log scale so the cross-model dynamic range is visible.
  Right (Panel B): per-layer log-max normalised load, in [0, 1].
                   y-axis linear; all 8 model curves now peak at 1 by
                   construction, showing the within-layer concentration *shape*
                   is preserved while the cross-model scale asymmetry vanishes.

Each panel: sorted load values plotted descending on a log x-axis (vertex rank).
8 lines, family-coloured.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "circuits" / "feature_ablation" / "load_curves_c4.json"
FIG_DIR = ROOT / "6a154a47401c9f4881c67a3f" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG = FIG_DIR / "load_distribution.pdf"


# Same family palette as plot_act_distribution.py.
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

    # ---- Panel A: raw load --------------------------------------------------
    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model]["load_raw_sorted"])
        ranks = np.arange(1, len(values) + 1)
        # Replace 0 with a tiny floor for log-scale plotting
        values = np.clip(values, 1e-6, None)
        ax_raw.plot(ranks, values, **style)

    # Reference line at load = 1 (the layer mean).
    ax_raw.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_raw.text(1.05, 1.0, " layer mean", fontsize=8.5, color="gray", va="center")

    ax_raw.set_xscale("log")
    ax_raw.set_yscale("log")
    ax_raw.set_xlabel(r"Vertex rank (sorted by $\mathrm{load}$, descending)", fontsize=11)
    ax_raw.set_ylabel(r"$\mathrm{load}_{\mathrm{raw}}(v) = n_{\mathrm{tok}}(v)\,/\,$mean$_{\ell}\,n_{\mathrm{tok}}$",
                      fontsize=11)
    ax_raw.set_title("Panel A: raw $\\mathrm{load}$ (current definition)",
                     fontsize=12, pad=8)
    ax_raw.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax_raw.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax_raw.set_axisbelow(True)
    ax_raw.legend(loc="lower left", framealpha=0.95, fontsize=9, ncol=1)

    # ---- Panel B: log-max normalised load ----------------------------------
    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model]["load_lognorm_sorted"])
        ranks = np.arange(1, len(values) + 1)
        ax_norm.plot(ranks, values, **style)

    ax_norm.set_xscale("log")
    ax_norm.set_xlabel(r"Vertex rank (sorted by $\widetilde{\mathrm{load}}$, descending)",
                       fontsize=11)
    ax_norm.set_ylabel(r"$\widetilde{\mathrm{load}}(v) = \log(1+\mathrm{load}_{\mathrm{raw}}(v))\,/\,\log(1+\max_{n'}\mathrm{load}_{\mathrm{raw}}(\ell, n'))$",
                       fontsize=10)
    ax_norm.set_title("Panel B: per-layer log-max normalised $\\widetilde{\\mathrm{load}}$",
                      fontsize=12, pad=8)
    ax_norm.set_ylim(0, 1.05)
    ax_norm.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax_norm.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax_norm.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIG, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG}")

    # Diagnostic: peak load per model (for the table in main.tex).
    print("\nDiagnostic (peak raw load and post-normalisation):")
    for model in MODEL_STYLE:
        raw = np.array(curves[model]["load_raw_sorted"])
        norm = np.array(curves[model]["load_lognorm_sorted"])
        print(f"  {model:<20s}  N={curves[model]['N']:>4d}  "
              f"raw_peak={raw[0]:>7.3f}  "
              f"norm_peak={norm[0]:.4f}  "
              f"norm_median={float(np.median(norm)):.4f}")


if __name__ == "__main__":
    main()
