"""Plot the per-vertex class-histogram diagnostics that motivate the
class feature in §3 of main.tex.

Reads:  results/circuits/feature_ablation/class_curves_c4.json
Writes: 6a154a47401c9f4881c67a3f/figures/class_distribution.pdf

Single 2-panel figure:

  Panel A (left)  -- WITHIN-vertex specialization.
    Per-vertex max_c class(v)_c sorted descending across all active vertices.
    Log x (vertex rank), linear y in [0.2, 1].  8 family-coloured curves.
    Mirrors plot_load_distribution / plot_act_distribution.

  Panel B (right) -- ACROSS-vertex diversity.
    For each model, the fraction of vertices whose argmax class is each of the
    5 macro-classes (content, functional, punctuation, numeric, special), as
    one horizontal stacked bar per model. Colorblind-safe palette.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "circuits" / "feature_ablation" / "class_curves_c4.json"
FIG_DIR = ROOT / "6a154a47401c9f4881c67a3f" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG = FIG_DIR / "class_distribution.pdf"


# Same palette as plot_load / plot_act for visual consistency.
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

# Colorblind-safe palette for the 5 token classes (Wong 2011 — used widely in
# scientific publications; each pair of colours remains distinguishable under
# deuteranopia / protanopia / tritanopia).
CLASS_COLORS = {
    "content":     "#0072B2",   # blue       (main lexical content)
    "functional":  "#009E73",   # bluish-green (syntactic glue)
    "punctuation": "#E69F00",   # orange
    "numeric":     "#CC79A7",   # reddish-purple
    "special":     "#999999",   # neutral grey (sentinel / unmapped)
}


def plot_specialization(ax) -> None:
    """Panel A: per-vertex max-class probability, sorted descending."""
    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model]["max_prob_sorted"])
        ranks = np.arange(1, len(values) + 1)
        ax.plot(ranks, values, **style)

    # Reference line at 0.2 = uniform-over-5 baseline.
    ax.axhline(y=0.2, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(1.05, 0.2, " uniform (0.2)", fontsize=8.5, color="gray", va="center")

    ax.set_xscale("log")
    ax.set_xlabel(r"Vertex rank (sorted by $\max_c \mathrm{ class}(v)_c$, descending)",
                  fontsize=13)
    ax.set_ylabel(r"$\max_c \mathrm{ class}(v)_c$", fontsize=13)
    ax.set_ylim(0.15, 1.02)
    ax.set_title(r"Panel A: within-vertex specialization", fontsize=13, pad=8)
    ax.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9, ncol=2)


def plot_diversity(ax, classes: list[str]) -> None:
    """Panel B: dominant-class composition across vertices (stacked bars)."""
    models = list(MODEL_STYLE.keys())
    y_pos = np.arange(len(models))[::-1]      # top-to-bottom = MODEL_STYLE order

    for yi, model in zip(y_pos, models):
        counts = curves[model]["argmax_counts"]
        total = sum(counts) or 1
        frac = np.array(counts) / total
        left = 0.0
        for ci, cls in enumerate(classes):
            ax.barh(yi, frac[ci], left=left,
                    color=CLASS_COLORS[cls],
                    edgecolor="white", linewidth=0.5,
                    label=cls.capitalize() if yi == y_pos[0] else None)
            left += frac[ci]

    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_STYLE[m]["label"] for m in models], fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of vertices", fontsize=13)
    ax.set_title(r"Panel B: across-vertex diversity (dominant class)",
                 fontsize=13, pad=8)
    ax.grid(True, axis="x", which="major", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    ax.legend(loc="lower left",
              bbox_to_anchor=(0.0, -0.35), ncol=len(classes),
              fontsize=9, frameon=False,
              title=r"dominant class $\arg\max_c \mathrm{ class}(v)_c$",
              title_fontsize=10)


def main() -> None:
    global curves
    with open(JSON) as f:
        curves = json.load(f)

    classes = curves[next(iter(MODEL_STYLE))]["classes"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.0),
                                            gridspec_kw={"wspace": 0.30})
    plot_specialization(ax_left)
    plot_diversity(ax_right, classes)

    plt.savefig(FIG, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG}")

    # Diagnostic printout for the §3 motivation text.
    print("\nDiagnostic (for §3 motivation text):")
    print(f"{'model':<20s}  {'rank1_max':>10s}  {'pct_specialized':>16s}")
    print("-" * 52)
    for model in MODEL_STYLE:
        d = curves[model]
        rank1 = d["max_prob_sorted"][0]
        pct_spec = 100.0 * d["n_specialized"] / max(d["n_active"], 1)
        print(f"  {model:<18s}  {rank1:>10.4f}  {pct_spec:>15.1f}%")


if __name__ == "__main__":
    main()
