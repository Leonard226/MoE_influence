"""Plot the per-vertex class-histogram diagnostics that motivate the
class feature in §3 of main.tex.

Reads:  results/circuits/feature_ablation/class_curves_c4.json
Writes: 6a154a47401c9f4881c67a3f/figures/class_specialization.pdf
        6a154a47401c9f4881c67a3f/figures/class_diversity.pdf

Two figures:

  Figure A (class_specialization.pdf)   -- WITHIN-vertex specialization.
    Per-vertex max_c class(v, c) sorted descending across all active vertices.
    Log x (vertex rank), linear y in [0.2, 1].  8 family-coloured curves.
    Mirrors plot_load_distribution / plot_act_distribution.

  Figure B (class_diversity.pdf)        -- ACROSS-vertex diversity.
    For each model, the fraction of vertices whose argmax class is each of the
    5 macro-classes, shown as one horizontal stacked bar per model. Entropy
    H[bits] of the argmax distribution is annotated on the right.

    Two stacked-bar groups:
      - all active vertices (left subplot)
      - vertices with max_c >= 0.5 only (right subplot)
    so the reader can see how diversity changes when restricted to the
    honestly-specialized subset.
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

FIG_A = FIG_DIR / "class_specialization.pdf"
FIG_B = FIG_DIR / "class_diversity.pdf"


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

# Distinct palette for the 5 token classes (categorical, not family-themed).
CLASS_COLORS = {
    "content":     "#2E7D32",   # green-dark
    "functional":  "#1565C0",   # blue
    "punctuation": "#EF6C00",   # orange
    "numeric":     "#6A1B9A",   # purple
    "special":     "#9E9E9E",   # grey (sentinel + unmapped)
}


def _entropy_bits(counts: list[int]) -> float:
    total = sum(counts) or 1
    frac = np.array(counts) / total
    return float(-np.sum(np.where(frac > 0, frac * np.log2(frac), 0.0)))


def plot_specialization(curves: dict) -> None:
    """Figure A: per-vertex max-class probability, sorted descending."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for model, style in MODEL_STYLE.items():
        values = np.array(curves[model]["max_prob_sorted"])
        ranks = np.arange(1, len(values) + 1)
        ax.plot(ranks, values, **style)

    # Reference line at 0.2 = uniform-over-5 baseline.
    ax.axhline(y=0.2, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(1.05, 0.2, " uniform (0.2)", fontsize=8.5, color="gray", va="center")

    ax.set_xscale("log")
    ax.set_xlabel(r"Vertex rank (sorted by $\max_c \mathrm{class}(v, c)$, descending)",
                  fontsize=14)
    ax.set_ylabel(r"$\max_c \mathrm{class}(v, c)$", fontsize=14)
    ax.set_ylim(0.15, 1.02)
    ax.set_title("Within-vertex class specialization on c4", fontsize=14, pad=8)
    ax.grid(True, which="major", alpha=0.35, linestyle="-",  linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":",  linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(FIG_A, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG_A}")


def _stacked_bars(ax, models: list[str], counts_key: str,
                  curves: dict, classes: list[str], title: str) -> None:
    """One horizontal stacked-bar subplot."""
    y_pos = np.arange(len(models))[::-1]   # top-to-bottom = model order

    for yi, model in zip(y_pos, models):
        counts = curves[model][counts_key]
        total = sum(counts) or 1
        frac = np.array(counts) / total
        ent = _entropy_bits(counts)

        left = 0.0
        for ci, cls in enumerate(classes):
            ax.barh(yi, frac[ci], left=left,
                    color=CLASS_COLORS[cls],
                    edgecolor="white", linewidth=0.5,
                    label=cls.capitalize() if yi == y_pos[0] else None)
            left += frac[ci]

        # Entropy + N annotation on the right.
        ax.text(1.01, yi, f"H={ent:4.2f}  N={total}",
                fontsize=8.5, va="center", ha="left",
                family="monospace", color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_STYLE[m]["label"] for m in models], fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of vertices", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, axis="x", which="major", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)


def plot_diversity(curves: dict) -> None:
    """Figure B: stacked bars of dominant-class fractions, two views."""
    classes = curves[next(iter(MODEL_STYLE))]["classes"]
    fig, (ax_all, ax_spec) = plt.subplots(1, 2, figsize=(14.0, 4.6),
                                          gridspec_kw={"wspace": 0.45})

    _stacked_bars(ax_all, list(MODEL_STYLE.keys()),
                  counts_key="argmax_counts",
                  curves=curves, classes=classes,
                  title="(A) all active vertices")
    _stacked_bars(ax_spec, list(MODEL_STYLE.keys()),
                  counts_key="argmax_counts_specialized",
                  curves=curves, classes=classes,
                  title=r"(B) specialized vertices ($\max_c \geq 0.5$)")

    # Single shared legend at top.
    handles, labels = ax_all.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(classes),
               bbox_to_anchor=(0.5, -0.02),
               fontsize=10, frameon=False,
               title="dominant class $\\arg\\max_c \\mathrm{class}(v, c)$",
               title_fontsize=10)

    fig.suptitle("Across-vertex diversity: dominant-class composition on c4",
                 fontsize=13, y=1.02)
    plt.savefig(FIG_B, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG_B}")


def main() -> None:
    with open(JSON) as f:
        curves = json.load(f)

    plot_specialization(curves)
    plot_diversity(curves)

    # Diagnostic printout for the LaTeX text.
    print("\nDiagnostic (for §3 motivation text):")
    print(f"{'model':<20s}  {'rank1_max':>10s}  {'pct_specialized':>16s}  "
          f"{'H_argmax_all':>13s}  {'H_argmax_spec':>14s}")
    print("-" * 84)
    for model in MODEL_STYLE:
        d = curves[model]
        rank1 = d["max_prob_sorted"][0]
        pct_spec = 100.0 * d["n_specialized"] / max(d["n_active"], 1)
        h_all  = _entropy_bits(d["argmax_counts"])
        h_spec = _entropy_bits(d["argmax_counts_specialized"])
        print(f"  {model:<18s}  {rank1:>10.4f}  {pct_spec:>15.1f}%  "
              f"{h_all:>13.3f}  {h_spec:>14.3f}")


if __name__ == "__main__":
    main()
