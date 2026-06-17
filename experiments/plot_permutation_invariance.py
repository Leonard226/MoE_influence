"""Make a 5-vertex illustrative figure for the appendix: two drawings of the
same abstract measure network, related by a composite permutation that
simultaneously does (a) a within-layer swap and (b) a cross-layer swap.

Setup: 3 layers, 2-2-1 vertices.
    Layer 0 (top):    v_0, v_1
    Layer 1 (middle): v_2, v_3
    Layer 2 (bottom): v_4

Permutation: pi = (v_0  v_1) (v_2  v_4); v_3 is the only fixed vertex.
    (v_0  v_1): within-layer transposition in layer 0
    (v_2  v_4): cross-layer transposition between layer 1 and layer 2

Only v_3 stays in its original drawing position. The edges live between
vertex IDENTITIES (not drawing slots) and therefore follow the vertices
through the swap: arrows between cross-layer-swapped vertices visually
``invert'' in Panel B, yet the abstract network (V, C, F, mass) is unchanged.

Writes: 6a154a47401c9f4881c67a3f/figures/permutation_invariance.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "6a154a47401c9f4881c67a3f" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG = FIG_DIR / "permutation_invariance.pdf"


# Each entry: (x, y, mass, colour). Top-down layout: y = 1 is layer 0 (top).
VERTICES_PANEL_A = {
    "v_0": (0.28, 1.00, 0.20, "#4477AA"),   # blue,   layer 0 left
    "v_1": (0.72, 1.00, 0.15, "#EE6677"),   # red,    layer 0 right
    "v_2": (0.28, 0.50, 0.25, "#228833"),   # green,  layer 1 left
    "v_3": (0.72, 0.50, 0.20, "#CCBB44"),   # yellow, layer 1 right (FIXED)
    "v_4": (0.50, 0.00, 0.20, "#AA3377"),   # purple, layer 2 (sink)
}

# Edges live between vertex IDENTITIES; they follow the vertices under any
# permutation.
EDGES = [
    ("v_0", "v_2", 0.6),
    ("v_0", "v_3", 0.4),
    ("v_1", "v_2", 0.3),
    ("v_1", "v_3", 0.7),
    ("v_2", "v_4", 0.8),
    ("v_3", "v_4", 0.5),
]

# Panel B: apply pi = (v_0  v_1) (v_2  v_4). v_3 is fixed.
VERTICES_PANEL_B = {
    "v_0": (0.72, 1.00, 0.20, "#4477AA"),   # v_0 <-> v_1 inside layer 0
    "v_1": (0.28, 1.00, 0.15, "#EE6677"),
    "v_2": (0.50, 0.00, 0.25, "#228833"),   # v_2 <-> v_4 across layers
    "v_3": (0.72, 0.50, 0.20, "#CCBB44"),   # FIXED
    "v_4": (0.28, 0.50, 0.20, "#AA3377"),
}


def _label_math(label: str) -> str:
    """Convert 'v_0' to math-formatted '$v_{0}$' for matplotlib mathtext."""
    if label.startswith("v_"):
        idx = label[2:]
        return r"$v_{" + idx + r"}$"
    return label


def draw_dag(ax, vertices: dict, edges: list, title: str,
             fixed: set[str] | None = None) -> None:
    fixed = fixed or set()

    # Edges first so the vertex disks sit on top.
    for src, tgt, w in edges:
        x1, y1, _, _ = vertices[src]
        x2, y2, _, _ = vertices[tgt]
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            color="#555555", lw=1.2,
            shrinkA=20, shrinkB=20,
            alpha=0.85,
        )
        ax.add_patch(arr)
        mx, my = 0.55 * x1 + 0.45 * x2, 0.55 * y1 + 0.45 * y2
        ax.text(mx + 0.02, my, r"$W = " + str(w) + r"$",
                fontsize=8, ha="left", va="center", color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": 0.7, "alpha": 0.92})

    # Vertices.
    for label, (x, y, mass, colour) in vertices.items():
        if label in fixed:
            # Highlight the fixed vertex with a heavier outline.
            ax.scatter([x], [y], s=2200, c=colour, zorder=3,
                       edgecolors="black", linewidths=2.6)
        else:
            ax.scatter([x], [y], s=2200, c=colour, zorder=3,
                       edgecolors="black", linewidths=1.2)
        ax.text(x, y, _label_math(label),
                fontsize=13, ha="center", va="center", zorder=4,
                color="white", weight="bold")
        ax.text(x + 0.10, y, r"$m = " + str(mass) + r"$",
                fontsize=8.5, ha="left", va="center", zorder=4, color="#222222")

    ax.set_xlim(-0.05, 1.18)
    ax.set_ylim(-0.18, 1.18)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 5.8),
                                     gridspec_kw={"wspace": 0.16})

    draw_dag(ax_a, VERTICES_PANEL_A, EDGES,
             r"Panel A: original network $N$")
    draw_dag(ax_b, VERTICES_PANEL_B, EDGES,
             r"Panel B: $N^\pi$, $\pi = (v_0\, v_1)(v_2\, v_4)$"
             r"   (only $v_3$ fixed)",
             fixed={"v_3"})

    plt.savefig(FIG, format="pdf", bbox_inches="tight")
    print(f"Saved: {FIG}")


if __name__ == "__main__":
    main()
