"""3 x 3 alpha x Q heatmap of the within-family minus cross-family same-task gap.

Reads the aggregated sweep tensor S_full_with_act.npz and produces one PDF
that summarises the full alpha x Q landscape in nine numbers: the
within-family same-task mean minus the cross-family same-task mean at each
(alpha, Q) cell.

Within-family same-task pairs: ordered family pairs (model_i, model_j) where
model_i and model_j belong to the same multi-member family (Mixtral,
DeepSeek, or Qwen3), evaluated on the same task. Singletons (OLMoE,
Phi-3.5-MoE) cannot form within-family pairs.

Cross-family same-task pairs: every other unordered pair on the same task,
including pairs involving singletons. Matches the n=24 / n=200 counts in
analyze_alpha_beta_sweep.py.

Usage:
    python experiments/plot_gap_landscape.py \\
        --input results/circuits/alpha_beta_sweep_logact_logload_conn_g0.5/S_full_with_act.npz

Output: gap_landscape.pdf written next to the input .npz unless --output is
given.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
TASKS = [
    "c4", "math", "code",
    "wikitext2", "gsm8k", "humaneval",
    "pile-arxiv", "pile-github",
]
TUPLES = [(m, t) for m in MODELS for t in TASKS]
TUPLE_IDX = {mt: i for i, mt in enumerate(TUPLES)}

FAMILIES = {
    "Mixtral":  ["mixtral-8x7b",     "mixtral-8x22b"],
    "DeepSeek": ["deepseek-v2-lite", "deepseek-v2"],
    "Qwen3":    ["qwen3-30b-a3b",    "qwen3-235b-a22b"],
}
MODEL_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}
ALPHAS    = [0.0, 0.5, 1.0]
QUANTILES = [0.9, 0.99, 0.999]


def _collect_gap(S_slice: np.ndarray) -> tuple[float, int, int]:
    """Return (within_mean - cross_mean, n_within, n_cross) on one (alpha, Q) slice."""
    within, cross = [], []
    for task in TASKS:
        # Within-family same-task: only multi-member families
        for fam_models in FAMILIES.values():
            for i, m_i in enumerate(fam_models):
                for m_j in fam_models[i + 1:]:
                    a = TUPLE_IDX[(m_i, task)]
                    b = TUPLE_IDX[(m_j, task)]
                    within.append(S_slice[a, b])
        # Cross-family same-task: every unordered cross-fam pair
        # (including singletons), same task
        for i, m_i in enumerate(MODELS):
            for m_j in MODELS[i + 1:]:
                fi = MODEL_TO_FAMILY.get(m_i)
                fj = MODEL_TO_FAMILY.get(m_j)
                if fi is not None and fj is not None and fi == fj:
                    continue   # within-family
                a = TUPLE_IDX[(m_i, task)]
                b = TUPLE_IDX[(m_j, task)]
                cross.append(S_slice[a, b])
    within = np.asarray(within, dtype=float)
    cross  = np.asarray(cross, dtype=float)
    return float(within.mean() - cross.mean()), len(within), len(cross)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, required=True,
                   help="Path to S_full_with_act.npz from the aggregator.")
    p.add_argument("--output", type=str, default=None,
                   help="Output PDF path (default: <input dir>/gap_landscape.pdf).")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.parent / "gap_landscape.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path}...")
    data = np.load(in_path, allow_pickle=True)
    S = data["S"].astype(float)
    assert S.shape == (len(TUPLES), len(TUPLES), len(ALPHAS), len(QUANTILES)), \
        f"unexpected shape {S.shape}"

    # 3 x 3 gap grid; rows = alpha, cols = Q (matches the table the
    # paper text uses).
    gaps = np.zeros((len(ALPHAS), len(QUANTILES)), dtype=float)
    for ai, _ in enumerate(ALPHAS):
        for qi, _ in enumerate(QUANTILES):
            gap, n_w, n_c = _collect_gap(S[:, :, ai, qi])
            gaps[ai, qi] = gap
            print(f"  alpha={ALPHAS[ai]} Q={QUANTILES[qi]} : "
                  f"gap={gap:+.4f}  (n_within={n_w}, n_cross={n_c})")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    vmax = float(np.max(np.abs(gaps)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=+vmax)
    im = ax.imshow(gaps, cmap="RdBu_r", norm=norm, aspect="auto")

    # Cell annotations.
    for ai in range(len(ALPHAS)):
        for qi in range(len(QUANTILES)):
            val = gaps[ai, qi]
            colour = "white" if abs(val) > 0.55 * vmax else "black"
            ax.text(qi, ai, f"{val:+.3f}", ha="center", va="center",
                    fontsize=13, color=colour, fontweight="bold")

    ax.set_xticks(range(len(QUANTILES)))
    ax.set_xticklabels([f"Q = {q}" for q in QUANTILES], fontsize=12)
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels([fr"$\alpha = {a}$" for a in ALPHAS], fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("within-family − cross-family same-task gap",
                   fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(r"$\alpha \times Q$ landscape of family discrimination",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
