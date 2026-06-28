"""Composite FGW similarity heatmap: 3 alpha x 3 Q grid of 64x64 matrices.

Each panel is the full 64x64 (model, task) x (model, task) similarity matrix
at one (alpha, Q) cell. Tuples are ordered (model, task), so the matrix has
8 super-blocks of 8 rows each (one super-block per model, 8 task rows
within). Thin white lines mark the 8-row model boundaries; thicker white
lines mark the family boundaries (Mixtral / DeepSeek / Qwen3 / singletons).

Single shared viridis colour scale in [0, 1]; single colorbar on the right.

Usage (local):
    python experiments/plot_sweep_heatmaps_composite.py \\
        --input results/circuits/alpha_beta_sweep_logact_logload_conn_g0.5/S_full_with_act.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Mirror run_alpha_beta_sweep.py — keep both in sync.
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
ALPHAS = [0.0, 0.5, 1.0]
QUANTILES = [0.9, 0.99, 0.999]

# Family group boundaries (between MODELS index i-1 and i): Mixtral [0,1],
# DeepSeek [2,3], Qwen3 [4,5], singletons [6,7]. Group breaks at 2, 4, 6.
FAMILY_BREAKS = [2, 4, 6]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Path to S_full_with_act.npz")
    p.add_argument("--output", default=None,
                   help="Output PDF (default: <input dir>/sweep_heatmaps_composite.pdf)")
    p.add_argument("--panel-size", type=float, default=2.6,
                   help="Per-panel inches (default 2.6).")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = (Path(args.output) if args.output
                else in_path.parent / "sweep_heatmaps_composite.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path, allow_pickle=True)
    S = data["S"].astype(float)
    N = len(MODELS) * len(TASKS)
    assert S.shape == (N, N, len(ALPHAS), len(QUANTILES)), \
        f"unexpected shape {S.shape}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_r = len(ALPHAS)
    n_c = len(QUANTILES)
    fig_w = args.panel_size * n_c + 1.4   # +1.4 for left labels + right colorbar
    fig_h = args.panel_size * n_r + 0.5   # +0.5 for column titles
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_r, n_c + 1,
        width_ratios=[1.0] * n_c + [0.06],
        wspace=0.07, hspace=0.10,
        left=0.06, right=0.95,
        top=1.0 - 0.30 / fig_h,
        bottom=0.02,
    )

    last_im = None
    n_tasks = len(TASKS)
    n_models = len(MODELS)
    for ai, alpha in enumerate(ALPHAS):
        for qi, q in enumerate(QUANTILES):
            ax = fig.add_subplot(gs[ai, qi])
            mat = S[:, :, ai, qi]
            im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0,
                           interpolation="nearest", rasterized=True)
            last_im = im
            # Per-model boundary lines (every 8 rows/cols).
            for k in range(1, n_models):
                lw = 1.1 if k in FAMILY_BREAKS else 0.4
                col = "white" if k in FAMILY_BREAKS else "#cccccc"
                ax.axvline(x=k * n_tasks - 0.5, color=col, linewidth=lw, alpha=0.85)
                ax.axhline(y=k * n_tasks - 0.5, color=col, linewidth=lw, alpha=0.85)
            ax.set_xticks([])
            ax.set_yticks([])
            if ai == 0:
                ax.set_title(f"$Q = {q}$", fontsize=11, pad=3, fontweight="bold")
            if qi == 0:
                ax.set_ylabel(fr"$\alpha = {alpha}$", fontsize=11, labelpad=6,
                              rotation=90, fontweight="bold",
                              ha="center", va="center")

    # Single shared colorbar in the right gridspec column, spanning all rows.
    cax = fig.add_subplot(gs[:, n_c])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"$\mathcal{S}_\alpha^Q$", fontsize=11, labelpad=2)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
