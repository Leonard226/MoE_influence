"""Composite C_conn heatmap figures for the routing-DAG paper.

Renders 8 models x 3 Q values in matplotlib subplots. By default writes ONE
PDF (8 rows). With --split, writes TWO PDFs of 4 rows each so each fits
comfortably on one page.

Row ordering keeps within-family models adjacent:
  1. Mixtral-8x7B     (family Mixtral)
  2. Mixtral-8x22B    (family Mixtral)
  3. DeepSeek-V2-Lite (family DeepSeek)
  4. DeepSeek-V2      (family DeepSeek)
  5. Qwen3-30B-A3B    (family Qwen3)
  6. Qwen3-235B-A22B  (family Qwen3)
  7. OLMoE            (singleton)
  8. Phi-3.5-MoE      (singleton)

Colour: viridis_r so bright = small C = tightly coupled = more connected,
dark = large C = decoupled. One thin colorbar per row on the right.

Reuses _compute_one and constants from inspect_conn_costs.py.

Usage (on piora, where the DAGs live):
    # single 8-row PDF
    python experiments/plot_cconn_composite.py --task c4 --gamma 0.5
    # split into two 4-row PDFs (for paper layout)
    python experiments/plot_cconn_composite.py --task c4 --gamma 0.5 --split

Output:
    ${result_path}/circuits/conn_inspection/C_conn_composite_<task>_g<gamma>.pdf
    or (with --split):
    ${result_path}/circuits/conn_inspection/C_conn_composite_<task>_g<gamma>_part{1,2}.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.inspect_conn_costs import _compute_one, DEFAULT_OUT_DIR


# Row order: within-family models adjacent, then singletons.
MODELS_ORDERED = [
    "mixtral-8x7b",
    "mixtral-8x22b",
    "deepseek-v2-lite",
    "deepseek-v2",
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
    "olmoe",
    "phi-3.5-moe",
]
MODEL_LABELS = {
    "mixtral-8x7b":     "Mixtral-8x7B",
    "mixtral-8x22b":    "Mixtral-8x22B",
    "deepseek-v2-lite": "DeepSeek-V2-Lite",
    "deepseek-v2":      "DeepSeek-V2",
    "qwen3-30b-a3b":    "Qwen3-30B-A3B",
    "qwen3-235b-a22b":  "Qwen3-235B-A22B",
    "olmoe":            "OLMoE",
    "phi-3.5-moe":      "Phi-3.5-MoE",
}
QUANTILES = [0.9, 0.99, 0.999]


def _panel(ax, C: np.ndarray, keep_mask: np.ndarray, N: int):
    """Render one upper-triangular heatmap into ax. Returns AxesImage."""
    import matplotlib.pyplot as plt
    if keep_mask is None or int(keep_mask.sum()) < 2:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    keep_idx = np.where(keep_mask)[0]
    V_eff = keep_idx.size
    C_eff = C[np.ix_(keep_idx, keep_idx)]
    display = C_eff.copy()
    tri_idx = np.tril_indices(V_eff, k=-1)
    display[tri_idx] = np.nan

    # viridis: bright = high C = decoupled; dark = low C = tightly coupled.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8d8d8")
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest", rasterized=True)

    # Layer boundary markers.
    original_layer = keep_idx // N
    diff_idx = np.where(np.diff(original_layer) > 0)[0]
    for b in diff_idx + 0.5:
        ax.axvline(x=b, color="white", linewidth=0.25, alpha=0.55)
        ax.axhline(y=b, color="white", linewidth=0.25, alpha=0.55)

    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _render_grid(models: list[str], cache: dict,
                 panel_size: float, gamma: float,
                 out_path: Path, dpi: int) -> None:
    """Render an n_rows x 3 composite into out_path. Models is the row order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_rows = len(models)
    n_cols = len(QUANTILES)

    # Figure dimensions. Very small top margin since there's no figure title.
    fig_w = panel_size * (n_cols + 0.15) + 0.7   # +0.7 for vertical row labels
    fig_h = panel_size * n_rows + 0.35           # +0.35 for column titles only

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.07],   # very thin colorbar column
        wspace=0.05, hspace=0.05,
        left=0.085, right=0.94,    # 6% right margin so colorbar tick labels aren't clipped
        top=1.0 - 0.30 / fig_h,    # ~0.3 inch top margin (just enough for Q labels)
        bottom=0.02,
    )

    for ri, m in enumerate(models):
        last_im = None
        for ci, q in enumerate(QUANTILES):
            ax = fig.add_subplot(gs[ri, ci])
            stats, C, keep_mask = cache[(m, q)]
            if stats.get("missing"):
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                last_im = _panel(ax, C, keep_mask, stats["N"])
                ax.text(0.02, 0.04,
                        f"$V_Q = {stats['V_eff']}$",
                        ha="left", va="bottom",
                        transform=ax.transAxes,
                        fontsize=7, color="black")
            if ri == 0:
                ax.set_title(f"$Q = {q}$", fontsize=10, pad=1)
            if ci == 0:
                ax.set_ylabel(MODEL_LABELS[m], fontsize=10, rotation=90,
                              ha="center", va="center", labelpad=4)
        # Thin per-row colorbar.
        cax = fig.add_subplot(gs[ri, n_cols])
        if last_im is not None:
            cbar = fig.colorbar(last_im, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("Structural cost $C(u,v)$\n(lower = stronger connectivity)", fontsize=7, labelpad=1)
            
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--panel-size", type=float, default=1.7,
                   help="Per-panel side in inches (default 1.7).")
    p.add_argument("--split", action="store_true",
                   help="Write two PDFs of 4 rows each instead of one 8-row PDF.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute every (model, Q) panel up front.
    cache: dict[tuple[str, float], tuple] = {}
    for m in MODELS_ORDERED:
        for q in QUANTILES:
            print(f"computing C_conn for {m} @ Q={q} ...", flush=True)
            stats, C, keep_mask, *_ = _compute_one(
                m, args.task, q, args.eps, args.gamma
            )
            cache[(m, q)] = (stats, C, keep_mask)

    if args.split:
        part1 = MODELS_ORDERED[:4]   # Mixtral + DeepSeek
        part2 = MODELS_ORDERED[4:]   # Qwen3 + singletons
        out1 = out_dir / f"C_conn_composite_{args.task}_g{args.gamma:g}_part1.pdf"
        out2 = out_dir / f"C_conn_composite_{args.task}_g{args.gamma:g}_part2.pdf"
        _render_grid(part1, cache, args.panel_size, args.gamma, out1, args.dpi)
        _render_grid(part2, cache, args.panel_size, args.gamma, out2, args.dpi)
    else:
        out = out_dir / f"C_conn_composite_{args.task}_g{args.gamma:g}.pdf"
        _render_grid(MODELS_ORDERED, cache, args.panel_size, args.gamma, out, args.dpi)


if __name__ == "__main__":
    main()
