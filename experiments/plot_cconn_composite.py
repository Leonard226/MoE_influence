"""Composite C_conn heatmap figure: 8 models (rows) x 3 Q values (columns)
in a single PDF.

Renders all 24 panels in one matplotlib figure so the LaTeX side just
includes one PDF. Model names are vertical labels at the left of each row;
one colorbar is drawn per row on the right (all panels share the same
[0, 1] colour scale, but per-row placement reads more cleanly than a
single global bar).

Reuses _compute_one and constants from inspect_conn_costs.py.

Usage (on piora, where the DAGs live):
    python experiments/plot_cconn_composite.py --task c4 --gamma 0.5

Output:
    ${result_path}/circuits/conn_inspection/C_conn_composite_<task>_g<gamma>.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.inspect_conn_costs import _compute_one, DEFAULT_OUT_DIR


# Rows: model order by ascending total expert count.
MODELS_ORDERED = [
    "mixtral-8x7b",
    "mixtral-8x22b",
    "phi-3.5-moe",
    "olmoe",
    "deepseek-v2-lite",
    "qwen3-30b-a3b",
    "deepseek-v2",
    "qwen3-235b-a22b",
]
MODEL_LABELS = {
    "mixtral-8x7b":     "Mixtral-8x7B",
    "mixtral-8x22b":    "Mixtral-8x22B",
    "phi-3.5-moe":      "Phi-3.5-MoE",
    "olmoe":            "OLMoE",
    "deepseek-v2-lite": "DeepSeek-V2-Lite",
    "qwen3-30b-a3b":    "Qwen3-30B-A3B",
    "deepseek-v2":      "DeepSeek-V2",
    "qwen3-235b-a22b":  "Qwen3-235B-A22B",
}
QUANTILES = [0.9, 0.99, 0.999]


def _panel(ax, C: np.ndarray, keep_mask: np.ndarray, L: int, N: int):
    """Render one upper-triangular heatmap into ax. Returns the AxesImage."""
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

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8d8d8")
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest", rasterized=True)

    # Layer boundary markers (white lines on the panel).
    original_layer = keep_idx // N
    diff_idx = np.where(np.diff(original_layer) > 0)[0]
    for b in diff_idx + 0.5:
        ax.axvline(x=b, color="white", linewidth=0.25, alpha=0.55)
        ax.axhline(y=b, color="white", linewidth=0.25, alpha=0.55)

    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--panel-size", type=float, default=1.7,
                   help="Per-panel width/height in inches (default 1.7).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(MODELS_ORDERED)
    n_cols = len(QUANTILES)

    # Compute every (model, Q) panel up front.
    cache: dict[tuple[str, float], tuple] = {}
    for m in MODELS_ORDERED:
        for q in QUANTILES:
            print(f"computing C_conn for {m} @ Q={q} ...", flush=True)
            stats, C, keep_mask, *_ = _compute_one(
                m, args.task, q, args.eps, args.gamma
            )
            cache[(m, q)] = (stats, C, keep_mask)

    # GridSpec layout: per row, 3 panels + small gap + thin colorbar column.
    # Width ratios: 1 1 1 0.15 (colorbar column).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Figure size: extra left margin for vertical row labels, no right margin
    # needed since the colorbars sit inside the gridspec.
    fig_w = args.panel_size * (n_cols + 0.30) + 0.9   # +0.9 inch for left labels
    fig_h = args.panel_size * n_rows + 0.6            # +0.6 inch for column titles
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.15],
        wspace=0.06, hspace=0.06,
        left=0.10, right=0.97, top=0.94, bottom=0.03,
    )

    im_per_row: list = []
    for ri, m in enumerate(MODELS_ORDERED):
        for ci, q in enumerate(QUANTILES):
            ax = fig.add_subplot(gs[ri, ci])
            stats, C, keep_mask = cache[(m, q)]
            if stats.get("missing"):
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                im_local = None
            else:
                im_local = _panel(ax, C, keep_mask, stats["L"], stats["N"])
                # V_eff annotation in the lower-right of each panel.
                ax.text(0.98, 0.04,
                        f"V$_{{\\mathrm{{eff}}}}$={stats['V_eff']}",
                        ha="right", va="bottom",
                        transform=ax.transAxes,
                        fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.45,
                                  edgecolor="none", pad=1.2))
            # Column titles on the top row.
            if ri == 0:
                ax.set_title(f"$Q = {q}$", fontsize=11, pad=4)
            # Vertical row label on the leftmost panel.
            if ci == 0:
                ax.set_ylabel(MODEL_LABELS[m], fontsize=10, rotation=90,
                              ha="center", va="center", labelpad=6)
            if im_local is not None and ci == n_cols - 1:
                im_per_row.append(im_local)
        # Per-row colorbar in the right column of this row.
        cax = fig.add_subplot(gs[ri, n_cols])
        if im_per_row and im_per_row[-1] is not None:
            cbar = fig.colorbar(im_per_row[-1], cax=cax)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(r"$C_{\mathrm{conn}}^{\gamma=" + f"{args.gamma:g}" + r"}$",
                           fontsize=8, labelpad=2)

    out_path = out_dir / f"C_conn_composite_{args.task}_g{args.gamma:g}.pdf"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
