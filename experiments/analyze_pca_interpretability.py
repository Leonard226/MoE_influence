"""Semantic interpretation of the principal axes of the pooled expert
feature space F_all on a given task.

Computes:
  - Pooled PCA on mean-centered F_all (all 8 models, 31,520 experts, 10 features).
  - Singular values, explained-variance ratio, cumulative variance.
  - Loadings V (rows = PC directions, cols = features).
  - Correlation matrix corr(PC_i score, feature_j) -- the more interpretable
    cousin of loadings, bounded in [-1, 1].
  - For each of PC1, PC2, PC3: top-10 and bottom-10 extreme experts.
  - For each expert, the (model, layer_idx, depth) breakdown.

Outputs (all under results/circuits/feature_inspection/):
  pca_scree_and_loadings.pdf      scree + cumvar + per-PC loading bars (PC1-3)
  pca_correlation_matrix.pdf      10-feature x 5-PC correlation heatmap
                                  (the headline "what does PC1 mean" figure)
  pca_scatter_by_feature.pdf      2x5 grid of PC1xPC2 scatter, one panel per
                                  feature (continuous colormap). Visual proof
                                  of the interpretation from the heatmap.
  pca_scatter_by_model.pdf        PC1xPC2 scatter, coloured by model. Companion
                                  to scatter_by_feature -- "where do the
                                  models sit in this semantic space".
  pca_extreme_experts.json        top/bottom 10 experts per PC1/PC2/PC3
  pca_interpretability_summary.json  full numerics (all 10 PCs).

Centering: pooled F is mean-centered before SVD (standard PCA convention,
matches the per-model effective-rank analysis in analyze_feature_rank.py).

Usage:
  python3 experiments/analyze_pca_interpretability.py --task c4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Clamp BLAS threads early.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
DEFAULT_OUT_DIR = Path(CFG["result_path"]) / "circuits" / "feature_inspection"

# Feature order is fixed by build_triple in experiments/fgw.py.
FEATURE_NAMES = ["depth", "out", "in", "load", "act",
                 "content", "functional", "punctuation", "numeric", "special"]


# -------------------- helpers ---------------------------------------------
def _load_embedding(task: str, out_dir: Path):
    emb_path = out_dir / f"embedding_{task}.npz"
    if not emb_path.exists():
        print(f"ERROR: cached embedding not found: {emb_path}", file=sys.stderr)
        print(f"  Run inspect_features.py --task {task} first.", file=sys.stderr)
        sys.exit(1)
    d = np.load(emb_path, allow_pickle=True)
    F_all     = np.asarray(d["F_all"]).astype(np.float64)
    model_idx = np.asarray(d["model_idx"]).astype(int)
    models    = list(d["models"])
    return F_all, model_idx, models


def _pca(F: np.ndarray) -> dict:
    """Mean-center, then SVD. Returns sigmas, var ratio, cumvar, V (D, D)."""
    mu = F.mean(axis=0, keepdims=True)
    Fc = F - mu
    # SVD: Fc = U diag(s) Vt, so Vt rows are right singular vectors (PC dirs).
    U, s, Vt = np.linalg.svd(Fc, full_matrices=False)
    var = s ** 2 / max(F.shape[0] - 1, 1)        # eigenvalues of cov
    ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    cumvar = np.cumsum(ratio)
    # PC scores Z[i, k] = projection of expert i onto PC k.
    Z = Fc @ Vt.T
    return {
        "mu": mu.ravel(),
        "sigmas":   s,
        "var":      var,
        "var_ratio": ratio,
        "cumvar":   cumvar,
        "loadings": Vt,    # (D, D) -- row k = PC_k direction in feature space
        "Z":        Z,     # (N, D) -- expert PC scores
        "Fc":       Fc,    # (N, D) -- centred features
    }


def _layer_index_per_expert(depth_col: np.ndarray,
                            model_idx: np.ndarray,
                            n_models: int) -> np.ndarray:
    """Recover per-expert layer index from the depth feature: depth = ell/(L-1)
    within each model. We recover L per model as len(unique(depth_m))."""
    layer_idx = np.zeros_like(model_idx, dtype=np.int64)
    for mi in range(n_models):
        mask = (model_idx == mi)
        d = depth_col[mask]
        uniq = np.unique(np.round(d, 6))
        L = uniq.size
        if L <= 1:
            layer_idx[mask] = 0
            continue
        layer_idx[mask] = np.round(d * (L - 1)).astype(np.int64)
    return layer_idx


# -------------------- plotting helpers ------------------------------------
def _figsize(w: float, h: float) -> tuple[float, float]:
    return (w, h)


def _plot_scree_and_loadings(pca: dict, out_path: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = len(pca["var_ratio"])
    fig = plt.figure(figsize=_figsize(15.5, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.28,
                          left=0.06, right=0.98, top=0.93, bottom=0.10)

    # ---- Top: scree bar + cumvar overlay -------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ks = np.arange(1, D + 1)
    ax.bar(ks, 100 * pca["var_ratio"], color="#4C78A8",
           edgecolor="black", linewidth=0.5, label="per-PC variance")
    ax.set_xticks(ks)
    ax.set_xlabel("Principal component $k$", fontsize=12)
    ax.set_ylabel("Variance explained (%)", fontsize=12, color="#4C78A8")
    ax.tick_params(axis="y", labelcolor="#4C78A8")
    ax.set_ylim(0, max(100 * pca["var_ratio"].max() * 1.15, 5))
    # Cumvar on secondary axis.
    ax2 = ax.twinx()
    ax2.plot(ks, 100 * pca["cumvar"], color="#E45756",
             marker="o", linewidth=2.0, markersize=5, label="cumulative variance")
    ax2.axhline(90, color="gray", linestyle=":", linewidth=0.8)
    ax2.axhline(95, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Cumulative variance (%)", fontsize=12, color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    ax2.set_ylim(0, 105)
    # Annotate individual % on bars for PCs 1-5.
    for k in range(min(5, D)):
        ax.text(ks[k], 100 * pca["var_ratio"][k] + 0.7,
                f"{100*pca['var_ratio'][k]:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Scree plot: variance explained per principal component "
                 "(pooled, mean-centred $\\mathbf{F}_\\mathrm{all}$)",
                 fontsize=13)

    # ---- Bottom row: loading bars for PC1, PC2, PC3 --------------------------
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        loadings_k = pca["loadings"][k]  # length D
        colors = ["#4C78A8" if v >= 0 else "#E45756" for v in loadings_k]
        ax.bar(range(D), loadings_k, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(D))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_title(
            f"PC{k+1} loadings  ({100*pca['var_ratio'][k]:.1f}% var)",
            fontsize=11,
        )
        ax.set_ylabel("loading $v_{" + str(k+1) + ",j}$" if k == 0 else "",
                       fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_correlation_matrix(pca: dict, F: np.ndarray, out_path: Path,
                             dpi: int, k_show: int = 5) -> np.ndarray:
    """Heatmap of corr(PC_i_score, feature_j) for i in 1..k_show, j in 1..D.
    Returns the correlation matrix (k_show, D) for the JSON summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = pca["Z"]                    # (N, D) -- PC scores
    D = F.shape[1]
    k_show = min(k_show, Z.shape[1])
    R = np.zeros((k_show, D))
    for i in range(k_show):
        for j in range(D):
            zi = Z[:, i]
            fj = F[:, j]
            sz, sf = zi.std(), fj.std()
            if sz < 1e-12 or sf < 1e-12:
                R[i, j] = 0.0
            else:
                R[i, j] = float(np.corrcoef(zi, fj)[0, 1])

    fig, ax = plt.subplots(figsize=_figsize(9, 5.5))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(k_show):
        for j in range(D):
            v = R[i, j]
            txt_col = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=10, color=txt_col)
    ax.set_xticks(range(D))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(k_show))
    ax.set_yticklabels(
        [f"PC{i+1} ({100*pca['var_ratio'][i]:.1f}%)" for i in range(k_show)],
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson correlation", fontsize=11)
    ax.set_title("PC--feature correlation: what does each principal axis mean?\n"
                 r"$r_{ij} = \mathrm{corr}(\mathrm{PC}_i\ \mathrm{score},\ "
                 r"\mathrm{feature}_j)$",
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return R


def _plot_scatter_by_feature(pca: dict, F: np.ndarray, out_path: Path,
                             dpi: int) -> None:
    """2x5 grid of PC1 vs PC2 scatter, one panel per feature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = pca["Z"]
    x, y = Z[:, 0], Z[:, 1]
    var1, var2 = 100 * pca["var_ratio"][0], 100 * pca["var_ratio"][1]
    pad = 0.04
    x_lim = (float(x.min()) - pad * (x.max() - x.min()),
             float(x.max()) + pad * (x.max() - x.min()))
    y_lim = (float(y.min()) - pad * (y.max() - y.min()),
             float(y.max()) + pad * (y.max() - y.min()))

    D = F.shape[1]
    n_cols = 5
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=_figsize(3.4 * n_cols, 3.5 * n_rows),
                              squeeze=False)
    for j in range(D):
        r, c = j // n_cols, j % n_cols
        ax = axes[r, c]
        vals = F[:, j]
        sc = ax.scatter(x, y, c=vals, s=1.6, alpha=0.55, linewidths=0,
                        cmap="viridis", rasterized=True,
                        vmin=float(vals.min()), vmax=float(vals.max()))
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_title(f"colour = {FEATURE_NAMES[j]}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    # Hide empty panels (D may be < n_rows*n_cols).
    for j in range(D, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r, c].set_visible(False)
    fig.suptitle(
        f"PC1 vs.\ PC2 scatter, coloured by each engineered feature "
        f"(pooled $\\mathbf{{F}}_\\mathrm{{all}}$, $n = {len(x):,d}$).\n"
        f"PC1 = {var1:.1f}% var, PC2 = {var2:.1f}% var.  "
        f"A clean horizontal gradient $\\Rightarrow$ PC1 \"is\" that feature; "
        f"vertical gradient $\\Rightarrow$ PC2 \"is\" that feature.",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_scatter_by_model(pca: dict, model_idx: np.ndarray, models: list[str],
                            out_path: Path, dpi: int) -> None:
    """PC1 vs PC2 scatter, coloured by model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import colorsys
    Z = pca["Z"]
    x, y = Z[:, 0], Z[:, 1]
    var1, var2 = 100 * pca["var_ratio"][0], 100 * pca["var_ratio"][1]
    n_models = len(models)
    colors = [colorsys.hsv_to_rgb(i / n_models, 0.85, 0.85)
              for i in range(n_models)]

    fig, ax = plt.subplots(figsize=_figsize(9, 7.5))
    # Draw smallest models last so they don't get buried.
    counts = np.bincount(model_idx, minlength=n_models)
    order = np.argsort(-counts)        # largest first
    for mi in order:
        m_mask = (model_idx == mi)
        ax.scatter(x[m_mask], y[m_mask], s=2.0, alpha=0.6, linewidths=0,
                   color=colors[mi], rasterized=True, label=models[mi])
    ax.set_xlabel(f"PC1  ({var1:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var2:.1f}% var)", fontsize=12)
    ax.set_title("PC1 vs.\ PC2 scatter coloured by model\n"
                 r"(where does each architecture sit in the semantic PC space?)",
                 fontsize=12)
    leg = ax.legend(loc="best", fontsize=9, markerscale=3.0,
                    framealpha=0.85)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _extreme_experts(pca: dict, F: np.ndarray, model_idx: np.ndarray,
                     models: list[str], top_n: int = 10,
                     ks: tuple[int, ...] = (0, 1, 2)) -> dict:
    """For each PC in ks, return top-N and bottom-N experts."""
    Z = pca["Z"]
    layer_idx = _layer_index_per_expert(F[:, 0], model_idx, len(models))
    out: dict[str, dict] = {}
    for k in ks:
        scores = Z[:, k]
        order = np.argsort(scores)         # ascending
        bottom = order[:top_n]
        top    = order[-top_n:][::-1]
        def _entry(idx):
            return {
                "global_idx":   int(idx),
                "model":        models[int(model_idx[idx])],
                "layer":        int(layer_idx[idx]),
                "depth":        float(F[idx, 0]),
                "pc_score":     float(scores[idx]),
                "feature_vec":  {FEATURE_NAMES[j]: float(F[idx, j])
                                 for j in range(F.shape[1])},
            }
        out[f"PC{k+1}"] = {
            "top":    [_entry(int(i)) for i in top],
            "bottom": [_entry(int(i)) for i in bottom],
        }
    return out


def _print_extremes_table(extr: dict, F: np.ndarray) -> None:
    print("\n=== Extreme experts at the tails of each PC ===")
    for pc_name, sides in extr.items():
        print(f"\n--- {pc_name} extremes ---")
        for side in ("top", "bottom"):
            print(f"  {side}:")
            print(f"    {'model':<18s} {'layer':>5s}  {'depth':>5s}  "
                  f"{'pc_score':>9s}  signature (top-3 feature deviations)")
            for e in sides[side]:
                feat = e["feature_vec"]
                global_mean = F.mean(axis=0)
                global_std  = F.std(axis=0).clip(min=1e-9)
                z = {fn: (feat[fn] - global_mean[i]) / global_std[i]
                     for i, fn in enumerate(FEATURE_NAMES)}
                top3 = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:3]
                sig  = ", ".join(f"{fn} (z={zv:+.2f})" for fn, zv in top3)
                print(f"    {e['model']:<18s} {e['layer']:>5d}  "
                      f"{e['depth']:>5.2f}  {e['pc_score']:>+9.3f}  {sig}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--top-n", type=int, default=10,
                   help="Top-N extreme experts per PC tail.")
    p.add_argument("--k-show", type=int, default=5,
                   help="Number of PCs in the correlation heatmap (default 5).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PCA interpretability analysis on task = {args.task} ===")
    F, model_idx, models = _load_embedding(args.task, out_dir)
    print(f"  pooled F: {F.shape}  ({len(models)} models)")

    pca = _pca(F)

    # ---- numbers to stdout ---------------------------------------------------
    D = F.shape[1]
    print(f"\n  PC explained variance:")
    print(f"    {'PC':>3s}  {'%var':>6s}  {'cum %':>7s}")
    for k in range(D):
        print(f"    {k+1:>3d}  {100*pca['var_ratio'][k]:>5.2f}%  "
              f"{100*pca['cumvar'][k]:>6.2f}%")

    # ---- figures -------------------------------------------------------------
    print("\n  building figures ...", flush=True)
    _plot_scree_and_loadings(pca, out_dir / "pca_scree_and_loadings.pdf",
                              args.dpi)
    R = _plot_correlation_matrix(pca, F, out_dir / "pca_correlation_matrix.pdf",
                                  args.dpi, k_show=args.k_show)
    _plot_scatter_by_feature(pca, F, out_dir / "pca_scatter_by_feature.pdf",
                              args.dpi)
    _plot_scatter_by_model(pca, model_idx, models,
                            out_dir / "pca_scatter_by_model.pdf",
                            args.dpi)

    # ---- extreme experts -----------------------------------------------------
    extr = _extreme_experts(pca, F, model_idx, models, top_n=args.top_n)
    _print_extremes_table(extr, F)
    out_extr = out_dir / "pca_extreme_experts.json"
    out_extr.write_text(json.dumps(extr, indent=2))
    print(f"  Saved {out_extr}")

    # ---- summary JSON --------------------------------------------------------
    summary = {
        "task":            args.task,
        "n_experts":       int(F.shape[0]),
        "n_features":      int(D),
        "n_models":        int(len(models)),
        "feature_names":   FEATURE_NAMES,
        "models":          models,
        "mean":            pca["mu"].tolist(),
        "sigmas":          pca["sigmas"].tolist(),
        "var_explained":   pca["var"].tolist(),
        "var_ratio":       pca["var_ratio"].tolist(),
        "cum_var":         pca["cumvar"].tolist(),
        "loadings":        pca["loadings"].tolist(),    # rows = PC dirs
        "corr_matrix":     R.tolist(),                  # (k_show, D)
        "corr_k_show":     args.k_show,
    }
    out_sum = out_dir / "pca_interpretability_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2))
    print(f"  Saved {out_sum}")


if __name__ == "__main__":
    main()
