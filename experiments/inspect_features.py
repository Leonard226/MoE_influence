"""Cross-model feature matrix F analysis.

For each model on the chosen task, builds F using the SAME normalisations as
the headline sweep (act_norm=log_max, load_norm=log_max) and combines all
experts from all 8 models into one big dataset (~32k experts × 10 features).

F's 10 dimensions:
    [depth, out_norm, in_norm, load, act,
     class_content, class_functional, class_punctuation, class_numeric, class_special]

Outputs (under {result_path}/circuits/feature_inspection/):
  - features_per_feature_ridge.pdf
      One row per feature × stacked per-model ridges, so you can see for each
      feature where each model's distribution sits.
  - features_correlation.pdf
      Pearson correlation matrix between feature dims (over all experts).
      Reveals redundancies (e.g. if act and load are highly correlated).
  - features_pca.pdf
      PCA of all experts onto 2D, coloured by model. Tests whether models form
      distinct clusters in feature space (-> features carry family info) or
      mix freely (-> features mostly encode intrinsic expert properties).
  - features_umap.pdf  (if umap-learn installed; else t-SNE fallback)
      Non-linear projection, often clearer than PCA.
  - features_kmeans.pdf
      Cluster all experts in feature space (k=8 by default). The cluster-by-
      model composition matrix shows whether clusters end up dominated by
      single models (clean family signal) or mixed (features dominated by
      cross-model invariants like depth).

Usage:
    python experiments/inspect_features.py
    python experiments/inspect_features.py --task math --skip-umap
    python experiments/inspect_features.py --kmeans-k 12 --subsample 5000
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Clamp BLAS thread fan-out BEFORE numpy / sklearn imports. piora GPU nodes
# have many more cores than OpenBLAS's 128-thread build cap; k-means with
# n_init>1 + a many-core box otherwise hits "too many memory regions" and
# segfaults. 4 threads is plenty for our problem sizes.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.fgw import build_triple             # noqa: E402
from experiments.run_alpha_beta_sweep import MODELS  # noqa: E402

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"
CLASSIFY_DIR = CIRCUITS_DIR / "classifications"
DEFAULT_OUT_DIR = CIRCUITS_DIR / "feature_inspection"

FEATURE_NAMES = [
    "depth", "out", "in", "load", "act",
    "content", "functional", "punctuation", "numeric", "special",
]


def _model_palette(n: int):
    """8 distinct, more-saturated qualitative colours (matplotlib Dark2),
    suitable for categorical scatter plots. Designed for varied hue AND
    saturation so models don't all read as 'kind of similar pastel'."""
    import matplotlib.pyplot as plt
    return list(plt.get_cmap("Dark2").colors)[:n]


# -------------------- per-model F construction -----------------------------
def _load_F(model: str, task: str) -> np.ndarray | None:
    """Build F (only) for one (model, task) using log-max normalisations.
    Calls build_triple with beta=1 so the (expensive) C_struct path is skipped."""
    dag_path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        return None
    cls_path = CLASSIFY_DIR / f"classify_{model}_{task}.pkl"
    classification = None
    if cls_path.exists():
        with open(cls_path, "rb") as f:
            classification = pickle.load(f)
    dag = torch.load(dag_path, weights_only=False, map_location="cpu")
    _C, F, _mass, _meta = build_triple(
        dag, classification,
        beta=1.0,                           # skip C_struct entirely
        act_norm_method="log_max",
        load_norm_method="log_max",
    )
    del dag
    return F


def _build_all(task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (F_all, model_idx, model_names). F_all is (sum_V, D); model_idx[i]
    is the index of the model that vertex i comes from (into MODELS)."""
    parts: list[tuple[int, np.ndarray]] = []
    for i, m in enumerate(MODELS):
        t0 = time.time()
        print(f"  loading {m}/{task} ...", end=" ", flush=True)
        F = _load_F(m, task)
        if F is None:
            print("MISSING")
            continue
        parts.append((i, F))
        print(f"V={F.shape[0]}, D={F.shape[1]}  ({time.time()-t0:.1f}s)",
              flush=True)
    F_all = np.concatenate([F for _, F in parts], axis=0)
    model_idx = np.concatenate(
        [np.full(F.shape[0], i, dtype=np.int32) for i, F in parts], axis=0)
    return F_all, model_idx, MODELS


# -------------------- plots ------------------------------------------------
def _save_cosine_similarity(F_all, model_idx, models, out_dir, dpi) -> Path:
    """Mean pairwise cosine similarity between expert feature vectors across
    every (model_a, model_b) pair. Uses the identity:

        (1/(N_a N_b)) Σ_{i,j} cos(F_a[i], F_b[j])  =  μ̂_a · μ̂_b

    where μ̂_m is the mean of model m's row-normalised features. So we don't
    need to materialise the N_a × N_b pairwise matrix — just take the mean
    of the normalised feature vectors per model and dot them.

    Diagonal cells (i, i): within-model expert similarity. Off-diagonal cells
    (i, j): cross-model expert similarity. Values are in [0, 1] because all
    features are in [0, 1] (so normalised vectors live in the non-negative
    orthant; cosine sim is >= 0).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norms = np.linalg.norm(F_all, axis=1, keepdims=True).clip(min=1e-12)
    F_norm = F_all / norms

    n = len(models)
    mean_dirs = np.zeros((n, F_all.shape[1]))
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = (model_idx == i)
        counts[i] = mask.sum()
        if counts[i] > 0:
            mean_dirs[i] = F_norm[mask].mean(axis=0)

    sim = mean_dirs @ mean_dirs.T               # (n, n), in [0, 1]
    # Auto-scale colour limits to the off-diagonal range so the diagonal
    # (saturated bright) doesn't squash the colour scale.
    eye = np.eye(n, dtype=bool)
    off_diag = sim[~eye]
    vmin = float(off_diag.min()) if off_diag.size else 0.0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="viridis", vmin=vmin, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            val = sim[i, j]
            # Pick text colour for contrast against the cell.
            frac = (val - vmin) / max(1.0 - vmin, 1e-12)
            txt_colour = "black" if frac > 0.55 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_colour)
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(models)
    fig.colorbar(im, ax=ax, label="mean pairwise cosine similarity")
    ax.set_title(
        "Mean pairwise cosine similarity between expert feature vectors\n"
        f"diagonal = within-model expert similarity; "
        f"off-diagonal = cross-model.   colour-range = [{vmin:.3f}, 1.0]"
    )
    fig.tight_layout()
    out_path = out_dir / "features_cosine_similarity.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Console summary: who's closest to whom?
    print("\n  Cross-model closest neighbours by mean cosine similarity:")
    for i in range(n):
        # Exclude self
        sims = sim[i].copy()
        sims[i] = -np.inf
        order = np.argsort(-sims)
        top3 = [(models[j], sim[i, j]) for j in order[:3]]
        s = ",  ".join(f"{m}: {v:.3f}" for m, v in top3)
        print(f"    {models[i]:<18s}  →  {s}")

    return out_path


def _save_correlation(F_all, out_dir, dpi) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = F_all.shape[1]
    corr = np.corrcoef(F_all.T)
    fnames = (FEATURE_NAMES[:D] if D <= len(FEATURE_NAMES)
              else [f"feat_{d}" for d in range(D)])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(D):
        for j in range(D):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax.set_xticks(range(D)); ax.set_yticks(range(D))
    ax.set_xticklabels(fnames, rotation=45, ha="right")
    ax.set_yticklabels(fnames)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Feature-feature correlation (pooled across all experts)")
    fig.tight_layout()
    out_path = out_dir / "features_correlation.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_pca(F_all, model_idx, models, out_dir, dpi) -> Path:
    """One PCA, 8 panels (one per model). Same global PCA, same x/y limits;
    each panel shows its model coloured against a faint background of all
    other experts so you can locate it inside the full point cloud."""
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    Z = pca.fit_transform(F_all)
    explained = pca.explained_variance_ratio_

    # Shared axis limits, derived from the full point cloud.
    pad = 0.04
    x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
    y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
    xlim = (x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ylim = (y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))

    base_colors = _model_palette(len(models))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    for i, (model, ax) in enumerate(zip(models, axes.flatten())):
        mask = (model_idx == i)
        # Faint background = all other models, so the user can see where this
        # model sits inside the full point cloud.
        ax.scatter(Z[~mask, 0], Z[~mask, 1], s=2, color="lightgray",
                   alpha=0.18, edgecolors="none", rasterized=True)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=5, color=base_colors[i],
                   alpha=0.7, edgecolors="none", rasterized=True)
        ax.set_title(f"{model}  (n={int(mask.sum())})", fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.15)

    # Shared axis labels on the outer edges only.
    for ax in axes[1, :]:
        ax.set_xlabel(f"PC1   ({100*explained[0]:.1f}% var)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2   ({100*explained[1]:.1f}% var)")
    fig.suptitle(
        "PCA of all experts in 10-D feature space — one panel per model\n"
        "(grey background = experts from the other 7 models; coloured = this model)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / "features_pca.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_embedding(F_all, model_idx, models, out_dir, dpi,
                    method: str, subsample: int) -> Path:
    """Non-linear 2D projection: UMAP if available, else t-SNE.
    Both can be slow on >10k points → subsample by default."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(F_all)
    if subsample > 0 and n > subsample:
        np.random.seed(0)
        sel = np.random.choice(n, subsample, replace=False)
        F_sub, idx_sub = F_all[sel], model_idx[sel]
    else:
        F_sub, idx_sub = F_all, model_idx

    if method == "umap":
        try:
            import umap                                  # noqa: F401
            from umap import UMAP
            print(f"  fitting UMAP on {len(F_sub)} points ...", flush=True)
            t0 = time.time()
            Z = UMAP(n_components=2, random_state=0,
                     n_neighbors=30, min_dist=0.1).fit_transform(F_sub)
            print(f"    done in {time.time()-t0:.1f}s", flush=True)
            title = (f"UMAP of all experts in feature space "
                     f"(n={len(F_sub)}), coloured by model")
            fname = "features_umap.pdf"
        except ImportError:
            print("  umap-learn not installed; falling back to t-SNE.")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        print(f"  fitting t-SNE on {len(F_sub)} points ...", flush=True)
        t0 = time.time()
        Z = TSNE(n_components=2, random_state=0, perplexity=30,
                 init="pca").fit_transform(F_sub)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)
        title = (f"t-SNE of all experts in feature space "
                 f"(n={len(F_sub)}), coloured by model")
        fname = "features_tsne.pdf"

    fig, ax = plt.subplots(figsize=(11, 9))
    base_colors = _model_palette(len(models))
    counts = np.bincount(idx_sub, minlength=len(models))
    order = np.argsort(-counts)
    for i in order:
        mask = (idx_sub == i)
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6,
                   color=base_colors[i], alpha=0.65,
                   label=f"{models[i]} (n={int(mask.sum())})",
                   edgecolors="none")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.85, markerscale=2.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{method.upper()}-1   (topological coordinate; no semantic units)")
    ax.set_ylabel(f"{method.upper()}-2   (only relative distance is meaningful)")
    fig.tight_layout()
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _kmeans_analysis(F_all, model_idx, models, out_dir, dpi, k) -> Path:
    """K-means on all experts → cluster composition by model."""
    from sklearn.cluster import KMeans
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"  k-means with k={k} on {len(F_all)} points ...", flush=True)
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=0, n_init=5, algorithm="lloyd")
    labels = km.fit_predict(F_all)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    n_models = len(models)
    M = np.zeros((k, n_models), dtype=np.int64)
    for c in range(k):
        for mi in range(n_models):
            M[c, mi] = int(((labels == c) & (model_idx == mi)).sum())

    # Column-normalise: each cell = "fraction of model M's experts in cluster C".
    # This is the per-model normalisation that correctly accounts for the very
    # different per-model expert counts (Mixtral 256 vs Qwen 12k). Each column
    # sums to 1. If model M's experts are spread uniformly, every cell of
    # column M is ~1/k. If concentrated, one cell of column M dominates.
    col_sum = M.sum(axis=0, keepdims=True).clip(min=1)
    M_frac = M / col_sum

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * k + 2)))
    im = ax.imshow(M_frac, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for c in range(k):
        for mi in range(n_models):
            if M[c, mi] == 0:
                continue
            colour = "white" if M_frac[c, mi] < 0.55 else "black"
            ax.text(mi, c, f"{100*M_frac[c, mi]:.0f}%\n({M[c, mi]})",
                     ha="center", va="center", fontsize=6.5, color=colour)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"cluster {c}" for c in range(k)])
    fig.colorbar(im, ax=ax, label="fraction of model's experts in this cluster")
    ax.set_title(
        f"K-means (k={k}) cluster composition, per-model normalisation\n"
        "colour = fraction of model's experts; text = fraction + raw count.\n"
        f"Uniform (no signal) ≈ {100/k:.0f}% in every cell of a column.   "
        "High signal = one cell of each column dominates."
    )
    fig.tight_layout()
    out_path = out_dir / "features_kmeans.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Console summary: per-model concentration (top cluster's fraction).
    print(f"\n  K-means (k={k}) per-model concentration:")
    print(f"    (uniform baseline = {1/k:.3f}; higher = experts cluster together)")
    for mi in range(n_models):
        col = M[:, mi]
        total = int(col.sum())
        if total == 0:
            continue
        top = col.argmax()
        frac = col[top] / total
        print(f"    {models[mi]:<18s} (n={total:6d}): top cluster = {top}, "
              f"holds {100*frac:.1f}% of this model's experts")

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="c4")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP/t-SNE (the slowest step).")
    parser.add_argument("--subsample", type=int, default=8000,
                        help="Random subsample size for UMAP/t-SNE "
                             "(default 8000; 0 = use all).")
    parser.add_argument("--kmeans-k", type=int, default=8,
                        help="K for k-means (default 8 = one per model).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Feature analysis on task = {args.task} ===\n")
    F_all, model_idx, models = _build_all(args.task)
    print(f"\n  Total: {len(F_all)} experts × {F_all.shape[1]} features\n")

    paths: list[Path] = []
    paths.append(_save_cosine_similarity(F_all, model_idx, models, out_dir, args.dpi))
    paths.append(_save_correlation(F_all, out_dir, args.dpi))
    paths.append(_save_pca(F_all, model_idx, models, out_dir, args.dpi))
    if not args.skip_umap:
        paths.append(_save_embedding(F_all, model_idx, models, out_dir,
                                     args.dpi, method="umap",
                                     subsample=args.subsample))
    paths.append(_kmeans_analysis(F_all, model_idx, models, out_dir,
                                  args.dpi, k=args.kmeans_k))

    print(f"\n  Saved:")
    for p in paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()
