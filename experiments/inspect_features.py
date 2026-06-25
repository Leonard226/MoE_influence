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
    """n colours at equally-spaced hues in HSV space (high saturation, high
    value). Mathematically guaranteed maximum hue separation; visually the
    most discriminative categorical palette in practice."""
    import colorsys
    return [colorsys.hsv_to_rgb(i / n, 0.92, 0.88) for i in range(n)]


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

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            val = sim[i, j]
            txt_colour = "black" if val > 0.55 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, color=txt_colour)
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(n))
    ax.set_yticklabels(models, fontsize=13)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    ax.set_title("Mean pairwise cosine similarity between expert feature vectors", fontsize=18)
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


def _save_correlation(F_all, model_idx, models, out_dir, dpi) -> Path:
    """8-panel 2x4 grid: per-model Pearson correlation of features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = F_all.shape[1]
    fnames = (FEATURE_NAMES[:D] if D <= len(FEATURE_NAMES)
              else [f"feat_{d}" for d in range(D)])

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")           # NaN cells (zero-variance feature)

    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    im = None
    for i, (ax, model) in enumerate(zip(axes.flatten(), models)):
        mask = (model_idx == i)
        if mask.sum() < 2:
            ax.set_title(f"{model}  (n<2)", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(F_all[mask].T)
        corr_masked = np.ma.masked_invalid(corr)
        im = ax.imshow(corr_masked, cmap=cmap, vmin=-1, vmax=1)
        for ii in range(D):
            for jj in range(D):
                v = corr[ii, jj]
                if np.isnan(v):
                    continue
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > 0.5 else "black")
        col = i % 4
        ax.set_xticks(range(D))
        ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=12)
        if col == 0:
            ax.set_yticks(range(D))
            ax.set_yticklabels(fnames, fontsize=12)
        else:
            ax.set_yticks([])
        ax.set_title(f"{model}  (n={int(mask.sum())})", fontsize=13)

    fig.suptitle("Pairwise Feature Correlations per Model", fontsize=18, y=0.99)
    # Reserve right margin for an EXPLICIT colorbar axes, then place the cax
    # manually. Avoids matplotlib's auto-layout pushing the colorbar into the
    # last column of subplots.
    fig.subplots_adjust(top=0.93, right=0.91, left=0.05,
                        wspace=0.18, hspace=0.30)
    if im is not None:
        cax = fig.add_axes([0.935, 0.10, 0.012, 0.78])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Pearson coefficient", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
    out_path = out_dir / "features_correlation.pdf"
    fig.savefig(out_path, dpi=dpi)            # don't use bbox_inches="tight"
                                              # (would re-shift the manual cax)
    plt.close(fig)
    return out_path


def _save_pca(F_all, model_idx, models, Z, explained, out_dir, dpi) -> Path:
    """3D PCA, 8 panels (one per model). Uses CANONICAL PCA coordinates from
    feature_embedding cache so every PCA visualisation in this project shares
    the same projection. Each panel shows its model in colour against a faint
    grey background of the other 7 models' experts."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Shared axis limits, derived from the full point cloud.
    pad = 0.04
    lims = []
    for d in range(3):
        lo, hi = float(Z[:, d].min()), float(Z[:, d].max())
        lims.append((lo - pad * (hi - lo), hi + pad * (hi - lo)))

    base_colors = _model_palette(len(models))
    fig = plt.figure(figsize=(24, 12))
    for i, model in enumerate(models):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        mask = (model_idx == i)
        # Faint background = experts from the other 7 models.
        ax.scatter(Z[~mask, 0], Z[~mask, 1], Z[~mask, 2],
                   s=1.5, color="lightgray", alpha=0.12,
                   edgecolors="none", depthshade=False, rasterized=True)
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                   s=5, color=base_colors[i], alpha=0.75,
                   edgecolors="none", depthshade=False, rasterized=True)
        ax.set_title(f"{model}  (n={int(mask.sum())})", fontsize=13, pad=2)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_xlabel(f"PC1 ({100*explained[0]:.1f}%)", fontsize=13, labelpad=2)
        ax.set_ylabel(f"PC2 ({100*explained[1]:.1f}%)", fontsize=13, labelpad=2)
        ax.set_zlabel(f"PC3 ({100*explained[2]:.1f}%)", fontsize=13, labelpad=2)
        ax.tick_params(labelsize=7)
        # Slight rotation away from the default for a more informative angle.
        ax.view_init(elev=22, azim=-55)

    fig.suptitle(
        "3D PCA of all experts in 10-D feature space",
        fontsize=18, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(top=0.93, hspace=0.05, wspace=0.05)
    out_path = out_dir / "features_pca.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_pca_per_model(F_all, model_idx, models, out_dir, dpi) -> Path:
    """Per-model 3D PCA. Unlike the global PCA, this fits a SEPARATE PCA(3)
    on each model's own features (F_m) so each panel has its own coordinate
    system and per-model variance-explained labels.

    Caveats (mention in the paper if used):
      - Coordinates ACROSS panels are NOT comparable (different bases).
      - Variance % differs per panel — it's the fraction of THAT model's
        internal variance captured, not the global variance.
      - Answers a different question than the global PCA: 'what are the
        dominant axes of variation WITHIN each model?', not 'where does
        each model sit in a shared feature space?'.
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_colors = _model_palette(len(models))
    fig = plt.figure(figsize=(24, 12))
    for i, model in enumerate(models):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        mask = (model_idx == i)
        n = int(mask.sum())
        if n < 4:
            ax.set_title(f"{model}  (n={n}; too few)", fontsize=13, pad=2)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            continue
        F_m = F_all[mask]
        pca = PCA(n_components=3)
        Z_m = pca.fit_transform(F_m)
        explained = pca.explained_variance_ratio_

        ax.scatter(Z_m[:, 0], Z_m[:, 1], Z_m[:, 2],
                   s=5, color=base_colors[i], alpha=0.75,
                   edgecolors="none", depthshade=False, rasterized=True)
        ax.set_title(f"{model}  (n={n})", fontsize=13, pad=2)
        ax.set_xlabel(f"PC1 ({100*explained[0]:.1f}%)", fontsize=13, labelpad=2)
        ax.set_ylabel(f"PC2 ({100*explained[1]:.1f}%)", fontsize=13, labelpad=2)
        ax.set_zlabel(f"PC3 ({100*explained[2]:.1f}%)", fontsize=13, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=22, azim=-55)

    fig.suptitle(
        "Per-model 3D PCA — each panel fitted independently on its own model's "
        "features (coordinates not comparable across panels)",
        fontsize=18, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(top=0.93, hspace=0.05, wspace=0.05)
    out_path = out_dir / "features_pca_per_model.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_embedding(F_all, model_idx, models, Z, out_dir, dpi,
                    method: str = "umap") -> Path:
    """Render the CANONICAL UMAP (from feature_embedding cache) coloured by
    model. No subsampling, no per-call UMAP fit — the projection is shared
    across all visualisation scripts so different colourings of the same
    embedding are strictly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = (f"{method.upper()} of all experts in feature space "
             f"(n={len(F_all)}), coloured by model")
    fname = f"features_{method}.pdf"

    fig, ax = plt.subplots(figsize=(11, 9))
    base_colors = _model_palette(len(models))
    counts = np.bincount(model_idx, minlength=len(models))
    order = np.argsort(-counts)
    for i in order:
        mask = (model_idx == i)
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=4,
                   color=base_colors[i], alpha=0.55,
                   label=f"{models[i]} (n={int(mask.sum())})",
                   edgecolors="none", rasterized=True)
    ax.set_title(title, fontsize=15)
    ax.legend(loc="best", fontsize=12, framealpha=0.9, markerscale=3.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{method.upper()}-1", fontsize=13)
    ax.set_ylabel(f"{method.upper()}-2", fontsize=13)
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

    fig, ax = plt.subplots(figsize=(13, max(5, 0.7 * k + 3)))
    im = ax.imshow(M_frac, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for c in range(k):
        for mi in range(n_models):
            colour = "white" if M_frac[c, mi] < 0.55 else "black"
            ax.text(mi, c, f"{100*M_frac[c, mi]:.0f}%",
                     ha="center", va="center", fontsize=13, color=colour)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"cluster {c}" for c in range(k)], fontsize=13)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fraction of model's experts in this cluster", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    ax.set_title(f"K-means (k={k}) cluster composition", fontsize=18)
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
                        help="Skip the UMAP figure (the canonical UMAP is "
                             "still computed and cached for downstream "
                             "scripts; this only skips the figure render).")
    parser.add_argument("--kmeans-k", type=int, default=8,
                        help="K for k-means (default 8 = one per model).")
    parser.add_argument("--recompute-embedding", action="store_true",
                        help="Recompute the canonical UMAP+PCA embedding "
                             "from scratch (default loads from cache if "
                             "present). Use after any change to the feature "
                             "construction.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Feature analysis on task = {args.task} ===\n")
    # Canonical embedding (cached). Single source of truth for UMAP/PCA
    # coordinates across all visualisation scripts.
    from experiments.feature_embedding import load_or_compute_embedding
    emb = load_or_compute_embedding(args.task, recompute=args.recompute_embedding)
    F_all = emb["F_all"]
    model_idx = emb["model_idx"]
    models = emb["models"]
    umap_2d = emb["umap_2d"]
    pca_3d = emb["pca_3d"]
    pca_explained = emb["pca_explained"]
    print(f"\n  Total: {len(F_all)} experts × {F_all.shape[1]} features\n")

    paths: list[Path] = []
    paths.append(_save_cosine_similarity(F_all, model_idx, models, out_dir, args.dpi))
    paths.append(_save_correlation(F_all, model_idx, models, out_dir, args.dpi))
    paths.append(_save_pca(F_all, model_idx, models, pca_3d, pca_explained,
                           out_dir, args.dpi))
    paths.append(_save_pca_per_model(F_all, model_idx, models, out_dir, args.dpi))
    if not args.skip_umap:
        paths.append(_save_embedding(F_all, model_idx, models, umap_2d,
                                     out_dir, args.dpi, method="umap"))
    paths.append(_kmeans_analysis(F_all, model_idx, models, out_dir,
                                  args.dpi, k=args.kmeans_k))

    print(f"\n  Saved:")
    for p in paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()
