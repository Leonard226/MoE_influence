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
    "depth", "out_norm", "in_norm", "load", "act",
    "class_content", "class_functional", "class_punctuation",
    "class_numeric", "class_special",
]


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
def _save_per_feature_ridge(F_all, model_idx, models, out_dir, dpi) -> Path:
    """One row per feature, ridge of all 8 models within the row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = F_all.shape[1]
    fnames = (FEATURE_NAMES[:D] if D <= len(FEATURE_NAMES)
              else [f"feat_{d}" for d in range(D)])
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(D, 1, figsize=(14, 1.3 * D + 1), sharex=False)
    if D == 1:
        axes = [axes]
    for d, ax in enumerate(axes):
        bins = np.linspace(0.0, 1.0, 60)
        centres = (bins[:-1] + bins[1:]) / 2
        for i, m in enumerate(models):
            mask = (model_idx == i)
            if mask.sum() == 0:
                continue
            counts, _ = np.histogram(F_all[mask, d], bins=bins)
            frac = counts / counts.sum() if counts.sum() > 0 else counts
            ax.fill_between(centres, 0, frac, step="mid",
                            facecolor=base_colors[i], edgecolor=base_colors[i],
                            linewidth=1.0, alpha=0.40,
                            label=m if d == 0 else None)
        ax.set_title(fnames[d], loc="left", x=0.01, y=0.78,
                      fontsize=10, fontweight="bold")
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("feature value (all features in [0, 1])")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    fig.suptitle("Feature distributions by model — one row per feature",
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / "features_per_feature_ridge.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
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
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    Z = pca.fit_transform(F_all)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11, 9))
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    # Plot small models last so they're visible above the dense Qwen scatter.
    counts = np.bincount(model_idx, minlength=len(models))
    order = np.argsort(-counts)             # largest first → drawn at bottom
    for i in order:
        mask = (model_idx == i)
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=4,
                   color=base_colors[i], alpha=0.45,
                   label=f"{models[i]} (n={int(mask.sum())})",
                   edgecolors="none")
    ax.set_xlabel(f"PC1  ({100*explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2  ({100*explained[1]:.1f}% var)")
    ax.set_title("PCA of all experts in feature space, coloured by model")
    ax.legend(loc="best", fontsize=8, framealpha=0.85, markerscale=2.0)
    fig.tight_layout()
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
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    counts = np.bincount(idx_sub, minlength=len(models))
    order = np.argsort(-counts)
    for i in order:
        mask = (idx_sub == i)
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], s=5,
                   color=base_colors[i], alpha=0.55,
                   label=f"{models[i]} (n={int(mask.sum())})",
                   edgecolors="none")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.85, markerscale=2.0)
    ax.set_xticks([]); ax.set_yticks([])
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
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(F_all)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    n_models = len(models)
    M = np.zeros((k, n_models), dtype=np.int64)
    for c in range(k):
        for mi in range(n_models):
            M[c, mi] = int(((labels == c) & (model_idx == mi)).sum())

    # Row-normalise to "fraction of this cluster from this model".
    row_sum = M.sum(axis=1, keepdims=True).clip(min=1)
    M_frac = M / row_sum

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * k + 2)))
    im = ax.imshow(M_frac, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for c in range(k):
        for mi in range(n_models):
            if M[c, mi] == 0:
                continue
            colour = "white" if M_frac[c, mi] < 0.55 else "black"
            ax.text(mi, c, f"{M[c, mi]}", ha="center", va="center",
                     fontsize=7, color=colour)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"cluster {c} (n={int(row_sum[c, 0])})" for c in range(k)])
    fig.colorbar(im, ax=ax, label="fraction of cluster from model")
    ax.set_title(f"K-means (k={k}) cluster-by-model composition\n"
                  f"colour = fraction; text = raw count")
    fig.tight_layout()
    out_path = out_dir / "features_kmeans.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Console summary: cluster purity.
    print(f"\n  K-means (k={k}) cluster purity (dominant-model fraction):")
    for c in range(k):
        sizes = M[c]
        total = int(sizes.sum())
        if total == 0:
            continue
        dom = sizes.argmax()
        purity = sizes[dom] / total
        print(f"    cluster {c} (n={total:6d}): purity={purity:.2f} "
              f"(dominant = {models[dom]})")

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
    paths.append(_save_per_feature_ridge(F_all, model_idx, models, out_dir, args.dpi))
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
