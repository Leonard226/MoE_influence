"""Discover and characterise expert clusters in feature space.

Pipeline:
  1. Build F_all (10-D per-expert features) across all 8 models on the chosen
     task, using the SAME log-max load + log-max act normalisation as the
     headline sweep. Pooled across models.
  2. Run HDBSCAN — auto-discovers the number of clusters and tags experts
     that don't fit any cluster as 'noise'.
  3. For each discovered cluster, compute:
       - Centroid feature vector
       - Z-score signature ((centroid − global_mean) / global_std per feature)
       - Per-model membership
       - Auto-suggested name from the top-3 most-deviating features.
  4. Visualise (a) cluster signatures as z-score bar charts, one panel per
     cluster, and (b) the UMAP scatter recoloured by cluster ID with cluster
     centroid labels drawn over each blob.
  5. Write a human-readable summary file with per-cluster signature + model
     composition.

Outputs (under {result_path}/circuits/feature_inspection/):
  - clusters_signatures.pdf   per-cluster z-score bar charts
  - clusters_umap.pdf         UMAP coloured by cluster (noise = grey)
  - clusters_summary.txt      per-cluster signature + model composition

Usage:
  python experiments/cluster_experts.py
  python experiments/cluster_experts.py --task math --min-cluster-size 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Clamp BLAS threads before numpy / sklearn imports (avoids OpenBLAS many-core
# segfault on piora). 4 threads is plenty for HDBSCAN + UMAP.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Reuse F-loading from inspect_features so we don't duplicate the build_triple
# wiring. _build_all returns (F_all, model_idx, MODELS).
from experiments.inspect_features import (   # noqa: E402
    _build_all, FEATURE_NAMES, DEFAULT_OUT_DIR,
)


# -------------------- HDBSCAN with graceful fallback -----------------------
def _run_hdbscan(F: np.ndarray, min_cluster_size: int,
                 min_samples: int | None,
                 cluster_selection_method: str = "eom") -> np.ndarray:
    """Returns labels: -1 = noise, 0..K-1 = cluster ids."""
    # sklearn 1.3+ has HDBSCAN built-in; fall back to the standalone package.
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
    except ImportError:
        try:
            import hdbscan as _hdb
            clusterer = _hdb.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
            )
        except ImportError:
            print("ERROR: neither sklearn.cluster.HDBSCAN nor hdbscan-learn "
                  "is installed.\n"
                  "Install with: pip install scikit-learn>=1.3   OR   "
                  "pip install hdbscan",
                  file=sys.stderr)
            sys.exit(1)
    print(f"  running HDBSCAN (min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, method={cluster_selection_method}) on "
          f"{len(F)} points...", flush=True)
    t0 = time.time()
    labels = clusterer.fit_predict(F)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return labels


# -------------------- signatures (z-score bars per cluster) ----------------
def _autoname_cluster(z: np.ndarray, fnames: list[str], top_k: int = 3) -> str:
    top = np.argsort(-np.abs(z))[:top_k]
    parts = [f"{'+' if z[t] > 0 else '-'}{fnames[t]}" for t in top]
    return ", ".join(parts)


def _save_signatures(F_all: np.ndarray, labels: np.ndarray,
                     model_idx: np.ndarray, models: list[str],
                     out_dir: Path, dpi: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = F_all.shape[1]
    fnames = (FEATURE_NAMES[:D] if D <= len(FEATURE_NAMES)
              else [f"feat_{d}" for d in range(D)])
    global_mean = F_all.mean(axis=0)
    global_std = F_all.std(axis=0).clip(min=1e-9)

    unique = sorted(set(int(c) for c in labels) - {-1})
    n_clusters = len(unique)
    if n_clusters == 0:
        out_path = out_dir / "clusters_signatures.pdf"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "HDBSCAN found 0 clusters\n(try a smaller "
                          "--min-cluster-size)", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path

    n_cols = min(4, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 5.2, n_rows * 4.3),
                              squeeze=False)

    # Track global z-range for shared y-axis.
    z_vals = []
    centroids = {}
    for c in unique:
        mask = labels == c
        centroid = F_all[mask].mean(axis=0)
        z = (centroid - global_mean) / global_std
        centroids[c] = z
        z_vals.append(z)
    z_max = float(np.max(np.abs(np.concatenate(z_vals))))
    ymax = max(1.0, 1.05 * z_max)

    for idx, c in enumerate(unique):
        ax = axes[idx // n_cols, idx % n_cols]
        z = centroids[c]
        mask = labels == c
        size = int(mask.sum())

        colours = ["#c8324c" if zi > 0 else "#1f6cb0" for zi in z]
        ax.bar(range(D), z, color=colours, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(range(D))
        ax.set_xticklabels(fnames, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("z-score vs global mean", fontsize=9)
        ax.set_ylim(-ymax, ymax)
        ax.grid(alpha=0.18, axis="y")

        sig = _autoname_cluster(z, fnames, top_k=3)
        # Per-model membership: top 3 contributors only, with raw counts.
        m_counts = np.bincount(model_idx[mask], minlength=len(models))
        m_order = np.argsort(-m_counts)
        m_top = [(models[m], int(m_counts[m])) for m in m_order
                 if m_counts[m] > 0][:3]
        m_str = ", ".join(f"{m}:{n}" for m, n in m_top)
        ax.set_title(f"Cluster {c}   n={size}\n"
                     f"signature: {sig}\n"
                     f"top models: {m_str}",
                     fontsize=9.5)

    # Hide unused subplots.
    for idx in range(n_clusters, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    n_noise = int((labels == -1).sum())
    fig.suptitle(
        f"HDBSCAN cluster signatures ({n_clusters} clusters; "
        f"{n_noise} noise points = {100*n_noise/len(labels):.1f}%)\n"
        "red bars = feature above global mean; blue = below; |z| ≥ 1 ≈ noteworthy",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / "clusters_signatures.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# -------------------- UMAP coloured by cluster -----------------------------
def _save_umap_by_cluster(Z: np.ndarray, labels: np.ndarray,
                          out_dir: Path, dpi: int) -> Path:
    """Render the CANONICAL UMAP (from feature_embedding cache) coloured by
    HDBSCAN cluster. Identical projection coordinates to the model-coloured
    UMAP in inspect_features.py, so the two are directly comparable.

    Two-panel layout:
      Left  — all experts; noise faded grey, clusters in colour.
      Right — clusters only (noise hidden), axes auto-zoomed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import colorsys

    unique = sorted(set(int(c) for c in labels) - {-1})
    palette = [colorsys.hsv_to_rgb(i / max(len(unique), 1), 0.92, 0.88)
               for i in range(len(unique))]
    cluster_sizes = {c: int((labels == c).sum()) for c in unique}
    draw_order = sorted(unique, key=lambda c: -cluster_sizes[c])

    def _draw_panel(ax, show_noise: bool, label_min_size: int):
        noise = labels == -1
        if show_noise and noise.any():
            ax.scatter(Z[noise, 0], Z[noise, 1], s=1.5, color="lightgray",
                       alpha=0.10, edgecolors="none", rasterized=True,
                       label=f"noise (n={int(noise.sum())})")
        for c in draw_order:
            col = palette[unique.index(c)]
            mask = labels == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=7, color=col, alpha=0.85,
                       edgecolors="none", rasterized=True,
                       label=f"cluster {c} (n={cluster_sizes[c]})")
        for c in unique:
            if cluster_sizes[c] < label_min_size:
                continue
            mask = labels == c
            cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
            ax.text(cx, cy, str(c), fontsize=12, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22",
                              facecolor="white", edgecolor="black",
                              linewidth=0.7, alpha=0.92))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1", fontsize=13)
        ax.set_ylabel("UMAP-2", fontsize=13)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    _draw_panel(axes[0], show_noise=True, label_min_size=50)
    axes[0].set_title(
        f"All experts on canonical UMAP — noise greyed, clusters in colour\n"
        f"(n={len(Z)}; "
        f"{100*sum(cluster_sizes.values())/len(Z):.1f}% clustered)",
        fontsize=13,
    )

    cluster_mask = labels != -1
    Z_clusters = Z[cluster_mask]
    _draw_panel(axes[1], show_noise=False, label_min_size=30)
    if len(Z_clusters) > 0:
        pad = 0.05
        x_min, x_max = Z_clusters[:, 0].min(), Z_clusters[:, 0].max()
        y_min, y_max = Z_clusters[:, 1].min(), Z_clusters[:, 1].max()
        axes[1].set_xlim(x_min - pad * (x_max - x_min),
                         x_max + pad * (x_max - x_min))
        axes[1].set_ylim(y_min - pad * (y_max - y_min),
                         y_max + pad * (y_max - y_min))
    axes[1].set_title(
        f"Clusters only (noise hidden, zoomed to cluster bbox)\n"
        f"{len(unique)} clusters, n={int(cluster_mask.sum())}",
        fontsize=13,
    )
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.92,
                   markerscale=2.0, ncol=2 if len(unique) > 8 else 1,
                   bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    out_path = out_dir / "clusters_umap.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# -------------------- composite: same UMAP, two colourings -----------------
def _save_composite_umap(Z: np.ndarray, model_idx: np.ndarray,
                         models: list[str], labels: np.ndarray,
                         out_dir: Path, dpi: int) -> Path:
    """The headline figure for the paper: SAME UMAP coordinates, side-by-side
    coloured by model (left) and by HDBSCAN cluster (right). Lets reviewers
    immediately verify "do same-model experts share clusters?" by aligning
    panels along the shared spatial layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import colorsys

    # Shared axis limits across both panels.
    pad = 0.04
    x_min, x_max = float(Z[:, 0].min()), float(Z[:, 0].max())
    y_min, y_max = float(Z[:, 1].min()), float(Z[:, 1].max())
    xlim = (x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ylim = (y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True, sharey=True)

    # ----- Left: coloured by model -----
    model_colors = [colorsys.hsv_to_rgb(i / len(models), 0.92, 0.88)
                    for i in range(len(models))]
    counts = np.bincount(model_idx, minlength=len(models))
    order_models = np.argsort(-counts)
    for i in order_models:
        mask = model_idx == i
        if not mask.any():
            continue
        axes[0].scatter(Z[mask, 0], Z[mask, 1], s=4,
                        color=model_colors[i], alpha=0.55,
                        edgecolors="none", rasterized=True,
                        label=f"{models[i]} (n={int(mask.sum())})")
    axes[0].set_title("Coloured by model", fontsize=14)
    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.92,
                   markerscale=3.0)

    # ----- Right: coloured by HDBSCAN cluster -----
    unique = sorted(set(int(c) for c in labels) - {-1})
    cluster_colors = [colorsys.hsv_to_rgb(i / max(len(unique), 1), 0.92, 0.88)
                      for i in range(len(unique))]
    cluster_sizes = {c: int((labels == c).sum()) for c in unique}

    noise = labels == -1
    if noise.any():
        axes[1].scatter(Z[noise, 0], Z[noise, 1], s=1.5, color="lightgray",
                        alpha=0.10, edgecolors="none", rasterized=True,
                        label=f"noise (n={int(noise.sum())})")
    for c in sorted(unique, key=lambda x: -cluster_sizes[x]):
        col = cluster_colors[unique.index(c)]
        mask = labels == c
        axes[1].scatter(Z[mask, 0], Z[mask, 1], s=7, color=col, alpha=0.85,
                        edgecolors="none", rasterized=True,
                        label=f"cluster {c} (n={cluster_sizes[c]})")
    # Cluster ID labels at centroids for clusters with n >= 50.
    for c in unique:
        if cluster_sizes[c] < 50:
            continue
        mask = labels == c
        cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
        axes[1].text(cx, cy, str(c), fontsize=11, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                               edgecolor="black", linewidth=0.7, alpha=0.92))
    axes[1].set_title("Coloured by HDBSCAN cluster", fontsize=14)
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.92,
                   markerscale=2.5,
                   ncol=2 if len(unique) > 8 else 1)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("UMAP-1", fontsize=13)
    axes[0].set_ylabel("UMAP-2", fontsize=13)

    fig.tight_layout()
    out_path = out_dir / "clusters_composite_umap.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# -------------------- text summary -----------------------------------------
def _save_summary(F_all: np.ndarray, labels: np.ndarray,
                  model_idx: np.ndarray, models: list[str],
                  args: argparse.Namespace, out_dir: Path) -> Path:
    D = F_all.shape[1]
    fnames = (FEATURE_NAMES[:D] if D <= len(FEATURE_NAMES)
              else [f"feat_{d}" for d in range(D)])
    global_mean = F_all.mean(axis=0)
    global_std = F_all.std(axis=0).clip(min=1e-9)

    unique = sorted(set(int(c) for c in labels) - {-1})
    n_noise = int((labels == -1).sum())

    lines: list[str] = []
    def w(s: str = ""):
        lines.append(s)
        print(s)

    w("=" * 78)
    w("HDBSCAN cluster analysis on expert feature space")
    w("=" * 78)
    w(f"  task                     : {args.task}")
    w(f"  total experts            : {len(labels)}")
    w(f"  feature dimensionality   : {D}  ({', '.join(fnames)})")
    w(f"  min_cluster_size         : {args.min_cluster_size}")
    w(f"  min_samples              : {args.min_samples}")
    w(f"  cluster_selection_method : {args.cluster_selection_method}")
    w(f"  → {len(unique)} clusters discovered; "
      f"{n_noise} noise points ({100*n_noise/len(labels):.2f}%)")
    w("")

    for c in unique:
        mask = labels == c
        size = int(mask.sum())
        centroid = F_all[mask].mean(axis=0)
        z = (centroid - global_mean) / global_std

        # Top features by |z|
        order = np.argsort(-np.abs(z))
        top_str = ", ".join(
            f"{fnames[t]}={centroid[t]:.3f} (z={z[t]:+.2f})"
            for t in order[:5]
        )

        sig = _autoname_cluster(z, fnames, top_k=3)

        # Per-model breakdown
        m_counts = np.bincount(model_idx[mask], minlength=len(models))
        m_order = np.argsort(-m_counts)
        m_total = int(m_counts.sum())
        m_lines = ", ".join(
            f"{models[m]}={int(m_counts[m])} ({100*m_counts[m]/m_total:.1f}%)"
            for m in m_order if m_counts[m] > 0
        )

        w("-" * 78)
        w(f"Cluster {c}   n = {size}   ({100*size/len(labels):.2f}% of all experts)")
        w(f"  signature (z-score top-3):  {sig}")
        w(f"  top features:               {top_str}")
        w(f"  model composition:          {m_lines}")
    w("=" * 78)

    out_path = out_dir / "clusters_summary.txt"
    out_path.write_text("\n".join(lines))
    return out_path


# -------------------- entry point ------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="c4")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--min-cluster-size", type=int, default=100,
                        help="HDBSCAN min cluster size. Larger → fewer / "
                             "coarser clusters (default 100).")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="HDBSCAN min_samples. None → defaults to "
                             "min_cluster_size (more conservative).")
    parser.add_argument("--cluster-selection-method", default="eom",
                        choices=["eom", "leaf"],
                        help="HDBSCAN cluster selection: 'eom' (excess of "
                             "mass, default) gives stable big clusters; "
                             "'leaf' returns finer-grained clusters.")
    parser.add_argument("--recompute-embedding", action="store_true",
                        help="Force-recompute the canonical UMAP+PCA from "
                             "scratch (default loads from cache if present).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== HDBSCAN cluster analysis on task = {args.task} ===\n")
    # Canonical embedding (cached). Same UMAP coordinates as
    # inspect_features.py, so all visualisations align.
    from experiments.feature_embedding import load_or_compute_embedding
    emb = load_or_compute_embedding(args.task,
                                    recompute=args.recompute_embedding)
    F_all = emb["F_all"]
    model_idx = emb["model_idx"]
    models = emb["models"]
    umap_2d = emb["umap_2d"]
    print(f"\n  Total: {len(F_all)} experts × {F_all.shape[1]} features\n")

    labels = _run_hdbscan(
        F_all,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
    )
    n_clusters = len(set(int(c) for c in labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  → {n_clusters} clusters, {n_noise} noise points "
          f"({100*n_noise/len(labels):.2f}%)\n")

    paths: list[Path] = []
    paths.append(_save_signatures(F_all, labels, model_idx, models,
                                  out_dir, args.dpi))
    paths.append(_save_umap_by_cluster(umap_2d, labels, out_dir, args.dpi))
    paths.append(_save_composite_umap(umap_2d, model_idx, models, labels,
                                      out_dir, args.dpi))
    paths.append(_save_summary(F_all, labels, model_idx, models, args, out_dir))

    print(f"\n  saved:")
    for p in paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()
