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
def _save_umap_by_cluster(F_all: np.ndarray, labels: np.ndarray,
                          out_dir: Path, dpi: int, subsample: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import colorsys

    try:
        from umap import UMAP
    except ImportError:
        print("WARNING: umap-learn not installed; falling back to t-SNE.")
        from sklearn.manifold import TSNE as _Embedding
        embedder_name = "t-SNE"
        embedder = _Embedding(n_components=2, random_state=0,
                              perplexity=30, init="pca")
    else:
        embedder_name = "UMAP"
        embedder = UMAP(n_components=2, random_state=0,
                        n_neighbors=30, min_dist=0.1)

    n = len(F_all)
    if subsample > 0 and n > subsample:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, subsample, replace=False)
        F_sub, lab_sub = F_all[sel], labels[sel]
    else:
        F_sub, lab_sub = F_all, labels

    print(f"  fitting {embedder_name} on {len(F_sub)} points ...", flush=True)
    t0 = time.time()
    Z = embedder.fit_transform(F_sub)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    noise = lab_sub == -1
    if noise.any():
        ax.scatter(Z[noise, 0], Z[noise, 1], s=3, color="lightgray",
                   alpha=0.35, edgecolors="none",
                   label=f"noise (n={int(noise.sum())})")

    unique = sorted(set(int(c) for c in lab_sub) - {-1})
    palette = [colorsys.hsv_to_rgb(i / max(len(unique), 1), 0.92, 0.88)
               for i in range(len(unique))]

    # Draw smaller clusters last so they stay visible above big ones.
    cluster_sizes = {c: int((lab_sub == c).sum()) for c in unique}
    draw_order = sorted(unique, key=lambda c: -cluster_sizes[c])
    for c in draw_order:
        col = palette[unique.index(c)]
        mask = lab_sub == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, color=col, alpha=0.75,
                   edgecolors="none",
                   label=f"cluster {c} (n={cluster_sizes[c]})")

    # Cluster ID labels at centroids — easy to cross-reference to signatures.
    for c in unique:
        mask = lab_sub == c
        if not mask.any():
            continue
        cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
        ax.text(cx, cy, str(c), fontsize=14, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="black",
                          linewidth=0.7, alpha=0.9))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{embedder_name}-1", fontsize=13)
    ax.set_ylabel(f"{embedder_name}-2", fontsize=13)
    ax.legend(loc="best", fontsize=10, framealpha=0.92, markerscale=2.5,
              ncol=2 if len(unique) > 8 else 1)
    ax.set_title(f"{embedder_name} of all experts, coloured by HDBSCAN cluster "
                  f"(n={len(F_sub)})", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "clusters_umap.pdf"
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
    parser.add_argument("--subsample", type=int, default=8000,
                        help="Random subsample for the UMAP scatter "
                             "(default 8000; 0 = use all).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== HDBSCAN cluster analysis on task = {args.task} ===\n")
    F_all, model_idx, models = _build_all(args.task)
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
    paths.append(_save_umap_by_cluster(F_all, labels, out_dir, args.dpi,
                                       args.subsample))
    paths.append(_save_summary(F_all, labels, model_idx, models, args, out_dir))

    print(f"\n  saved:")
    for p in paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()
