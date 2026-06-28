"""Per-model HDBSCAN clustering of the expert feature space F_m.

Counterpart to cluster_experts.py: instead of clustering the pooled
31,520-expert F matrix, we cluster each model's F_m independently. This
exposes within-model substructure that gets washed out by the pooled
analysis (where the dominant axis is "which model are you from").

Per-model HDBSCAN params:
  - cluster_selection_method = leaf (matches the pooled run)
  - min_cluster_size scales with the model's expert count:
      max(10, floor(0.02 * n_experts))
    so small models (Mixtral) get min_cluster_size = 10, and large models
    (Qwen3-235B) get min_cluster_size = 240.

Reads:
  ${result_path}/circuits/feature_inspection/embedding_<task>.npz
  produced by inspect_features.py (contains F_all, model_idx, models).

Writes (to ${result_path}/circuits/feature_inspection/):
  - clusters_per_model_umap.pdf      2 x 4 grid, one panel per model,
                                      UMAP coloured by within-model cluster
  - clusters_per_model_summary.txt   per-model cluster signatures + counts

Usage:
  python experiments/cluster_experts_per_model.py
  python experiments/cluster_experts_per_model.py --task math --min-frac 0.03
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
DEFAULT_OUT_DIR = Path(CFG["result_path"]) / "circuits" / "feature_inspection"

FEATURE_NAMES = ["depth", "out", "in", "load", "act",
                 "content", "functional", "punctuation", "numeric", "special"]


def _hdbscan(F: np.ndarray, min_cluster_size: int, method: str = "leaf"
             ) -> np.ndarray:
    """Returns labels (-1 = noise, 0..K-1 = cluster ids)."""
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size,
                       cluster_selection_method=method).fit_predict(F)
    except ImportError:
        import hdbscan as _hdb
        return _hdb.HDBSCAN(min_cluster_size=min_cluster_size,
                            cluster_selection_method=method).fit_predict(F)


def _umap_2d(F: np.ndarray, seed: int = 0) -> np.ndarray:
    """Returns (n, 2) UMAP coords for F."""
    import umap
    n_neighbors = min(15, max(2, F.shape[0] - 1))
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2,
                     random_state=seed).fit_transform(F)


def _signature(F_cluster: np.ndarray, F_model: np.ndarray, top_k: int = 3
               ) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Returns (mean_vec, top-k (name, mean, z) sorted by |z|).
    Z-score is centred / scaled by the MODEL's mean / std, not global."""
    mu_cluster = F_cluster.mean(axis=0)
    mu_model = F_model.mean(axis=0)
    sd_model = F_model.std(axis=0)
    sd_model = np.where(sd_model < 1e-8, 1e-8, sd_model)
    z = (mu_cluster - mu_model) / sd_model
    order = np.argsort(-np.abs(z))[:top_k]
    return mu_cluster, [(FEATURE_NAMES[i], float(mu_cluster[i]), float(z[i]))
                        for i in order]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--min-frac", type=float, default=0.02,
                   help="min_cluster_size = max(10, floor(min_frac * n_experts)).")
    p.add_argument("--min-floor", type=int, default=10,
                   help="Floor on min_cluster_size (default 10).")
    p.add_argument("--method", default="leaf",
                   choices=["leaf", "eom"])
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / f"embedding_{args.task}.npz"
    if not emb_path.exists():
        print(f"ERROR: cached embedding not found: {emb_path}", file=sys.stderr)
        print(f"  Run inspect_features.py --task {args.task} first.")
        sys.exit(1)
    print(f"Loading {emb_path} ...")
    d = np.load(emb_path, allow_pickle=True)
    F_all = np.asarray(d["F_all"]).astype(np.float32)
    model_idx = np.asarray(d["model_idx"]).astype(int)
    MODELS = list(d["models"])
    n_models = len(MODELS)
    print(f"  pooled F: {F_all.shape}; models: {MODELS}")

    # Per-model clustering.
    per_model: dict[str, dict] = {}
    for mi, model in enumerate(MODELS):
        mask = (model_idx == mi)
        F_m = F_all[mask]
        n = F_m.shape[0]
        mcs = max(args.min_floor, int(args.min_frac * n))
        print(f"\n[{model}] n_experts={n}; min_cluster_size={mcs}")
        labels = _hdbscan(F_m, mcs, args.method)
        print(f"  HDBSCAN → {labels.max() + 1} clusters, "
              f"{int((labels == -1).sum())} noise "
              f"({100 * (labels == -1).mean():.1f}%)")
        print(f"  computing per-model UMAP on {n} points ...", flush=True)
        umap2 = _umap_2d(F_m)
        per_model[model] = {
            "F_m": F_m, "labels": labels, "umap": umap2,
            "min_cluster_size": mcs,
        }

    # --- composite UMAP grid (2 x 4, one per model) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows),
                              squeeze=False)
    for mi, model in enumerate(MODELS):
        r, c = mi // n_cols, mi % n_cols
        ax = axes[r, c]
        labels = per_model[model]["labels"]
        umap2 = per_model[model]["umap"]
        n_clusters = int(labels.max() + 1)
        # Noise first (grey, behind), then clusters on top.
        noise_mask = labels == -1
        ax.scatter(umap2[noise_mask, 0], umap2[noise_mask, 1],
                   s=3, c="#cccccc", alpha=0.5, linewidths=0,
                   rasterized=True)
        if n_clusters > 0:
            cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
            for k in range(n_clusters):
                m = labels == k
                ax.scatter(umap2[m, 0], umap2[m, 1],
                           s=6, color=cmap(k % cmap.N), alpha=0.85,
                           linewidths=0, label=f"c{k}",
                           rasterized=True)
        ax.set_title(f"{model}\n($n = {len(labels)}$, $K = {n_clusters}$, "
                      f"noise $= {100*(labels==-1).mean():.0f}\\%$)",
                      fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if n_clusters and n_clusters <= 8:
            ax.legend(loc="best", fontsize=7, framealpha=0.85,
                      markerscale=1.5, handletextpad=0.3)

    # Hide unused panels.
    for ax in axes.flat[n_models:]:
        ax.set_visible(False)

    fig.tight_layout()
    out_pdf = out_dir / "clusters_per_model_umap.pdf"
    fig.savefig(out_pdf, dpi=args.dpi)
    plt.close(fig)
    print(f"\nSaved {out_pdf}")

    # --- summary text ---
    out_txt = out_dir / "clusters_per_model_summary.txt"
    with open(out_txt, "w") as f:
        f.write("=" * 78 + "\n")
        f.write(f"Per-model HDBSCAN clustering on F_m  (task = {args.task})\n")
        f.write("=" * 78 + "\n")
        f.write(f"  cluster_selection_method : {args.method}\n")
        f.write(f"  min_cluster_size formula : max({args.min_floor}, "
                f"{args.min_frac} * n_experts)\n\n")
        for model in MODELS:
            info = per_model[model]
            F_m, labels = info["F_m"], info["labels"]
            n = F_m.shape[0]
            n_clusters = int(labels.max() + 1)
            n_noise = int((labels == -1).sum())
            f.write("-" * 78 + "\n")
            f.write(f"[{model}] n_experts={n}  min_cluster_size={info['min_cluster_size']}\n")
            f.write(f"  → {n_clusters} clusters; "
                    f"noise = {n_noise} ({100*n_noise/n:.1f}%)\n")
            for k in range(n_clusters):
                F_k = F_m[labels == k]
                n_k = F_k.shape[0]
                mu, top = _signature(F_k, F_m, top_k=3)
                sig = ", ".join(
                    f"{name}={val:.3f} (z={z:+.2f})" for name, val, z in top
                )
                f.write(f"   c{k}  n={n_k:5d} ({100*n_k/n:5.1f}%)  {sig}\n")
            f.write("\n")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
