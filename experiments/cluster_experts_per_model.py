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


def _hdbscan(F: np.ndarray, min_cluster_size: int,
             min_samples: int | None = None,
             method: str = "leaf"
             ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, probabilities).
    labels[i]:        -1 = noise, 0..K-1 = cluster id.
    probabilities[i]: HDBSCAN membership strength in [0, 1] (0 for noise,
                      near 1 for points deep inside a cluster's core)."""
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_method=method)
    except ImportError:
        import hdbscan as _hdb
        clusterer = _hdb.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples,
                                 cluster_selection_method=method)
    clusterer.fit(F)
    labels = clusterer.labels_.astype(np.int64)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = (labels >= 0).astype(np.float64)
    return labels, np.asarray(probs, dtype=np.float64)


# -------------------- layer localisation per cluster -----------------------
def _layer_localisation(depth_vals: np.ndarray) -> dict:
    """Per-cluster depth (layer-position) statistics. depth_vals are the
    'depth' feature column values for experts in the cluster, each in [0, 1].
    Returns:
        mean, std, range = max - min
        H_layer_norm: normalised entropy of the discrete depth distribution
            (each layer has a unique depth value = layer_idx / (L-1)).
            H / log(K_unique), where K_unique = number of distinct depth values
            seen in the cluster. Range [0, 1]; 0 = all from one layer,
            1 = uniform across the layers actually represented.
        n_layers: number of distinct layers (= unique depth values) in cluster.
    """
    if depth_vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "range": float("nan"), "H_layer_norm": float("nan"),
                "n_layers": 0}
    unique, counts = np.unique(np.round(depth_vals, 6), return_counts=True)
    n_layers = int(unique.size)
    p = counts / counts.sum()
    p_nz = p[p > 0]
    H = -float((p_nz * np.log(p_nz)).sum())
    H_norm = float(H / np.log(n_layers)) if n_layers > 1 else 0.0
    return {
        "mean":         float(depth_vals.mean()),
        "std":          float(depth_vals.std()),
        "range":        float(depth_vals.max() - depth_vals.min()),
        "H_layer_norm": H_norm,
        "n_layers":     n_layers,
    }


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


def _run_sweep(F_all: np.ndarray, model_idx: np.ndarray, MODELS: list[str],
               min_frac_values: list[float], min_floor: int,
               min_samples: int | None, method: str,
               out_dir: Path) -> Path:
    """Sweep min_frac for every model. Produces a per-model comparison table
    and a JSON dump. No plots — pick a setting from the table, then re-run in
    single-value mode to get plots."""
    import json
    rows: list[dict] = []
    print(f"\n=== Per-model sweep over min_frac ===\n")
    for mi, model in enumerate(MODELS):
        F_m = F_all[model_idx == mi].astype(np.float32)
        n = F_m.shape[0]
        for f in min_frac_values:
            mcs = max(min_floor, int(f * n))
            labels, probs = _hdbscan(F_m, mcs, min_samples=min_samples,
                                     method=method)
            unique = sorted(set(int(c) for c in labels) - {-1})
            n_clusters = len(unique)
            n_noise = int((labels == -1).sum())
            sizes = [int((labels == c).sum()) for c in unique]
            p_means = [float(probs[labels == c].mean()) for c in unique]
            rows.append({
                "model":            model,
                "n_experts":        n,
                "min_frac":         f,
                "min_cluster_size": mcs,
                "n_clusters":       n_clusters,
                "n_noise":          n_noise,
                "noise_pct":        100.0 * n_noise / n,
                "mean_size":  float(np.mean(sizes)) if sizes else float("nan"),
                "min_size":   int(np.min(sizes)) if sizes else 0,
                "max_size":   int(np.max(sizes)) if sizes else 0,
                "mean_prob":  float(np.mean(p_means)) if p_means else float("nan"),
            })

    # ---- pretty-print ------------------------------------------------------
    lines: list[str] = []
    def w(s: str = ""):
        lines.append(s); print(s)

    w("")
    w("=" * 112)
    w(f"Per-model HDBSCAN sweep — method = {method}, "
      f"min_samples = {min_samples}, min_floor = {min_floor}")
    w("=" * 112)
    w(f"  {'model':<18s} {'n':>6s}  {'frac':>5s}  {'mcs':>5s}  "
      f"{'K':>3s}  {'noise (%)':>14s}  {'mean_size':>10s}  "
      f"{'(min..max)':>13s}  {'mean_prob':>10s}")
    w("-" * 112)
    last_model = None
    for r in rows:
        if last_model is not None and r["model"] != last_model:
            w("")
        last_model = r["model"]
        if r["n_clusters"] > 0:
            sz = f"{r['mean_size']:>10.1f}"
            mm = f"({r['min_size']}..{r['max_size']})"
            pr = f"{r['mean_prob']:>10.3f}"
        else:
            sz = "       n/a"; mm = "           --"; pr = "        --"
        w(f"  {r['model']:<18s} {r['n_experts']:>6d}  "
          f"{r['min_frac']:>5.2f}  {r['min_cluster_size']:>5d}  "
          f"{r['n_clusters']:>3d}  "
          f"{r['n_noise']:>5d} ({r['noise_pct']:>5.2f}%)  "
          f"{sz}  {mm:>13s}  {pr}")
    w("=" * 112)
    w("Columns:")
    w("  frac = min_frac value used; mcs = max(min_floor, floor(frac * n_experts)).")
    w("  K = number of clusters discovered; noise = experts assigned label -1.")
    w("  mean_prob = mean HDBSCAN membership probability averaged across clusters.")

    out_txt = out_dir / "clusters_per_model_sweep.txt"
    out_txt.write_text("\n".join(lines))
    out_json = out_dir / "clusters_per_model_sweep.json"
    out_json.write_text(json.dumps({
        "method": method, "min_samples": min_samples,
        "min_floor": min_floor, "results": rows,
    }, indent=2))
    print(f"\n  Saved {out_txt}")
    print(f"  Saved {out_json}")
    return out_txt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--min-frac", type=float, default=0.02,
                   help="Default min_frac used for any model not overridden by "
                        "--min-frac-file. min_cluster_size = "
                        "max(min_floor, floor(min_frac * n_experts)).")
    p.add_argument("--min-frac-file", default=None,
                   help="Optional path to a JSON file mapping "
                        "{model_name: min_frac} to override --min-frac on a "
                        "per-model basis. Missing models fall back to --min-frac. "
                        "Use this to generate a figure where each panel uses its "
                        "model's sweet-spot from the sweep table.")
    p.add_argument("--min-floor", type=int, default=10,
                   help="Floor on min_cluster_size (default 10).")
    p.add_argument("--min-samples", type=int, default=15,
                   help="HDBSCAN min_samples (default 15, matches pooled "
                        "cluster_experts.py).")
    p.add_argument("--method", default="leaf",
                   choices=["leaf", "eom"])
    p.add_argument("--sweep", action="store_true",
                   help="Sweep min_frac ∈ {0.01, 0.02, 0.05, 0.1, 0.2} and "
                        "print a per-model comparison table. Skips plots and "
                        "the per-cluster signature pass.")
    p.add_argument("--sweep-values", type=float, nargs="+",
                   default=[0.01, 0.02, 0.05, 0.1, 0.2],
                   help="Override min_frac values for --sweep.")
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

    # Sweep mode: comparison table only, no plots.
    if args.sweep:
        _run_sweep(
            F_all, model_idx, MODELS,
            min_frac_values=list(args.sweep_values),
            min_floor=args.min_floor,
            min_samples=args.min_samples,
            method=args.method,
            out_dir=out_dir,
        )
        return

    # Optional per-model min_frac overrides.
    frac_overrides: dict[str, float] = {}
    if args.min_frac_file:
        import json
        fp = Path(args.min_frac_file)
        if not fp.exists():
            print(f"ERROR: --min-frac-file not found: {fp}", file=sys.stderr)
            sys.exit(1)
        raw = json.loads(fp.read_text())
        # Skip keys starting with "_" (documentation / comment keys).
        frac_overrides = {k: float(v) for k, v in raw.items()
                          if not k.startswith("_")}
        print(f"Loaded per-model min_frac overrides from {fp}")
        for k, v in frac_overrides.items():
            print(f"  {k}: {v}")

    # Per-model clustering.
    per_model: dict[str, dict] = {}
    for mi, model in enumerate(MODELS):
        mask = (model_idx == mi)
        F_m = F_all[mask]
        n = F_m.shape[0]
        frac = frac_overrides.get(model, args.min_frac)
        mcs = max(args.min_floor, int(frac * n))
        override_flag = " (override)" if model in frac_overrides else ""
        print(f"\n[{model}] n_experts={n}; min_frac={frac}{override_flag}; "
              f"min_cluster_size={mcs}")
        labels, probs = _hdbscan(F_m, mcs, min_samples=args.min_samples,
                                 method=args.method)
        print(f"  HDBSCAN → {labels.max() + 1} clusters, "
              f"{int((labels == -1).sum())} noise "
              f"({100 * (labels == -1).mean():.1f}%)")
        print(f"  computing per-model UMAP on {n} points ...", flush=True)
        umap2 = _umap_2d(F_m)
        per_model[model] = {
            "F_m": F_m, "labels": labels, "probs": probs, "umap": umap2,
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
        f.write(f"  min_samples              : {args.min_samples}\n")
        f.write(f"  min_cluster_size formula : max({args.min_floor}, "
                f"{args.min_frac} * n_experts)\n\n")
        f.write("Per-cluster columns:\n")
        f.write("  prob    = HDBSCAN membership probability, mean ± std (median),\n"
                "            in [0, 1]. Near 1 = points deep in cluster core.\n")
        f.write("  depth   = mean ± std of the 'depth' feature (normalised layer\n"
                "            position in [0, 1]); 'layers' = number of distinct\n"
                "            layers seen in cluster; 'H_layer' = normalised\n"
                "            entropy of the layer distribution in [0, 1] (0 =\n"
                "            cluster localised to a single layer; 1 = uniform\n"
                "            across the layers it spans).\n\n")
        for model in MODELS:
            info = per_model[model]
            F_m, labels, probs = info["F_m"], info["labels"], info["probs"]
            n = F_m.shape[0]
            n_clusters = int(labels.max() + 1)
            n_noise = int((labels == -1).sum())
            f.write("-" * 78 + "\n")
            f.write(f"[{model}] n_experts={n}  min_cluster_size={info['min_cluster_size']}\n")
            f.write(f"  → {n_clusters} clusters; "
                    f"noise = {n_noise} ({100*n_noise/n:.1f}%)\n")
            for k in range(n_clusters):
                mask_k = labels == k
                F_k = F_m[mask_k]
                p_k = probs[mask_k]
                n_k = int(F_k.shape[0])
                _mu, top = _signature(F_k, F_m, top_k=3)
                sig = ", ".join(
                    f"{name}={val:.3f} (z={z:+.2f})" for name, val, z in top
                )
                # 'depth' is column 0 of F by construction.
                loc = _layer_localisation(F_k[:, 0])
                f.write(f"   c{k}  n={n_k:5d} ({100*n_k/n:5.1f}%)  {sig}\n")
                f.write(f"        prob:  {p_k.mean():.3f} ± {p_k.std():.3f}  "
                        f"(median {float(np.median(p_k)):.3f})\n")
                f.write(f"        depth: {loc['mean']:.3f} ± {loc['std']:.3f}  "
                        f"(range {loc['range']:.3f}, layers {loc['n_layers']}, "
                        f"H_layer {loc['H_layer_norm']:.3f})\n")
            f.write("\n")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
