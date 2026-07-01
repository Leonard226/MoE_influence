"""Cross-model expert-cluster archetypes.

For each model, re-runs HDBSCAN on its own F_m using the sweet-spot
min_frac values (from experiments/per_model_min_frac_sweet_spots.json), then:
  1. Computes the centroid of each per-model cluster in the original
     10-dimensional feature space.
  2. Projects each centroid onto the pooled PCA basis
     (from pca_interpretability_summary.json) to give every cluster
     an interpretable coordinate on PC1..PC5.
  3. Runs agglomerative clustering on the cluster centroids in PC1..PC5
     space to auto-detect cross-model archetypes.
  4. Writes:
       per_model_cluster_archetypes.pdf   scatter of centroids in PC1xPC2
       per_model_archetypes_summary.txt   table + archetype groupings
       per_model_archetypes.json          full numerics

The archetype-count k defaults to 3 (based on eyeballing the c4 pattern:
early-layer specialists, late-layer content, late-layer functional). Tune
via --n-archetypes if needed.

Usage:
  python3 experiments/analyze_per_model_archetypes.py \
      --task c4 \
      --min-frac-file experiments/per_model_min_frac_sweet_spots.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
DEFAULT_OUT_DIR = Path(CFG["result_path"]) / "circuits" / "feature_inspection"

FEATURE_NAMES = ["depth", "out", "in", "load", "act",
                 "content", "functional", "punctuation", "numeric", "special"]
PC_NAMES = {
    0: "PC1 (specialisation: content-/functional+)",
    1: "PC2 (depth: deep-/shallow+)",
    2: "PC3 (load: heavy-/light+)",
    3: "PC4 (punctuation-/other+)",
    4: "PC5 (out-strength: hub-/low+)",
}


def _hdbscan(F: np.ndarray, mcs: int, min_samples: int, method: str
             ) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples,
                            cluster_selection_method=method)
    except ImportError:
        import hdbscan as _hdb
        clusterer = _hdb.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples,
                                 cluster_selection_method=method)
    clusterer.fit(F)
    labels = clusterer.labels_.astype(np.int64)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = (labels >= 0).astype(np.float64)
    return labels, np.asarray(probs, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", default="c4")
    p.add_argument("--min-frac-file",
                   default=str(ROOT / "experiments" /
                                "per_model_min_frac_sweet_spots.json"),
                   help="JSON dict {model_name: min_frac}. Defaults to the "
                        "sweet-spot file in experiments/.")
    p.add_argument("--default-min-frac", type=float, default=0.02,
                   help="Fallback min_frac for models not in the file.")
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--method", default="leaf", choices=["leaf", "eom"])
    p.add_argument("--min-floor", type=int, default=10)
    p.add_argument("--n-archetypes", type=int, default=3,
                   help="Number of cross-model archetypes to group into (via "
                        "agglomerative clustering on PC1..PC5). Default 3.")
    p.add_argument("--pc-k", type=int, default=5,
                   help="Number of PC axes to use for centroid projection "
                        "(default 5).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load data ----------------------------------------------------
    emb_path = out_dir / f"embedding_{args.task}.npz"
    if not emb_path.exists():
        print(f"ERROR: embedding cache not found: {emb_path}", file=sys.stderr)
        sys.exit(1)
    d = np.load(emb_path, allow_pickle=True)
    F_all = np.asarray(d["F_all"]).astype(np.float64)
    model_idx = np.asarray(d["model_idx"]).astype(int)
    MODELS = list(d["models"])
    print(f"Loaded F_all: {F_all.shape}; models: {MODELS}")

    pca_path = out_dir / "pca_interpretability_summary.json"
    if not pca_path.exists():
        print(f"ERROR: PCA summary not found: {pca_path}\n"
              f"  Run analyze_pca_interpretability.py first.", file=sys.stderr)
        sys.exit(1)
    pca_sum = json.load(open(pca_path))
    V = np.array(pca_sum["loadings"])         # (D, D), row k = PC_k dir
    mu_global = np.array(pca_sum["mean"])
    var_ratio = pca_sum["var_ratio"]

    frac_overrides = {}
    if Path(args.min_frac_file).exists():
        raw = json.loads(Path(args.min_frac_file).read_text())
        frac_overrides = {k: float(v) for k, v in raw.items()
                          if not k.startswith("_")}
        print(f"Loaded per-model min_frac from {args.min_frac_file}")
    else:
        print(f"WARN: min_frac_file not found; using default "
              f"{args.default_min_frac} for all models")

    # -------- Per-model HDBSCAN + centroid extraction ----------------------
    all_clusters: list[dict] = []
    for mi, m in enumerate(MODELS):
        mask = model_idx == mi
        F_m = F_all[mask].astype(np.float32)
        n = F_m.shape[0]
        frac = frac_overrides.get(m, args.default_min_frac)
        mcs = max(args.min_floor, int(frac * n))
        print(f"\n[{m}]  n={n}  f_m={frac}  mcs={mcs}")
        labels, probs = _hdbscan(F_m, mcs, args.min_samples, args.method)
        K = int(labels.max() + 1)
        n_noise = int((labels == -1).sum())
        print(f"  -> {K} clusters, noise={n_noise} ({100*n_noise/n:.1f}%)")
        for c in range(K):
            m_mask = labels == c
            F_c = F_m[m_mask].astype(np.float64)
            centroid = F_c.mean(axis=0)
            centroid_c = centroid - mu_global   # centred vs pooled mean
            pc_coords = centroid_c @ V.T        # project onto pooled PCs
            all_clusters.append({
                "model":   m,
                "cluster_id": int(c),
                "size":    int(m_mask.sum()),
                "prob_mean": float(probs[m_mask].mean()),
                "centroid_raw":  centroid.tolist(),
                "pc":            pc_coords[:args.pc_k].tolist(),
            })

    if not all_clusters:
        print("No clusters discovered; aborting.")
        sys.exit(0)

    # -------- Agglomerative archetype clustering ---------------------------
    from scipy.cluster.hierarchy import linkage, fcluster
    X = np.array([c["pc"] for c in all_clusters])  # (n_clusters, pc_k)
    Z = linkage(X, method="ward")
    archetype_labels = fcluster(Z, t=args.n_archetypes,
                                 criterion="maxclust")
    for i, c in enumerate(all_clusters):
        c["archetype"] = int(archetype_labels[i])

    # -------- Archetype summaries -----------------------------------------
    archetypes: dict[int, dict] = {}
    for c in all_clusters:
        a = c["archetype"]
        archetypes.setdefault(a, {"members": [], "pc_centroid": []})
        archetypes[a]["members"].append(c)
    for a, info in archetypes.items():
        pc_stack = np.array([m["pc"] for m in info["members"]])
        info["pc_centroid"] = pc_stack.mean(axis=0).tolist()

    # -------- Human-readable name for each archetype -----------------------
    # Compose from the top-2 PC coordinates of the archetype centroid.
    axis_desc = {
        (0, "+"): "functional-leaning",
        (0, "-"): "content-leaning",
        (1, "+"): "shallow / early-layer",
        (1, "-"): "deep / late-layer",
        (2, "+"): "low-load / lightly-utilised",
        (2, "-"): "high-load / heavy",
        (3, "+"): "non-punctuation",
        (3, "-"): "punctuation-specialist",
        (4, "+"): "low-out",
        (4, "-"): "high-out / hub",
    }
    for a, info in archetypes.items():
        pc = np.array(info["pc_centroid"])
        order = np.argsort(-np.abs(pc))
        top2 = order[:2]
        parts = [axis_desc.get((int(i), "+" if pc[i] > 0 else "-"),
                                f"PC{i+1}{'+' if pc[i] > 0 else '-'}")
                 for i in top2]
        info["auto_name"] = " + ".join(parts)

    # -------- Print + save summary -----------------------------------------
    lines: list[str] = []
    def w(s: str = ""):
        lines.append(s); print(s)

    w("")
    w("=" * 84)
    w(f"Cross-model per-model cluster archetypes  (task = {args.task})")
    w("=" * 84)
    w(f"  Per-model HDBSCAN: leaf, min_samples={args.min_samples}, "
      f"per-model min_frac from {args.min_frac_file}")
    w(f"  Archetype clustering: Ward linkage on PC1..PC{args.pc_k} "
      f"cluster centroids; k={args.n_archetypes}")
    w("")
    w(f"Total per-model clusters discovered: {len(all_clusters)}")
    w("")
    for a in sorted(archetypes):
        info = archetypes[a]
        pc_c = info["pc_centroid"]
        w("-" * 84)
        w(f"Archetype {a}: {info['auto_name']}  "
          f"({len(info['members'])} clusters from "
          f"{len({m['model'] for m in info['members']})} models)")
        w(f"  Centroid in PC space: " +
          "  ".join(f"PC{i+1}={pc_c[i]:+.3f}" for i in range(args.pc_k)))
        w(f"  {'model':<20s} {'c#':>3s} {'size':>5s}  " +
          "  ".join(f"{'PC' + str(i+1):>7s}" for i in range(args.pc_k)))
        for m in sorted(info["members"], key=lambda x: (x["model"], x["cluster_id"])):
            w(f"  {m['model']:<20s} {m['cluster_id']:>3d} {m['size']:>5d}  " +
              "  ".join(f"{v:+7.3f}" for v in m["pc"]))
    w("=" * 84)

    out_txt = out_dir / "per_model_archetypes_summary.txt"
    out_txt.write_text("\n".join(lines))
    print(f"\n  Saved {out_txt}")

    out_json = out_dir / "per_model_archetypes.json"
    out_json.write_text(json.dumps({
        "task":          args.task,
        "n_archetypes":  args.n_archetypes,
        "pc_k":          args.pc_k,
        "feature_names": FEATURE_NAMES,
        "var_ratio":     var_ratio,
        "clusters":      all_clusters,
        "archetypes":    {str(a): {
            "auto_name":   info["auto_name"],
            "pc_centroid": info["pc_centroid"],
            "n_members":   len(info["members"]),
            "member_ids":  [f"{m['model']}:c{m['cluster_id']}"
                            for m in info["members"]],
        } for a, info in archetypes.items()},
    }, indent=2))
    print(f"  Saved {out_json}")

    # -------- Figure: scatter in PC1 x PC2 (with background) ---------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import colorsys
    n_models = len(MODELS)
    colors = [colorsys.hsv_to_rgb(i / n_models, 0.85, 0.85)
              for i in range(n_models)]

    Fc_all = F_all - mu_global
    Z_all = Fc_all @ V.T                   # (N, D)
    x_all, y_all = Z_all[:, 0], Z_all[:, 1]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.scatter(x_all, y_all, s=1.0, c="lightgray", alpha=0.15, linewidths=0,
               rasterized=True, label="all experts (background)")

    seen_labels = set()
    for c in all_clusters:
        mi = MODELS.index(c["model"])
        pc = c["pc"]
        size = 30 + 3.0 * np.sqrt(c["size"])
        label = c["model"] if c["model"] not in seen_labels else None
        seen_labels.add(c["model"])
        ax.scatter(pc[0], pc[1], s=size, color=colors[mi],
                   edgecolor="black", linewidth=0.9, alpha=0.9,
                   label=label, zorder=3)
        ax.annotate(f"{c['model'].split('-')[0][:6]}:c{c['cluster_id']}",
                    (pc[0], pc[1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, alpha=0.85, zorder=4)

    # Archetype centroids as large open circles.
    for a, info in archetypes.items():
        pc = info["pc_centroid"]
        ax.scatter(pc[0], pc[1], s=350, facecolor="none",
                   edgecolor="black", linewidth=2.0, zorder=2,
                   marker="o")
        ax.annotate(f"Archetype {a}: {info['auto_name']}",
                    (pc[0], pc[1]),
                    xytext=(0, -18), textcoords="offset points",
                    fontsize=9, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="white", edgecolor="black",
                               linewidth=0.5, alpha=0.85),
                    zorder=5)

    ax.set_xlabel(f"PC1  ({100*var_ratio[0]:.1f}% var, "
                   f"specialisation: content$-$/functional$+$)",
                  fontsize=11)
    ax.set_ylabel(f"PC2  ({100*var_ratio[1]:.1f}% var, "
                   f"depth: deep$-$/shallow$+$)",
                  fontsize=11)
    ax.set_title(
        f"Per-model cluster centroids in the pooled PCA basis (task = {args.task})\n"
        f"Do the same expert archetypes emerge across architectures?",
        fontsize=12,
    )
    leg = ax.legend(loc="best", fontsize=8, markerscale=1.2, framealpha=0.85)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    out_fig = out_dir / "per_model_cluster_archetypes.pdf"
    fig.savefig(out_fig, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved {out_fig}")


if __name__ == "__main__":
    main()
