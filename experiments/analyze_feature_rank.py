"""Per-model effective rank and cross-model subspace alignment of F.

For each model m on task `--task` (default c4):
  1. Load F_m of shape (N_m, 10) via inspect_features._load_F.
  2. Mean-center: F_m_c = F_m - mean(F_m, axis=0).
  3. SVD: F_m_c = U_m S_m V_m^T.
  4. Report:
       - Singular value spectrum sigma_1, ..., sigma_10.
       - Cumulative variance explained: cumsum(sigma_k^2) / sum(sigma_k^2).
       - Two scalar effective-rank measures:
           r_PR  (participation ratio)  := (sum sigma_k^2)^2 / sum(sigma_k^4)
           r_ENT (entropy)              := exp(-sum p_k log p_k),  p_k = sigma_k^2 / sum
         Both are in [1, D] (D=10). Higher = variance spread over more directions.

Cross-model subspace alignment:
  - For each (a, b), measure how well model a's top-k right-singular subspace
    aligns with model b's top-k. Two views, both in [0, 1]:
       grassmann_overlap(a, b, k) = (1/k) sum_i cos^2(theta_i),
         where theta_1 <= ... <= theta_k are the principal angles between
         span(V_a[:, :k]) and span(V_b[:, :k]). 1 = identical k-dim subspace,
         0 = orthogonal.
    Default k = 3 (covers >90% var on most models; configurable via --k).

Outputs (under results/circuits/feature_inspection/):
  features_rank_spectrum.pdf            -- 2-panel: (a) singular values per
                                           model, (b) cumulative variance
  features_subspace_overlap.pdf         -- (n_models x n_models) heatmap of
                                           grassmann_overlap
  features_shared_axis_distributions.pdf -- per-model histograms of expert
                                           projections onto the top-k axes of
                                           the GLOBAL (pooled-F) SVD. Same axes
                                           for every model, so where models
                                           disagree on the distribution shape
                                           is visible at a glance.
  features_rank_summary.json            -- per-model effective ranks, sigmas,
                                           pairwise grassmann_overlap matrix,
                                           and the global cumulative variance.

Usage (on piora, where the DAGs live):
    python3 experiments/analyze_feature_rank.py
    python3 experiments/analyze_feature_rank.py --task c4 --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.inspect_features import _load_F, MODELS  # noqa: E402

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
OUT_DIR = Path(CFG["result_path"]) / "circuits" / "feature_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _svd_spectrum(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (singular_values, right_singular_vectors). Centers F first."""
    F_c = F - F.mean(axis=0, keepdims=True)
    # full_matrices=False -> S shape (min(N, D),) = (D,) since N >> D
    _, S, Vt = np.linalg.svd(F_c, full_matrices=False)
    return S, Vt   # Vt rows are right singular vectors


def _participation_ratio(sigmas: np.ndarray) -> float:
    s2 = sigmas ** 2
    if s2.sum() < 1e-30:
        return float("nan")
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def _entropy_rank(sigmas: np.ndarray) -> float:
    s2 = sigmas ** 2
    if s2.sum() < 1e-30:
        return float("nan")
    p = s2 / s2.sum()
    p = p[p > 0]
    H = -float((p * np.log(p)).sum())
    return float(np.exp(H))


def _grassmann_overlap(Va: np.ndarray, Vb: np.ndarray) -> float:
    """Mean squared cosine of principal angles between span(Va) and span(Vb).
    Va, Vb are (k, D) matrices whose ROWS are orthonormal basis vectors.
    Returns scalar in [0, 1]: 1 = identical k-dim subspace, 0 = orthogonal."""
    M = Va @ Vb.T                # (k, k)
    cos_thetas = np.linalg.svd(M, compute_uv=False)
    return float((cos_thetas ** 2).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--k", type=int, default=3,
                   help="Subspace dimension for cross-model alignment (default 3).")
    args = p.parse_args()

    # ---------- 1. Load F_m for every model -----------------------------------
    print(f"Loading F for {len(MODELS)} models on task={args.task} ...")
    Fs: dict[str, np.ndarray] = {}
    for m in MODELS:
        F = _load_F(m, args.task)
        if F is None:
            print(f"  {m}: MISSING (no DAG)")
            continue
        Fs[m] = F
        print(f"  {m}: F shape {F.shape}")

    if not Fs:
        print("No DAGs found; aborting.")
        sys.exit(1)

    # ---------- 2. Per-model SVD ----------------------------------------------
    spectra: dict[str, dict] = {}
    Vs: dict[str, np.ndarray] = {}        # top-k right-singular vectors per model
    for m, F in Fs.items():
        S, Vt = _svd_spectrum(F)
        cum = np.cumsum(S ** 2) / (S ** 2).sum()
        spectra[m] = {
            "sigmas": S.tolist(),
            "cum_var": cum.tolist(),
            "r_PR":  _participation_ratio(S),
            "r_ENT": _entropy_rank(S),
            "N": int(F.shape[0]),
            "D": int(F.shape[1]),
        }
        Vs[m] = Vt[:args.k]   # rows are top-k right singular vectors

    # ---------- 3. Per-model summary table ------------------------------------
    print("\nPer-model effective rank (D = 10, smaller = more concentrated):")
    print(f"  {'model':<18s}  {'N':>6s}  {'r_PR':>5s}  {'r_ENT':>5s}  "
          f"{'cumvar@1':>8s} {'cumvar@3':>8s} {'cumvar@5':>8s}")
    for m, s in spectra.items():
        cv = s["cum_var"]
        print(f"  {m:<18s}  {s['N']:>6d}  {s['r_PR']:>5.2f}  {s['r_ENT']:>5.2f}  "
              f"{cv[0]:>8.3f}  {cv[2]:>8.3f}  {cv[4]:>8.3f}")

    # ---------- 4. Cross-model subspace alignment -----------------------------
    print(f"\nCross-model Grassmann overlap (top-k={args.k} subspaces):")
    model_list = list(Fs.keys())
    n = len(model_list)
    G = np.zeros((n, n))
    for i, mi in enumerate(model_list):
        for j, mj in enumerate(model_list):
            G[i, j] = _grassmann_overlap(Vs[mi], Vs[mj])
    print(f"  {'':<18s}  " + "  ".join(f"{m[:8]:>8s}" for m in model_list))
    for i, mi in enumerate(model_list):
        print(f"  {mi:<18s}  " + "  ".join(f"{G[i, j]:>8.3f}" for j in range(n)))

    # ---------- 5. Save figures -----------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import colorsys
    colors = [colorsys.hsv_to_rgb(i / n, 0.85, 0.85) for i in range(n)]

    # (a) singular value spectrum + (b) cumulative variance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ks = np.arange(1, 11)
    for i, m in enumerate(model_list):
        S = np.array(spectra[m]["sigmas"])
        S_norm = S / S[0]            # normalise so leading sigma = 1
        axes[0].plot(ks[:len(S)], S_norm, marker="o", color=colors[i], label=m)
        axes[1].plot(ks[:len(S)], spectra[m]["cum_var"], marker="o", color=colors[i], label=m)
    axes[0].set_xlabel("singular value index $k$", fontsize=12)
    axes[0].set_ylabel(r"$\sigma_k / \sigma_1$", fontsize=12)
    axes[0].set_yscale("log")
    axes[0].set_title("Singular value spectrum per model\n(F mean-centered, normalised by $\\sigma_1$)")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("number of components $k$", fontsize=12)
    axes[1].set_ylabel("cumulative variance explained", fontsize=12)
    axes[1].set_title("Cumulative variance per model")
    axes[1].axhline(0.9, color="gray", lw=0.5, ls="--")
    axes[1].axhline(0.95, color="gray", lw=0.5, ls="--")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_a = OUT_DIR / "features_rank_spectrum.pdf"
    fig.savefig(out_a, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_a}")

    # (c) subspace overlap heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # Added import
    
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(G, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            txt_col = "black" if G[i, j] > 0.55 else "white"
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=10, color=txt_col)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(model_list, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(model_list, fontsize=12)
    
    # --- Updated Colorbar Logic ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)  # Creates an axis with locked height
    cbar = fig.colorbar(im, cax=cax)                         # Draws colorbar inside the new axis
    cbar.set_label("Grassmann overlap", fontsize=12)         # Set label on cbar object directly
    # ------------------------------
    
    ax.set_title(f"Similarity of principal subspaces across models", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out_b = OUT_DIR / "features_subspace_overlap.pdf"
    fig.savefig(out_b, dpi=200)
    plt.close(fig)
    print(f"Saved {out_b}")

    # ---------- 5b. Distribution along shared axes ---------------------------
    # Build a COMMON basis by SVD on the pooled, globally-centered F_all.
    # Project each model's experts onto the top-k pooled PCs and plot the
    # per-model marginal distribution along each PC. If models truly share
    # principal axes, the structure (modes, spread) of these distributions is
    # where their disagreements actually live.
    F_list = [Fs[m] for m in model_list]
    n_per_model = [F.shape[0] for F in F_list]
    F_all = np.concatenate(F_list, axis=0)
    mu_global = F_all.mean(axis=0, keepdims=True)
    F_all_c = F_all - mu_global
    _, S_global, Vt_global = np.linalg.svd(F_all_c, full_matrices=False)
    V_top = Vt_global[:args.k]   # (k, D) rows are global PC directions
    cum_var_global = np.cumsum(S_global ** 2) / (S_global ** 2).sum()

    fig, axes = plt.subplots(1, args.k, figsize=(5.2 * args.k, 4.4), sharey=True)
    if args.k == 1:
        axes = [axes]
    start = 0
    for i, m in enumerate(model_list):
        F_m = F_list[i]
        F_m_c = F_m - mu_global    # use GLOBAL mean to keep axes common
        proj = F_m_c @ V_top.T     # (N_m, k)
        for pc_idx in range(args.k):
            axes[pc_idx].hist(
                proj[:, pc_idx], bins=60, histtype="step", density=True,
                color=colors[i], label=m, linewidth=1.4,
            )
        start += F_m.shape[0]
    for pc_idx in range(args.k):
        axes[pc_idx].set_xlabel(
            f"projection on global PC{pc_idx+1}"
            f"  ({100*S_global[pc_idx]**2/(S_global**2).sum():.1f}% var)",
            fontsize=11,
        )
        axes[pc_idx].grid(alpha=0.3)
    axes[0].set_ylabel("density (per model)", fontsize=11)
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Expert distributions along the global top-{} principal axes\n"
        "(common basis from pooled, mean-centered F_all)".format(args.k),
        fontsize=12,
    )
    fig.tight_layout()
    out_c = OUT_DIR / "features_shared_axis_distributions.pdf"
    fig.savefig(out_c, dpi=200)
    plt.close(fig)
    print(f"Saved {out_c}")

    # ---------- 6. JSON summary ----------------------------------------------
    summary = {
        "task": args.task,
        "k_subspace": args.k,
        "models": model_list,
        "per_model": spectra,
        "grassmann_overlap": {
            mi: {mj: float(G[i, j]) for j, mj in enumerate(model_list)}
            for i, mi in enumerate(model_list)
        },
        "global_pooled": {
            "sigmas":  S_global.tolist(),
            "cum_var": cum_var_global.tolist(),
        },
    }
    out_json = OUT_DIR / "features_rank_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
