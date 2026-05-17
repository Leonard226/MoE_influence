"""Null-model sanity check for the spectral DAG-similarity metric.

Question: does subspace_similarity(U_A, U_B) — the metric currently used in
spectral.ipynb to compare cross-task DAGs — actually measure circuit identity,
or is it picking up degree-distribution / layer-block-structure artefacts?

Test:
    real:  subspace_similarity(U_A, U_B)
    null:  subspace_similarity(U_A, U_{B'})  for B' = B with expert labels
           permuted *within each layer*.

Within-layer permutation preserves:
    - the layer x layer block structure (which is a model property),
    - the per-block weight distribution and sparsity pattern's marginals.
It destroys:
    - the identity of individual experts within each layer.

So if the real-vs-null gap is large, the metric responds to per-expert identity
(what we want). If the gap is small, the metric is reading off degree
distribution / layer structure and the cross-model project needs a different
comparison metric.

Sweep:
    - threshold quantile q in {0.90, 0.95, 0.98, 0.99, 0.995, 0.999}
    - subspace dim k in {5, 10, 25, 50, 100, 200}
    - edge weight in {"AARV", "APS"}  (run both; decide after seeing results)

Run cell-by-cell on the cluster (designed as # %% cells).
"""
# %%
from __future__ import annotations
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = "/scratch/sleonard/routing_decision"
sys.path.insert(0, ROOT)
# Also make helper.py importable directly (matches spectral.ipynb's pattern).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

from helper import thresholding_routing_graph, get_thresholds

N_LAYERS = 16
N_EXPERTS = 64
N_NODES = N_LAYERS * N_EXPERTS

DAG_DIR = os.path.join(config["result_path"], "circuits")

# %% Load DAGs
dag_c4   = torch.load(os.path.join(DAG_DIR, "dag_c4.pt"),   map_location="cpu")
dag_math = torch.load(os.path.join(DAG_DIR, "dag_math.pt"), map_location="cpu")
dag_code = torch.load(os.path.join(DAG_DIR, "dag_code.pt"), map_location="cpu")
print("Loaded DAGs. Keys:", list(dag_c4.keys()))

# %% Utilities ---------------------------------------------------------------

def permute_within_layer(W4: torch.Tensor, seed: int):
    """Randomly relabel experts within each layer.

    W4: [L, N, L, N] tensor (sender_layer, sender_expert, recv_layer, recv_expert).
    For each layer l, draws one random permutation sigma_l of {0, .., N-1} and
    applies it to BOTH the sender axis (1) when sender_layer == l and the
    receiver axis (3) when recv_layer == l. This keeps the (l, n) identity
    consistent across the expert's two roles.

    Preserves: layer-block structure, per-block weight multiset.
    Destroys:  per-expert identity within each layer.
    """
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(N_EXPERTS) for _ in range(N_LAYERS)]
    W = W4.clone()
    # Sender axis (axis=1) per sender layer (axis=0).
    for c in range(N_LAYERS):
        W[c] = W[c][perms[c]]
    # Receiver axis (axis=3) per receiver layer (axis=2).
    for r in range(N_LAYERS):
        W[:, :, r, :] = W[:, :, r, :][:, :, perms[r]]
    return W, perms


def dag_with_permuted_weight(dag: dict, weight: str, seed: int) -> dict:
    """Return a shallow-copied dag dict with dag[weight] permuted within layers."""
    new_dag = dict(dag)
    new_dag[weight], _ = permute_within_layer(dag[weight], seed=seed)
    return new_dag


def adjacency_from_dag(dag: dict, weight: str, quantile: float) -> np.ndarray:
    """Threshold dag[weight] at the given |w|-quantile and return [N, N] adjacency."""
    thr = get_thresholds(dag, weight, [quantile])[quantile]
    g = thresholding_routing_graph(dag, weight, thr)
    return np.array(g.get_adjacency(attribute="weight").data, dtype=float)


def top_k_left_singular(A: np.ndarray, k: int) -> np.ndarray:
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, :k]


def subspace_similarity(U1: np.ndarray, U2: np.ndarray) -> float:
    """Mean cosine of principal angles between column spaces of U1 and U2."""
    M = U1.T @ U2
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.clip(s, -1.0, 1.0).mean())

# %% Sweep config -----------------------------------------------------------

WEIGHTS = ["APS", "ANS", "AVG", "VAR", "AARV"]  # all five
QUANTILES = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
KS = [5, 10, 25, 50, 100, 200]
N_NULL_SEEDS = 5

PAIRS = {
    "c4_math":   (dag_c4,   dag_math),
    "c4_code":   (dag_c4,   dag_code),
    "math_code": (dag_math, dag_code),
}

# %% Run sweep --------------------------------------------------------------

def run_sweep(weight: str) -> dict:
    """results[pair][q][k] = {real, nulls, null_mean, null_std, z}."""
    out = {p: {q: {} for q in QUANTILES} for p in PAIRS}
    for pair_name, (dagA, dagB) in PAIRS.items():
        print(f"[{weight}] pair={pair_name}")
        for q in QUANTILES:
            A = adjacency_from_dag(dagA, weight, q)
            B = adjacency_from_dag(dagB, weight, q)
            k_max = max(KS)
            UA_full = top_k_left_singular(A, k_max)
            UB_full = top_k_left_singular(B, k_max)
            # Permuted-B baselines (compute once across seeds, reuse over k).
            UBn_full = []
            for seed in range(N_NULL_SEEDS):
                dagB_perm = dag_with_permuted_weight(dagB, weight, seed)
                B_perm = adjacency_from_dag(dagB_perm, weight, q)
                UBn_full.append(top_k_left_singular(B_perm, k_max))
            for k in KS:
                UA, UB = UA_full[:, :k], UB_full[:, :k]
                sim_real = subspace_similarity(UA, UB)
                sims_null = [subspace_similarity(UA, Un[:, :k]) for Un in UBn_full]
                mu = float(np.mean(sims_null))
                sd = float(np.std(sims_null)) + 1e-9
                out[pair_name][q][k] = {
                    "real": sim_real,
                    "nulls": [float(x) for x in sims_null],
                    "null_mean": mu,
                    "null_std": sd,
                    "z": (sim_real - mu) / sd,
                }
            print("  q={:.3f}: ".format(q) + " | ".join(
                "k={:>3d} real={:.3f} null={:.3f} z={:+.1f}".format(
                    k, out[pair_name][q][k]["real"],
                    out[pair_name][q][k]["null_mean"],
                    out[pair_name][q][k]["z"])
                for k in KS))
    return out


results = {w: run_sweep(w) for w in WEIGHTS}

# %% Save -------------------------------------------------------------------
out_path = os.path.join(DAG_DIR, "null_check.pt")
torch.save(results, out_path)
print("saved:", out_path)

# %% Plot — z-score heatmaps -----------------------------------------------

def plot_z_heatmaps(results_w: dict, weight_name: str):
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 4))
    if len(PAIRS) == 1:
        axes = [axes]
    for ax, pair_name in zip(axes, PAIRS):
        Z = np.array([[results_w[pair_name][q][k]["z"] for k in KS] for q in QUANTILES])
        vmax = max(abs(Z.min()), abs(Z.max()))
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
        ax.set_yticks(range(len(QUANTILES))); ax.set_yticklabels(QUANTILES)
        ax.set_xlabel("k (subspace dim)"); ax.set_ylabel("threshold quantile")
        ax.set_title(f"{weight_name} — {pair_name}\n(real - null_mean) / null_std")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.show()


for w in WEIGHTS:
    plot_z_heatmaps(results[w], w)

# %% Plot — real vs null curves --------------------------------------------

def plot_real_vs_null(results_w: dict, weight_name: str,
                      q_to_show=(0.95, 0.99, 0.999)):
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 4))
    if len(PAIRS) == 1:
        axes = [axes]
    for ax, pair_name in zip(axes, PAIRS):
        for q in q_to_show:
            real = np.array([results_w[pair_name][q][k]["real"]      for k in KS])
            mean = np.array([results_w[pair_name][q][k]["null_mean"] for k in KS])
            std  = np.array([results_w[pair_name][q][k]["null_std"]  for k in KS])
            line, = ax.plot(KS, real, marker="o", label=f"real q={q}")
            ax.plot(KS, mean, marker="x", linestyle="--",
                    color=line.get_color(), alpha=0.55, label=f"null q={q}")
            ax.fill_between(KS, mean - std, mean + std,
                            color=line.get_color(), alpha=0.1)
        ax.set_xlabel("k"); ax.set_ylabel("subspace similarity")
        ax.set_title(f"{weight_name} — {pair_name}")
        ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.show()


for w in WEIGHTS:
    plot_real_vs_null(results[w], w)

# %% Quick summary table ----------------------------------------------------
print("\n=== summary at (q=0.99, k=100) ===")
print(f"{'weight':>6}  {'pair':>10}  {'real':>6}  {'null':>6}  {'z':>6}")
for w in WEIGHTS:
    for p in PAIRS:
        r = results[w][p][0.99][100]
        print(f"{w:>6}  {p:>10}  {r['real']:>6.3f}  {r['null_mean']:>6.3f}  {r['z']:>+6.1f}")

# %% Cross-weight ranking — which edge weight gives the biggest real-vs-null gap?
print("\n=== best (real - null_mean) gap over (q, k), per weight per pair ===")
print(f"{'weight':>6}  {'pair':>10}  {'q*':>6}  {'k*':>4}  {'real':>6}  {'null':>6}  {'z':>6}")
for w in WEIGHTS:
    for p in PAIRS:
        best = max(
            ((q, k, results[w][p][q][k]) for q in QUANTILES for k in KS),
            key=lambda x: x[2]["real"] - x[2]["null_mean"],
        )
        q, k, r = best
        print(f"{w:>6}  {p:>10}  {q:>6.3f}  {k:>4d}  "
              f"{r['real']:>6.3f}  {r['null_mean']:>6.3f}  {r['z']:>+6.1f}")
