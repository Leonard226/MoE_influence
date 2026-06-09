"""Q-sweep extension of the metric-validation SNR test.

Pair: Mixtral-8x7B / c4   vs   Qwen3-30B-A3B / c4
Settings: alpha = 1, beta = 0 (pure path-geometry channel of the structural cost)
Axis:     Q in {0.0, 0.9, 0.95, 0.99, 0.999} - per-graph quantile sparsification
          (Q = 0 means dense forward DAG; Q = 0.9 keeps top 10% of edges; etc.)

For each Q we compare:
  - real x real:   d_mix vs d_qwen at the same Q-threshold, 5 FGW-solver seeds
  - real x random: d_mix vs (qwen with its W_softmax entries i.i.d. shuffled),
                   5 permutation seeds, Q-threshold recomputed per-perm

Outputs JSON to <project>/results/snr_q_sweep_results.json plus a printed table.

Runs single-threaded; ~1 CPU-hour total on piora4.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Make experiments/ importable when run from project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.fgw import build_triple, fgw_distance  # noqa: E402

# Single-thread BLAS/OpenMP to match how the sweep is run on the cluster.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
PAIR = ("mixtral-8x7b", "qwen3-30b-a3b")
DATASET = "c4"
QUANTILES = [0.0, 0.9, 0.95, 0.99, 0.999]
N_SOLVER_SEEDS = 5
N_PERM_SEEDS = 5
N_INIT = 10                # FGW random initialisations per solver call
ALPHA = 1.0
BETA = 0.0


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def q_threshold(dag: dict, Q: float) -> float:
    """Per-graph Q-quantile of positive forward |W_softmax| edges.

    Matches the convention in run_alpha_beta_sweep.py: drop edges with
    |W| < theta_Q before computing C_path.
    """
    if Q == 0.0:
        return 0.0
    W = dag["W_softmax"]
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    vals = W[fwd].abs()
    vals = vals[vals > 0]
    return float(torch.quantile(vals, Q))


def shuffle_edges(dag: dict, seed: int) -> dict:
    """Return a copy of dag with W_softmax entries i.i.d. shuffled.

    Preserves the marginal distribution of edge weights (each value still
    appears exactly once) but destroys the (sender, receiver) coupling.
    """
    out = copy.deepcopy(dag)
    g = torch.Generator().manual_seed(seed)
    W = out["W_softmax"].clone()
    flat = W.flatten()
    out["W_softmax"] = flat[torch.randperm(flat.numel(), generator=g)].reshape(W.shape)
    return out


def _load_dag(model: str, dataset: str, result_path: str) -> dict:
    path = Path(result_path) / "circuits" / f"dag_{model}_{dataset}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Main sweep.
# ---------------------------------------------------------------------------
def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "snr_q_sweep_results.json"

    print(f"Loading DAGs from {result_path}/circuits/ ...")
    d_a = _load_dag(PAIR[0], DATASET, result_path)
    d_b = _load_dag(PAIR[1], DATASET, result_path)
    print(f"  {PAIR[0]}/{DATASET}: W_softmax shape={tuple(d_a['W_softmax'].shape)}")
    print(f"  {PAIR[1]}/{DATASET}: W_softmax shape={tuple(d_b['W_softmax'].shape)}")
    print(f"Settings: alpha={ALPHA}, beta={BETA}, n_init={N_INIT}, "
          f"solver_seeds={N_SOLVER_SEEDS}, perm_seeds={N_PERM_SEEDS}")
    print(f"Q axis: {QUANTILES}")
    print()

    results: dict[str, dict] = {}
    print(f"{'Q':>6s}  {'realS_mean':>10s}  {'realS_std':>10s}  "
          f"{'randS_mean':>10s}  {'randS_std':>10s}  {'gap_d':>8s}  {'SNR':>7s}  "
          f"{'theta_a':>9s}  {'theta_b':>9s}  {'elapsed':>8s}")
    print("-" * 110)

    for Q in QUANTILES:
        t0 = time.time()
        theta_a = q_threshold(d_a, Q)
        theta_b = q_threshold(d_b, Q)

        # --- real x real (FGW solver seeds only; DAGs deterministic) ---
        real_d: list[float] = []
        for s in range(N_SOLVER_SEEDS):
            t1 = build_triple(d_a, beta=BETA, edge_threshold=theta_a)
            t2 = build_triple(d_b, beta=BETA, edge_threshold=theta_b)
            d, _ = fgw_distance(t1, t2, alpha=ALPHA, n_init=N_INIT, seed=s)
            real_d.append(d)

        # --- real x random (one solver seed per perm; perm = source of variance) ---
        rand_d: list[float] = []
        for s in range(N_PERM_SEEDS):
            d_b_rand = shuffle_edges(d_b, s + 1000)
            theta_b_rand = q_threshold(d_b_rand, Q)
            t1 = build_triple(d_a, beta=BETA, edge_threshold=theta_a)
            t2 = build_triple(d_b_rand, beta=BETA, edge_threshold=theta_b_rand)
            d, _ = fgw_distance(t1, t2, alpha=ALPHA, n_init=N_INIT, seed=0)
            rand_d.append(d)

        real_d_arr = np.array(real_d)
        rand_d_arr = np.array(rand_d)
        real_S = np.exp(-real_d_arr)
        rand_S = np.exp(-rand_d_arr)

        gap_d = float(rand_d_arr.mean() - real_d_arr.mean())
        noise = max(float(real_d_arr.std()), float(rand_d_arr.std()), 1e-9)
        snr = gap_d / noise

        results[f"{Q}"] = dict(
            Q=Q,
            theta_a=theta_a, theta_b=theta_b,
            real_d_mean=float(real_d_arr.mean()), real_d_std=float(real_d_arr.std()),
            rand_d_mean=float(rand_d_arr.mean()), rand_d_std=float(rand_d_arr.std()),
            real_S_mean=float(real_S.mean()), real_S_std=float(real_S.std()),
            rand_S_mean=float(rand_S.mean()), rand_S_std=float(rand_S.std()),
            gap_d=gap_d, snr=snr,
            real_d=[float(x) for x in real_d_arr],
            rand_d=[float(x) for x in rand_d_arr],
            elapsed_s=time.time() - t0,
        )
        print(f"{Q:>6.3f}  {real_S.mean():>10.4f}  {real_S.std():>10.4f}  "
              f"{rand_S.mean():>10.4f}  {rand_S.std():>10.4f}  "
              f"{gap_d:>+8.4f}  {snr:>7.1f}  {theta_a:>9.4f}  {theta_b:>9.4f}  "
              f"{results[f'{Q}']['elapsed_s']:>7.0f}s")

    meta = dict(
        pair=list(PAIR), dataset=DATASET,
        alpha=ALPHA, beta=BETA, n_init=N_INIT,
        n_solver_seeds=N_SOLVER_SEEDS, n_perm_seeds=N_PERM_SEEDS,
        quantiles=QUANTILES,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
