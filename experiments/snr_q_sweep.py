"""(alpha, Q)-sweep validation of the FGW metric (parallel).

Pair: Mixtral-8x7B / c4   vs   Qwen3-30B-A3B / c4
Settings: beta = 0.5 fixed (matches the headline sweep).
Axes:
  alpha in {0.0, 0.5, 1.0}                       - feature vs structure mix
  Q     in {0.0, 0.9, 0.95, 0.99, 0.999}         - per-graph quantile sparsification

For each (alpha, Q) we compare:
  - real x real:   d_mix vs d_qwen at the same Q-threshold, N_SOLVER_SEEDS FGW-solver seeds
  - real x random: d_mix vs make_null(d_qwen, seed), N_PERM_SEEDS permutation seeds.
                   The null applies (i) a forward-triangle-only shuffle of
                   W_softmax entries, AND (ii) independent cross-layer
                   permutations of each per-vertex feature column (act,
                   n_tokens_selected, top_*-bundle). See make_null() docstring.
                   Q-threshold recomputed per-null.

Parallelism: the 50 independent FGW calls are dispatched to a ProcessPoolExecutor.
Set SLURM_CPUS_PER_TASK (or pass --workers N) to control the worker count.

Outputs: <project>/results/snr_q_sweep_results.json + printed summary table.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Single-thread BLAS/OpenMP BEFORE importing torch/numpy so each worker stays
# at 1 thread (we get parallelism from the worker pool, not from BLAS).
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.fgw import build_triple, fgw_distance  # noqa: E402
from experiments.run_alpha_beta_sweep import _subset_triple  # noqa: E402


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
PAIR = ("mixtral-8x7b", "qwen3-30b-a3b")
DATASET = "c4"
QUANTILES  = [0.0, 0.9, 0.95, 0.99, 0.999]
ALPHA_AXIS = [0.0, 0.5, 1.0]
N_SOLVER_SEEDS = 5
N_PERM_SEEDS = 5
N_INIT = 10           # FGW random initialisations per solver call
BETA = 0.5            # matches the headline sweep


# ---------------------------------------------------------------------------
# Helpers (shared by main + workers).
# ---------------------------------------------------------------------------
def q_threshold(dag: dict, Q: float) -> float:
    """Per-graph Q-quantile of positive forward |W_softmax| edges."""
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


def _shuffle_forward_edges(dag: dict, seed: int) -> dict:
    """Permute W_softmax entries WITHIN the forward triangle (s < r).
    Backward triangle stays zero. Preserves the marginal distribution of
    forward edge magnitudes exactly; randomises which (sender, receiver)
    pair holds which weight.
    """
    out = copy.deepcopy(dag)
    g = torch.Generator().manual_seed(seed)
    W = out["W_softmax"].clone()
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    fwd_vals = W[fwd]
    perm = torch.randperm(fwd_vals.numel(), generator=g)
    W_new = torch.zeros_like(W)
    W_new[fwd] = fwd_vals[perm]
    out["W_softmax"] = W_new
    return out


def _shuffle_features(dag: dict, seed: int) -> dict:
    """Independent cross-layer permutation of each per-vertex feature column.

    Three independent permutations of shape (L*N,):
      - pi_act      applied to dag["act"]
      - pi_load     applied to dag["n_tokens_selected"]
      - pi_class    applied jointly to dag["top_weight"], ["top_prompt"],
                    ["top_pos"], ["top_token"] (they form a bundle from
                    which class_hist is computed; must permute together
                    to keep that bundle internally consistent).

    Depth is positional (not stored) and stays correct after this shuffle.
    Out-strength / in-strength are recomputed from the shuffled W inside
    build_triple, so they remain consistent with the shuffled structure.

    Per-feature-column independence prevents the transport plan from
    'decrypting' the permutation: no single pi pairs vertices in a way
    that zeroes any feature column.
    """
    out = copy.deepcopy(dag)
    g = torch.Generator().manual_seed(seed)
    L, N = out["W_softmax"].shape[0], out["W_softmax"].shape[1]
    n_verts = L * N

    def _apply(key, perm):
        t = out[key]
        arr = t.reshape(n_verts, *t.shape[2:])
        out[key] = arr[perm].reshape(t.shape)

    # Independent permutation per feature group.
    if "act" in out:
        _apply("act", torch.randperm(n_verts, generator=g))
    if "n_tokens_selected" in out:
        _apply("n_tokens_selected", torch.randperm(n_verts, generator=g))

    # Class-hist bundle: same permutation across the four top_* tensors so
    # that each null-vertex's top-K token bundle stays internally coherent.
    class_perm = torch.randperm(n_verts, generator=g)
    for key in ("top_weight", "top_prompt", "top_pos", "top_token"):
        if key in out:
            _apply(key, class_perm)

    return out


def make_null(dag: dict, seed: int) -> dict:
    """Combined null: forward-only edge shuffle + cross-layer per-column
    feature shuffle. See _shuffle_forward_edges and _shuffle_features for
    the construction rationale."""
    out = _shuffle_forward_edges(dag, seed)
    out = _shuffle_features(out, seed + 1_000_003)
    return out


# ---------------------------------------------------------------------------
# Worker: per-process DAG loading + single-job compute.
# ---------------------------------------------------------------------------
_WORKER_DAGS: tuple[dict, dict] | None = None


def _worker_init(result_path: str, pair: tuple[str, str], dataset: str) -> None:
    """Called once per worker process: load DAGs + pin BLAS threads."""
    global _WORKER_DAGS
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)
    d_a = torch.load(
        Path(result_path) / "circuits" / f"dag_{pair[0]}_{dataset}.pt",
        weights_only=False,
    )
    d_b = torch.load(
        Path(result_path) / "circuits" / f"dag_{pair[1]}_{dataset}.pt",
        weights_only=False,
    )
    _WORKER_DAGS = (d_a, d_b)


def _isolated_keep_mask(dag: dict, theta: float) -> np.ndarray:
    """Same filter as run_alpha_beta_sweep.build_triple_at_Q:
    keep a vertex iff it has at least one surviving forward in- or out-edge
    above threshold."""
    W = dag["W_softmax"]
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    survive = (torch.abs(W) > theta) & fwd
    out_sparse = survive.sum(dim=(2, 3)).reshape(-1).cpu().numpy()
    in_sparse  = survive.sum(dim=(0, 1)).reshape(-1).cpu().numpy()
    return (out_sparse > 0) | (in_sparse > 0)


def _build_filtered_triple(dag: dict, theta: float):
    """Build a triple at BETA, edge_threshold=theta, then drop vertices
    isolated in the sparsified graph (matches the headline sweep)."""
    triple = build_triple(dag, beta=BETA, edge_threshold=theta)
    keep_mask = _isolated_keep_mask(dag, theta)
    return _subset_triple(triple, keep_mask), int(keep_mask.sum())


def _worker_compute(job: tuple[float, float, str, int]) -> dict:
    """One FGW call: (alpha, Q, kind, seed) -> distance + thresholds + |V|s."""
    assert _WORKER_DAGS is not None
    alpha, Q, kind, seed = job
    d_a, d_b = _WORKER_DAGS

    theta_a = q_threshold(d_a, Q)
    t1, n_keep_a = _build_filtered_triple(d_a, theta_a)
    if kind == "real":
        theta_b = q_threshold(d_b, Q)
        t2, n_keep_b = _build_filtered_triple(d_b, theta_b)
        d, _ = fgw_distance(t1, t2, alpha=alpha, n_init=N_INIT, seed=seed)
    elif kind == "random":
        d_b_rand = make_null(d_b, seed + 1000)
        theta_b = q_threshold(d_b_rand, Q)
        t2, n_keep_b = _build_filtered_triple(d_b_rand, theta_b)
        d, _ = fgw_distance(t1, t2, alpha=alpha, n_init=N_INIT, seed=0)
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    return {"alpha": alpha, "Q": Q, "kind": kind, "seed": seed, "d": d,
            "theta_a": theta_a, "theta_b": theta_b,
            "n_keep_a": n_keep_a, "n_keep_b": n_keep_b}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: SLURM_CPUS_PER_TASK or 8)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "snr_q_sweep_results.json"

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    print(f"Project root : {ROOT}")
    print(f"Result path  : {result_path}")
    print(f"Pair         : {PAIR}  on  {DATASET}")
    print(f"Settings     : beta={BETA}, n_init={N_INIT}, "
          f"solver_seeds={N_SOLVER_SEEDS}, perm_seeds={N_PERM_SEEDS}")
    print(f"alpha axis   : {ALPHA_AXIS}")
    print(f"Q axis       : {QUANTILES}")
    print(f"Workers      : {n_workers}")

    # Build job list: |alpha_axis| * |QUANTILES| * (real + random) seeds.
    jobs: list[tuple[float, float, str, int]] = []
    for alpha in ALPHA_AXIS:
        for Q in QUANTILES:
            for s in range(N_SOLVER_SEEDS):
                jobs.append((alpha, Q, "real", s))
            for s in range(N_PERM_SEEDS):
                jobs.append((alpha, Q, "random", s))
    print(f"Total jobs   : {len(jobs)}\n")

    # Run pool. raw[(alpha, Q)] holds the per-cell aggregation state.
    t0 = time.time()
    raw: dict[tuple[float, float], dict] = {
        (a, Q): {"real": {}, "random": {}, "theta_a": None,
                 "theta_b_real": None, "n_keep_a": None,
                 "n_keep_b_real": None, "n_keep_b_rand": []}
        for a in ALPHA_AXIS for Q in QUANTILES
    }
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init,
                             initargs=(result_path, PAIR, DATASET)) as ex:
        futs = [ex.submit(_worker_compute, job) for job in jobs]
        for f in as_completed(futs):
            r = f.result()
            alpha, Q, kind, seed = r["alpha"], r["Q"], r["kind"], r["seed"]
            cell = raw[(alpha, Q)]
            cell[kind][seed] = (r["d"], r["theta_b"])
            cell["theta_a"] = r["theta_a"]
            cell["n_keep_a"] = r["n_keep_a"]
            if kind == "real":
                cell["theta_b_real"] = r["theta_b"]
                cell["n_keep_b_real"] = r["n_keep_b"]
            else:
                cell["n_keep_b_rand"].append(r["n_keep_b"])
            completed += 1
            print(f"[{completed:>3}/{len(jobs)}]  alpha={alpha:.2f}  Q={Q:>6.3f}  "
                  f"{kind:>6}  seed={seed}  d={r['d']:.4f}  "
                  f"|V|=({r['n_keep_a']},{r['n_keep_b']})  "
                  f"(t={time.time() - t0:.0f}s)",
                  flush=True)

    # Aggregate per (alpha, Q).
    results: dict[str, dict] = {}
    for alpha in ALPHA_AXIS:
        for Q in QUANTILES:
            cell = raw[(alpha, Q)]
            real_d = np.array([cell["real"][s][0] for s in range(N_SOLVER_SEEDS)])
            rand_d = np.array([cell["random"][s][0] for s in range(N_PERM_SEEDS)])
            real_S = np.exp(-real_d)
            rand_S = np.exp(-rand_d)
            gap_d = float(rand_d.mean() - real_d.mean())
            noise = max(float(real_d.std()), float(rand_d.std()), 1e-9)
            snr = gap_d / noise

            results[f"a{alpha}_Q{Q}"] = dict(
                alpha=alpha, Q=Q,
                theta_a=float(cell["theta_a"]),
                theta_b_real=float(cell["theta_b_real"]),
                n_keep_a=int(cell["n_keep_a"]),
                n_keep_b_real=int(cell["n_keep_b_real"]),
                n_keep_b_rand_mean=float(np.mean(cell["n_keep_b_rand"])),
                real_d_mean=float(real_d.mean()), real_d_std=float(real_d.std()),
                rand_d_mean=float(rand_d.mean()), rand_d_std=float(rand_d.std()),
                real_S_mean=float(real_S.mean()), real_S_std=float(real_S.std()),
                rand_S_mean=float(rand_S.mean()), rand_S_std=float(rand_S.std()),
                gap_d=gap_d, snr=snr,
                real_d=[float(x) for x in real_d],
                rand_d=[float(x) for x in rand_d],
            )

    # Print summary table: one block per alpha, rows = Q.
    print()
    for alpha in ALPHA_AXIS:
        print(f"--- alpha = {alpha} ---")
        print(f"{'Q':>6s}  {'realS':>8s}  {'randS':>8s}  {'gap_S':>+8s}  "
              f"{'gap_d':>+8s}  {'SNR':>7s}  {'|V|_a':>6s}  "
              f"{'|V|_b_r':>8s}  {'|V|_b_p':>8s}")
        print("-" * 84)
        for Q in QUANTILES:
            r = results[f"a{alpha}_Q{Q}"]
            gap_S = r["real_S_mean"] - r["rand_S_mean"]
            print(f"{Q:>6.3f}  {r['real_S_mean']:>8.4f}  {r['rand_S_mean']:>8.4f}  "
                  f"{gap_S:>+8.4f}  {r['gap_d']:>+8.4f}  {r['snr']:>7.1f}  "
                  f"{r['n_keep_a']:>6d}  {r['n_keep_b_real']:>8d}  "
                  f"{r['n_keep_b_rand_mean']:>8.0f}")
        print()

    meta = dict(
        pair=list(PAIR), dataset=DATASET,
        beta=BETA, n_init=N_INIT,
        n_solver_seeds=N_SOLVER_SEEDS, n_perm_seeds=N_PERM_SEEDS,
        alpha_axis=ALPHA_AXIS, quantiles=QUANTILES,
        n_workers=n_workers,
        wallclock_s=time.time() - t0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
