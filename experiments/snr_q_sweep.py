"""Q-sweep extension of the metric-validation SNR test (parallel).

Pair: Mixtral-8x7B / c4   vs   Qwen3-30B-A3B / c4
Settings: alpha = 1, beta = 0 (pure path-geometry channel of the structural cost)
Axis:     Q in {0.0, 0.9, 0.95, 0.99, 0.999} - per-graph quantile sparsification

For each Q we compare:
  - real x real:   d_mix vs d_qwen at the same Q-threshold, N_SOLVER_SEEDS FGW-solver seeds
  - real x random: d_mix vs (qwen with W_softmax entries i.i.d. shuffled),
                   N_PERM_SEEDS permutation seeds, Q-threshold recomputed per-perm

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


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
PAIR = ("mixtral-8x7b", "qwen3-30b-a3b")
DATASET = "c4"
QUANTILES = [0.0, 0.9, 0.95, 0.99, 0.999]
N_SOLVER_SEEDS = 5
N_PERM_SEEDS = 5
N_INIT = 10           # FGW random initialisations per solver call
ALPHA = 1.0
BETA = 0.0


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


def shuffle_edges(dag: dict, seed: int) -> dict:
    """Copy `dag` with W_softmax entries i.i.d. shuffled."""
    out = copy.deepcopy(dag)
    g = torch.Generator().manual_seed(seed)
    W = out["W_softmax"].clone()
    flat = W.flatten()
    out["W_softmax"] = flat[torch.randperm(flat.numel(), generator=g)].reshape(W.shape)
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


def _worker_compute(job: tuple[float, str, int]) -> dict:
    """One FGW call: (Q, kind, seed) -> distance + the two thetas used."""
    assert _WORKER_DAGS is not None
    Q, kind, seed = job
    d_a, d_b = _WORKER_DAGS

    theta_a = q_threshold(d_a, Q)
    if kind == "real":
        theta_b = q_threshold(d_b, Q)
        t1 = build_triple(d_a, beta=BETA, edge_threshold=theta_a)
        t2 = build_triple(d_b, beta=BETA, edge_threshold=theta_b)
        d, _ = fgw_distance(t1, t2, alpha=ALPHA, n_init=N_INIT, seed=seed)
    elif kind == "random":
        d_b_rand = shuffle_edges(d_b, seed + 1000)
        theta_b = q_threshold(d_b_rand, Q)
        t1 = build_triple(d_a, beta=BETA, edge_threshold=theta_a)
        t2 = build_triple(d_b_rand, beta=BETA, edge_threshold=theta_b)
        d, _ = fgw_distance(t1, t2, alpha=ALPHA, n_init=N_INIT, seed=0)
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    return {"Q": Q, "kind": kind, "seed": seed, "d": d,
            "theta_a": theta_a, "theta_b": theta_b}


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
    print(f"Settings     : alpha={ALPHA}, beta={BETA}, n_init={N_INIT}, "
          f"solver_seeds={N_SOLVER_SEEDS}, perm_seeds={N_PERM_SEEDS}")
    print(f"Q axis       : {QUANTILES}")
    print(f"Workers      : {n_workers}")

    # Build job list (50 FGW calls = 5 Q x (5 real + 5 random)).
    jobs: list[tuple[float, str, int]] = []
    for Q in QUANTILES:
        for s in range(N_SOLVER_SEEDS):
            jobs.append((Q, "real", s))
        for s in range(N_PERM_SEEDS):
            jobs.append((Q, "random", s))
    print(f"Total jobs   : {len(jobs)}\n")

    # Run pool.
    t0 = time.time()
    raw: dict[float, dict] = {Q: {"real": {}, "random": {}, "theta_a": None,
                                  "theta_b_real": None} for Q in QUANTILES}
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init,
                             initargs=(result_path, PAIR, DATASET)) as ex:
        futs = [ex.submit(_worker_compute, job) for job in jobs]
        for f in as_completed(futs):
            r = f.result()
            Q, kind, seed = r["Q"], r["kind"], r["seed"]
            raw[Q][kind][seed] = (r["d"], r["theta_b"])
            raw[Q]["theta_a"] = r["theta_a"]
            if kind == "real":
                raw[Q]["theta_b_real"] = r["theta_b"]
            completed += 1
            print(f"[{completed:>3}/{len(jobs)}]  Q={Q:>6.3f}  {kind:>6}  "
                  f"seed={seed}  d={r['d']:.4f}  "
                  f"theta_b={r['theta_b']:.4f}  (t={time.time() - t0:.0f}s)",
                  flush=True)

    # Aggregate per Q.
    results: dict[str, dict] = {}
    for Q in QUANTILES:
        real_d = np.array([raw[Q]["real"][s][0] for s in range(N_SOLVER_SEEDS)])
        rand_d = np.array([raw[Q]["random"][s][0] for s in range(N_PERM_SEEDS)])
        real_S = np.exp(-real_d)
        rand_S = np.exp(-rand_d)
        gap_d = float(rand_d.mean() - real_d.mean())
        noise = max(float(real_d.std()), float(rand_d.std()), 1e-9)
        snr = gap_d / noise

        results[f"{Q}"] = dict(
            Q=Q,
            theta_a=float(raw[Q]["theta_a"]),
            theta_b_real=float(raw[Q]["theta_b_real"]),
            real_d_mean=float(real_d.mean()), real_d_std=float(real_d.std()),
            rand_d_mean=float(rand_d.mean()), rand_d_std=float(rand_d.std()),
            real_S_mean=float(real_S.mean()), real_S_std=float(real_S.std()),
            rand_S_mean=float(rand_S.mean()), rand_S_std=float(rand_S.std()),
            gap_d=gap_d, snr=snr,
            real_d=[float(x) for x in real_d],
            rand_d=[float(x) for x in rand_d],
        )

    # Print summary table.
    print()
    print(f"{'Q':>6s}  {'realS_mean':>10s}  {'realS_std':>10s}  "
          f"{'randS_mean':>10s}  {'randS_std':>10s}  {'gap_d':>8s}  "
          f"{'SNR':>7s}  {'theta_a':>9s}  {'theta_b':>9s}")
    print("-" * 105)
    for Q in QUANTILES:
        r = results[f"{Q}"]
        print(f"{Q:>6.3f}  {r['real_S_mean']:>10.4f}  {r['real_S_std']:>10.4f}  "
              f"{r['rand_S_mean']:>10.4f}  {r['rand_S_std']:>10.4f}  "
              f"{r['gap_d']:>+8.4f}  {r['snr']:>7.1f}  "
              f"{r['theta_a']:>9.4f}  {r['theta_b_real']:>9.4f}")

    meta = dict(
        pair=list(PAIR), dataset=DATASET,
        alpha=ALPHA, beta=BETA, n_init=N_INIT,
        n_solver_seeds=N_SOLVER_SEEDS, n_perm_seeds=N_PERM_SEEDS,
        quantiles=QUANTILES, n_workers=n_workers,
        wallclock_s=time.time() - t0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
