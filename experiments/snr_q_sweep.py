"""Self-perturbation panel for the FGW metric (parallel).

For each model M in MODELS we measure how much the FGW similarity drops when
we compare M against a randomised copy of ITSELF. The upper anchor is
S(M, M) = 1 by construction (identity); the lower anchor is

  X_M(variant, alpha, Q) = S_{alpha, Q}(M, null_variant(M)),

which directly measures how sensitive the metric is to routing identity
within a single model, with marginal distributions held exactly fixed.

This sidesteps the OT 'uniform-target-is-easier-to-match' confound that
breaks cross-model permutation nulls at alpha < 1.

Axes:
  variant in {'edge', 'feature', 'both'}     - which channel is perturbed
  alpha   in {0.0, 0.5, 1.0}                 - feature vs structure mix
  Q       in {0.0, 0.9, 0.95, 0.99, 0.999}   - per-graph quantile sparsification
  model   in MODELS                          - the model being self-perturbed

For each cell we compute S(M, null_variant(M)) over N_PERM_SEEDS permutations
of the null. real-vs-real is skipped (S = 1 by construction).

Parallelism: independent FGW calls dispatched to a ProcessPoolExecutor; set
SLURM_CPUS_PER_TASK (or pass --workers N) to control the worker count.

Outputs: <result_path>/snr_q_sweep_results.json + printed summary tables.
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
MODELS = [
    "mixtral-8x7b",
    "qwen3-30b-a3b",
    "deepseek-v2-lite",
    "olmoe",
]
DATASET = "c4"
QUANTILES  = [0.0, 0.9, 0.95, 0.99, 0.999]
ALPHA_AXIS = [0.0, 0.5, 1.0]
VARIANTS   = ["edge", "feature", "both"]
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
    Preserves the marginal distribution of forward edge magnitudes exactly;
    randomises which (sender, receiver) pair holds which weight.
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

    Three independent permutations of length L*N:
      - pi_act      applied to dag["act"]
      - pi_load     applied to dag["n_tokens_selected"]
      - pi_class    applied JOINTLY to the top_* tensors (they form the
                    bundle that compute_class_histogram consumes; must
                    permute together to stay internally coherent)

    depth is positional (not stored); out_strength / in_strength are
    recomputed from the (unchanged here) W inside build_triple, so this
    helper alone touches only the dag entries that are *stored* features.
    """
    out = copy.deepcopy(dag)
    g = torch.Generator().manual_seed(seed)
    L, N = out["W_softmax"].shape[0], out["W_softmax"].shape[1]
    n_verts = L * N

    def _apply(key, perm):
        t = out[key]
        arr = t.reshape(n_verts, *t.shape[2:])
        out[key] = arr[perm].reshape(t.shape)

    if "act" in out:
        _apply("act", torch.randperm(n_verts, generator=g))
    if "n_tokens_selected" in out:
        _apply("n_tokens_selected", torch.randperm(n_verts, generator=g))

    class_perm = torch.randperm(n_verts, generator=g)
    for key in ("top_weight", "top_prompt", "top_pos", "top_token"):
        if key in out:
            _apply(key, class_perm)

    return out


def _make_null_edge(dag: dict, seed: int) -> dict:
    """Edge-only perturbation: forward-only edge shuffle, features intact."""
    return _shuffle_forward_edges(dag, seed)


def _make_null_feature(dag: dict, seed: int) -> dict:
    """Feature-only perturbation: cross-layer per-column feature shuffle,
    edges intact."""
    return _shuffle_features(dag, seed)


def _make_null_both(dag: dict, seed: int) -> dict:
    """Combined perturbation: both edge and feature shuffles."""
    out = _shuffle_forward_edges(dag, seed)
    out = _shuffle_features(out, seed + 1_000_003)
    return out


NULL_FNS = {
    "edge":    _make_null_edge,
    "feature": _make_null_feature,
    "both":    _make_null_both,
}


# ---------------------------------------------------------------------------
# Worker: per-process DAG loading + single-job compute.
# ---------------------------------------------------------------------------
_WORKER_DAGS: dict[str, dict] | None = None


def _worker_init(result_path: str, models: list[str], dataset: str) -> None:
    """Called once per worker process: load ALL models' DAGs + pin BLAS."""
    global _WORKER_DAGS
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)
    _WORKER_DAGS = {}
    for m in models:
        p = Path(result_path) / "circuits" / f"dag_{m}_{dataset}.pt"
        _WORKER_DAGS[m] = torch.load(p, weights_only=False)


def _isolated_keep_mask(dag: dict, theta: float) -> np.ndarray:
    """Same filter as run_alpha_beta_sweep.build_triple_at_Q:
    keep a vertex iff it has at least one surviving forward in- or out-edge
    above the threshold."""
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


def _worker_compute(job: tuple[str, str, float, float, int]) -> dict:
    """One self-perturbation FGW call:
       (model, variant, alpha, Q, seed) -> distance + kept-vertex counts."""
    assert _WORKER_DAGS is not None
    model, variant, alpha, Q, seed = job
    dag = _WORKER_DAGS[model]

    theta_real = q_threshold(dag, Q)
    t_real, n_keep_real = _build_filtered_triple(dag, theta_real)

    dag_null = NULL_FNS[variant](dag, seed + 1000)
    theta_null = q_threshold(dag_null, Q)
    t_null, n_keep_null = _build_filtered_triple(dag_null, theta_null)

    d, _ = fgw_distance(t_real, t_null, alpha=alpha, n_init=N_INIT, seed=0)

    return {"model": model, "variant": variant, "alpha": alpha, "Q": Q,
            "seed": seed, "d": d,
            "theta_real": theta_real, "theta_null": theta_null,
            "n_keep_real": n_keep_real, "n_keep_null": n_keep_null}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: SLURM_CPUS_PER_TASK or 8)")
    parser.add_argument("--variant", choices=list(NULL_FNS) + ["all"],
                        default="all",
                        help="run a single null variant or all three (default: all)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "snr_q_sweep_results.json"

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    variants = VARIANTS if args.variant == "all" else [args.variant]

    print(f"Project root : {ROOT}")
    print(f"Result path  : {result_path}")
    print(f"Models       : {MODELS}  on  {DATASET}")
    print(f"Settings     : beta={BETA}, n_init={N_INIT}, "
          f"perm_seeds={N_PERM_SEEDS}")
    print(f"Variants     : {variants}")
    print(f"alpha axis   : {ALPHA_AXIS}")
    print(f"Q axis       : {QUANTILES}")
    print(f"Workers      : {n_workers}")

    # Job list: per-model x per-variant x per-(alpha, Q) x perm-seeds.
    jobs: list[tuple[str, str, float, float, int]] = []
    for model in MODELS:
        for variant in variants:
            for alpha in ALPHA_AXIS:
                for Q in QUANTILES:
                    for s in range(N_PERM_SEEDS):
                        jobs.append((model, variant, alpha, Q, s))
    print(f"Total jobs   : {len(jobs)}\n")

    # raw[(model, variant, alpha, Q)] -> {"d": [seed -> d], "theta_real",
    # "theta_null_avg", "n_keep_real", "n_keep_null_avg"}.
    t0 = time.time()
    raw: dict[tuple[str, str, float, float], dict] = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_worker_init,
                             initargs=(result_path, MODELS, DATASET)) as ex:
        futs = [ex.submit(_worker_compute, job) for job in jobs]
        for f in as_completed(futs):
            r = f.result()
            key = (r["model"], r["variant"], r["alpha"], r["Q"])
            cell = raw.setdefault(key, {
                "d": {}, "theta_real": r["theta_real"],
                "n_keep_real": r["n_keep_real"],
                "theta_null_per_seed": [],
                "n_keep_null_per_seed": [],
            })
            cell["d"][r["seed"]] = r["d"]
            cell["theta_null_per_seed"].append(r["theta_null"])
            cell["n_keep_null_per_seed"].append(r["n_keep_null"])
            completed += 1
            print(f"[{completed:>4}/{len(jobs)}]  {r['model']:<18s}  "
                  f"{r['variant']:>7s}  alpha={r['alpha']:.2f}  "
                  f"Q={r['Q']:>6.3f}  seed={r['seed']}  d={r['d']:.4f}  "
                  f"|V|=({r['n_keep_real']},{r['n_keep_null']})  "
                  f"(t={time.time() - t0:.0f}s)",
                  flush=True)

    # Aggregate per (model, variant, alpha, Q).
    results: dict[str, dict] = {}
    for (model, variant, alpha, Q), cell in raw.items():
        d_arr = np.array([cell["d"][s] for s in range(N_PERM_SEEDS)])
        S_arr = np.exp(-d_arr)
        results[f"{model}_{variant}_a{alpha}_Q{Q}"] = dict(
            model=model, variant=variant, alpha=alpha, Q=Q,
            theta_real=float(cell["theta_real"]),
            theta_null_mean=float(np.mean(cell["theta_null_per_seed"])),
            n_keep_real=int(cell["n_keep_real"]),
            n_keep_null_mean=float(np.mean(cell["n_keep_null_per_seed"])),
            d_mean=float(d_arr.mean()), d_std=float(d_arr.std()),
            S_mean=float(S_arr.mean()), S_std=float(S_arr.std()),
            d=[float(x) for x in d_arr],
            S=[float(x) for x in S_arr],
        )

    # Save JSON FIRST so any print failure doesn't cost us data.
    meta = dict(
        models=MODELS, dataset=DATASET,
        beta=BETA, n_init=N_INIT,
        n_perm_seeds=N_PERM_SEEDS,
        variants=variants, alpha_axis=ALPHA_AXIS, quantiles=QUANTILES,
        n_workers=n_workers,
        wallclock_s=time.time() - t0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}\n")

    # ----- Summary tables -----
    # For each variant: mean +- std over models, per (alpha, Q).
    for variant in variants:
        print(f"=== variant: {variant}  (X = S(M, null_{variant}(M)), "
              f"mean +- std over {len(MODELS)} models) ===")
        header = f"  {'alpha':>5s}  {'Q':>6s}  {'mean X':>8s}  "\
                 f"{'std X':>8s}  per-model X (mixtral / qwen3-30b / "\
                 f"dsl / olmoe)"
        print(header)
        print("-" * len(header))
        for alpha in ALPHA_AXIS:
            for Q in QUANTILES:
                S_per_model = []
                for model in MODELS:
                    key = f"{model}_{variant}_a{alpha}_Q{Q}"
                    if key not in results:
                        continue
                    S_per_model.append(results[key]["S_mean"])
                S_arr = np.array(S_per_model)
                per_model = "  ".join(f"{s:.4f}" for s in S_arr)
                print(f"  {alpha:>5.2f}  {Q:>6.3f}  "
                      f"{S_arr.mean():>8.4f}  {S_arr.std():>8.4f}  "
                      f"{per_model}")
            print()
        print()

    print(f"Total wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
