"""Spectral baseline for the FGW structural metric.

For each (model, task) DAG and each Q in QUANTILES:
  1. Build the Q-sparsified forward adjacency A in [L*N, L*N].
  2. Apply the isolated-vertex filter (same rule as the headline sweep).
  3. Normalise: A_tilde = D_out^{-1/2} A D_in^{-1/2}.
  4. Compute the top-k singular values via scipy sparse SVD.
     The signature is sigma_1 >= sigma_2 >= ... >= sigma_k >= 0 (padded to
     top_k with zeros if rank < top_k).

For each unordered cross-pair (G_a, G_b) at each Q:
  d_spec = || sigma_a - sigma_b ||_2          # L2 on top-k singular values
  S_spec = exp(-d^2)                          # convert to [0, 1] similarity

The script then loads S_full_with_act.npz and reports the Pearson correlation
between the off-diagonal S_spec(., ., Q) and FGW S_alpha=1, S_alpha=0.5,
S_alpha=0 at each Q. This is the publishable comparison: structural-channel
agreement at alpha=1 (we expect strong corr), divergence at alpha<1 (we expect
weaker corr -- that's the "FGW does more than spectral" finding).

Outputs:
  <result_path>/spectral_baseline/S_spec.npz       # (64, 64, 3) similarity tensor
  <result_path>/spectral_baseline/signatures.npz   # per-(m, t, Q) top-k singular vectors
  printed comparison table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Single-thread BLAS BEFORE importing scipy/numpy/torch so each worker is at 1 thread
# (parallelism comes from the pool, not from BLAS).
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Config (mirrors run_alpha_beta_sweep.py).
# ---------------------------------------------------------------------------
MODELS = [
    "mixtral-8x7b",
    "mixtral-8x22b",
    "deepseek-v2-lite",
    "deepseek-v2",
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
    "olmoe",
    "phi-3.5-moe",
]
TASKS = [
    "c4", "math", "code",
    "wikitext2", "gsm8k", "humaneval",
    "pile-arxiv", "pile-github",
]
TUPLES = [(m, t) for m in MODELS for t in TASKS]
N_TUPLES = len(TUPLES)
QUANTILES = [0.9, 0.99, 0.999]
TOP_K = 50


# ---------------------------------------------------------------------------
# Spectral signature: top-k singular values of the normalised, Q-sparsified
# forward adjacency.
# ---------------------------------------------------------------------------
def spectral_signature(dag: dict, Q: float, top_k: int = TOP_K) -> tuple[np.ndarray, int]:
    """Return (signature, n_keep). The signature is a length-top_k vector of
    singular values, padded with zeros if rank < top_k."""
    W = dag["W_softmax"].float()
    L, N = W.shape[0], W.shape[1]

    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)

    # Per-graph Q-threshold over POSITIVE forward edges (matches the sweep).
    # Use numpy.quantile, not torch.quantile: torch errors out above ~16M
    # elements and Qwen3-235B has ~72M forward edges. Same workaround as
    # run_alpha_beta_sweep._edge_quantile_threshold.
    if Q == 0.0:
        theta = 0.0
    else:
        edge_vals = torch.abs(W[fwd]).cpu().numpy().astype(np.float64)
        nz = edge_vals[edge_vals > 0]
        theta = float(np.quantile(nz, Q)) if len(nz) else 0.0

    # Build the sparsified forward adjacency.
    keep_edge = (torch.abs(W) > theta) & fwd
    A_full = (W.abs() * keep_edge.float()).reshape(L * N, L * N).cpu().numpy().astype(np.float64)

    # Isolated-vertex filter (same as run_alpha_beta_sweep._isolated_keep_mask):
    # keep iff at least one surviving in- or out-edge above threshold.
    out_sparse = keep_edge.sum(dim=(2, 3)).reshape(-1).cpu().numpy()
    in_sparse  = keep_edge.sum(dim=(0, 1)).reshape(-1).cpu().numpy()
    keep_mask  = (out_sparse > 0) | (in_sparse > 0)
    keep_idx   = np.where(keep_mask)[0]
    n_keep     = int(len(keep_idx))

    if n_keep < 2:
        # Degenerate case: zero or one vertex left after filter.
        return np.zeros(top_k, dtype=np.float64), n_keep

    A = A_full[np.ix_(keep_idx, keep_idx)]

    # Normalisation: A_tilde = D_out^{-1/2} A D_in^{-1/2}.
    # d_out[u] = sum_v A[u, v]    (outgoing weight from u)
    # d_in[v]  = sum_u A[u, v]    (incoming weight to v)
    d_out = A.sum(axis=1)
    d_in  = A.sum(axis=0)
    eps = 1e-12
    inv_sqrt_d_out = 1.0 / np.sqrt(d_out + eps)
    inv_sqrt_d_in  = 1.0 / np.sqrt(d_in  + eps)
    A_tilde = (inv_sqrt_d_out[:, None] * A) * inv_sqrt_d_in[None, :]

    # Top-k singular values. svds wants k <= n_keep - 1; pad result with zeros
    # to length top_k for cross-graph compatibility.
    k = min(top_k, n_keep - 1)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds
        # svds is faster on sparse input; A_tilde is generally dense post-norm but
        # the sparse Lanczos path still wins for large matrices.
        sigma = svds(csr_matrix(A_tilde), k=k, which="LM", return_singular_vectors=False)
    except Exception:
        # Fall back to dense SVD if Lanczos fails (small matrix, ill-conditioned, etc.).
        sigma = np.linalg.svd(A_tilde, compute_uv=False)[:k]

    sigma = np.sort(np.asarray(sigma, dtype=np.float64))[::-1]  # descending

    signature = np.zeros(top_k, dtype=np.float64)
    signature[: len(sigma)] = sigma

    return signature, n_keep


# ---------------------------------------------------------------------------
# Worker: compute the signature for one (model, task, Q) triple.
#
# Per-worker DAG cache: if the same DAG is needed again for a different Q on
# the same worker, the second call reuses the in-memory copy. This makes the
# fine-grained 192-job split free in terms of redundant disk I/O.
# ---------------------------------------------------------------------------
_WORKER_DAG_CACHE: dict[tuple[str, str], dict] = {}


def _worker_compute(args: tuple[int, str, str, float, str]) -> dict:
    """Compute the spectral signature for ONE (model, task, Q) job."""
    idx, model, task, Q, result_path = args
    key = (model, task)
    dag = _WORKER_DAG_CACHE.get(key)
    if dag is None:
        path = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
        dag = torch.load(path, weights_only=False)
        _WORKER_DAG_CACHE[key] = dag
    sig, n_keep = spectral_signature(dag, Q, top_k=TOP_K)
    return {"idx": idx, "model": model, "task": task, "Q": Q,
            "signature": sig, "n_keep": n_keep}


# ---------------------------------------------------------------------------
# Main: compute signatures, pairwise distances, and the FGW comparison.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: SLURM_CPUS_PER_TASK or 8)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_dir = Path(result_path) / "spectral_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    print(f"Project root : {ROOT}")
    print(f"Result path  : {result_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Models       : {len(MODELS)}, tasks: {len(TASKS)}, "
          f"|tuples| = {N_TUPLES}")
    print(f"Q axis       : {QUANTILES}")
    print(f"top_k        : {TOP_K}")
    print(f"Workers      : {n_workers}")
    print()

    # Stage 1: compute signatures in parallel.
    # 192 fine-grained jobs (one per (model, task, Q)) so the 128-core pool
    # stays saturated. The per-worker DAG cache means re-using the same DAG
    # for multiple Q values is free in terms of disk I/O.
    print("Stage 1: computing spectral signatures (one SVD per DAG-Q) ...")
    t0 = time.time()
    jobs = [(i, m, t, Q, result_path)
            for i, (m, t) in enumerate(TUPLES)
            for Q in QUANTILES]
    print(f"  Total jobs: {len(jobs)}  (fine-grained: 1 per (model, task, Q))\n")

    # signatures[idx][Q] = ndarray of length TOP_K
    signatures: dict[int, dict[float, np.ndarray]] = {i: {} for i in range(N_TUPLES)}
    n_keep: dict[int, dict[float, int]] = {i: {} for i in range(N_TUPLES)}
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_worker_compute, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            idx = r["idx"]
            Q = r["Q"]
            signatures[idx][Q] = r["signature"]
            n_keep[idx][Q] = r["n_keep"]
            completed += 1
            print(f"  [{completed:>3d}/{len(jobs)}] {r['model']:<18s}/{r['task']:<12s}  "
                  f"Q={Q:>6.3f}  |V|_keep={r['n_keep']:>5d}  "
                  f"sig[0:3]={r['signature'][:3]}  "
                  f"(t={time.time() - t0:.0f}s)", flush=True)

    print(f"\nStage 1 done in {time.time() - t0:.0f}s.\n")

    # Stage 2: pairwise spectral similarity (fast, sequential, ~milliseconds).
    print("Stage 2: pairwise spectral similarities ...")
    t1 = time.time()
    S_spec = np.zeros((N_TUPLES, N_TUPLES, len(QUANTILES)), dtype=np.float64)
    for i in range(N_TUPLES):
        S_spec[i, i, :] = 1.0   # self-similarity
    for qi, Q in enumerate(QUANTILES):
        for i in range(N_TUPLES):
            sig_i = signatures[i][Q]
            for j in range(i + 1, N_TUPLES):
                sig_j = signatures[j][Q]
                d = float(np.linalg.norm(sig_i - sig_j))
                S = float(np.exp(-d * d))
                S_spec[i, j, qi] = S
                S_spec[j, i, qi] = S
    print(f"Stage 2 done in {time.time() - t1:.2f}s.\n")

    # Save the similarity tensor + signatures.
    np.savez(
        out_dir / "S_spec.npz",
        S=S_spec,
        tuples=np.array([f"{m}/{t}" for m, t in TUPLES], dtype=object),
        models=np.array(MODELS, dtype=object),
        tasks=np.array(TASKS, dtype=object),
        quantiles=np.array(QUANTILES),
        top_k=TOP_K,
    )
    print(f"Saved: {out_dir / 'S_spec.npz'}")

    # Per-DAG signatures, indexed (tuple_idx, Q_idx, k).
    sigs_arr = np.zeros((N_TUPLES, len(QUANTILES), TOP_K), dtype=np.float64)
    nk_arr   = np.zeros((N_TUPLES, len(QUANTILES)), dtype=np.int64)
    for i in range(N_TUPLES):
        for qi, Q in enumerate(QUANTILES):
            sigs_arr[i, qi, :] = signatures[i][Q]
            nk_arr[i, qi] = n_keep[i][Q]
    np.savez(
        out_dir / "signatures.npz",
        signatures=sigs_arr,
        n_keep=nk_arr,
        tuples=np.array([f"{m}/{t}" for m, t in TUPLES], dtype=object),
        quantiles=np.array(QUANTILES),
    )
    print(f"Saved: {out_dir / 'signatures.npz'}\n")

    # Stage 3: Pearson correlation with FGW for each (alpha, Q) cell.
    fgw_path = Path(result_path) / "circuits" / "alpha_beta_sweep" / "S_full_with_act.npz"
    if not fgw_path.exists():
        print(f"WARNING: {fgw_path} not found; skipping FGW comparison.")
        print(f"Total wallclock: {time.time() - t0:.0f}s")
        return

    print(f"Stage 3: comparison vs FGW similarities from {fgw_path.name} ...\n")
    fgw_S = np.load(fgw_path)["S"]    # (64, 64, 3, 3): (src, tgt, alpha, Q)

    mask = ~np.eye(N_TUPLES, dtype=bool)

    print(f"{'alpha':>6s}  {'Q':>6s}  {'Pearson(FGW, spec)':>22s}  "
          f"{'Spearman(FGW, spec)':>22s}")
    print("-" * 70)
    from scipy.stats import pearsonr, spearmanr
    for ai, alpha in enumerate([0.0, 0.5, 1.0]):
        for qi, Q in enumerate(QUANTILES):
            fgw_vals = fgw_S[..., ai, qi][mask]
            spec_vals = S_spec[..., qi][mask]
            # Drop NaNs if any.
            ok = ~(np.isnan(fgw_vals) | np.isnan(spec_vals))
            r_p, _ = pearsonr(fgw_vals[ok], spec_vals[ok])
            r_s, _ = spearmanr(fgw_vals[ok], spec_vals[ok])
            print(f"{alpha:>6.2f}  {Q:>6.3f}  {r_p:>22.4f}  {r_s:>22.4f}")
        print()

    # Headline summary: spectral should correlate with FGW alpha=1 (structure)
    # MORE than with FGW alpha=0 (features). Quantify and print.
    print("\nHeadline check (correlations averaged over Q):")
    for ai, alpha in enumerate([0.0, 0.5, 1.0]):
        rs = []
        for qi in range(len(QUANTILES)):
            fgw_vals = fgw_S[..., ai, qi][mask]
            spec_vals = S_spec[..., qi][mask]
            ok = ~(np.isnan(fgw_vals) | np.isnan(spec_vals))
            r_p, _ = pearsonr(fgw_vals[ok], spec_vals[ok])
            rs.append(r_p)
        print(f"  alpha = {alpha:.1f}:  Pearson(FGW, spec) averaged over Q "
              f"= {np.mean(rs):.4f}")

    # Save the comparison numbers for the report.
    summary = {
        "spec_vs_fgw_pearson":  {},
        "spec_vs_fgw_spearman": {},
    }
    for ai, alpha in enumerate([0.0, 0.5, 1.0]):
        for qi, Q in enumerate(QUANTILES):
            fgw_vals = fgw_S[..., ai, qi][mask]
            spec_vals = S_spec[..., qi][mask]
            ok = ~(np.isnan(fgw_vals) | np.isnan(spec_vals))
            r_p, _ = pearsonr(fgw_vals[ok], spec_vals[ok])
            r_s, _ = spearmanr(fgw_vals[ok], spec_vals[ok])
            key = f"alpha={alpha}_Q={Q}"
            summary["spec_vs_fgw_pearson"][key]  = float(r_p)
            summary["spec_vs_fgw_spearman"][key] = float(r_s)
    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'comparison_summary.json'}")
    print(f"\nTotal wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
