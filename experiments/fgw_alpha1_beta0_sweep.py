"""FGW at alpha=1, beta=0 over all 64x64 cross-pairs at 3 Q values.

Path-only structural channel of FGW (no depth in C, no features in the
Wasserstein term -- alpha=1 cancels the Wasserstein term entirely). This is
the apples-to-apples target for the spectral baseline:
  - both see only the Q-sparsified W_softmax forward path structure
  - both ignore depth, vertex features, and mass weighting beyond the
    structural-cost matrix

For each unordered cross-pair at each Q in {0.9, 0.99, 0.999}:
  d = FGW^2(triple_a, triple_b, alpha=1.0)
  S = exp(-d)

Output: <result_path>/circuits/alpha_beta_sweep/S_alpha1_beta0.npz
        shape (64, 64, 3) indexed (src, tgt, Q).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
ALPHA = 1.0
N_INIT = 3         # baseline-comparison run; slightly noisier per pair but fine
                   # for Pearson/Spearman vs spectral (was 5 in headline sweep)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def q_threshold(dag: dict, Q: float) -> float:
    """Per-graph Q-quantile of positive forward |W_softmax| edges.
    Uses np.quantile (torch errors above ~16M elements; Qwen3-235B has ~72M)."""
    if Q == 0.0:
        return 0.0
    W = dag["W_softmax"]
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    edge_vals = torch.abs(W[fwd]).cpu().numpy().astype(np.float64)
    nz = edge_vals[edge_vals > 0]
    return float(np.quantile(nz, Q)) if len(nz) else 0.0


def _isolated_keep_mask(dag: dict, theta: float) -> np.ndarray:
    """Drop vertices that have zero surviving forward in- or out-edges above
    threshold. Same rule as run_alpha_beta_sweep._isolated_keep_mask."""
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
    """Build triple at edge_threshold=theta, then drop isolated vertices."""
    triple = build_triple(dag, edge_threshold=theta)
    keep_mask = _isolated_keep_mask(dag, theta)
    return _subset_triple(triple, keep_mask), int(keep_mask.sum())


# ---------------------------------------------------------------------------
# Worker: per-process triple cache keyed by (tuple_idx, Q_idx).
# Each entry has the (triple, n_keep) ready for FGW.
# ---------------------------------------------------------------------------
_WORKER_DAGS: dict[tuple[str, str], dict] = {}
_WORKER_TRIPLES: dict[tuple[int, int], tuple] = {}


def _get_dag(model: str, task: str, result_path: str) -> dict:
    key = (model, task)
    dag = _WORKER_DAGS.get(key)
    if dag is None:
        path = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
        dag = torch.load(path, weights_only=False)
        _WORKER_DAGS[key] = dag
    return dag


def _get_triple(idx: int, q_idx: int, result_path: str):
    key = (idx, q_idx)
    triple = _WORKER_TRIPLES.get(key)
    if triple is None:
        model, task = TUPLES[idx]
        Q = QUANTILES[q_idx]
        dag = _get_dag(model, task, result_path)
        theta = q_threshold(dag, Q)
        triple, _ = _build_filtered_triple(dag, theta)
        _WORKER_TRIPLES[key] = triple
    return triple


def _worker_compute(args: tuple[int, int, int, str]) -> dict:
    """Compute one FGW pair: (src_idx, tgt_idx, q_idx, result_path) -> distance."""
    src_idx, tgt_idx, q_idx, result_path = args
    t_src = _get_triple(src_idx, q_idx, result_path)
    t_tgt = _get_triple(tgt_idx, q_idx, result_path)
    d, _ = fgw_distance(t_src, t_tgt, alpha=ALPHA, n_init=N_INIT, seed=0)
    return {"src": src_idx, "tgt": tgt_idx, "q": q_idx, "d": float(d)}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "circuits" / "alpha_beta_sweep" / "S_alpha1_beta0.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    print(f"Project root : {ROOT}")
    print(f"Result path  : {result_path}")
    print(f"Output       : {out_path}")
    print(f"Settings     : alpha={ALPHA}, n_init={N_INIT}")
    print(f"Q axis       : {QUANTILES}")
    print(f"Pair space   : {N_TUPLES} x {N_TUPLES} = {N_TUPLES * N_TUPLES} ordered "
          f"(unordered cross-pairs = {N_TUPLES * (N_TUPLES - 1) // 2})")
    print(f"Workers      : {n_workers}")

    # Build job list: unordered cross-pairs (src_idx < tgt_idx) x QUANTILES.
    jobs: list[tuple[int, int, int, str]] = []
    for src in range(N_TUPLES):
        for tgt in range(src + 1, N_TUPLES):
            for qi in range(len(QUANTILES)):
                jobs.append((src, tgt, qi, result_path))
    print(f"Total jobs   : {len(jobs)}\n")

    # S_full[src, tgt, q] = similarity. Symmetric; diagonal = 1.
    S = np.full((N_TUPLES, N_TUPLES, len(QUANTILES)), np.nan, dtype=np.float64)
    for i in range(N_TUPLES):
        S[i, i, :] = 1.0

    t0 = time.time()
    ctx = mp.get_context("spawn")
    completed = 0
    failed = 0
    last_print = 0.0
    last_save = 0.0
    save_interval = 60.0   # checkpoint to disk at most once a minute

    def _save_checkpoint(tag: str = "") -> None:
        np.savez(
            out_path,
            S=S,
            tuples=np.array([f"{m}/{t}" for m, t in TUPLES], dtype=object),
            models=np.array(MODELS, dtype=object),
            tasks=np.array(TASKS, dtype=object),
            quantiles=np.array(QUANTILES),
            alpha=ALPHA, n_init=N_INIT,
        )
        if tag:
            print(f"    [checkpoint {tag}] saved {out_path.name}  "
                  f"NaN={int(np.isnan(S).sum())}",
                  flush=True)

    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker_compute, j): j for j in jobs}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    src, tgt, qi, d = r["src"], r["tgt"], r["q"], r["d"]
                    s = float(np.exp(-d))
                    S[src, tgt, qi] = s
                    S[tgt, src, qi] = s
                except Exception as e:
                    # Per-future failure (worker OOM-killed, solver crash, etc.):
                    # leave the cell as NaN and continue.
                    failed += 1
                    job = futs[f]
                    src, tgt, qi, _ = job
                    src_m, src_t = TUPLES[src]
                    tgt_m, tgt_t = TUPLES[tgt]
                    if failed <= 20 or failed % 50 == 0:
                        print(f"  FAIL [{failed}]: src={src_m}/{src_t}  "
                              f"tgt={tgt_m}/{tgt_t}  Q={QUANTILES[qi]}  "
                              f"reason={type(e).__name__}: {str(e)[:120]}",
                              flush=True)
                completed += 1
                now = time.time()
                if now - last_print > 5.0 or completed in (1, len(jobs)):
                    last_print = now
                    print(f"  [{completed:>5d}/{len(jobs)}]  "
                          f"ok={completed - failed}  fail={failed}  "
                          f"(t={now - t0:.0f}s)", flush=True)
                if now - last_save > save_interval:
                    last_save = now
                    _save_checkpoint(tag=f"{completed}/{len(jobs)}")
    except Exception as e:
        print(f"\n!! Pool died: {type(e).__name__}: {e}")
        print(f"   Salvaging {completed - failed} completed cells.")

    print(f"\nDone in {time.time() - t0:.0f}s.")
    print(f"  Completed: {completed - failed}/{len(jobs)}")
    print(f"  Failed:    {failed}/{len(jobs)}")
    print(f"  NaN cells remaining: {int(np.isnan(S).sum())}")

    _save_checkpoint()
    print(f"Saved: {out_path}\n")

    # Quick comparison vs spectral baseline.
    spec_path = Path(result_path) / "spectral_baseline" / "S_spec.npz"
    if spec_path.exists():
        from scipy.stats import pearsonr, spearmanr
        print(f"Comparison vs spectral baseline ({spec_path.name}):")
        S_spec = np.load(spec_path)["S"]  # (64, 64, 3)
        mask = ~np.eye(N_TUPLES, dtype=bool)
        print(f"\n{'Q':>6s}  {'Pearson':>10s}  {'Spearman':>10s}")
        print("-" * 32)
        summary = {}
        for qi, Q in enumerate(QUANTILES):
            fgw_vals = S[..., qi][mask]
            spec_vals = S_spec[..., qi][mask]
            ok = ~(np.isnan(fgw_vals) | np.isnan(spec_vals))
            r_p, _ = pearsonr(fgw_vals[ok], spec_vals[ok])
            r_s, _ = spearmanr(fgw_vals[ok], spec_vals[ok])
            print(f"{Q:>6.3f}  {r_p:>10.4f}  {r_s:>10.4f}")
            summary[f"Q={Q}"] = {"pearson": float(r_p), "spearman": float(r_s)}
        with open(out_path.parent / "comparison_alpha1_beta0_vs_spec.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path.parent / 'comparison_alpha1_beta0_vs_spec.json'}")
    else:
        print(f"Skipping spectral comparison: {spec_path} not found.")


if __name__ == "__main__":
    main()
