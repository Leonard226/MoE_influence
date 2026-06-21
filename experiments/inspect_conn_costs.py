"""Inspect the C_conn cost-matrix distribution on a few DAGs.

Reports for each (model, task, Q):
  - raw Phi = ((I - W_sparse)^-1)_uv distribution: percentiles + histogram of
    log10(Phi) over upper-triangular forward pairs.
  - C_conn = -log(max(Phi, eps)) / -log(eps) distribution: percentiles +
    fraction near 1 (saturated / unreachable), fraction near 0 (strong
    coupling), histogram.
  - mean / median / fraction-of-zeros / fraction-of-unities, to flag whether
    the transform we picked is overly compressing or saturating.

Usage:
    python experiments/inspect_conn_costs.py \\
        --model mixtral-8x7b --task c4
    python experiments/inspect_conn_costs.py \\
        --model qwen3-235b-a22b --task c4 --q 0.999    # one Q only

Default: all three Qs on the requested (model, task).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.fgw import _conn_costs                 # noqa: E402
from experiments.run_alpha_beta_sweep import (          # noqa: E402
    QUANTILES, EDGE_TENSOR, _edge_quantile_threshold,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"


def _summarise(name: str, x: np.ndarray, log_scale: bool = False) -> None:
    """Print percentiles + histogram of a 1D array."""
    if log_scale:
        # Hide non-positives for log-scale display.
        pos = x[x > 0]
        print(f"  {name}  (n={len(x)}, positive={len(pos)})")
        if len(pos) == 0:
            print(f"    all zero / negative")
            return
        logx = np.log10(pos)
        pcts = np.percentile(logx, [1, 5, 25, 50, 75, 95, 99])
        print(f"    log10 percentiles  [1,5,25,50,75,95,99]:")
        print(f"      " + "  ".join(f"{p:+7.2f}" for p in pcts))
        # Coarse histogram of log10.
        bins = np.linspace(np.floor(logx.min()), np.ceil(logx.max()), 11)
        hist, _ = np.histogram(logx, bins=bins)
        print(f"    log10 histogram (bins, counts):")
        for b0, b1, c in zip(bins[:-1], bins[1:], hist):
            bar = "#" * int(40 * c / max(hist.max(), 1))
            print(f"      [{b0:+6.2f}, {b1:+6.2f}]  {c:>9d}  {bar}")
    else:
        print(f"  {name}  (n={len(x)})")
        pcts = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
        print(f"    percentiles  [1,5,25,50,75,95,99]:")
        print(f"      " + "  ".join(f"{p:6.4f}" for p in pcts))
        print(f"    mean={x.mean():.4f}  std={x.std():.4f}  "
              f"min={x.min():.6f}  max={x.max():.6f}")
        # Bucket counts.
        edges = np.array([0.0, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-3, 1.0 + 1e-12])
        for lo, hi in zip(edges[:-1], edges[1:]):
            c = int(((x >= lo) & (x < hi)).sum())
            frac = c / max(len(x), 1)
            bar = "#" * int(40 * frac)
            print(f"    [{lo:8.5f}, {hi:8.5f})   {c:>9d}  ({100*frac:5.2f}%)  {bar}")
        # Exact-1 and exact-0 counts.
        n_zero = int((x == 0).sum())
        n_one = int((x == 1).sum())
        print(f"    exact zero  : {n_zero:>9d}  ({100*n_zero/len(x):.2f}%)")
        print(f"    exact one   : {n_one:>9d}  ({100*n_one/len(x):.2f}%)")


def _run_one(model: str, task: str, Q: float) -> None:
    dag_path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        print(f"  [SKIP] missing DAG: {dag_path}")
        return
    dag = torch.load(dag_path, weights_only=False, map_location="cpu")
    W = dag[EDGE_TENSOR].float()
    L, N = W.shape[0], W.shape[1]
    V = L * N
    threshold = _edge_quantile_threshold(W, Q)

    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_fwd = W * fwd.float()

    print(f"\n{'=' * 78}")
    print(f"model = {model}   task = {task}   Q = {Q}")
    print(f"  L = {L}   N = {N}   V = {V}   threshold = {threshold:.6g}")
    print(f"{'=' * 78}")

    # Compute Phi directly (mirror what _conn_costs does internally), so we can
    # report both the raw and the transformed distributions.
    import scipy.linalg
    W_abs = W_fwd.abs().cpu().numpy().reshape(V, V).astype(np.float64)
    W_sparse = np.where(W_abs > threshold, W_abs, 0.0)
    print(f"  W_sparse: nnz = {int((W_sparse > 0).sum()):,}  "
          f"max = {W_sparse.max():.4f}  "
          f"mean (nz) = {W_sparse[W_sparse > 0].mean():.4f}", flush=True)

    print(f"  building Phi = (I - W_sparse)^-1 via triangular solve ...", flush=True)
    import time
    t0 = time.time()
    A = np.eye(V, dtype=np.float64) - W_sparse
    Phi = scipy.linalg.solve_triangular(A, np.eye(V, dtype=np.float64), lower=False)
    print(f"    solve took {time.time() - t0:.1f}s", flush=True)
    Phi = np.maximum(Phi, Phi.T)

    # Report on upper-triangular off-diagonal entries (the meaningful pairs).
    iu, ju = np.triu_indices(V, k=1)
    phi_pairs = Phi[iu, ju]

    print(f"\n--- Raw Phi distribution (upper-triangular, off-diagonal) ---")
    _summarise("Phi", phi_pairs, log_scale=True)
    print(f"\n  Phi raw stats:")
    print(f"    min     = {phi_pairs.min():.6e}")
    print(f"    max     = {phi_pairs.max():.6e}")
    print(f"    median  = {np.median(phi_pairs):.6e}")
    n_zero = int((phi_pairs == 0).sum())
    n_gt1 = int((phi_pairs > 1).sum())
    n_le_eps = int((phi_pairs <= 1e-12).sum())
    n_total = len(phi_pairs)
    print(f"    fraction Phi == 0       (unreachable)     : "
          f"{n_zero:>9d}  ({100*n_zero/n_total:5.2f}%)")
    print(f"    fraction Phi <= 1e-12   (clipped by eps)  : "
          f"{n_le_eps:>9d}  ({100*n_le_eps/n_total:5.2f}%)")
    print(f"    fraction Phi >  1       (clipped to C=0)  : "
          f"{n_gt1:>9d}  ({100*n_gt1/n_total:5.2f}%)")

    # Now compute the transformed C_conn (call the real function so we test
    # exactly what the sweep uses).
    print(f"\n--- C_conn distribution (after -log transform + clip + mirror) ---")
    # Re-derive symbolically so we don't lose the symmetric matrix we just built.
    eps = 1e-12
    log_floor = -np.log(eps)
    C = -np.log(np.clip(Phi, eps, None)) / log_floor
    C = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(C, 0.0)
    c_pairs = C[iu, ju]
    _summarise("C_conn", c_pairs, log_scale=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="e.g. mixtral-8x7b, qwen3-235b-a22b")
    parser.add_argument("--task", default="c4")
    parser.add_argument("--q", type=float, default=None,
                        help="Single Q to inspect; default = all three")
    args = parser.parse_args()

    qs = [args.q] if args.q is not None else QUANTILES
    for q in qs:
        _run_one(args.model, args.task, q)


if __name__ == "__main__":
    main()
