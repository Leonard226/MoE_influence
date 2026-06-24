"""Inspect the C_conn cost-matrix distribution on one or all DAGs.

Reports for each (model, task, Q):
  - raw Phi = ((I - W_sparse)^-1)_uv distribution: percentiles + histogram of
    log10(Phi) over upper-triangular forward pairs.
  - C_conn = -log(max(Phi, eps)) / -log(eps) distribution: percentiles +
    fraction near 1 (saturated / unreachable), fraction near 0 (strong
    coupling), histogram.
  - mean / median / fraction-of-zeros / fraction-of-unities, to flag whether
    the transform is overly compressing or saturating.

Modes:
  --model <name>          : run on one model (verbose per-Q output)
  --all-models            : run on all 8 archs at one or all Q values and print
                            a compact cross-model comparison table.

Extras:
  --eps <float>           : floor for -log transform (default 1e-12). Lower eps
                            stretches the "very weak path" tail.
  --heatmap               : save C_conn heatmap PNG to --out-dir. For graphs
                            with V > HEATMAP_VERTEX_CAP, aggregate to per-layer
                            (L x L) means so the image is manageable.
  --out-dir <path>        : where to save heatmaps (default: result_path/
                            circuits/conn_inspection/).

Usage:
    python experiments/inspect_conn_costs.py --all-models --q 0.9
    python experiments/inspect_conn_costs.py --model qwen3-235b-a22b --heatmap
    python experiments/inspect_conn_costs.py --all-models --eps 1e-25
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_alpha_beta_sweep import (          # noqa: E402
    MODELS, QUANTILES, EDGE_TENSOR, _edge_quantile_threshold,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"
DEFAULT_OUT_DIR = CIRCUITS_DIR / "conn_inspection"

HEATMAP_VERTEX_CAP = 2000          # plot full matrix below this; else aggregate


# -------------------- per-graph computation --------------------------------
def _build_phi(W_fwd: torch.Tensor, threshold: float) -> np.ndarray:
    """Build symmetric Phi = (I - W_sparse)^-1 via triangular solve, then
    direction-mirror. W is upper-triangular (forward DAG)."""
    import scipy.linalg
    L, N = W_fwd.shape[0], W_fwd.shape[1]
    V = L * N
    W_abs = W_fwd.abs().cpu().numpy().reshape(V, V).astype(np.float64)
    W_sparse = np.where(W_abs > threshold, W_abs, 0.0)
    A = np.eye(V, dtype=np.float64) - W_sparse
    Phi = scipy.linalg.solve_triangular(A, np.eye(V, dtype=np.float64), lower=False)
    Phi = np.maximum(Phi, Phi.T)
    return Phi, W_sparse


def _phi_to_C(Phi: np.ndarray, eps: float) -> np.ndarray:
    log_floor = -np.log(eps)
    C = -np.log(np.clip(Phi, eps, None)) / log_floor
    C = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(C, 0.0)
    return C


def _compute_one(model: str, task: str, Q: float, eps: float
                 ) -> tuple[dict, np.ndarray | None]:
    """Compute summary stats for (model, task, Q). Returns (stats_dict, C_matrix
    or None if the build failed). The C_matrix is returned only when needed
    downstream for heatmap plotting; otherwise discarded by caller."""
    dag_path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        return {"model": model, "task": task, "Q": Q, "missing": True}, None
    dag = torch.load(dag_path, weights_only=False, map_location="cpu")
    W = dag[EDGE_TENSOR].float()
    L, N = W.shape[0], W.shape[1]
    V = L * N
    threshold = _edge_quantile_threshold(W, Q)
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_fwd = W * fwd.float()

    t0 = time.time()
    Phi, W_sparse = _build_phi(W_fwd, threshold)
    solve_time = time.time() - t0
    C = _phi_to_C(Phi, eps)

    iu, ju = np.triu_indices(V, k=1)
    phi_pairs = Phi[iu, ju]
    c_pairs = C[iu, ju]
    nnz = int((W_sparse > 0).sum())
    n_total = len(phi_pairs)
    n_unreach = int((phi_pairs == 0).sum())
    n_clip = int((phi_pairs <= eps).sum())
    n_sat = int((c_pairs >= 1.0 - 1e-12).sum())

    stats = {
        "model": model, "task": task, "Q": Q, "missing": False,
        "L": L, "N": N, "V": V,
        "threshold": float(threshold),
        "W_nnz": nnz,
        "W_max": float(W_sparse.max() if nnz > 0 else 0.0),
        "W_mean_nz": float(W_sparse[W_sparse > 0].mean() if nnz > 0 else 0.0),
        "solve_sec": float(solve_time),
        "phi_min": float(phi_pairs.min()),
        "phi_max": float(phi_pairs.max()),
        "phi_median": float(np.median(phi_pairs)),
        "phi_p99": float(np.percentile(phi_pairs, 99)),
        "phi_log10_p50": float(np.log10(np.clip(np.median(phi_pairs[phi_pairs > 0])
                                                if (phi_pairs > 0).any() else eps,
                                                eps, None))),
        "c_mean": float(c_pairs.mean()),
        "c_std": float(c_pairs.std()),
        "c_median": float(np.median(c_pairs)),
        "frac_unreach": n_unreach / n_total,
        "frac_clipped": n_clip / n_total,
        "frac_saturated": n_sat / n_total,
    }
    del dag, W, W_fwd, Phi, W_sparse
    return stats, C


# -------------------- formatting -------------------------------------------
def _print_compact_table(rows: list[dict]) -> None:
    """One row per (model, Q). Sorted by V ascending."""
    rows = [r for r in rows if not r.get("missing")]
    rows.sort(key=lambda r: (r["V"], r["Q"]))
    if not rows:
        print("(no rows)")
        return
    print("\n" + "=" * 124)
    print(f"  {'model':<18s}  {'V':>6s}  {'Q':>6s}  {'thr':>8s}  "
          f"{'nnz':>10s}  {'C mean':>7s}  {'C std':>7s}  {'C med':>7s}  "
          f"{'%unreach':>9s}  {'%clipped':>9s}  {'%C=1':>7s}  {'sec':>5s}")
    print("  " + "-" * 122)
    for r in rows:
        print(f"  {r['model']:<18s}  {r['V']:>6d}  {r['Q']:>6.3g}  "
              f"{r['threshold']:>8.2e}  {r['W_nnz']:>10,}  "
              f"{r['c_mean']:>7.4f}  {r['c_std']:>7.4f}  {r['c_median']:>7.4f}  "
              f"{100*r['frac_unreach']:>8.2f}%  "
              f"{100*r['frac_clipped']:>8.2f}%  "
              f"{100*r['frac_saturated']:>6.2f}%  {r['solve_sec']:>5.1f}")
    print()


def _print_verbose_one(stats: dict, phi_pairs: np.ndarray, c_pairs: np.ndarray,
                       eps: float) -> None:
    print(f"\n{'=' * 78}")
    print(f"model = {stats['model']}   task = {stats['task']}   Q = {stats['Q']}")
    print(f"  L = {stats['L']}   N = {stats['N']}   V = {stats['V']}   "
          f"threshold = {stats['threshold']:.6g}")
    print(f"{'=' * 78}")
    print(f"  W_sparse: nnz = {stats['W_nnz']:,}  max = {stats['W_max']:.4f}  "
          f"mean (nz) = {stats['W_mean_nz']:.4f}")
    print(f"  solve took {stats['solve_sec']:.1f}s")

    print(f"\n--- Raw Phi distribution (upper-triangular, off-diagonal) ---")
    pos = phi_pairs[phi_pairs > 0]
    if len(pos) > 0:
        logx = np.log10(pos)
        pcts = np.percentile(logx, [1, 5, 25, 50, 75, 95, 99])
        print(f"  log10(Phi) percentiles [1,5,25,50,75,95,99]:")
        print(f"    " + "  ".join(f"{p:+7.2f}" for p in pcts))
        bins = np.linspace(np.floor(logx.min()), np.ceil(logx.max()), 11)
        hist, _ = np.histogram(logx, bins=bins)
        print(f"  log10(Phi) histogram:")
        for b0, b1, c in zip(bins[:-1], bins[1:], hist):
            bar = "#" * int(40 * c / max(hist.max(), 1))
            print(f"    [{b0:+6.2f}, {b1:+6.2f}]  {c:>9d}  {bar}")
    print(f"  fraction Phi == 0       (unreachable)     : "
          f"{100*stats['frac_unreach']:5.2f}%")
    print(f"  fraction Phi <= {eps:.0e} (clipped by eps)  : "
          f"{100*stats['frac_clipped']:5.2f}%")

    print(f"\n--- C_conn distribution ---")
    pcts = np.percentile(c_pairs, [1, 5, 25, 50, 75, 95, 99])
    print(f"  percentiles [1,5,25,50,75,95,99]: "
          + "  ".join(f"{p:6.4f}" for p in pcts))
    print(f"  mean = {stats['c_mean']:.4f}  std = {stats['c_std']:.4f}  "
          f"median = {stats['c_median']:.4f}")
    print(f"  fraction at C == 1 (saturated): "
          f"{100*stats['frac_saturated']:5.2f}%")


# -------------------- heatmap ----------------------------------------------
def _save_heatmap(C: np.ndarray, model: str, task: str, Q: float,
                  L: int, N: int, eps: float, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    V = C.shape[0]
    if V <= HEATMAP_VERTEX_CAP:
        img = C
        kind = f"full {V}x{V}"
    else:
        # Aggregate per-(layer_i, layer_j): mean over the N x N block.
        C_layer = C.reshape(L, N, L, N).mean(axis=(1, 3))
        img = C_layer
        kind = f"layer-aggregated {L}x{L} (from {V}x{V})"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="C_conn")
    ax.set_title(f"C_conn  ({model}/{task})  Q={Q}  eps={eps:.0e}\n{kind}")
    ax.set_xlabel("vertex j (receiver)" if V <= HEATMAP_VERTEX_CAP
                  else "layer j (receiver)")
    ax.set_ylabel("vertex i (sender)" if V <= HEATMAP_VERTEX_CAP
                  else "layer i (sender)")
    out_path = out_dir / f"C_conn_{model}_{task}_Q{Q:g}_eps{eps:.0e}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# -------------------- entry point ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", help="e.g. mixtral-8x7b, qwen3-235b-a22b")
    grp.add_argument("--all-models", action="store_true",
                     help=f"Loop over all 8 archs: {MODELS}")
    parser.add_argument("--task", default="c4")
    parser.add_argument("--q", type=float, default=None,
                        help="Single Q to inspect; default = all three "
                             f"({QUANTILES})")
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="Floor for -log transform (default 1e-12). Lower "
                             "eps stretches the very-weak-path tail.")
    parser.add_argument("--heatmap", action="store_true",
                        help="Save C_conn heatmap PNG per (model, Q).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory for heatmap PNGs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    qs = [args.q] if args.q is not None else list(QUANTILES)
    models = MODELS if args.all_models else [args.model]

    rows: list[dict] = []
    for model in models:
        for Q in qs:
            stats, C = _compute_one(model, args.task, Q, args.eps)
            rows.append(stats)
            if stats.get("missing"):
                print(f"  [SKIP missing] {model}/{args.task} Q={Q}")
                continue
            if not args.all_models:
                # Verbose mode (single model): show histogram + distributions.
                dag_path = CIRCUITS_DIR / f"dag_{model}_{args.task}.pt"
                # Recompute Phi from C to print log10 histogram cleanly.
                eps = args.eps
                log_floor = -np.log(eps)
                Phi_recovered = np.exp(-C * log_floor)
                iu, ju = np.triu_indices(C.shape[0], k=1)
                _print_verbose_one(stats, Phi_recovered[iu, ju],
                                   C[iu, ju], eps)
            if args.heatmap and C is not None:
                p = _save_heatmap(C, model, args.task, Q,
                                  stats["L"], stats["N"], args.eps, out_dir)
                print(f"  heatmap -> {p}")
            del C

    if args.all_models:
        print(f"\n=== Cross-model summary  (task={args.task}, eps={args.eps:.0e}) ===")
        _print_compact_table(rows)


if __name__ == "__main__":
    main()
