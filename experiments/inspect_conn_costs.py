"""Inspect the C_conn cost-matrix distribution on one or all DAGs.

The metric pipeline (matching what run_alpha_beta_sweep.py does):
  1. Q-sparsify W on forward edges (|W| > threshold).
  2. Compute Phi = (I - W_sparse)^-1 via upper-triangular solve.
  3. C = -log(max(Phi, eps)) / -log(eps),  clipped to [0, 1].
  4. ISOLATION FILTER: drop vertices with no surviving in- or out-edge in
     the sparsified graph. FGW only ever sees the surviving sub-matrix.

For each (model, task, Q) we report C statistics computed on the FILTERED
sub-matrix (matching what FGW sees), and the heatmap shows the full V x V
matrix with isolated rows/cols masked in grey so you can see which experts
the filter dropped.

Modes:
  --model <name>          : run on one model (verbose per-Q output)
  --all-models            : run on all 8 archs at one or all Q values and
                            print a compact cross-model comparison table.

Extras:
  --eps <float>           : floor for -log transform (default 1e-12).
  --heatmap               : save C_conn heatmap PDF per (model, Q). Full
                            V x V always; lower triangle and isolated rows/
                            cols are masked grey.
  --out-dir <path>        : where to save heatmaps (default: result_path/
                            circuits/conn_inspection/).
  --dpi <int>             : raster dpi for the heatmap PDF (default 200; for
                            full Qwen3-235B-A22B resolution try 600+).

Usage:
    python experiments/inspect_conn_costs.py --all-models
    python experiments/inspect_conn_costs.py --all-models --heatmap
    python experiments/inspect_conn_costs.py --model qwen3-235b-a22b \\
        --heatmap --dpi 600
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


# -------------------- per-graph computation --------------------------------
def _compute_one(model: str, task: str, Q: float, eps: float
                 ) -> tuple[dict, np.ndarray | None, np.ndarray | None,
                            np.ndarray | None, np.ndarray | None]:
    """Returns (stats, C_full, keep_mask, phi_pairs_eff, c_pairs_eff) where
    phi/c-pairs are taken on the FILTERED V_eff x V_eff sub-matrix, matching
    what FGW receives. C_full and keep_mask are returned for heatmap plotting."""
    import scipy.linalg

    dag_path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        return ({"model": model, "task": task, "Q": Q, "missing": True},
                None, None, None, None)
    dag = torch.load(dag_path, weights_only=False, map_location="cpu")
    W = dag[EDGE_TENSOR].float()
    L, N = W.shape[0], W.shape[1]
    V = L * N
    threshold = _edge_quantile_threshold(W, Q)
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_fwd = W * fwd.float()

    W_abs = W_fwd.abs().cpu().numpy().reshape(V, V).astype(np.float64)
    W_sparse = np.where(W_abs > threshold, W_abs, 0.0)
    nnz = int((W_sparse > 0).sum())

    # Isolation filter (matches run_alpha_beta_sweep.py): drop vertices with
    # no surviving forward in- or out-edge above threshold.
    survive = W_sparse > 0
    out_deg = survive.sum(axis=1)
    in_deg = survive.sum(axis=0)
    keep_mask = (out_deg > 0) | (in_deg > 0)
    V_eff = int(keep_mask.sum())

    t0 = time.time()
    A = np.eye(V, dtype=np.float64) - W_sparse
    Phi = scipy.linalg.solve_triangular(A, np.eye(V, dtype=np.float64), lower=False)
    solve_time = time.time() - t0
    Phi = np.maximum(Phi, Phi.T)

    log_floor = -np.log(eps)
    C = -np.log(np.clip(Phi, eps, None)) / log_floor
    C = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(C, 0.0)

    # Stats on the FILTERED sub-matrix (what FGW sees).
    keep_idx = np.where(keep_mask)[0]
    if V_eff >= 2:
        C_eff = C[np.ix_(keep_idx, keep_idx)]
        Phi_eff = Phi[np.ix_(keep_idx, keep_idx)]
        iu, ju = np.triu_indices(V_eff, k=1)
        phi_pairs = Phi_eff[iu, ju]
        c_pairs = C_eff[iu, ju]
        n_total = len(phi_pairs)
        n_unreach = int((phi_pairs == 0).sum())
        n_clip = int((phi_pairs <= eps).sum())
        n_sat = int((c_pairs >= 1.0 - 1e-12).sum())
        stats_block = {
            "phi_min": float(phi_pairs.min()),
            "phi_max": float(phi_pairs.max()),
            "phi_median": float(np.median(phi_pairs)),
            "c_mean": float(c_pairs.mean()),
            "c_std": float(c_pairs.std()),
            "c_median": float(np.median(c_pairs)),
            "frac_unreach": n_unreach / max(n_total, 1),
            "frac_clipped": n_clip / max(n_total, 1),
            "frac_saturated": n_sat / max(n_total, 1),
        }
    else:
        phi_pairs = np.array([])
        c_pairs = np.array([])
        stats_block = {
            "phi_min": float("nan"), "phi_max": float("nan"),
            "phi_median": float("nan"),
            "c_mean": float("nan"), "c_std": float("nan"),
            "c_median": float("nan"),
            "frac_unreach": float("nan"),
            "frac_clipped": float("nan"),
            "frac_saturated": float("nan"),
        }

    stats = {
        "model": model, "task": task, "Q": Q, "missing": False,
        "L": L, "N": N, "V": V, "V_eff": V_eff,
        "threshold": float(threshold),
        "W_nnz": nnz,
        "W_max": float(W_sparse.max() if nnz > 0 else 0.0),
        "W_mean_nz": float(W_sparse[W_sparse > 0].mean() if nnz > 0 else 0.0),
        "solve_sec": float(solve_time),
        **stats_block,
    }
    del dag, W, W_fwd, W_sparse, Phi
    return stats, C, keep_mask, phi_pairs, c_pairs


# -------------------- formatting -------------------------------------------
def _print_compact_table(rows: list[dict]) -> None:
    rows = [r for r in rows if not r.get("missing")]
    rows.sort(key=lambda r: (r["V"], r["Q"]))
    if not rows:
        print("(no rows)")
        return
    print("\n" + "=" * 134)
    print(f"  {'model':<18s}  {'V':>6s}  {'V_eff':>6s}  {'%kept':>6s}  "
          f"{'Q':>6s}  {'thr':>8s}  {'nnz':>10s}  "
          f"{'C mean':>7s}  {'C std':>7s}  {'C med':>7s}  "
          f"{'%C=1':>7s}  {'sec':>5s}")
    print("  " + "-" * 132)
    for r in rows:
        kept_pct = 100 * r["V_eff"] / max(r["V"], 1)
        sat = (100 * r["frac_saturated"]) if not np.isnan(r["frac_saturated"]) else float("nan")
        print(f"  {r['model']:<18s}  {r['V']:>6d}  {r['V_eff']:>6d}  "
              f"{kept_pct:>5.1f}%  {r['Q']:>6.3g}  {r['threshold']:>8.2e}  "
              f"{r['W_nnz']:>10,}  "
              f"{r['c_mean']:>7.4f}  {r['c_std']:>7.4f}  {r['c_median']:>7.4f}  "
              f"{sat:>6.2f}%  {r['solve_sec']:>5.1f}")
    print()


def _print_verbose_one(stats: dict, phi_pairs: np.ndarray,
                       c_pairs: np.ndarray, eps: float) -> None:
    print(f"\n{'=' * 78}")
    print(f"model = {stats['model']}   task = {stats['task']}   Q = {stats['Q']}")
    print(f"  L = {stats['L']}   N = {stats['N']}   V = {stats['V']}   "
          f"V_eff (after isolation filter) = {stats['V_eff']}   "
          f"({100*stats['V_eff']/max(stats['V'],1):.1f}% kept)")
    print(f"  threshold = {stats['threshold']:.6g}")
    print(f"{'=' * 78}")
    print(f"  W_sparse: nnz = {stats['W_nnz']:,}  max = {stats['W_max']:.4f}  "
          f"mean (nz) = {stats['W_mean_nz']:.4f}")
    print(f"  solve took {stats['solve_sec']:.1f}s")

    if len(phi_pairs) == 0:
        print("  V_eff < 2, no off-diagonal pairs to summarise.")
        return

    print(f"\n--- Raw Phi distribution (V_eff x V_eff upper-triangular) ---")
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

    print(f"\n--- C_conn distribution (V_eff x V_eff) ---")
    pcts = np.percentile(c_pairs, [1, 5, 25, 50, 75, 95, 99])
    print(f"  percentiles [1,5,25,50,75,95,99]: "
          + "  ".join(f"{p:6.4f}" for p in pcts))
    print(f"  mean = {stats['c_mean']:.4f}  std = {stats['c_std']:.4f}  "
          f"median = {stats['c_median']:.4f}")
    print(f"  fraction at C == 1 (saturated): "
          f"{100*stats['frac_saturated']:5.2f}%")


# -------------------- heatmap (full V x V, upper triangular, isolated masked)
def _save_heatmap(C: np.ndarray, keep_mask: np.ndarray, V_eff: int,
                  model: str, task: str, Q: float,
                  out_dir: Path, dpi: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    V = C.shape[0]

    # Build display matrix: NaN for lower triangle AND isolated rows/cols.
    display = C.copy()
    tri_idx = np.tril_indices(V, k=-1)
    display[tri_idx] = np.nan
    iso = ~keep_mask
    display[iso, :] = np.nan
    display[:, iso] = np.nan

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8d8d8")    # light grey for NaN (lower tri + isolated)
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, label="C_conn")
    ax.set_title(f"C_conn  ({model}/{task})  Q={Q}  "
                 f"V={V}, V_eff={V_eff} ({100*V_eff/V:.1f}% kept)")
    ax.set_xlabel("vertex j (receiver)")
    ax.set_ylabel("vertex i (sender)")
    out_path = out_dir / f"C_conn_{model}_{task}_Q{Q:g}.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
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
                        help="Floor for -log transform (default 1e-12).")
    parser.add_argument("--heatmap", action="store_true",
                        help="Save C_conn heatmap PDF per (model, Q).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory for heatmap PDFs.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Raster dpi for the heatmap PDF (default 200).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    qs = [args.q] if args.q is not None else list(QUANTILES)
    models = MODELS if args.all_models else [args.model]

    rows: list[dict] = []
    for model in models:
        for Q in qs:
            stats, C, keep_mask, phi_pairs, c_pairs = _compute_one(
                model, args.task, Q, args.eps)
            rows.append(stats)
            if stats.get("missing"):
                print(f"  [SKIP missing] {model}/{args.task} Q={Q}")
                continue
            if not args.all_models:
                _print_verbose_one(stats, phi_pairs, c_pairs, args.eps)
            if args.heatmap and C is not None:
                p = _save_heatmap(C, keep_mask, stats["V_eff"],
                                  model, args.task, Q, out_dir, args.dpi)
                print(f"  heatmap -> {p}")
            del C, keep_mask, phi_pairs, c_pairs

    if args.all_models:
        print(f"\n=== Cross-model summary  (task={args.task}, eps={args.eps:.0e}) ===")
        _print_compact_table(rows)


if __name__ == "__main__":
    main()
