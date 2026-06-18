"""Spectral baseline for the cross-model FGW comparison.

For each (model, task) routing DAG at each Q in {0.9, 0.99, 0.999}:
  1. Apply Q-sparsification on |W_softmax| (matches FGW headline sweep).
  2. Drop vertices that are isolated in the sparsified forward DAG.
  3. Build the magnetic Laplacian L^(q) on the surviving vertices.
  4. Compute the NetLSD heat-trace signature h(t) on a log-spaced t-grid.

For each ordered pair (G_1, G_2) at each Q:
  d_struct = ||h_{G_1, Q} - h_{G_2, Q}||_2
  S_struct(G_1, G_2; Q) = exp(-d_struct^2)

The output tensor S has shape (N_TUPLES, N_TUPLES, len(QUANTILES)) and is
symmetric with diagonal 1 by construction.

References:
  - Fanuel, Alaiz, Suykens (2018, arXiv:1812.02160): magnetic Laplacian for
    weighted directed graphs.
  - Tsitsulin et al., KDD 2018 (arXiv:1805.10712): NetLSD heat-trace signature.

Output:
  ${result_path}/circuits/spectral_baseline/S_struct.npz
    S         : (N_TUPLES, N_TUPLES, n_q) float64
    signatures: (N_TUPLES, n_q, NETLSD_N_T) float64
    v_effs    : (N_TUPLES, n_q) int -- effective vertex count per graph after
                Q-sparsification and isolated-vertex filtering
    quantiles, t_grid, q_phase, edge_tensor : metadata
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_alpha_beta_sweep import (   # noqa: E402
    MODELS, TASKS, TUPLES, N_TUPLES, QUANTILES,
)


# --- Configuration --------------------------------------------------------
Q_PHASE = 0.25                           # magnetic-Laplacian phase parameter
NETLSD_N_T = 250                         # number of t values in heat-trace signature
NETLSD_T_MIN = 1e-2                      # smallest t (probes the large eigenvalues)
NETLSD_T_MAX = 1e2                       # largest t (probes the small eigenvalues)
EDGE_TENSOR = "W_softmax"                # matches FGW headline sweep
OUT_SUBDIR = "spectral_baseline"

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"

T_GRID = np.logspace(np.log10(NETLSD_T_MIN), np.log10(NETLSD_T_MAX), NETLSD_N_T)


# --- Sparsification + Magnetic Laplacian ---------------------------------
def _edge_quantile_threshold(W_full: torch.Tensor, Q: float) -> float:
    """Q-quantile of |W| over forward edges only (sender_layer < receiver_layer).

    Matches the threshold definition in run_alpha_beta_sweep.py."""
    L = W_full.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W_full)
    edge_vals = torch.abs(W_full[fwd]).cpu().numpy().astype(np.float64)
    nz = edge_vals[edge_vals > 0]
    return float(np.quantile(nz, Q)) if len(nz) else 0.0


def _build_magnetic_laplacian(W_full: torch.Tensor, threshold: float,
                              q_phase: float = Q_PHASE) -> tuple[np.ndarray, int]:
    """Construct the magnetic Laplacian L^(q) of the Q-sparsified forward DAG.

    Definition (Fanuel et al. 2018):
        A_sym[u, v] = (W[u, v] + W[v, u]) / 2          (symmetric magnitude)
        Theta[u, v] = 2 pi q (W[u, v] - W[v, u])       (antisymmetric phase)
        H[u, v]     = A_sym[u, v] * exp(i * Theta[u, v])  (Hermitian adjacency)
        D_s[u, u]   = sum_v A_sym[u, v]                (symmetric degree)
        L^(q)       = D_s - H                          (Hermitian Laplacian)

    For a forward-only DAG (W[u, v] >= 0 and W[v, u] = 0 whenever u != v):
        A_sym[u, v] = W[u, v] / 2,  Theta[u, v] = 2 pi q W[u, v].

    Returns the dense complex (V_eff, V_eff) matrix and the effective vertex
    count after dropping vertices isolated in the sparsified forward DAG.
    """
    L_layers, N_experts = W_full.shape[0], W_full.shape[1]
    V = L_layers * N_experts

    W = W_full.float().reshape(V, V).cpu().numpy().astype(np.float64)

    # Forward-edge mask (sender layer < receiver layer).
    layer_of = np.arange(V) // N_experts
    fwd_mask = layer_of[:, None] < layer_of[None, :]
    W_fwd = W * fwd_mask

    # Magnitude threshold.
    W_sparse = np.where(np.abs(W_fwd) > threshold, W_fwd, 0.0)

    # Drop vertices with no surviving forward in- or out-edge.
    has_out = (np.abs(W_sparse) > 0).any(axis=1)
    has_in = (np.abs(W_sparse) > 0).any(axis=0)
    keep = has_out | has_in
    keep_idx = np.where(keep)[0]
    V_eff = int(len(keep_idx))
    if V_eff == 0:
        # Degenerate: empty surviving graph. Return a 1x1 zero matrix.
        return np.zeros((1, 1), dtype=np.complex128), 0

    W_kept = W_sparse[np.ix_(keep_idx, keep_idx)]

    A_sym = (W_kept + W_kept.T) / 2.0
    Theta = 2.0 * np.pi * q_phase * (W_kept - W_kept.T)
    H = A_sym * np.exp(1j * Theta)
    D_s_diag = A_sym.sum(axis=1)
    L_mag = np.diag(D_s_diag).astype(np.complex128) - H

    return L_mag, V_eff


def _netlsd_signature(L_mag: np.ndarray) -> np.ndarray:
    """NetLSD heat-trace signature h(t) = (1/V) sum_k exp(-t lambda_k).

    Eigenvalues of the magnetic Laplacian are real (Hermitian), so we use
    eigvalsh which is faster and more stable than the general eig solver."""
    V = L_mag.shape[0]
    if V <= 1:
        return np.zeros(NETLSD_N_T)
    eigvals = scipy.linalg.eigvalsh(L_mag)
    # Outer product: h_j = (1/V) sum_k exp(-t_j * lambda_k).
    h = np.exp(-T_GRID[:, None] * eigvals[None, :]).sum(axis=1) / V
    return h


# --- Main ----------------------------------------------------------------
def main() -> None:
    out_dir = CIRCUITS_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "S_struct.npz"

    n_q = len(QUANTILES)
    signatures = np.zeros((N_TUPLES, n_q, NETLSD_N_T), dtype=np.float64)
    v_effs = np.zeros((N_TUPLES, n_q), dtype=np.int64)

    print(f"Computing NetLSD signatures on magnetic Laplacian "
          f"(q_phase={Q_PHASE}, n_t={NETLSD_N_T})")
    print(f"  {N_TUPLES} graphs x {n_q} Q values = {N_TUPLES * n_q} signatures total\n")
    print(f"{'idx':>3s}  {'model':<18s}  {'task':<11s}  {'Q':>6s}  "
          f"{'V_eff':>6s}  {'sec':>6s}")
    print("-" * 65)

    t_start = time.time()
    for ti, (model, task) in enumerate(TUPLES):
        dag_path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
        if not dag_path.exists():
            raise FileNotFoundError(f"missing DAG: {dag_path}")
        dag = torch.load(dag_path, weights_only=False, map_location="cpu")
        W = dag[EDGE_TENSOR].float()

        for qi, Q in enumerate(QUANTILES):
            t0 = time.time()
            threshold = _edge_quantile_threshold(W, Q)
            L_mag, V_eff = _build_magnetic_laplacian(W, threshold)
            h = _netlsd_signature(L_mag)
            signatures[ti, qi] = h
            v_effs[ti, qi] = V_eff
            print(f"{ti:>3d}  {model:<18s}  {task:<11s}  "
                  f"{Q:>6.3g}  {V_eff:>6d}  {time.time() - t0:>6.1f}",
                  flush=True)
        del dag, W

    print(f"\nTotal signature time: {(time.time() - t_start) / 60:.1f} min\n")

    # Pairwise structural similarity per Q.
    print("Computing pairwise distances ...")
    S = np.full((N_TUPLES, N_TUPLES, n_q), np.nan, dtype=np.float64)
    for qi in range(n_q):
        h_q = signatures[:, qi, :]                                   # (N, T)
        # ||h_i - h_j||^2 = ||h_i||^2 + ||h_j||^2 - 2 <h_i, h_j>
        sq_norms = (h_q ** 2).sum(axis=1)
        d_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * h_q @ h_q.T
        d_sq = np.clip(d_sq, 0.0, None)
        S[:, :, qi] = np.exp(-d_sq)

    np.savez(out_path,
             S=S,
             signatures=signatures,
             v_effs=v_effs,
             quantiles=np.array(QUANTILES),
             t_grid=T_GRID,
             q_phase=Q_PHASE,
             edge_tensor=EDGE_TENSOR,
             models=np.array(MODELS, dtype=object),
             tasks=np.array(TASKS, dtype=object))
    print(f"\nSaved: {out_path}")
    print(f"  S shape: {S.shape}")
    print(f"  diagonal (Q=0.9):   "
          f"min={S.diagonal(axis1=0, axis2=1)[0].min():.6f}  "
          f"max={S.diagonal(axis1=0, axis2=1)[0].max():.6f}  (should be 1.0)")
    print(f"  off-diagonal range (Q=0.9):  "
          f"[{S[~np.eye(N_TUPLES, dtype=bool), 0].min():.4f}, "
          f"{S[~np.eye(N_TUPLES, dtype=bool), 0].max():.4f}]")


if __name__ == "__main__":
    main()
