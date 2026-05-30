"""Pairwise α × Q sweep across all (model, task) tuples at β = FIXED_BETA.

One worker = one (source_idx, target_chunk) work unit. Run as a SLURM array
via experiments/launch_alpha_beta_sweep.sh.

For each (source, target) pair, each quantile Q, and each α, computes
    S_α(G_source^Q, G_target^Q)         at fixed β = FIXED_BETA
and writes a slice file
    {output_dir}/sweep_src{SS}_chunk{CC}.npz
that the aggregator stitches into
    S[src, tgt, alpha, Q].

Graph: the routing DAG used throughout is P_flip = P_add + P_rem. The
per-graph quantile threshold is taken on |P_flip| over forward edges.

Threshold semantics (per main.tex §3.6): features (depth, out_norm, in_norm,
load, class_hist) and mass are computed on the DENSE graph -- they are the
expert's intrinsic profile and should not be artifacts of the threshold.
The threshold only modifies the STRUCTURAL cost C_path (the all-pairs
shortest-path distance on edges with |W| > θ_Q). Vertices that have no
surviving incident edge in the sparsified graph are dropped (they would
otherwise contribute only uniform-diameter distance rows = structural
noise) together with the special-token-dominated vertices.

β is fixed, so only ONE triple is built per (model, task, Q). The metric
is CPU-bound (POT + scipy); no GPU needed.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.fgw import build_triple, fgw_similarity

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Sweep configuration. To add models/tasks, append to these lists; aggregator
# uses the same order to index the output.
# ---------------------------------------------------------------------------
MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
TASKS = [
    "c4", "math", "code",
    "wikitext2", "gsm8k", "humaneval",
    "pile-arxiv", "pile-github",
]
TUPLES   = [(m, t) for m in MODELS for t in TASKS]   # 8 × 8 = 64
N_TUPLES = len(TUPLES)

ALPHAS    = [0.0, 0.5, 1.0]
QUANTILES = [0.9, 0.99, 0.999]    # per-edge |W| quantile; drop edges below it.
FIXED_BETA = 0.5
N_INIT = 5

CACHE_DIR = os.path.join(config["result_path"], "circuits", "classifications")
DAG_DIR   = os.path.join(config["result_path"], "circuits")


def _subset_triple(triple, keep_mask):
    """Restrict triple to vertices where keep_mask is True (renormalises mass)."""
    C, F, mass, meta = triple
    keep_idx = np.where(keep_mask)[0]
    sub_mass = mass[keep_idx]
    return (
        C[np.ix_(keep_idx, keep_idx)],
        F[keep_idx],
        sub_mass / sub_mass.sum(),
        {**meta, "n_verts": len(keep_idx), "keep_idx": keep_idx},
    )


def _edge_quantile_threshold(W_combined: torch.Tensor, Q: float) -> float:
    """Q-quantile of |W| over forward edges only (sender_layer < receiver_layer).
    Returns the absolute weight threshold to pass to the masking step."""
    L = W_combined.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W_combined)
    edge_vals = torch.abs(W_combined)[fwd].cpu().numpy().astype(np.float64)
    nz = edge_vals[edge_vals > 0]
    if len(nz) == 0:
        return 0.0
    # numpy.quantile (not torch) — torch errors out above ~16M elements,
    # qwen3-235b-a22b has ~72M forward edges.
    return float(np.quantile(nz, Q))


def load_classification(model, task):
    path = os.path.join(CACHE_DIR, f"classify_{model}_{task}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dag(model, task):
    path = os.path.join(DAG_DIR, f"dag_{model}_{task}.pt")
    return torch.load(path, map_location="cpu")


def build_triple_at_Q(model, task, classification, Q):
    """Build one triple at β = FIXED_BETA with:
      - F, mass        : computed on the DENSE routing DAG (intrinsic features).
      - C_path         : computed on the SPARSIFIED graph (edges with
                         |P_flip| > θ_Q only).
      - vertex set     : drop vertices isolated in the sparsified graph
                         (no surviving in- or out-edge).

    Note: this baseline does NOT apply the special-token (class_hist[special]
    > 0.5) mask. The legacy dense sweep applied it; we drop it here so the
    only discretionary vertex filter is Q-isolation. Re-enable later if a
    masked variant is wanted for comparison.

    Returns (triple, threshold)."""
    dag = load_dag(model, task)
    P_combined = (dag["P_add"] + dag["P_rem"]).float()
    L = P_combined.shape[0]

    # Forward-edge mask.
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(P_combined)

    # Per-graph absolute threshold from quantile Q.
    threshold = _edge_quantile_threshold(P_combined, Q)

    # build_triple keeps F and mass on the DENSE graph; edge_threshold only
    # filters edges fed into the C_path shortest-path computation.
    triple = build_triple(dag, classification,
                          beta=FIXED_BETA, edge_threshold=threshold)

    # Identify vertices isolated in the SPARSIFIED graph (no surviving in-
    # or out-edge above threshold). This is the ONLY vertex filter.
    survive = (torch.abs(P_combined) >= threshold) & fwd
    out_sparse = survive.sum(dim=(2, 3)).reshape(-1).cpu().numpy()
    in_sparse  = survive.sum(dim=(0, 1)).reshape(-1).cpu().numpy()
    keep_mask = (out_sparse > 0) | (in_sparse > 0)

    triple = _subset_triple(triple, keep_mask)
    triple[3]["quantile"] = Q
    triple[3]["edge_threshold"] = threshold

    del dag, P_combined, survive, fwd
    return triple, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-idx", type=int, required=True,
                        help=f"0..{N_TUPLES - 1} index into TUPLES")
    parser.add_argument("--target-chunk", type=int, default=0,
                        help="Target chunk index (0..num-chunks-1)")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks to split the 63 targets across")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-init", type=int, default=N_INIT,
                        help="Number of random initialisations for FGW")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        config["result_path"], "circuits", "alpha_beta_sweep"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    src_model, src_task = TUPLES[args.source_idx]

    all_other = [i for i in range(N_TUPLES) if i != args.source_idx]
    chunk_size = (len(all_other) + args.num_chunks - 1) // args.num_chunks
    chunk_start = args.target_chunk * chunk_size
    chunk_end = min(chunk_start + chunk_size, len(all_other))
    target_indices = all_other[chunk_start:chunk_end]
    n_tgts = len(target_indices)

    output_path = os.path.join(
        output_dir,
        f"sweep_src{args.source_idx:02d}_chunk{args.target_chunk:02d}.npz",
    )

    n_a, n_q = len(ALPHAS), len(QUANTILES)
    n_cells = n_a * n_q

    print(f"=== α × Q Sweep at β = {FIXED_BETA} ===")
    print(f"  source : {src_model}/{src_task} (idx {args.source_idx})")
    print(f"  targets: chunk {args.target_chunk}/{args.num_chunks - 1}  -> {n_tgts} tuples")
    print(f"  α      : {ALPHAS}")
    print(f"  Q      : {QUANTILES}")
    print(f"  cells  : {n_cells} (α × Q) per pair")
    print(f"  total  : {n_tgts * n_cells} FGW calls")
    print(f"  output : {output_path}")

    S_mat = np.full((n_tgts, n_a, n_q), np.nan)
    if os.path.exists(output_path):
        try:
            existing = np.load(output_path, allow_pickle=True)
            if existing["S"].shape == S_mat.shape:
                S_mat = existing["S"]
                n_done = int(np.sum(~np.isnan(S_mat[:, 0, 0])))
                if n_done == n_tgts:
                    print(f"[skip] all {n_tgts} targets already done")
                    return
                if n_done > 0:
                    print(f"[resume] {n_done}/{n_tgts} targets already done")
        except Exception as e:
            print(f"[warn] could not resume from {output_path}: {e}; starting fresh")
            S_mat = np.full((n_tgts, n_a, n_q), np.nan)

    # --- Source triples: one per Q at β = FIXED_BETA. ---
    print(f"\n[1/2] Building source triples at each Q ...", flush=True)
    src_class = load_classification(src_model, src_task)
    src_triples_by_Q = {}
    for Q in QUANTILES:
        t0 = time.time()
        tri, theta = build_triple_at_Q(src_model, src_task, src_class, Q)
        src_triples_by_Q[Q] = tri
        print(f"  Q={Q:5.3g}: built in {time.time() - t0:7.1f}s  "
              f"n_verts={tri[3]['n_verts']:6d}  θ={theta:.4g}", flush=True)

    def _save():
        np.savez(output_path, S=S_mat,
                 alphas=np.array(ALPHAS),
                 quantiles=np.array(QUANTILES),
                 fixed_beta=np.float64(FIXED_BETA),
                 source_idx=args.source_idx,
                 source_model=src_model, source_task=src_task,
                 target_indices=np.array(target_indices),
                 target_tuples=np.array(
                     [f"{TUPLES[i][0]}/{TUPLES[i][1]}" for i in target_indices],
                     dtype=object))

    # --- Per-target loop ---
    print(f"\n[2/2] Sweeping {n_tgts} targets × {n_cells} (α, Q) cells ...", flush=True)
    t_start = time.time()
    for local_t, tgt_global_idx in enumerate(target_indices):
        if not np.isnan(S_mat[local_t, 0, 0]):
            continue
        tgt_model, tgt_task = TUPLES[tgt_global_idx]
        print(f"\n  [{local_t + 1:2d}/{n_tgts}] {tgt_model}/{tgt_task} ...", flush=True)
        t_tgt = time.time()

        try:
            tgt_class = load_classification(tgt_model, tgt_task)
        except FileNotFoundError as e:
            print(f"    [WARN] missing classification: {e}")
            S_mat[local_t, :, :] = -1.0
            _save()
            continue

        for qi, Q in enumerate(QUANTILES):
            try:
                tgt_tri, _ = build_triple_at_Q(tgt_model, tgt_task, tgt_class, Q)
            except FileNotFoundError as e:
                print(f"    [WARN] missing DAG at Q={Q}: {e}")
                S_mat[local_t, :, qi] = -1.0
                continue

            src_tri = src_triples_by_Q[Q]
            for ai, alpha in enumerate(ALPHAS):
                try:
                    S, _ = fgw_similarity(src_tri, tgt_tri,
                                          alpha=alpha, n_init=args.n_init)
                except Exception as e:
                    print(f"    ERROR α={alpha} Q={Q}: {type(e).__name__}: {e}")
                    S = -1.0
                S_mat[local_t, ai, qi] = S
            del tgt_tri

        dt = time.time() - t_tgt
        avg_s = float(np.nanmean(np.where(S_mat[local_t] >= 0, S_mat[local_t], np.nan)))
        elapsed_min = (time.time() - t_start) / 60
        print(f"    done in {dt:.1f}s  avg_S={avg_s:.4f}  "
              f"(elapsed {elapsed_min:.1f}min)", flush=True)

        _save()
        del tgt_class

    print(f"\n=== Worker done in {(time.time() - t_start) / 60:.1f} min ===")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()