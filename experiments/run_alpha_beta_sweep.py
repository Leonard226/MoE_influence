"""Pairwise α × β sweep across all (model, task) tuples. No nulls.

One worker = one (source_idx, target_chunk) work unit. Run as a SLURM array
via experiments/launch_alpha_beta_sweep.sh.

For each (source, target) pair and each (α, β) config, computes
    S_α,β(G_source, G_target)
and writes a slice file
    {output_dir}/sweep_src{SS}_chunk{CC}.npz
that the aggregator stitches into the full
    S[src, tgt, alpha, beta]
array.

The metric is CPU-bound (POT + scipy); no GPU needed. Worker triple builds
amortise across the α × β grid via β-linear-combo of pre-built (β=0, β=1)
components -- so each (model, task) pair pays one shortest-path computation
regardless of how many β values are swept.
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

from experiments.fgw import build_triple, fgw_similarity, TOKEN_CLASSES

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

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BETAS  = [0.0, 0.25, 0.5, 0.75, 1.0]
N_INIT = 5

SPECIAL_IDX = TOKEN_CLASSES.index("special")
MASK_THRESH = 0.5
CACHE_DIR = os.path.join(config["result_path"], "circuits", "classifications")
DAG_DIR   = os.path.join(config["result_path"], "circuits")


def mask_triple(triple, class_idx=SPECIAL_IDX, threshold=MASK_THRESH):
    """Drop vertices with class_hist[class_idx] > threshold."""
    C, F, mass, meta = triple
    keep = F[:, 4 + class_idx] <= threshold
    keep_idx = np.where(keep)[0]
    return (
        C[np.ix_(keep_idx, keep_idx)],
        F[keep_idx],
        (mass[keep_idx] / mass[keep_idx].sum()),
        {**meta, "n_verts": len(keep_idx), "keep_idx": keep_idx},
    )


def load_classification(model, task):
    path = os.path.join(CACHE_DIR, f"classify_{model}_{task}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dag(model, task):
    path = os.path.join(DAG_DIR, f"dag_{model}_{task}.pt")
    return torch.load(path, map_location="cpu")


def build_two_triples(model, task, classification):
    """Build β=0 (path-only C) and β=1 (depth-only C) triples for one tuple.
    These two are linearly mixed at FGW time to recover arbitrary β."""
    dag = load_dag(model, task)
    triple_p = mask_triple(build_triple(dag, classification, beta=0.0, edge_threshold=0.0))
    triple_d = mask_triple(build_triple(dag, classification, beta=1.0, edge_threshold=0.0))
    del dag
    return triple_p, triple_d


def mixed_triple(triple_p, triple_d, beta):
    """Return a triple with C = β·C_depth + (1-β)·C_path. F and mass are
    identical between triple_p and triple_d (only C differs)."""
    C_mix = beta * triple_d[0] + (1.0 - beta) * triple_p[0]
    return (C_mix, triple_p[1], triple_p[2], triple_p[3])


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

    # Targets in this chunk.
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

    print(f"=== Alpha-Beta Sweep ===")
    print(f"  source : {src_model}/{src_task} (idx {args.source_idx})")
    print(f"  targets: chunk {args.target_chunk}/{args.num_chunks - 1}  -> {n_tgts} tuples")
    print(f"  α × β  : {len(ALPHAS)} × {len(BETAS)} = {len(ALPHAS) * len(BETAS)} configs")
    print(f"  total  : {n_tgts * len(ALPHAS) * len(BETAS)} FGW calls")
    print(f"  output : {output_path}")

    # --- Initialise / resume the S slice ---
    S_mat = np.full((n_tgts, len(ALPHAS), len(BETAS)), np.nan)
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
            S_mat = np.full((n_tgts, len(ALPHAS), len(BETAS)), np.nan)

    # --- Source triples (one shortest-path computation; reused for everything) ---
    print(f"\n[1/2] Building source β=0 / β=1 triples ...", flush=True)
    src_class = load_classification(src_model, src_task)
    t0 = time.time()
    src_p, src_d = build_two_triples(src_model, src_task, src_class)
    print(f"  source triples built in {time.time() - t0:.1f}s  "
          f"(n_verts={src_p[3]['n_verts']})", flush=True)

    # --- Loop over targets ---
    print(f"\n[2/2] Sweeping {n_tgts} target tuples × 25 (α, β) configs ...", flush=True)
    t_start = time.time()
    for local_t, tgt_global_idx in enumerate(target_indices):
        if not np.isnan(S_mat[local_t, 0, 0]):
            continue  # already done
        tgt_model, tgt_task = TUPLES[tgt_global_idx]
        print(f"\n  [{local_t + 1:2d}/{n_tgts}] {tgt_model}/{tgt_task} ...", flush=True)
        t_tgt = time.time()

        try:
            tgt_class = load_classification(tgt_model, tgt_task)
            tgt_p, tgt_d = build_two_triples(tgt_model, tgt_task, tgt_class)
        except FileNotFoundError as e:
            print(f"    [WARN] missing DAG or classification: {e}")
            S_mat[local_t, :, :] = -1.0  # sentinel
            np.savez(output_path, S=S_mat,
                     alphas=np.array(ALPHAS), betas=np.array(BETAS),
                     source_idx=args.source_idx,
                     source_model=src_model, source_task=src_task,
                     target_indices=np.array(target_indices),
                     target_tuples=np.array(
                         [f"{TUPLES[i][0]}/{TUPLES[i][1]}" for i in target_indices],
                         dtype=object))
            continue

        for bi, beta in enumerate(BETAS):
            src_tri = mixed_triple(src_p, src_d, beta)
            tgt_tri = mixed_triple(tgt_p, tgt_d, beta)
            for ai, alpha in enumerate(ALPHAS):
                try:
                    S, _ = fgw_similarity(src_tri, tgt_tri,
                                          alpha=alpha, n_init=args.n_init)
                except Exception as e:
                    print(f"    ERROR α={alpha} β={beta}: {type(e).__name__}: {e}")
                    S = -1.0
                S_mat[local_t, ai, bi] = S

        dt = time.time() - t_tgt
        avg_s = float(np.nanmean(np.where(S_mat[local_t] >= 0, S_mat[local_t], np.nan)))
        elapsed_min = (time.time() - t_start) / 60
        print(f"    done in {dt:.1f}s  avg_S={avg_s:.4f}  "
              f"(elapsed {elapsed_min:.1f}min)", flush=True)

        # Checkpoint after each target so a crash mid-chunk doesn't lose progress.
        np.savez(output_path, S=S_mat,
                 alphas=np.array(ALPHAS), betas=np.array(BETAS),
                 source_idx=args.source_idx,
                 source_model=src_model, source_task=src_task,
                 target_indices=np.array(target_indices),
                 target_tuples=np.array(
                     [f"{TUPLES[i][0]}/{TUPLES[i][1]}" for i in target_indices],
                     dtype=object))

        del tgt_class, tgt_p, tgt_d, src_tri, tgt_tri

    print(f"\n=== Worker done in {(time.time() - t_start) / 60:.1f} min ===")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()