"""Stitch per-(source, chunk) sweep slices into a full S[src, tgt, α, β] array.

Usage:
    python experiments/aggregate_alpha_beta_sweep.py

Reads {result_path}/circuits/alpha_beta_sweep/sweep_src*_chunk*.npz files
and writes the merged S_full.npz to the same directory. Reports any missing
or failed pairs.
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.run_alpha_beta_sweep import (
    MODELS, TASKS, TUPLES, N_TUPLES, ALPHAS, BETAS,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_dir = args.input_dir or os.path.join(
        config["result_path"], "circuits", "alpha_beta_sweep"
    )
    output_path = args.output or os.path.join(input_dir, "S_full.npz")

    n_a, n_b = len(ALPHAS), len(BETAS)
    S_full = np.full((N_TUPLES, N_TUPLES, n_a, n_b), np.nan)

    # Diagonal = 1.0 by construction (verified in §3 of the notebook).
    for i in range(N_TUPLES):
        S_full[i, i, :, :] = 1.0

    files = sorted(glob(os.path.join(input_dir, "sweep_src*_chunk*.npz")))
    print(f"Found {len(files)} slice files under {input_dir}")
    if not files:
        print("No slice files; nothing to aggregate.")
        return

    n_filled, n_failed = 0, 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        src = int(d["source_idx"])
        target_indices = d["target_indices"]
        S_slice = d["S"]  # [n_targets_in_chunk, n_a, n_b]
        for local_t, tgt in enumerate(target_indices):
            tgt = int(tgt)
            S_full[src, tgt, :, :] = S_slice[local_t]
            S_full[tgt, src, :, :] = S_slice[local_t]   # symmetric
            n_filled += 1
            if np.any(S_slice[local_t] < 0):
                n_failed += int(np.sum(S_slice[local_t] < 0))

    n_off_diag_total = N_TUPLES * (N_TUPLES - 1)   # ordered pairs
    n_off_diag_done = int(np.sum(~np.isnan(S_full)) - N_TUPLES * n_a * n_b)  # subtract diag
    print(f"Filled {n_filled} (src, tgt) entries from slices.")
    print(f"S_full has {n_off_diag_done}/{n_off_diag_total * n_a * n_b} non-NaN off-diagonal cells.")
    n_nan = int(np.isnan(S_full).sum())
    if n_nan:
        # Identify which (src, tgt) pairs are missing.
        missing_pairs = set()
        for i in range(N_TUPLES):
            for j in range(N_TUPLES):
                if i == j:
                    continue
                if np.isnan(S_full[i, j, 0, 0]):
                    missing_pairs.add((min(i, j), max(i, j)))
        print(f"Missing {len(missing_pairs)} (src, tgt) pairs (unordered):")
        for (i, j) in sorted(missing_pairs)[:20]:
            print(f"  {TUPLES[i][0]}/{TUPLES[i][1]}  <->  {TUPLES[j][0]}/{TUPLES[j][1]}")
        if len(missing_pairs) > 20:
            print(f"  ... and {len(missing_pairs) - 20} more")
    if n_failed:
        print(f"WARNING: {n_failed} cells are -1 (FGW solver failures)")

    np.savez(
        output_path,
        S=S_full,
        tuples=np.array([f"{m}/{t}" for m, t in TUPLES], dtype=object),
        alphas=np.array(ALPHAS),
        betas=np.array(BETAS),
        models=np.array(MODELS, dtype=object),
        tasks=np.array(TASKS, dtype=object),
    )
    print(f"\nSaved: {output_path}")
    print(f"  S.shape = {S_full.shape}  (src, tgt, alpha, beta)")


if __name__ == "__main__":
    main()