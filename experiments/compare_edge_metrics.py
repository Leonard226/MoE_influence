"""Post-pilot sanity: compare P_flip = P_add + P_rem against the new W_softmax.

For one dag_*.pt file (the rebuilt schema), compute the Pearson correlation
between P_flip and W_softmax restricted to forward edges (S < R), report
decile distributions of both, and list the top-10 edges where the two
metrics most disagree (z-score gap).

Decision criterion from the design plan:
  Pearson > 0.95     → new metric ~= old, rebuild not informative
  Pearson 0.5 – 0.8  → genuinely new signal, scale rebuild to all models
  Pearson < 0.5      → dramatic decorrelation; investigate before scaling

Usage:
    python experiments/compare_edge_metrics.py /path/to/dag_mixtral-8x7b_c4.pt
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dag_path", type=str)
    args = ap.parse_args()

    d = torch.load(args.dag_path, map_location="cpu", weights_only=False)
    L, N = d["P_add"].shape[0], d["P_add"].shape[1]

    p_flip = (d["P_add"] + d["P_rem"]).numpy()              # [L, N, L, N]
    w_sm   = d["W_softmax"].numpy()                          # [L, N, L, N]

    # Forward-edge mask: S < R.
    S_idx = np.arange(L)[:, None, None, None]
    R_idx = np.arange(L)[None, None, :, None]
    fwd = np.broadcast_to(S_idx < R_idx, p_flip.shape)
    p_flat = p_flip[fwd]
    w_flat = w_sm[fwd]

    # Pearson over the forward-edge population.
    if p_flat.std() > 0 and w_flat.std() > 0:
        r = float(np.corrcoef(p_flat, w_flat)[0, 1])
    else:
        r = float("nan")

    qs = np.linspace(0, 1, 11)
    pq = np.quantile(p_flat, qs)
    wq = np.quantile(w_flat, qs)

    print(f"file:     {args.dag_path}")
    print(f"model:    {d.get('model', '?')}")
    print(f"dataset:  {d.get('dataset', '?')}")
    print(f"n_prompts:{d.get('n_prompts', '?')}")
    print(f"shape:    L={L}, N={N}   forward edges = {p_flat.size}")
    print()
    print(f"Pearson(P_flip, W_softmax) = {r:.4f}")
    print()
    print(f"P_flip    deciles: {np.array2string(pq, precision=4, floatmode='fixed')}")
    print(f"W_softmax deciles: {np.array2string(wq, precision=6, floatmode='fixed')}")
    print()

    # Top-10 disagreement edges by |z(P_flip) − z(W_softmax)|.
    pz = (p_flat - p_flat.mean()) / (p_flat.std() + 1e-12)
    wz = (w_flat - w_flat.mean()) / (w_flat.std() + 1e-12)
    gap = np.abs(pz - wz)
    top = np.argsort(-gap)[:10]
    fwd_idx = np.argwhere(fwd)
    print("Top-10 disagreement edges (largest |z(P_flip) − z(W_softmax)|):")
    print(f"  {'S':>3} {'j':>3} {'R':>3} {'n':>3}   {'P_flip':>8} {'W_softmax':>10}   {'gap':>6}")
    for k in top:
        S, j, R, n = fwd_idx[k]
        print(f"  {S:>3} {j:>3} {R:>3} {n:>3}   "
              f"{p_flat[k]:>8.4f} {w_flat[k]:>10.6f}   {gap[k]:>6.2f}")


if __name__ == "__main__":
    main()