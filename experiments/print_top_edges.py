"""Print the top-K strongest influence edges per model on a fixed task.

Edge influence is the primary edge weight W_softmax(v -> v') = E_i[|p_orig(v')
- p_pert_{v->v'}(v')|], the downstream gating-probability displacement caused
by ablating v's contribution to v' (see main.tex). out(v) = sum of v's
outgoing W_softmax; this script instead reports the single strongest edges,
i.e. the dominant routing pathways v -> v'.

For each model, forward edges (sender layer < receiver layer) are ranked by
W_softmax descending and the top-K are printed as

    LsEa -> LrEb   (weight)

with model-absolute layer indices (DeepSeek models: DAG layer + 1, since the
raw DAG omits the leading dense layer).

Shared experts (DeepSeek-V2/V2-Lite): if the dag has a W_softmax_shared entry
([L, L, N] -- sender layer x receiver layer x receiver expert, one
shared-expert vertex per layer, no sender-expert axis), its edges are ranked
the same way and merged into the SAME top-K as the regular expert-to-expert
edges (top-K taken independently from each population first, then merged --
cheap even for the largest models, since we never materialize the full edge
space for the merge, only the small per-population top-K). A shared-expert
sender is printed as "LsS" instead of "LsEj".

Usage:
    python experiments/print_top_edges.py
    python experiments/print_top_edges.py --task math --top-k 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
RESULTS = Path(CFG["result_path"])

MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe",
    "deepseek-v2-lite", "olmoe",
    "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]

# Model-absolute layer offset (DeepSeek family omits the leading dense layer
# in the raw MoE-only DAG); matches print_top_acts.py / cross_rank_analysis.py.
NUM_DENSE = {
    "mixtral-8x7b": 0, "mixtral-8x22b": 0, "phi-3.5-moe": 0,
    "deepseek-v2-lite": 1, "deepseek-v2": 1,
    "olmoe": 0, "qwen3-30b-a3b": 0, "qwen3-235b-a22b": 0,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--top-k", type=int, default=10,
                   help="How many strongest edges to print per model (default 10).")
    args = p.parse_args()

    for m in MODELS:
        path = RESULTS / "dags" / args.task / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"\n[{m}]  MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        if "W_softmax" not in dag:
            print(f"\n[{m}]  no 'W_softmax' field")
            continue
        W = dag["W_softmax"].to(torch.float64)          # [L, N, L, N]
        L, N = W.shape[0], W.shape[1]
        nd = NUM_DENSE.get(m, 0)

        # Forward-edge mask: sender layer < receiver layer.
        s_idx = torch.arange(L).view(-1, 1, 1, 1)
        r_idx = torch.arange(L).view(1, 1, -1, 1)
        fwd = (s_idx < r_idx).expand_as(W)
        Wabs = W.abs()
        Wabs = torch.where(fwd, Wabs, torch.zeros_like(Wabs))

        flat = Wabs.reshape(-1)
        k = min(args.top_k, int((flat > 0).sum().item()))
        top_vals, top_idx = torch.topk(flat, k)

        candidates = []  # (weight, edge_str), merged across both populations below
        for rank in range(k):
            idx = int(top_idx[rank])
            sl = idx // (N * L * N)
            se = (idx // (L * N)) % N
            rl = (idx // N) % L
            re = idx % N
            edge = f"L{sl + nd}E{se} -> L{rl + nd}E{re}"
            candidates.append((float(top_vals[rank]), edge))

        if "W_softmax_shared" in dag:
            Ws = dag["W_softmax_shared"].to(torch.float64)   # [L, L, N]
            s_idx3 = torch.arange(L).view(-1, 1, 1)
            r_idx3 = torch.arange(L).view(1, -1, 1)
            fwd3 = (s_idx3 < r_idx3).expand_as(Ws)
            Wsabs = Ws.abs()
            Wsabs = torch.where(fwd3, Wsabs, torch.zeros_like(Wsabs))
            sflat = Wsabs.reshape(-1)
            ksh = min(args.top_k, int((sflat > 0).sum().item()))
            if ksh > 0:
                top_vals_sh, top_idx_sh = torch.topk(sflat, ksh)
                for rank in range(ksh):
                    idx = int(top_idx_sh[rank])
                    sl = idx // (L * N)
                    rl = (idx // N) % L
                    re = idx % N
                    edge = f"L{sl + nd}S -> L{rl + nd}E{re}"
                    candidates.append((float(top_vals_sh[rank]), edge))

        # Merge the (small) per-population top-K sets and take the overall top-K.
        candidates.sort(key=lambda c: -c[0])
        candidates = candidates[:args.top_k]

        print(f"\n[{m}]  (L={L} MoE layers, N={N} experts/layer; "
              f"model-absolute layers shown, dense offset +{nd})")
        print(f"  {'rank':>4s}  {'edge':>22s}  {'W_softmax':>10s}")
        for rank, (val, edge) in enumerate(candidates):
            print(f"  {rank + 1:>4d}  {edge:>22s}  {val:>10.4g}")


if __name__ == "__main__":
    main()
