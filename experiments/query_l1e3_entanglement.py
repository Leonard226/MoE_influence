"""One-off query of the saved DAG tensor for candidate L1E3 entanglement
partners in Mixtral-8x7B.

build_dag.py saves the FULL pairwise edge tensor W_softmax[c,j,l,n] (every
sender (c,j) -> receiver (l,n), not just the top-10 already in
tab:topk-match-c4). This script just indexes into that tensor -- no model
load, no forward pass, CPU-only, reads dag_mixtral-8x7b_c4.pt directly.

Reports:
  1. L1E3's full outgoing edge row, top-20 by weight (any receiver layer >1).
  2. L1E3's incoming edges from layer-0 senders only, top-10 by weight --
     the only architecturally-possible reverse-entanglement candidates,
     since routing at layer 1 cannot depend on anything at layer >=1.

These are linearized, pairwise-isolated estimates (see Section on Routing
DAG construction) -- a strong edge here is a CANDIDATE for real joint-
ablation entanglement, not proof of it.

Usage:
    python experiments/query_l1e3_entanglement.py
"""
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DAG_PATH = Path(CFG["result_path"]) / "circuits" / "dag_mixtral-8x7b_c4.pt"

SENDER_LAYER, SENDER_EXPERT = 1, 3  # L1E3

d = torch.load(DAG_PATH, map_location="cpu")
W = d["W_softmax"]  # [c, j, l, n]
L, N = W.shape[2], W.shape[3]
print(f"Loaded {DAG_PATH}: W_softmax shape {tuple(W.shape)} (L={L}, N={N})")

print(f"\n=== L{SENDER_LAYER}E{SENDER_EXPERT} outgoing edges, top-20 ===")
row = W[SENDER_LAYER, SENDER_EXPERT].clone()  # [l, n]
row[:SENDER_LAYER + 1, :] = float("-inf")  # mask non-causal (l <= c) entries, regardless
                                           # of how the underlying accumulator initialised them
flat = row.flatten()
topk = torch.topk(flat, 20)
for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
    l, n = idx // N, idx % N
    print(f"  {rank:2d}. L{SENDER_LAYER}E{SENDER_EXPERT} -> L{l}E{n}: {val:.4f}")

print(f"\n=== Incoming edges to L{SENDER_LAYER}E{SENDER_EXPERT} from layer-0 senders, top-10 ===")
col = W[0, :, SENDER_LAYER, SENDER_EXPERT]  # [j] -- only c=0 senders can reach l=1
topk_in = torch.topk(col, min(10, col.numel()))
for rank, (val, idx) in enumerate(zip(topk_in.values.tolist(), topk_in.indices.tolist()), 1):
    print(f"  {rank:2d}. L0E{idx} -> L{SENDER_LAYER}E{SENDER_EXPERT}: {val:.4f}")

print(f"\n(for reference) L{SENDER_LAYER}E{SENDER_EXPERT}'s own act magnitude: "
      f"{d['act'][SENDER_LAYER, SENDER_EXPERT].item():.4g}")
