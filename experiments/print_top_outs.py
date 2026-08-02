"""Print top-K per-expert outgoing-strength values per model on a fixed task.

out(v) = sum over outgoing forward edges of |W_softmax|
       = sum_{v'} |W_softmax^{m,d}(v -> v')|
         restricted to v -> v' with sender_layer < receiver_layer

This is the raw (unnormalised) "global influence" of each expert. The
log-max-normalised version (out / max_v' out) is what feeds the F matrix.

For each model, experts are sorted by out(v) descending and the values
at ranks 1..10, 20, plus the graph-wide median are reported.

Usage:
    python experiments/print_top_outs.py
    python experiments/print_top_outs.py --task math
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
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--top-n-ids", type=int, default=10,
                   help="Also print (layer, expert_idx_in_layer, out) for the "
                        "top-N experts per model (default 10). Useful for "
                        "identifying which specific experts sit at the head "
                        "of the out-strength distribution.")
    args = p.parse_args()

    # -------- Block 1: the rank-value summary (as before) ------------------
    header = (f"{'model':<18s} {'V':>6s}  "
              + "  ".join(f"{'top' + str(k):>11s}" for k in RANKS)
              + f"  {'median':>11s}")
    print(header)
    print("-" * len(header))

    top_id_records: dict[str, list[dict]] = {}
    for m in MODELS:
        path = RESULTS / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"{m:<18s}  MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        if "W_softmax" not in dag:
            print(f"{m:<18s}  no 'W_softmax' field ({path})")
            continue
        W = dag["W_softmax"].cpu().to(torch.float64)
        L, N, _, _ = W.shape
        # Forward-edge mask: sender_layer < receiver_layer
        s_idx = torch.arange(L).view(-1, 1, 1, 1)
        r_idx = torch.arange(L).view(1, 1, -1, 1)
        fwd = (s_idx < r_idx).expand_as(W)
        W_fwd = W * fwd.to(W.dtype)
        # out(v) = sum over receivers (l, n) of |W(v -> l, n)|
        out_LN = W_fwd.abs().sum(dim=(2, 3)).cpu().numpy()   # shape (L, N)
        out_flat = out_LN.reshape(-1)                        # flat[l*N + n]
        V = int(out_flat.size)
        # Rank all experts descending by out-strength.
        order = np.argsort(-out_flat)
        sorted_desc = out_flat[order]
        tops = [float(sorted_desc[k - 1]) if k <= V else float("nan")
                for k in RANKS]
        med = float(np.median(sorted_desc))
        print(f"{m:<18s} {V:>6d}  "
              + "  ".join(f"{t:>11.4g}" for t in tops)
              + f"  {med:>11.4g}")

        # Stash top-N (layer, expert_idx, value) records for Block 2 below.
        records = []
        for rank in range(min(args.top_n_ids, V)):
            flat = int(order[rank])
            layer_idx = flat // N
            expert_idx = flat % N
            records.append({
                "rank": rank + 1,
                "layer": layer_idx,
                "expert_in_layer": expert_idx,
                "out": float(out_flat[flat]),
                "L": L, "N": N,
            })
        top_id_records[m] = records

    # -------- Block 2: (layer, expert_idx) of top-N experts per model ------
    print()
    print("=" * 70)
    print(f"Top-{args.top_n_ids} experts per model with (layer, expert_idx_in_layer)")
    print("=" * 70)
    for m in MODELS:
        if m not in top_id_records:
            continue
        recs = top_id_records[m]
        L = recs[0]["L"]; N = recs[0]["N"]
        print(f"\n[{m}]  (L = {L} layers, N = {N} experts per layer)")
        print(f"  {'rank':>4s}  {'layer':>5s}  {'expert':>6s}  {'out(v)':>10s}")
        for r in recs:
            print(f"  {r['rank']:>4d}  {r['layer']:>5d}  "
                  f"{r['expert_in_layer']:>6d}  {r['out']:>10.4g}")


if __name__ == "__main__":
    main()
