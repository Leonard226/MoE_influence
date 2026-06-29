"""Print top-K per-expert outgoing-strength values per model on a fixed task.

out(v) = sum over outgoing forward edges of |W_softmax|
       = sum_{v'} |W_softmax^{m,d}(v -> v')|
         restricted to v -> v' with sender_layer < receiver_layer

This is the raw (unnormalised) "global influence" of each expert. The
log-max-normalised version (out / max_v' out) is what feeds the F matrix.

For each model, experts are sorted by out(v) descending and the values
at ranks 1, 2, 3, 5, 10 are reported.

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
CIRCUITS = Path(CFG["result_path"]) / "circuits"

MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe",
    "deepseek-v2-lite", "olmoe",
    "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]
RANKS = [1, 2, 3, 5, 10]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    args = p.parse_args()

    header = (f"{'model':<18s} {'V':>6s}  "
              + "  ".join(f"{'top' + str(k):>11s}" for k in RANKS))
    print(header)
    print("-" * len(header))
    for m in MODELS:
        path = CIRCUITS / f"dag_{m}_{args.task}.pt"
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
        out_strength = W_fwd.abs().sum(dim=(2, 3)).reshape(-1).numpy()
        V = int(out_strength.size)
        sorted_desc = np.sort(out_strength)[::-1]
        tops = [float(sorted_desc[k - 1]) for k in RANKS]
        print(f"{m:<18s} {V:>6d}  "
              + "  ".join(f"{t:>11.4g}" for t in tops))


if __name__ == "__main__":
    main()
