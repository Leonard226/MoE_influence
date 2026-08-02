"""Print top-K per-expert load values per model on a fixed task.

load(v) = n_tokens_selected(v) / (1/N * sum_{n'} n_tokens_selected(layer_v, n'))
        = how many times the layer mean of tokens that expert v received.

By construction, the mean of load over experts in a layer is 1. Reported
values are the top 1, 2, 3, 4, 5, 10 ranked load values across the whole
graph (descending). Equivalent unit to the load values in
Table~tab:load-distribution.

Usage:
    python experiments/print_top_loads.py
    python experiments/print_top_loads.py --task math
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
RANKS = [1, 2, 3, 4, 5, 10]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    args = p.parse_args()

    header = (f"{'model':<18s} {'V':>6s}  "
              + "  ".join(f"{'top' + str(k):>9s}" for k in RANKS))
    print(header)
    print("-" * len(header))
    for m in MODELS:
        path = RESULTS / "dags" / args.task / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"{m:<18s}  MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        n_tok = dag["n_tokens_selected"].cpu().numpy().astype(np.float64)
        L, N = n_tok.shape
        V = int(L * N)
        # Per-layer mean (clipped) and layer-mean-normalised load.
        layer_mean = n_tok.mean(axis=1, keepdims=True).clip(min=1e-12)
        load = (n_tok / layer_mean).reshape(-1)
        sorted_desc = np.sort(load)[::-1]
        tops = [float(sorted_desc[k - 1]) for k in RANKS]
        print(f"{m:<18s} {V:>6d}  "
              + "  ".join(f"{t:>9.3f}" for t in tops))


if __name__ == "__main__":
    main()
