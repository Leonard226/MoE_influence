"""Print raw top-K token-count loads per model on a fixed task.

For each model, sort experts by raw n_tokens_selected (descending) and
print the values at ranks 1, 2, 3, 4, 5, 10. No normalisation -- these
are absolute token counts on the calibration corpus.

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
CIRCUITS = Path(CFG["result_path"]) / "circuits"

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
              + "  ".join(f"{'top' + str(k):>7s}" for k in RANKS))
    print(header)
    print("-" * len(header))
    for m in MODELS:
        path = CIRCUITS / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"{m:<18s}  MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        n_tok = dag["n_tokens_selected"].cpu().numpy().reshape(-1)
        V = int(n_tok.size)
        sorted_desc = np.sort(n_tok)[::-1]
        tops = [int(sorted_desc[k - 1]) for k in RANKS]
        print(f"{m:<18s} {V:>6d}  "
              + "  ".join(f"{t:>7d}" for t in tops))


if __name__ == "__main__":
    main()
