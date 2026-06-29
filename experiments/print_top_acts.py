"""Print top-K raw act(v) values per model on a fixed task.

act(v) = max_{i in d : v in TopK_i^l} || e_{out, i}^v ||_inf

This is the L_inf norm of expert v's output, taken as the maximum over
tokens routed to v during the forward pass -- the "super-expert"
indicator from Su et al. (ICLR, 2026). No normalisation; values are the
raw L_inf magnitudes (the values that feed log-max normalisation in the
feature matrix F).

For each model, experts are sorted by act(v) descending and the values
at ranks 1, 2, 3, 4, 5, 10 are reported.

Usage:
    python experiments/print_top_acts.py
    python experiments/print_top_acts.py --task math
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
              + "  ".join(f"{'top' + str(k):>11s}" for k in RANKS)
              + f"  {'rank@0.5%':>11s}  {'P99.5':>11s}")
    print(header)
    print("-" * len(header))
    for m in MODELS:
        path = CIRCUITS / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"{m:<18s}  MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        if "act" not in dag:
            print(f"{m:<18s}  no 'act' field in DAG ({path})")
            continue
        act = dag["act"].cpu().numpy().astype(np.float64).reshape(-1)
        V = int(act.size)
        sorted_desc = np.sort(act)[::-1]
        tops = [float(sorted_desc[k - 1]) for k in RANKS]
        # P_99.5: threshold such that exactly 0.5% of experts exceed it.
        # Use rank = ceil(0.005 * V) (1-indexed), so sorted_desc[rank-1].
        rank_05 = max(1, int(np.ceil(0.005 * V)))
        p_995 = float(sorted_desc[rank_05 - 1])
        print(f"{m:<18s} {V:>6d}  "
              + "  ".join(f"{t:>11.4g}" for t in tops)
              + f"  {rank_05:>11d}  {p_995:>11.4g}")


if __name__ == "__main__":
    main()
