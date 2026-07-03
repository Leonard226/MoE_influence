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
    p.add_argument("--top-n-ids", type=int, default=30,
                   help="Also print (layer, expert_idx_in_layer, act) for the "
                        "top-N experts per model (default 30, enough to cover "
                        "all Super Experts even for Qwen3 which has "
                        "~30 experts above the P_99.5 threshold).")
    args = p.parse_args()

    header = (f"{'model':<18s} {'V':>6s}  "
              + "  ".join(f"{'top' + str(k):>11s}" for k in RANKS)
              + f"  {'rank@0.5%':>11s}  {'P99.5':>11s}  {'top1x0.1':>11s}")
    print(header)
    print("-" * len(header))

    top_id_records: dict[str, list[dict]] = {}
    thresholds: dict[str, dict] = {}
    for m in MODELS:
        path = CIRCUITS / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"{m:<18s}  MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        if "act" not in dag:
            print(f"{m:<18s}  no 'act' field in DAG ({path})")
            continue
        act = dag["act"].cpu().numpy().astype(np.float64)     # shape (L, N)
        L, N = act.shape
        act_flat = act.reshape(-1)                            # flat[l*N + n]
        V = int(act_flat.size)
        order = np.argsort(-act_flat)
        sorted_desc = act_flat[order]
        tops = [float(sorted_desc[k - 1]) for k in RANKS]
        # P_99.5: threshold such that exactly 0.5% of experts exceed it.
        rank_05 = max(1, int(np.ceil(0.005 * V)))
        p_995 = float(sorted_desc[rank_05 - 1])
        top1_x_01 = float(sorted_desc[0]) * 0.1
        print(f"{m:<18s} {V:>6d}  "
              + "  ".join(f"{t:>11.4g}" for t in tops)
              + f"  {rank_05:>11d}  {p_995:>11.4g}  {top1_x_01:>11.4g}")

        # Stash top-N (layer, expert_idx, act) records for Block 2.
        records = []
        for rank in range(min(args.top_n_ids, V)):
            flat = int(order[rank])
            layer_idx = flat // N
            expert_idx = flat % N
            act_val = float(act_flat[flat])
            is_se = act_val > max(p_995, top1_x_01)
            records.append({
                "rank": rank + 1,
                "layer": layer_idx,
                "expert_in_layer": expert_idx,
                "act": act_val,
                "is_se": is_se,
                "L": L, "N": N,
            })
        top_id_records[m] = records
        thresholds[m] = {"p_995": p_995, "top1_x_01": top1_x_01,
                          "se_threshold": max(p_995, top1_x_01),
                          "V": V, "L": L, "N": N}

    # -------- Block 2: (layer, expert_idx) of top-N experts per model ------
    print()
    print("=" * 78)
    print(f"Top-{args.top_n_ids} experts per model with (layer, expert_idx_in_layer)")
    print("'SE' marks experts satisfying Su et al.'s criterion:")
    print("      act(v) > max(P_99.5,  top-1 * 0.1)   for that model.")
    print("=" * 78)
    for m in MODELS:
        if m not in top_id_records:
            continue
        recs = top_id_records[m]
        thr = thresholds[m]
        print(f"\n[{m}]  (L={thr['L']} layers, N={thr['N']} experts/layer, "
              f"V={thr['V']} total)")
        print(f"  SE threshold = max(P_99.5={thr['p_995']:.4g}, "
              f"top1*0.1={thr['top1_x_01']:.4g})  =  {thr['se_threshold']:.4g}")
        n_se = sum(1 for r in recs if r["is_se"])
        print(f"  Super Experts in this top-{args.top_n_ids}: {n_se}")
        print(f"  {'rank':>4s}  {'layer':>5s}  {'expert':>6s}  {'act(v)':>10s}  SE")
        for r in recs:
            marker = "SE" if r["is_se"] else ""
            print(f"  {r['rank']:>4d}  {r['layer']:>5d}  "
                  f"{r['expert_in_layer']:>6d}  {r['act']:>10.4g}  {marker}")


if __name__ == "__main__":
    main()
