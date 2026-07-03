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

# Su et al.'s include_layers filter uses model-absolute layer indices
# (num_hidden_layers). Our DAG's L excludes first_k_dense_replace dense
# layers for the DeepSeek family, so we add that offset back to align.
NUM_DENSE = {
    "mixtral-8x7b": 0, "mixtral-8x22b": 0,
    "phi-3.5-moe": 0,
    "deepseek-v2-lite": 1, "deepseek-v2": 1,
    "olmoe": 0,
    "qwen3-30b-a3b": 0, "qwen3-235b-a22b": 0,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--top-n-ids", type=int, default=30,
                   help="Also print (layer, expert_idx_in_layer, act) for the "
                        "top-N experts per model (default 30, enough to cover "
                        "all Super Experts even for Qwen3 which has "
                        "~30 experts above the P_99.5 threshold).")
    p.add_argument("--include-layers", type=float, default=1.0,
                   help="Su et al.'s layer-depth cutoff (default 1.0 = no "
                        "filter; Su's default is 0.75). Keeps model layers "
                        "with index < round(num_hidden_layers * frac); "
                        "P_99.5 and top1//10 thresholds are computed over "
                        "this filtered subset only (matches "
                        "_identify_super_experts). Uses np.percentile linear "
                        "interpolation and floor division to match Su exactly.")
    args = p.parse_args()
    if not 0.0 < args.include_layers <= 1.0:
        p.error("--include-layers must be in (0, 1]")

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
        # Apply Su's include_layers cutoff on model-absolute layer indices,
        # translated to our DAG's dense-shifted range. include_up_to_dag
        # is exclusive: DAG rows [0, include_up_to_dag) are retained.
        num_dense = NUM_DENSE.get(m, 0)
        total_model_L = L + num_dense
        include_up_to_model = round(total_model_L * args.include_layers)
        include_up_to_dag = max(0, min(L, include_up_to_model - num_dense))
        act_filt = act[:include_up_to_dag]                    # (L', N)
        L_filt = act_filt.shape[0]
        act_flat = act_filt.reshape(-1)                       # flat[l*N + n]
        V = int(act_flat.size)
        if V == 0:
            print(f"{m:<18s}  empty after --include-layers filter "
                  f"(include_up_to_dag={include_up_to_dag})")
            continue
        order = np.argsort(-act_flat)
        sorted_desc = act_flat[order]
        tops = [float(sorted_desc[k - 1]) for k in RANKS if k <= V]
        # Su-exact thresholds: linear-interp percentile + floor division,
        # both over the (possibly filtered) subset only.
        p_995 = float(np.percentile(act_flat, 99.5))
        top1_x_01 = float(np.max(act_flat) // 10)
        # For the Block-1 rank@0.5% column, keep the empirical rank index
        # (interpretation-friendly), but note p_995 is now interpolated.
        rank_05 = max(1, int(np.ceil(0.005 * V)))
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
                          "V": V, "L": L, "L_filt": L_filt, "N": N,
                          "num_dense": num_dense,
                          "include_up_to_model": include_up_to_model,
                          "include_up_to_dag": include_up_to_dag}

    # -------- Block 2: (layer, expert_idx) of top-N experts per model ------
    print()
    print("=" * 78)
    print(f"Top-{args.top_n_ids} experts per model with (layer, expert_idx_in_layer)")
    print("'SE' marks experts satisfying Su et al.'s criterion (Su-exact math):")
    print("      act(v) > np.percentile(subset, 99.5)")
    print("  AND act(v) > np.max(subset) // 10")
    print(f"where subset = layer-filtered DAG rows (include_layers={args.include_layers}).")
    print("=" * 78)
    for m in MODELS:
        if m not in top_id_records:
            continue
        recs = top_id_records[m]
        thr = thresholds[m]
        print(f"\n[{m}]  (L={thr['L']} DAG layers, dense+={thr['num_dense']}, "
              f"N={thr['N']} experts/layer)")
        print(f"  include_layers={args.include_layers}: keep model layers "
              f"[0, {thr['include_up_to_model']})  ->  DAG rows "
              f"[0, {thr['include_up_to_dag']})  ->  L_filt={thr['L_filt']}, "
              f"V={thr['V']}")
        print(f"  SE threshold = max(P_99.5={thr['p_995']:.4g}, "
              f"top1//10={thr['top1_x_01']:.4g})  =  {thr['se_threshold']:.4g}")
        n_se = sum(1 for r in recs if r["is_se"])
        print(f"  Super Experts in this top-{args.top_n_ids}: {n_se}")
        print(f"  {'rank':>4s}  {'layer':>5s}  {'expert':>6s}  {'act(v)':>10s}  SE")
        for r in recs:
            marker = "SE" if r["is_se"] else ""
            print(f"  {r['rank']:>4d}  {r['layer']:>5d}  "
                  f"{r['expert_in_layer']:>6d}  {r['act']:>10.4g}  {marker}")


if __name__ == "__main__":
    main()
