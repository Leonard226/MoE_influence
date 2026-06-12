"""Decompose the LOO ablation result by family and by task.

Two questions:
1. Is the load-removal collapse uniform across the 3 families
   (Mixtral, DeepSeek, Qwen3) or driven by one family?
2. Is the within/cross gap uniform across the 8 tasks or driven
   by a particular task?

Reads:  results/circuits/feature_ablation/S_loo.npz       # (7, 3, 8, 8, 8)
Writes: results/circuits/feature_ablation/breakdown.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
TASKS = ["c4", "math", "code", "wikitext2", "gsm8k", "humaneval", "pile-arxiv", "pile-github"]
FAMILIES = {
    "Mixtral":  [0, 1],   # mixtral-8x7b, mixtral-8x22b
    "DeepSeek": [2, 3],   # dsv2-lite, dsv2
    "Qwen3":    [4, 5],   # qwen3-30b-a3b, qwen3-235b-a22b
}
MODEL_TO_FAMILY = {m: f for f, idxs in FAMILIES.items() for m in idxs}

QUANTILES = [0.9, 0.99, 0.999]
ABLATIONS = ["full", "no_depth", "no_out", "no_in", "no_load", "no_act", "no_class"]


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    npz_path = Path(result_path) / "circuits" / "feature_ablation" / "S_loo.npz"
    json_out = Path(result_path) / "circuits" / "feature_ablation" / "breakdown.json"

    d = np.load(npz_path, allow_pickle=True)
    S = d["S"]   # (7, 3, 8, 8, 8): (abl, q, task, mi, mj)
    n_abl, n_q, n_t, n_m, _ = S.shape

    print(f"S shape: {S.shape}")
    print(f"NaN cells: {int(np.isnan(S).sum())} / {S.size}")
    print(f"(Expected: 7 abl x 3 Q x 8 tasks x 8 diag = {7 * 3 * 8 * 8} diag NaNs)\n")

    summary: dict = {"per_family": {}, "per_task": {}}

    # -- 1. Per-family within-mean for each ablation x Q -----------------
    # For each family, the within-family pair is unique: there are exactly
    # 2 same-family models in each, so 1 unordered pair. The "within mean"
    # for a family is then averaged over the 8 tasks (8 samples).
    print("=" * 92)
    print("Per-family within-family mean S (across 8 tasks per family):")
    print("=" * 92)
    print(f"{'ablation':<10s}  {'Q':>6s}  {'Mixtral':>10s}  {'DeepSeek':>10s}  "
          f"{'Qwen3':>10s}  {'all-fam mean':>14s}")
    print("-" * 92)

    for abl_idx, abl in enumerate(ABLATIONS):
        per_q = {}
        for q_idx, Q in enumerate(QUANTILES):
            row_family = {}
            for fam, (i, j) in FAMILIES.items():
                # Within-family pair (i, j) at each of 8 tasks; symmetric so use [..., i, j].
                vals = S[abl_idx, q_idx, :, i, j]
                row_family[fam] = float(np.nanmean(vals))
            all_fam_mean = float(np.mean(list(row_family.values())))
            per_q[f"Q={Q}"] = row_family | {"mean_3_families": all_fam_mean}
            print(f"{abl:<10s}  {Q:>6.3f}  "
                  f"{row_family['Mixtral']:>10.4f}  "
                  f"{row_family['DeepSeek']:>10.4f}  "
                  f"{row_family['Qwen3']:>10.4f}  "
                  f"{all_fam_mean:>14.4f}")
        summary["per_family"][abl] = per_q
        print()

    # -- 2. Per-task gap (within - cross) for each ablation x Q ----------
    print()
    print("=" * 92)
    print("Per-task gap (within-family mean - cross-family mean), averaged over models:")
    print("=" * 92)
    print(f"{'ablation':<10s}  {'Q':>6s}  " + "  ".join(f"{t[:8]:>9s}" for t in TASKS))
    print("-" * 92)

    for abl_idx, abl in enumerate(ABLATIONS):
        per_q = {}
        for q_idx, Q in enumerate(QUANTILES):
            row_task = {}
            for t_idx, task in enumerate(TASKS):
                within_vals = []
                cross_vals = []
                for mi in range(n_m):
                    for mj in range(mi + 1, n_m):
                        s = S[abl_idx, q_idx, t_idx, mi, mj]
                        if np.isnan(s):
                            continue
                        fam_i = MODEL_TO_FAMILY.get(mi)
                        fam_j = MODEL_TO_FAMILY.get(mj)
                        same_fam = (fam_i is not None and fam_i == fam_j)
                        (within_vals if same_fam else cross_vals).append(s)
                w = float(np.mean(within_vals)) if within_vals else float("nan")
                c = float(np.mean(cross_vals))  if cross_vals  else float("nan")
                gap = w - c
                row_task[task] = {"within": w, "cross": c, "gap": gap,
                                  "n_within": len(within_vals),
                                  "n_cross": len(cross_vals)}
            per_q[f"Q={Q}"] = row_task
            gaps = [row_task[t]["gap"] for t in TASKS]
            print(f"{abl:<10s}  {Q:>6.3f}  "
                  + "  ".join(f"{g:>+9.4f}" for g in gaps))
        summary["per_task"][abl] = per_q
        print()

    # -- 3. Variance summary: how uniform are the per-task gaps? ---------
    print()
    print("=" * 92)
    print("Per-task gap dispersion (std across the 8 tasks):")
    print("=" * 92)
    print(f"{'ablation':<10s}  {'Q':>6s}  {'mean gap':>10s}  {'std':>8s}  "
          f"{'min task':>16s}  {'max task':>16s}")
    print("-" * 92)
    for abl in ABLATIONS:
        for q_idx, Q in enumerate(QUANTILES):
            per_t = summary["per_task"][abl][f"Q={Q}"]
            gaps = np.array([per_t[t]["gap"] for t in TASKS])
            mean_gap = float(np.mean(gaps))
            std_gap  = float(np.std(gaps))
            min_t = TASKS[int(np.argmin(gaps))]
            max_t = TASKS[int(np.argmax(gaps))]
            print(f"{abl:<10s}  {Q:>6.3f}  {mean_gap:>+10.4f}  {std_gap:>8.4f}  "
                  f"{min_t:>11s} ({np.min(gaps):+.3f})  "
                  f"{max_t:>11s} ({np.max(gaps):+.3f})")
        print()

    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_out}")


if __name__ == "__main__":
    main()
