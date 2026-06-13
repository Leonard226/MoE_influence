"""Audit top-K storage in DAGs for char-fingerprint feasibility (Option A).

For each (model, task) DAG we examine the per-vertex `top_weight`, `top_prompt`,
`top_pos` arrays. Each vertex stores its top-K tokens by routing weight, where
K is fixed at DAG-build time (see build_dag.py).

The audit reports three properties per vertex (then aggregated across the
active vertices of each (model, task)):

  1. Concentration (frac_top1):  top_weight[0] / sum(top_weight[:K])
     - Close to 1 means a single token dominates the vertex; fingerprint
       is effectively that one token.
     - Spread across top-K is good for a meaningful fingerprint.

  2. Active slots (n_active_in_topk): how many of the K stored slots actually
     carry non-zero weight (i.e., K minus padding).

  3. Diversity (n_unique_(prompt, pos) pairs): how many distinct
     (prompt_idx, position_in_prompt) tuples appear among the top-K. This is
     the size of the char-fingerprint set for that vertex (modulo char-offset
     mapping).

Interpretation thresholds for Option A viability:
  - median(n_unique_(prompt, pos)) >= 10 -> fingerprints are substantive
  - median(frac_top1) <= 0.5             -> mass not pathologically concentrated
  - p10(n_unique_(prompt, pos)) >= 3     -> even sparse vertices have some signal

If diversity is uniformly low or top-1 dominates, the char-fingerprint approach
is too sparse and we should fall back to BERT-cluster classification (Option B).

Writes: results/circuits/feature_ablation/topk_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
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


def audit_dag(dag: dict) -> dict:
    """Compute summary stats for one DAG. Returns dict of aggregates."""
    tw = dag["top_weight"]    # [L, N, K]
    tp = dag["top_prompt"]    # [L, N, K]
    ts = dag["top_pos"]       # [L, N, K]
    L, N, K = tw.shape

    tw_np = tw.cpu().numpy().reshape(L * N, K).astype(np.float64)
    tp_np = tp.cpu().numpy().reshape(L * N, K).astype(np.int64)
    ts_np = ts.cpu().numpy().reshape(L * N, K).astype(np.int64)

    # Only audit vertices that received at least one token.
    sums = tw_np.sum(axis=1)
    active_mask = sums > 1e-9
    n_active = int(active_mask.sum())
    if n_active == 0:
        return {"K_stored": int(K), "n_active_vertices": 0}

    tw_a = tw_np[active_mask]   # [n_active, K]
    tp_a = tp_np[active_mask]
    ts_a = ts_np[active_mask]
    sums_a = sums[active_mask]

    # Per-vertex computations.
    frac_top1 = tw_a[:, 0] / sums_a    # [n_active]

    n_unique_pairs = np.zeros(n_active, dtype=np.int64)
    n_active_in_topk = np.zeros(n_active, dtype=np.int64)
    n_unique_prompts = np.zeros(n_active, dtype=np.int64)
    for i in range(n_active):
        active_k = tw_a[i] > 1e-9
        n_active_in_topk[i] = int(active_k.sum())
        if not active_k.any():
            continue
        pairs = set(zip(tp_a[i, active_k].tolist(), ts_a[i, active_k].tolist()))
        n_unique_pairs[i] = len(pairs)
        n_unique_prompts[i] = len(set(tp_a[i, active_k].tolist()))

    def quartiles(arr: np.ndarray) -> dict:
        return {
            "p10":    float(np.percentile(arr, 10)),
            "median": float(np.percentile(arr, 50)),
            "p90":    float(np.percentile(arr, 90)),
        }

    return {
        "K_stored":           int(K),
        "n_active_vertices":  n_active,
        "n_total_vertices":   int(L * N),
        "frac_top1":          quartiles(frac_top1),
        "n_active_in_topk":   quartiles(n_active_in_topk.astype(np.float64)),
        "n_unique_pairs":     quartiles(n_unique_pairs.astype(np.float64)),
        "n_unique_prompts":   quartiles(n_unique_prompts.astype(np.float64)),
    }


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "circuits" / "feature_ablation" / "topk_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'model':<20s}  {'task':<12s}  {'K':>4s}  {'V_act':>6s}  "
          f"{'top1_med':>9s}  {'pairs_p10':>9s}  {'pairs_med':>9s}  "
          f"{'pairs_p90':>9s}  {'active_med':>10s}")
    print("-" * 110)

    summary: dict = {}
    for model in MODELS:
        for task in TASKS:
            dag_path = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
            try:
                dag = torch.load(dag_path, weights_only=False)
            except Exception as e:
                print(f"{model:<20s}  {task:<12s}  load failed: {type(e).__name__}: {e}")
                continue
            stats = audit_dag(dag)
            summary[f"{model}/{task}"] = stats
            if "frac_top1" not in stats:
                print(f"{model:<20s}  {task:<12s}  no active vertices")
                continue
            print(f"{model:<20s}  {task:<12s}  {stats['K_stored']:>4d}  "
                  f"{stats['n_active_vertices']:>6d}  "
                  f"{stats['frac_top1']['median']:>9.3f}  "
                  f"{stats['n_unique_pairs']['p10']:>9.1f}  "
                  f"{stats['n_unique_pairs']['median']:>9.1f}  "
                  f"{stats['n_unique_pairs']['p90']:>9.1f}  "
                  f"{stats['n_active_in_topk']['median']:>10.1f}")
        print()

    # Aggregate per-model summary (averaged across tasks).
    print("=" * 110)
    print("Per-model summary (averaged across the 8 tasks):")
    print("=" * 110)
    print(f"{'model':<20s}  {'K':>4s}  {'top1_avg':>9s}  "
          f"{'pairs_p10_avg':>14s}  {'pairs_med_avg':>14s}  {'verdict':>10s}")
    print("-" * 92)
    per_model_avg = {}
    for model in MODELS:
        keys = [k for k in summary if k.startswith(f"{model}/")
                                   and "frac_top1" in summary[k]]
        if not keys:
            continue
        top1 = np.mean([summary[k]["frac_top1"]["median"]      for k in keys])
        pp10 = np.mean([summary[k]["n_unique_pairs"]["p10"]    for k in keys])
        pmed = np.mean([summary[k]["n_unique_pairs"]["median"] for k in keys])
        K = summary[keys[0]]["K_stored"]
        verdict = "viable" if (pmed >= 10 and top1 <= 0.5) else "marginal"
        if pmed < 3 or top1 >= 0.8:
            verdict = "too sparse"
        per_model_avg[model] = {
            "frac_top1_avg":         float(top1),
            "n_unique_pairs_p10_avg":  float(pp10),
            "n_unique_pairs_med_avg":  float(pmed),
            "verdict":               verdict,
        }
        print(f"{model:<20s}  {K:>4d}  {top1:>9.3f}  "
              f"{pp10:>14.1f}  {pmed:>14.1f}  {verdict:>10s}")

    summary["_per_model_avg"] = per_model_avg
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final decision text.
    print()
    print("=" * 110)
    print("Decision summary:")
    print("=" * 110)
    n_viable = sum(1 for v in per_model_avg.values() if v["verdict"] == "viable")
    n_marginal = sum(1 for v in per_model_avg.values() if v["verdict"] == "marginal")
    n_sparse = sum(1 for v in per_model_avg.values() if v["verdict"] == "too sparse")
    print(f"  viable     : {n_viable}/{len(per_model_avg)} models")
    print(f"  marginal   : {n_marginal}/{len(per_model_avg)} models")
    print(f"  too sparse : {n_sparse}/{len(per_model_avg)} models")
    if n_sparse > 0:
        print()
        print("  >>> At least one model has too-sparse top-K. Consider Option B")
        print("  >>> (BERT clusters), or revisit build_dag.py to increase K.")
    elif n_marginal > len(per_model_avg) / 2:
        print()
        print("  >>> Most models are marginal. Option A is workable but expect")
        print("  >>> noisier fingerprints; may need K-larger DAGs for robustness.")
    else:
        print()
        print("  >>> Option A is viable across models. Proceed with char-fingerprint")
        print("  >>> feature implementation.")


if __name__ == "__main__":
    main()
