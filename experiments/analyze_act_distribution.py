"""Diagnostic: distribution of raw `act` values per (model, task).

Helps choose between normalisation schemes for hat{act}:
  - rank (current):     erases distribution shape entirely (all graphs uniform on [0, 1])
  - max-linear:         act / max(act)              [outlier-sensitive]
  - log-max:            log(1+act) / log(1+max(act)) [robust to outliers]
  - quantile-99:        act / p99(act)              [robust, scale-comparable]

We report per (model, task):
  - Raw scale: min, p50, p90, p95, p99, max  (so we can see how heavy the tail is)
  - max / p99 ratio   (single-vertex outlier dominance)
  - max / p50 ratio   (overall skew)
  - mass fraction in top-5 and top-50 vertices  (concentration)

Cross-model summary per task:
  - distribution of max(act) across the 8 models  (do scales vary by 100x or 100000x?)

Writes:
  results/feature_ablation/act_distribution.json
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


def stats_for_act(act_vals: np.ndarray) -> dict:
    """Compute summary statistics for a flattened act vector. Drops zeros
    (unused vertices) before computing — they're not informative for
    distribution-shape questions."""
    a = act_vals[act_vals > 0].astype(np.float64)
    if len(a) == 0:
        return {"n_nonzero": 0}
    a_sorted = np.sort(a)
    total = float(a_sorted.sum())
    top5  = float(a_sorted[-5:].sum())  if len(a) >= 5  else float(a_sorted.sum())
    top50 = float(a_sorted[-50:].sum()) if len(a) >= 50 else float(a_sorted.sum())
    return {
        "n_nonzero": int(len(a)),
        "min":   float(a.min()),
        "p50":   float(np.percentile(a, 50)),
        "p75":   float(np.percentile(a, 75)),
        "p90":   float(np.percentile(a, 90)),
        "p95":   float(np.percentile(a, 95)),
        "p99":   float(np.percentile(a, 99)),
        "max":   float(a.max()),
        "ratio_max_p99":  float(a.max() / max(np.percentile(a, 99), 1e-30)),
        "ratio_max_p50":  float(a.max() / max(np.percentile(a, 50), 1e-30)),
        "frac_top5":  top5  / max(total, 1e-30),
        "frac_top50": top50 / max(total, 1e-30),
    }


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_dir = Path(result_path) / "circuits" / "feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "act_distribution.json"

    summary: dict = {"per_dag": {}, "per_task": {}}

    print(f"{'model':<20s}  {'task':<12s}  {'|V|':>6s}  "
          f"{'p50':>9s}  {'p99':>9s}  {'max':>9s}  "
          f"{'max/p99':>9s}  {'max/p50':>9s}  "
          f"{'top5%':>7s}  {'top50%':>7s}")
    print("-" * 124)

    for model in MODELS:
        for task in TASKS:
            dag_path = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
            try:
                dag = torch.load(dag_path, weights_only=False)
            except Exception as e:
                print(f"{model}/{task}: load failed - {e}")
                continue
            if "act" not in dag:
                print(f"{model}/{task}: no 'act' key in DAG")
                continue
            act = dag["act"].cpu().numpy().reshape(-1)
            s = stats_for_act(act)
            summary["per_dag"][f"{model}/{task}"] = s
            print(f"{model:<20s}  {task:<12s}  {s['n_nonzero']:>6d}  "
                  f"{s['p50']:>9.3e}  {s['p99']:>9.3e}  {s['max']:>9.3e}  "
                  f"{s['ratio_max_p99']:>9.2f}  {s['ratio_max_p50']:>9.2f}  "
                  f"{s['frac_top5']*100:>6.1f}%  {s['frac_top50']*100:>6.1f}%")
        print()

    # Per-task cross-model summary: how does max(act) scale vary?
    print()
    print("Cross-model max(act) variation per task:")
    print(f"{'task':<14s}  {'min(max)':>11s}  {'median(max)':>13s}  "
          f"{'max(max)':>11s}  {'spread_ratio':>14s}")
    print("-" * 78)
    for task in TASKS:
        maxes = [summary["per_dag"][f"{m}/{task}"]["max"]
                 for m in MODELS
                 if f"{m}/{task}" in summary["per_dag"]]
        if not maxes:
            continue
        mn = float(min(maxes))
        med = float(np.median(maxes))
        mx = float(max(maxes))
        spread = mx / max(mn, 1e-30)
        summary["per_task"][task] = {
            "min_of_max": mn, "median_of_max": med, "max_of_max": mx,
            "spread_ratio": spread,
        }
        print(f"{task:<14s}  {mn:>11.3e}  {med:>13.3e}  {mx:>11.3e}  {spread:>14.2f}")
    print()

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
