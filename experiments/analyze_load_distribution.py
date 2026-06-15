"""Audit raw load distribution per (model, task).

For each DAG, computes the current definition
    load(v) = n_tok(v) / (1/N * sum_{n'} n_tok(ell, n'))
i.e. layer-mean-normalised token count (mean = 1 per layer, range [0, N]).

We report:
  - Per-layer max(load): how unbalanced is the heaviest expert in each layer
  - Per-layer top-1 mass fraction: what share of layer mass goes to the
    heaviest expert (1/N = uniform; 1 = degenerate)
  - Per-vertex load distribution: median, p99, max across the whole graph
  - max / p99 ratio: per-vertex tail heaviness (high = single super-expert)
  - Cross-model summary

Verdict logic:
  - If max_per_layer typically ≤ 2-3 and p99 ≈ max, the layer is near-balanced
    → simple max-normalisation (A) is sufficient
  - If max_per_layer often ≥ 5 and p99 << max, there is a heavy tail
    → log-max normalisation (B) is justified
  - If concentration (top-1 fraction) is close to 1/N for most layers,
    routing is balanced; if often >> 1/N, routing is unbalanced

Writes: results/circuits/feature_ablation/load_distribution.json
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
    """Compute load distribution summary for one DAG."""
    n_tok = dag["n_tokens_selected"].cpu().numpy().astype(np.float64)  # [L, N]
    L, N = n_tok.shape

    # Layer mean (with min clamp to match the fgw.py computation).
    layer_mean = n_tok.mean(axis=1, keepdims=True).clip(min=1e-12)  # [L, 1]
    load = n_tok / layer_mean  # [L, N], mean = 1 per layer, range [0, N]

    # Per-layer stats.
    max_per_layer = load.max(axis=1)  # [L]
    min_per_layer = load.min(axis=1)
    # Top-1 mass fraction per layer: fraction of layer's tokens to heaviest expert.
    layer_sums = n_tok.sum(axis=1).clip(min=1e-12)
    layer_max_tok = n_tok.max(axis=1)
    top1_frac = layer_max_tok / layer_sums  # [L]; uniform = 1/N

    # Per-vertex flat stats.
    load_flat = load.reshape(-1)
    load_pos = load_flat[load_flat > 0]
    if len(load_pos) == 0:
        return {"N": N, "L": L}

    p50 = float(np.percentile(load_pos, 50))
    p90 = float(np.percentile(load_pos, 90))
    p99 = float(np.percentile(load_pos, 99))
    p99_safe = max(p99, 1e-12)
    vmax = float(load_pos.max())

    return {
        "N": int(N),
        "L": int(L),
        "uniform_frac": float(1.0 / N),
        "load_p50": p50,
        "load_p90": p90,
        "load_p99": p99,
        "load_max": vmax,
        "max_over_p99": vmax / p99_safe,
        # Per-layer:
        "max_per_layer_median":  float(np.median(max_per_layer)),
        "max_per_layer_p90":     float(np.percentile(max_per_layer, 90)),
        "max_per_layer_max":     float(max_per_layer.max()),
        "min_per_layer_median":  float(np.median(min_per_layer)),
        # Top-1 layer concentration:
        "top1_frac_median":      float(np.median(top1_frac)),
        "top1_frac_p90":         float(np.percentile(top1_frac, 90)),
        "top1_frac_max":         float(top1_frac.max()),
    }


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "circuits" / "feature_ablation" / "load_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict = {}

    print(f"{'model':<20s}  {'task':<12s}  {'N':>4s}  {'unif':>6s}  "
          f"{'p50':>6s}  {'p99':>7s}  {'max':>7s}  "
          f"{'max/p99':>8s}  {'mxL_med':>8s}  {'mxL_p90':>8s}  "
          f"{'top1_med':>8s}  {'top1_p90':>8s}")
    print("-" * 130)

    for model in MODELS:
        for task in TASKS:
            dag_path = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
            try:
                dag = torch.load(dag_path, weights_only=False)
            except Exception as e:
                print(f"  {model}/{task}: load failed: {e}")
                continue
            s = audit_dag(dag)
            summary[f"{model}/{task}"] = s
            if "load_p50" not in s:
                continue
            print(f"{model:<20s}  {task:<12s}  {s['N']:>4d}  "
                  f"{s['uniform_frac']:>6.4f}  "
                  f"{s['load_p50']:>6.3f}  {s['load_p99']:>7.3f}  {s['load_max']:>7.3f}  "
                  f"{s['max_over_p99']:>8.2f}  "
                  f"{s['max_per_layer_median']:>8.3f}  {s['max_per_layer_p90']:>8.3f}  "
                  f"{s['top1_frac_median']:>8.3f}  {s['top1_frac_p90']:>8.3f}")
        print()

    # Per-model summary averaged across tasks.
    print("=" * 130)
    print("Per-model summary (averaged across 8 tasks):")
    print("=" * 130)
    print(f"{'model':<20s}  {'N':>4s}  {'unif':>6s}  "
          f"{'mxL_med_avg':>12s}  {'mxL_p90_avg':>12s}  "
          f"{'top1_med_avg':>13s}  {'top1_p90_avg':>13s}  "
          f"{'load_max_avg':>13s}  {'verdict':>14s}")
    print("-" * 130)
    per_model: dict = {}
    for model in MODELS:
        keys = [k for k in summary if k.startswith(f"{model}/") and "load_p50" in summary[k]]
        if not keys:
            continue
        N = summary[keys[0]]["N"]
        unif = 1.0 / N
        mxL_med = float(np.mean([summary[k]["max_per_layer_median"] for k in keys]))
        mxL_p90 = float(np.mean([summary[k]["max_per_layer_p90"]    for k in keys]))
        t1_med  = float(np.mean([summary[k]["top1_frac_median"]     for k in keys]))
        t1_p90  = float(np.mean([summary[k]["top1_frac_p90"]        for k in keys]))
        lmax    = float(np.mean([summary[k]["load_max"]             for k in keys]))

        # Verdict for normalisation choice:
        #   balanced     : per-layer max ~ 2x or less → simple max-norm is enough
        #   moderate     : per-layer max 2x-5x → either max-norm or log-max works
        #   heavy-tailed : per-layer max ≥ 5x → log-max recommended
        if mxL_p90 <= 2.0:
            verdict = "balanced"
        elif mxL_p90 <= 5.0:
            verdict = "moderate"
        else:
            verdict = "heavy-tailed"
        per_model[model] = {
            "N": N, "uniform_frac": unif,
            "max_per_layer_median_avg": mxL_med,
            "max_per_layer_p90_avg":    mxL_p90,
            "top1_frac_median_avg":     t1_med,
            "top1_frac_p90_avg":        t1_p90,
            "load_max_avg":             lmax,
            "verdict":                  verdict,
        }
        print(f"{model:<20s}  {N:>4d}  {unif:>6.4f}  "
              f"{mxL_med:>12.3f}  {mxL_p90:>12.3f}  "
              f"{t1_med:>13.4f}  {t1_p90:>13.4f}  "
              f"{lmax:>13.2f}  {verdict:>14s}")

    summary["_per_model"] = per_model

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final decision text.
    print()
    print("=" * 130)
    print("Decision support:")
    print("=" * 130)
    verdicts = [v["verdict"] for v in per_model.values()]
    n_bal = verdicts.count("balanced")
    n_mod = verdicts.count("moderate")
    n_hvy = verdicts.count("heavy-tailed")
    print(f"  balanced     ({n_bal}/{len(per_model)}): per-layer max load ≤ 2× the mean — routing is close to uniform")
    print(f"  moderate     ({n_mod}/{len(per_model)}): per-layer max load 2×-5× — some unbalance, manageable tail")
    print(f"  heavy-tailed ({n_hvy}/{len(per_model)}): per-layer max load ≥ 5× — log compression needed")
    print()
    if n_hvy >= 1:
        print("  >>> At least one model has heavy-tailed load. log-max (Option B) recommended.")
    elif n_mod == 0:
        print("  >>> All models near-balanced. Simple max-norm (Option A) is sufficient.")
        print("      log-max adds no value here.")
    else:
        print("  >>> Mixed. Either A or B works; A is simpler if you prefer that.")


if __name__ == "__main__":
    main()
