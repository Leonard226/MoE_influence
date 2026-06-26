"""Diagnostic: is the 5-bin token-class histogram too coarse to discriminate?

Tests two claims:
  (Q1) One bin dominates       — if most mass lives in a single class, the
                                 5-bin resolution is mostly wasted.
  (Q2) Variation across models — if the per-bin fractions don't differ across
                                 models, class cannot discriminate routing
                                 policies regardless of resolution.

For each (model, task), we compute the LOAD-WEIGHTED model-level mean of the
per-vertex class histogram (the fraction of routed mass in each class) by:
  H_model[c] = (sum_v load(v) * class_hist(v, c)) / (sum_v load(v))
This is the overall routing-mass distribution over the 5 classes for that
(model, task). We then compare the 8 H_model vectors per task.

Outputs:
  results/circuits/feature_ablation/class_resolution.json
  printed table per task
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.fgw import build_triple, TOKEN_CLASSES  # noqa: E402


MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
TASKS = ["c4", "math", "code", "wikitext2", "gsm8k", "humaneval", "pile-arxiv", "pile-github"]
N_CLASSES = len(TOKEN_CLASSES)
CLASS_BLOCK_COLS = slice(5, 10)   # cols of F where class_hist lives (see fgw.py:324-340)


def _model_class_distribution(model: str, task: str, result_path: str) -> np.ndarray:
    """Load-weighted mean class histogram for one (model, task).
    Returns a length-5 numpy array summing to 1."""
    dag_path = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
    cls_path = Path(result_path) / "circuits" / "classifications" / f"classify_{model}_{task}.pkl"
    dag = torch.load(dag_path, weights_only=False)
    with open(cls_path, "rb") as f:
        classification = pickle.load(f)

    # We only need F and the mass vector here; C is discarded.
    _, F, mass, _ = build_triple(dag, classification, edge_threshold=0.0)
    # F is np.ndarray of shape (L*N, 10): cols 5-9 are the class histogram.
    class_block = np.asarray(F[:, CLASS_BLOCK_COLS])   # (L*N, 5)
    mass = np.asarray(mass)
    # Load-weighted average.
    w = mass / mass.sum() if mass.sum() > 0 else np.ones_like(mass) / len(mass)
    H = (w[:, None] * class_block).sum(axis=0)
    # Renormalise to handle the special-default rows (mass=0 vertices get class
    # 'special' by default; the weighting drops them but we re-check).
    if H.sum() > 0:
        H = H / H.sum()
    return H.astype(np.float64)


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "circuits" / "feature_ablation" / "class_resolution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Classes: {TOKEN_CLASSES}")
    print(f"Models : {MODELS}")
    print(f"Tasks  : {TASKS}\n")

    summary: dict = {"classes": TOKEN_CLASSES, "per_task": {}}

    for task in TASKS:
        H = np.zeros((len(MODELS), N_CLASSES), dtype=np.float64)
        for mi, model in enumerate(MODELS):
            try:
                H[mi] = _model_class_distribution(model, task, result_path)
            except Exception as e:
                print(f"  FAIL: {model}/{task}: {type(e).__name__}: {str(e)[:120]}")
                H[mi] = np.nan
        # Per-class summary across models.
        with np.errstate(invalid="ignore"):
            mean_per_class = np.nanmean(H, axis=0)
            std_per_class  = np.nanstd(H, axis=0)
            min_per_class  = np.nanmin(H, axis=0)
            max_per_class  = np.nanmax(H, axis=0)

        # Print model-level table for this task.
        print("=" * 86)
        print(f"Task: {task}")
        print("=" * 86)
        print(f"{'model':<20s}  " + "  ".join(f"{c:>10s}" for c in TOKEN_CLASSES) + "   ent")
        print("-" * 86)
        for mi, model in enumerate(MODELS):
            row = H[mi]
            ent = float(-np.sum(row * np.log(np.clip(row, 1e-12, None))) / np.log(N_CLASSES))
            print(f"{model:<20s}  " + "  ".join(f"{v:>10.4f}" for v in row)
                  + f"   {ent:>.3f}")
        print(f"{'mean':<20s}  " + "  ".join(f"{v:>10.4f}" for v in mean_per_class))
        print(f"{'std (across 8m)':<20s}  " + "  ".join(f"{v:>10.4f}" for v in std_per_class))
        print(f"{'max - min':<20s}  " + "  ".join(f"{v:>10.4f}" for v in (max_per_class - min_per_class)))
        print(f"{'dominant bin':<20s}  {TOKEN_CLASSES[int(np.argmax(mean_per_class))]:>10s}  "
              f"({mean_per_class.max():.3f})")
        print()

        summary["per_task"][task] = {
            "H_per_model":    {MODELS[mi]: H[mi].tolist() for mi in range(len(MODELS))},
            "mean_per_class": mean_per_class.tolist(),
            "std_per_class":  std_per_class.tolist(),
            "range_per_class": (max_per_class - min_per_class).tolist(),
            "dominant_bin": TOKEN_CLASSES[int(np.argmax(mean_per_class))],
            "dominant_fraction": float(mean_per_class.max()),
        }

    # Final summary.
    print("=" * 86)
    print("Across all tasks, the dominant bin and its fraction:")
    print("=" * 86)
    for task in TASKS:
        s = summary["per_task"][task]
        print(f"  {task:<14s}  dominant={s['dominant_bin']:<12s}  "
              f"frac={s['dominant_fraction']:.3f}  "
              f"max(std)={max(s['std_per_class']):.4f}")

    print()
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
