"""Extract per-vertex class-histogram summary statistics per model (c4 task)
for the 2-figure class-distribution motivation (Figures fig:class-specialization
and fig:class-diversity).

For each model on c4, computes class_hist[v] in R^5 via fgw.compute_class_histogram
(top-100 routed events per vertex, UPOS-mapped to 5 macro-classes, normalised to
sum=1 per vertex). Then derives:

  - max_prob_sorted     : sorted descending of max_c class_hist[v, c] over all
                          vertices. Range [0.2, 1]. Used for Figure A
                          (within-vertex specialization).
  - argmax_counts       : 5-vector, count of vertices whose argmax class is each
                          of {content, functional, punctuation, numeric, special}.
                          Used for Figure B (across-vertex diversity).
  - argmax_counts_specialized : same as above, restricted to vertices with
                                max_c >= 0.5 (the "honestly specialized" subset).

The "special" class catches unmapped tokens AND vertices with no routed events
(see compute_class_histogram fallback at fgw.py:185). To avoid that fallback
inflating the diversity count, vertices whose top_weight buffer is fully empty
are filtered out before all aggregations.

Writes: results/circuits/feature_ablation/class_curves_c4.json
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
from experiments.fgw import TOKEN_CLASSES, compute_class_histogram  # noqa: E402


MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
TASK = "c4"
SPECIALIZED_THRESHOLD = 0.5   # vertices with max_c >= this are "specialized"


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "circuits" / "feature_ablation" / "class_curves_c4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict] = {}
    print(f"Extracting class histogram summaries on task='{TASK}'")
    print(f"  classes: {TOKEN_CLASSES}")
    print(f"  specialized threshold: max_c >= {SPECIALIZED_THRESHOLD}\n")
    print(f"{'model':<20s}  {'n_active':>9s}  {'n_special':>10s}  "
          f"{'rank1_max':>10s}  {'median_max':>11s}")
    print("-" * 70)

    for model in MODELS:
        dag_path = Path(result_path) / "circuits" / f"dag_{model}_{TASK}.pt"
        cls_path = Path(result_path) / "circuits" / "classifications" / f"classify_{model}_{TASK}.pkl"
        dag = torch.load(dag_path, weights_only=False)
        with open(cls_path, "rb") as f:
            classification = pickle.load(f)

        hist = compute_class_histogram(
            dag["top_weight"], dag["top_prompt"], dag["top_pos"], classification,
        )                                                  # [L, N, 5]
        L, N, _ = hist.shape
        hist_flat = hist.reshape(L * N, -1).cpu().numpy()  # [L*N, 5]

        # Filter out vertices whose top_weight buffer is fully empty: those got
        # the "all mass on special" fallback (compute_class_histogram, line 185)
        # and are not real specialization signal.
        top_weight = dag["top_weight"].reshape(L * N, -1).cpu().numpy()
        any_routed = (top_weight > 0).any(axis=1)         # [L*N]
        hist_active = hist_flat[any_routed]               # [n_active, 5]
        n_active = int(any_routed.sum())

        max_prob = hist_active.max(axis=1)                # [n_active]
        argmax = hist_active.argmax(axis=1)               # [n_active]

        # Argmax counts over all active vertices.
        argmax_counts = np.bincount(argmax, minlength=len(TOKEN_CLASSES)).tolist()

        # Argmax counts restricted to specialized vertices.
        specialized = max_prob >= SPECIALIZED_THRESHOLD
        argmax_counts_specialized = np.bincount(
            argmax[specialized], minlength=len(TOKEN_CLASSES)
        ).tolist()

        max_prob_sorted = np.sort(max_prob)[::-1]         # descending

        # Diagnostic: how many vertices fell back to the "all special" sentinel
        # (these were already filtered).
        n_fallback_filtered = int((~any_routed).sum())

        curves[model] = {
            "n_vertices":              L * N,
            "n_active":                n_active,
            "n_fallback_filtered":     n_fallback_filtered,
            "specialized_threshold":   SPECIALIZED_THRESHOLD,
            "n_specialized":           int(specialized.sum()),
            "classes":                 TOKEN_CLASSES,
            "max_prob_sorted":         max_prob_sorted.tolist(),
            "argmax_counts":           argmax_counts,
            "argmax_counts_specialized": argmax_counts_specialized,
        }
        print(f"{model:<20s}  {n_active:>9d}  {n_fallback_filtered:>10d}  "
              f"{max_prob_sorted[0]:>10.4f}  "
              f"{float(np.median(max_prob_sorted)):>11.4f}")

    with open(out_path, "w") as f:
        json.dump(curves, f)
    print(f"\nSaved: {out_path}")
    size_kb = out_path.stat().st_size / 1024
    print(f"File size: {size_kb:.0f} KB")

    # Cross-model summary: argmax-class fractions per model.
    print("\nDominant-class composition across vertices (fraction; all active):")
    header = "  " + " " * 20 + "  " + "  ".join(f"{c[:9]:>9s}" for c in TOKEN_CLASSES) + f"  {'H[bits]':>8s}"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        d = curves[model]
        total = sum(d["argmax_counts"]) or 1
        frac = np.array(d["argmax_counts"]) / total
        # Entropy in bits; treat zero-fraction classes as 0 contribution.
        ent = float(-np.sum(np.where(frac > 0, frac * np.log2(frac), 0.0)))
        cells = "  ".join(f"{f:>9.3f}" for f in frac)
        print(f"  {model:<20s}  {cells}  {ent:>8.3f}")


if __name__ == "__main__":
    main()
