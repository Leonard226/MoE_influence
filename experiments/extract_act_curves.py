"""Extract sorted raw activation values per model (c4 task) for the
Zipf-style distribution plot.

For each model, loads the c4 DAG, extracts the per-vertex `act` values,
drops zeros (unused vertices), sorts descending, and saves the resulting
array as a JSON.

The output is read by experiments/plot_act_distribution.py locally to
render the per-model distribution figure.

Writes: results/circuits/feature_ablation/act_curves_c4.json
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
TASK = "c4"


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    out_path = Path(result_path) / "circuits" / "feature_ablation" / "act_curves_c4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curves: dict[str, list[float]] = {}
    print(f"Extracting raw act curves on task='{TASK}'")
    print(f"{'model':<20s}  {'n_nonzero':>10s}  {'min':>10s}  {'max':>10s}")
    print("-" * 60)
    for model in MODELS:
        dag_path = Path(result_path) / "circuits" / f"dag_{model}_{TASK}.pt"
        dag = torch.load(dag_path, weights_only=False)
        act = dag["act"].cpu().numpy().reshape(-1)
        a = act[act > 0].astype(np.float64)
        a_sorted = np.sort(a)[::-1]  # descending: rank 1 = max
        curves[model] = a_sorted.tolist()
        print(f"{model:<20s}  {len(a):>10d}  {a.min():>10.3e}  {a.max():>10.3e}")

    with open(out_path, "w") as f:
        json.dump(curves, f)
    print(f"\nSaved: {out_path}")
    size_kb = out_path.stat().st_size / 1024
    print(f"File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
