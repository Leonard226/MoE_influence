"""Extract sorted activation values per model (c4 task) for the
2-panel activation-distribution visualisation.

For each model on c4, computes:
  - act_raw(v) : raw activation magnitude from dag["act"], zeros dropped
                 (unused vertices). Range spans 3-5 orders of magnitude within
                 a graph and another 100-10000x across models.
  - act_lognorm(v) = log(1 + act_raw(v)) / log(1 + max(act_raw))
                 Global log-max normalisation matching fgw.py's "log_max"
                 mode. Range [0, 1]; preserves super-expert dominance shape.

Both arrays are sorted descending and saved as JSON. Plotted by
experiments/plot_act_distribution.py locally.

Writes: results/distributions/act_curves_c4.json
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
    out_path = Path(result_path) / "distributions" / "act_curves_c4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict] = {}
    print(f"Extracting act curves on task='{TASK}'")
    print(f"{'model':<20s}  {'n_active':>10s}  "
          f"{'raw_max':>11s}  {'raw_min':>11s}  "
          f"{'lognorm_max':>12s}  {'lognorm_min':>12s}")
    print("-" * 88)
    for model in MODELS:
        dag_path = Path(result_path) / f"dag_{model}_{TASK}.pt"
        dag = torch.load(dag_path, weights_only=False)
        act = dag["act"].cpu().numpy().reshape(-1).astype(np.float64)

        active = act > 0
        a = act[active]

        # Global log-max normalisation: matches fgw.py "log_max" path.
        a_max = float(a.max())
        log_max = float(np.log1p(a_max))
        log_max = log_max if log_max > 1e-12 else 1e-12
        a_lognorm = np.log1p(a) / log_max     # in [0, 1]

        a_sorted        = np.sort(a)[::-1]            # descending
        a_lognorm_sorted = np.sort(a_lognorm)[::-1]   # descending

        curves[model] = {
            "n_active":            int(a.size),
            "n_total":             int(act.size),
            "act_raw_sorted":      a_sorted.tolist(),
            "act_lognorm_sorted":  a_lognorm_sorted.tolist(),
        }
        print(f"{model:<20s}  {a.size:>10d}  "
              f"{a_sorted[0]:>11.3e}  {a_sorted[-1]:>11.3e}  "
              f"{a_lognorm_sorted[0]:>12.4f}  {a_lognorm_sorted[-1]:>12.4f}")

    with open(out_path, "w") as f:
        json.dump(curves, f)
    print(f"\nSaved: {out_path}")
    size_kb = out_path.stat().st_size / 1024
    print(f"File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
