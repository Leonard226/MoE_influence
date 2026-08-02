"""Extract sorted load values per model (c4 task) for the load-distribution
visualisation (Figure for §3 of main.tex).

For each model on c4, we compute:
  - load_raw(v) = n_tok(v) / mean_in_layer(n_tok)
      Current definition; layer mean = 1, range up to N.
  - load_lognorm(v) = log(1 + load_raw(v)) / log(1 + max_n' load_raw(ell, n'))
      Per-layer log-max-normalisation; range [0, 1] per layer.

Both arrays are sorted descending and saved as JSON. Plotted by
experiments/plot_load_distribution.py locally.

Writes: results/distributions/load_curves_c4.json
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
    out_path = Path(result_path) / "distributions" / "load_curves_c4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict] = {}
    print(f"Extracting load curves on task='{TASK}'")
    print(f"{'model':<20s}  {'N':>4s}  {'n_active':>8s}  "
          f"{'raw_max':>10s}  {'lognorm_max':>12s}  {'lognorm_min':>12s}")
    print("-" * 80)

    for model in MODELS:
        dag_path = Path(result_path) / f"dag_{model}_{TASK}.pt"
        dag = torch.load(dag_path, weights_only=False)

        n_tok = dag["n_tokens_selected"].cpu().numpy().astype(np.float64)  # [L, N]
        L, N = n_tok.shape

        layer_mean = n_tok.mean(axis=1, keepdims=True).clip(min=1e-12)  # [L, 1]
        load_raw = n_tok / layer_mean                                    # [L, N]

        # Per-layer log-max normalisation.
        log_load = np.log1p(load_raw)                                    # [L, N]
        log_max  = np.log1p(load_raw.max(axis=1, keepdims=True)).clip(min=1e-12)  # [L, 1]
        load_lognorm = log_load / log_max                                # [L, N], in [0, 1]

        # Flatten, drop the very-rare zero-load vertices (unused experts).
        raw_flat = load_raw.reshape(-1)
        lognorm_flat = load_lognorm.reshape(-1)
        active = raw_flat > 0
        raw_pos = raw_flat[active]
        lognorm_pos = lognorm_flat[active]

        raw_sorted = np.sort(raw_pos)[::-1]      # descending
        lognorm_sorted = np.sort(lognorm_pos)[::-1]

        curves[model] = {
            "N": int(N),
            "L": int(L),
            "load_raw_sorted":     raw_sorted.tolist(),
            "load_lognorm_sorted": lognorm_sorted.tolist(),
        }
        print(f"{model:<20s}  {N:>4d}  {len(raw_pos):>8d}  "
              f"{raw_sorted[0]:>10.3f}  "
              f"{lognorm_sorted[0]:>12.4f}  {lognorm_sorted[-1]:>12.4f}")

    with open(out_path, "w") as f:
        json.dump(curves, f)
    print(f"\nSaved: {out_path}")
    size_kb = out_path.stat().st_size / 1024
    print(f"File size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
