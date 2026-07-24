"""Per-model layer x layer influence heatmap from the saved DAG tensor.

build_dag.py saves the full pairwise edge tensor W_softmax[c,j,l,n] (sender
layer c, sender expert j, receiver layer l, receiver expert n) to
dag_{model}_{dataset}.pt. This script collapses it down to a depth-only view:
for each (sending layer, receiving layer) pair, sum the raw out(v)-style
edge weight over every sender expert in the sending layer and every
receiver expert in the receiving layer --

    H[c, l] = sum_j sum_n W_softmax[c, j, l, n]

No normalization -- these are the raw accumulated W_softmax values. Entries
where l <= c are not causally valid edges (E = {(c,j)->(l,n): c<l}) and are
masked (rendered as light gray, distinct from genuinely-small-but-valid
values) rather than left at whatever the tensor happened to initialise them
to.

One heatmap per model, each with its OWN color scale (raw out(v) magnitudes
span ~100x across models per tab:top-ins-c4, so a shared scale would wash
out all but the largest model). y-axis = sending layer, x-axis = receiving
layer, both starting at 0; origin='lower' so valid (l>c) entries form the
familiar upper-triangle-above-the-diagonal shape.

Usage:
    python experiments/plot_layer_influence_heatmap.py
    python experiments/plot_layer_influence_heatmap.py --models mixtral-8x7b,olmoe
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
CIRCUITS = Path(CFG["result_path"]) / "circuits"

ALL_MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe", "olmoe",
    "deepseek-v2-lite", "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]


def layer_heatmap(model: str, dataset: str, out_dir: Path) -> None:
    dag_path = CIRCUITS / f"dag_{model}_{dataset}.pt"
    if not dag_path.exists():
        print(f"{model}: MISSING {dag_path}, skipping")
        return

    d = torch.load(dag_path, map_location="cpu")
    W = d["W_softmax"]  # [c, j, l, n]
    L = W.shape[2]
    H = W.sum(dim=(1, 3)).numpy()  # [c, l], raw accumulated out(v)

    mask = np.tril(np.ones((L, L), dtype=bool), k=0)  # l <= c: non-causal
    H_masked = np.ma.masked_array(H, mask=mask)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H_masked, cmap=cmap, origin="lower", aspect="equal")
    ax.set_xlabel("receiving layer")
    ax.set_ylabel("sending layer")
    ax.set_title(f"{model}: accumulated out(v) by (sending, receiving) layer")
    fig.colorbar(im, ax=ax, label="accumulated raw out(v)")
    fig.tight_layout()

    out_path = out_dir / f"layer_influence_heatmap_{model}_{dataset}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"{model}: saved {out_path} (L={L})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=None,
                   help="Comma-separated model keys (default: all 8).")
    p.add_argument("--dataset", default="c4")
    p.add_argument("--out-dir", default=None,
                   help="Default: results/circuits/layer_influence_heatmaps/")
    args = p.parse_args()

    models = args.models.split(",") if args.models else ALL_MODELS
    out_dir = Path(args.out_dir) if args.out_dir else CIRCUITS / "layer_influence_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        layer_heatmap(model.strip(), args.dataset, out_dir)


if __name__ == "__main__":
    main()
