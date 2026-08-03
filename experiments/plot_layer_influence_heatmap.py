"""Per-model layer x layer influence heatmap from the saved DAG tensor.

build_dag.py saves the full pairwise edge tensor W_softmax[c,j,l,n] (sender
layer c, sender expert j, receiver layer l, receiver expert n) to
dags/{dataset}/dag_{model}_{dataset}.pt. This script collapses it down to a depth-only view:
for each (sending layer, receiving layer) pair, sum the raw out(v)-style
edge weight over every sender expert in the sending layer and every
receiver expert in the receiving layer --

    H[c, l] = sum_j sum_n W_softmax[c, j, l, n]  +  sum_n W_softmax_shared[c, l, n]

The second term folds in each sending layer's shared-expert vertex (main.tex
"Shared experts"), when the dag has a W_softmax_shared entry ([c, l, n] --
sender layer x receiver layer x receiver expert, no sender-expert axis since
there's one shared-expert vertex per layer, not N-many) -- currently
DeepSeek-V2 and DeepSeek-V2-Lite. Absent for models without shared experts,
in which case H is exactly the old regular-experts-only sum.

No normalization -- these are the raw accumulated W_softmax values. Entries
where l <= c are not causally valid edges (E = {(c,j)->(l,n): c<l}) and are
masked (rendered as plain white, distinct from genuinely-small-but-valid
values) rather than left at whatever the tensor happened to initialise them
to.

Also prints, per model, the top-5 SINGLE-expert out(v) values (W[c,j,:,:]
summed over every receiver) -- a cross-check against tab:topk-match-c4 /
tab:top-ins-c4's reported out(v) numbers. A heatmap cell only captures the
slice of one sender's influence landing in ONE receiving layer (and also
pools in every other sender of that layer), so it is expected and NOT a bug
for every individual cell to be well below a single expert's total out(v)
if that total is spread across many receiving layers.

One heatmap per model, each with its OWN color scale (raw out(v) magnitudes
span ~100x across models per tab:top-ins-c4, so a shared scale would wash
out all but the largest model); Blues colormap, light = low, dark = high.
y-axis = sending layer, x-axis = receiving layer, every layer index labeled,
both starting at 0; origin='lower' so valid (l>c) entries form the familiar
upper-triangle-above-the-diagonal shape. Light gray minor-tick gridlines
delineate cells against a white background.

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
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
RESULTS = Path(CFG["result_path"])

ALL_MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe", "olmoe",
    "deepseek-v2-lite", "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]


def layer_heatmap(model: str, dataset: str, out_dir: Path) -> None:
    dag_path = RESULTS / "dags" / dataset / f"dag_{model}_{dataset}.pt"
    if not dag_path.exists():
        print(f"{model}: MISSING {dag_path}, skipping")
        return

    d = torch.load(dag_path, map_location="cpu")
    W = d["W_softmax"]  # [c, j, l, n]
    L, N = W.shape[2], W.shape[1]
    H = W.sum(dim=(1, 3)).numpy()  # [c, l], raw accumulated out(v)

    if "W_softmax_shared" in d:
        H_shared = d["W_softmax_shared"].sum(dim=-1).numpy()  # [c, l]
        H = H + H_shared
        print(f"{model}: folded in shared-expert contribution to H[c,l]")

    # Cross-check: out(v) for a SINGLE expert is W[c,j,:,:].sum() -- the same
    # quantity reported in tab:topk-match-c4/tab:top-ins-c4. A heatmap cell
    # H[c,l] only captures the slice of that landing in ONE receiving layer,
    # and also pools in the other N-1 senders of layer c, so no single cell
    # has to reach a sender's full out(v) even when the aggregation is
    # correct. Printing the top single-expert out(v) values lets that be
    # checked directly against the paper's tables rather than assumed.
    per_expert_out = W.sum(dim=(2, 3))  # [c, j]
    flat = per_expert_out.flatten()
    topk = torch.topk(flat, min(5, flat.numel()))
    print(f"{model}: top-5 single-expert out(v) (for cross-check against the paper's tables):")
    for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        c, j = idx // N, idx % N
        print(f"    L{c}E{j}: out(v)={val:.4g}")

    mask = np.tril(np.ones((L, L), dtype=bool), k=0)  # l <= c: non-causal
    H_masked = np.ma.masked_array(H, mask=mask)

    cmap = plt.get_cmap("Blues").copy()  # light blue = low, dark blue = high
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(max(6, L * 0.35), max(5, L * 0.35)))
    ax.set_facecolor("white")
    im = ax.imshow(H_masked, cmap=cmap, origin="lower", aspect="equal")
    ax.set_xlabel("receiving layer")
    ax.set_ylabel("sending layer")
    ax.set_title(model)

    ax.set_xticks(range(L))
    ax.set_xticklabels(range(L), fontsize=6, rotation=90)
    ax.set_yticks(range(L))
    ax.set_yticklabels(range(L), fontsize=6)
    ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # make_axes_locatable ties the colorbar's height to the main axes' actual
    # displayed height (plain fig.colorbar(im, ax=ax) does not, and ends up
    # visibly taller/shorter than the matrix once aspect="equal" is applied).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.15)
    fig.colorbar(im, cax=cax, label="accumulated raw out(v)")
    fig.tight_layout()

    out_path = out_dir / f"{model}_layer_influence_heatmap_{dataset}.pdf"
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
                   help="Default: results/layer_influence_heatmaps/")
    args = p.parse_args()

    models = args.models.split(",") if args.models else ALL_MODELS
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / "layer_influence_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        layer_heatmap(model.strip(), args.dataset, out_dir)


if __name__ == "__main__":
    main()
