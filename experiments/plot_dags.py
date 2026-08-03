"""Edge-first routing-DAG visualisation, one layered graph per model.

Replaces the former experiments/spectral.ipynb (§1), which did the same thing
but carried its rendered output inside the .ipynb.

Sparsification: keep the forward edges whose |W_softmax| exceeds the EDGE_Q
quantile of all forward-edge magnitudes (i.e. the top 1 - EDGE_Q fraction).
No per-vertex threshold, no vertex-first pass. The SAME EDGE_Q is used for
every model; large models simply have far more edges clearing that quantile
(bigger L*N), so their survivors are additionally subsampled down to
--max-edges (uniform random, fixed seed) purely for legibility. The quantile
threshold itself is never touched by that subsampling.

    W_softmax(v -> v') = E_i[|p_orig(v') - p_pert(v')|] under an ablation of v
                         (see build_dag.py / main.tex); bounded in [0, 1].

Vertex handling: show_enhanced_layered_graph already drops isolated vertices
(degree == 0) from the drawing. No `is_super` vertex attribute is set, which
suppresses the gold/red highlighting inside the drawing routine -- every node
renders uniformly (white fill, black border). The colour axis is pinned to
[0, 1] so the same |w| renders the same shade in every model.

Reads:  {result_path}/dags/{task}/dag_{model}_{task}.pt
Writes: {result_path}/dag_visualizations/{model}_W_softmax_EDGE-first_{task}_q{EDGE_Q}.pdf

Usage:
    python experiments/plot_routing_graphs.py
    python experiments/plot_routing_graphs.py --models mixtral-8x7b,olmoe
    python experiments/plot_routing_graphs.py --edge-q 0.9999
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
RESULTS = Path(CFG["result_path"])

from experiments.helper import (  # noqa: E402
    sparsify_edges, subsample_edges, thresholding_routing_graph,
    show_enhanced_layered_graph,
)

TARGET = "W_softmax"

ALL_MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "olmoe", "phi-3.5-moe",
    "deepseek-v2-lite", "deepseek-v2", "qwen3-30b-a3b", "qwen3-235b-a22b",
]

# Models whose surviving edge set is too dense to read without subsampling.
LARGE_MODELS = {"deepseek-v2", "qwen3-30b-a3b", "qwen3-235b-a22b"}


def plot_one(model: str, task: str, out_dir: Path, edge_q: float,
             max_edges: int) -> None:
    dag_path = RESULTS / "dags" / task / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        print(f"{model}: MISSING {dag_path}, skipping")
        return

    # weights_only=False: the dag dict holds plain-Python entries (model
    # string, moe_layers list, dataset name) alongside the tensors.
    dag = torch.load(dag_path, map_location="cpu", weights_only=False)

    W_e, einfo = sparsify_edges(dag[TARGET], edge_q=edge_q, edge_floor_frac=0.0)
    print(f"[EDGE] {model}: edges_kept={einfo['n_edges_kept']}/"
          f"{einfo['n_edges_total']}, t_edge={einfo['t_edge']:.4g}")

    if model in LARGE_MODELS:
        W_e, sinfo = subsample_edges(W_e, max_edges=max_edges, seed=0)
        print(f"[SAMPLE] {model}: sampled {sinfo['n_edges_sampled']}/"
              f"{sinfo['n_edges_before_sample']} edges down to cap={max_edges}")

    dag["_vis_edge"] = W_e
    g = thresholding_routing_graph(dag, "_vis_edge", 1e-9)

    show_enhanced_layered_graph(
        g, quantile=edge_q,
        target=f"{TARGET}/EDGE-first",
        model=model, dataset=task, n_prompts=dag["n_prompts"],
        layer_labels=dag["moe_layers"],
        color_vmin=0.0, color_vmax=1.0,
        save_path=out_dir / f"{model}_{TARGET}_EDGE-first_{task}_q{edge_q}.pdf",
    )
    plt.close("all")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(ALL_MODELS),
                   help="comma-separated model names")
    p.add_argument("--task", default="c4")
    p.add_argument("--edge-q", type=float, default=0.999,
                   help="quantile of forward-edge |W| kept (default: 0.999)")
    p.add_argument("--max-edges", type=int, default=300,
                   help="legibility cap for large models (default: 300)")
    p.add_argument("--out-dir", default=None,
                   help="default: {result_path}/dag_visualizations/")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / "dag_visualizations"
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        plot_one(model, args.task, out_dir, args.edge_q, args.max_edges)


if __name__ == "__main__":
    main()
