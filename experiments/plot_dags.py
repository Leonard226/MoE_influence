"""Edge-first routing-DAG visualisation, one layered graph per model.

Replaces the former experiments/spectral.ipynb (§1), which did the same thing
but carried its rendered output inside the .ipynb.

Sparsification: keep the forward edges whose |W_softmax| exceeds the EDGE_Q
quantile of all forward-edge magnitudes (i.e. the top 1 - EDGE_Q fraction).
No per-vertex threshold, no vertex-first pass. The SAME EDGE_Q is used for
every model; large models simply have far more edges clearing that quantile
(bigger L*N), so their survivors are additionally subsampled down to
--max-edges purely for legibility. The quantile threshold itself is never
touched by that subsampling. Subsampling is hybrid (see TOP_K_GUARANTEED):
the TOP_K_GUARANTEED largest-magnitude survivors are always kept -- so the
plot's max never silently disagrees with print_top_edges.py's true top-K --
and the rest of the cap is filled with a uniform random sample (fixed seed)
of the remainder, for a sense of the broader structure beyond the single
strongest cluster.

    W_softmax(v -> v') = E_i[|p_orig(v') - p_pert(v')|] under an ablation of v
                         (see build_dag.py / main.tex); bounded in [0, 1] --
                         except deepseek-v2, see COLOR_RANGE_OVERRIDE below.

Vertex handling: show_enhanced_layered_graph already drops isolated vertices
(degree == 0) from the drawing. No `is_super` vertex attribute is set, which
suppresses the gold/red highlighting inside the drawing routine -- every node
renders uniformly (white fill, black border). The colour axis is pinned to
[0, 1] so the same |w| renders the same shade in every model, except
deepseek-v2 (see COLOR_RANGE_OVERRIDE).

Shared experts (DeepSeek-V2/V2-Lite): if the dag has a W_softmax_shared entry
([L, L, N] -- sender layer x receiver layer x receiver expert, no
sender-expert axis since there's one shared-expert vertex per layer), it's
sparsified independently (own quantile pass -- it's a different statistical
quantity from W_softmax: an unconditional mean over all tokens, not
conditional on a selection event) and rendered as an extra light-green
row/column per layer, one slot past the last regular expert.

Vertex guarantee (see TOP_K_VERTICES/VERTEX_MASS_FRAC/VERTEX_MAX_EDGES): the
edge-first quantile cutoff can make a vertex with diffuse influence (many
modest outgoing edges, no single one in the global extreme tail) vanish from
the plot entirely, despite topping the out(v) ranking (main.tex "Global
Influence" / print_top_outs.py). The top-K vertices by out(v) always get a
representative slice of their own outgoing edges force-included -- the
smallest prefix (by magnitude) whose cumulative sum reaches VERTEX_MASS_FRAC
of that vertex's own out(v), capped at VERTEX_MAX_EDGES. These
guarantee-only edges (not already present via the quantile/subsample path)
render dashed, so the plot stays honest about why they're visible.

Reads:  {result_path}/dags/{task}/dag_{model}_{task}.pt
Writes: {result_path}/dag_visualizations/{model}_{task}_q{EDGE_Q}.pdf

Usage:
    python experiments/plot_dags.py
    python experiments/plot_dags.py --models mixtral-8x7b,olmoe
    python experiments/plot_dags.py --edge-q 0.9999
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
    sparsify_edges, sparsify_shared_edges, subsample_edges,
    thresholding_routing_graph, show_enhanced_layered_graph,
    guarantee_top_vertices,
)

TARGET = "W_softmax"
SHARED_TARGET = "W_softmax_shared"

ALL_MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "olmoe", "phi-3.5-moe",
    "deepseek-v2-lite", "deepseek-v2", "qwen3-30b-a3b", "qwen3-235b-a22b",
]

# Natural-casing display names for plot titles (filenames keep the raw key).
MODEL_DISPLAY = {
    "mixtral-8x7b": "Mixtral-8x7B",
    "mixtral-8x22b": "Mixtral-8x22B",
    "olmoe": "OLMoE",
    "phi-3.5-moe": "Phi-3.5-MoE",
    "deepseek-v2-lite": "DeepSeek-V2-Lite",
    "deepseek-v2": "DeepSeek-V2",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen3-235B-A22B",
}

# Models whose surviving edge set is too dense to read without subsampling.
LARGE_MODELS = {"deepseek-v2-lite", "deepseek-v2", "qwen3-30b-a3b", "qwen3-235b-a22b"}

# Fixed subsampling cap for all LARGE_MODELS, overriding --max-edges.
MAX_EDGES_OVERRIDE = {
    "deepseek-v2-lite": 500,
    "deepseek-v2": 500,
    "qwen3-30b-a3b": 500,
    "qwen3-235b-a22b": 500,
}

# Hybrid subsampling: this many of the largest-magnitude survivors are always
# kept (so the plot's max never silently disagrees with print_top_edges.py),
# with the rest of the cap filled by uniform random sample of the remainder.
TOP_K_GUARANTEED = 100

# Per-model colorbar (color_vmin, color_vmax) override. Default (0.0, 1.0) is
# shared across models for cross-model comparability. deepseek-v2 uses
# routing_scale=16 (main.tex "Scaled routing weight") -- its own gating
# weight genuinely isn't bounded [0,1] (there's no unscaled version anywhere
# in the real model to fall back to), so W_softmax for deepseek-v2 can
# legitimately exceed 1. (None, None) falls back to that graph's own
# min/max magnitude, giving proper shade differentiation among its edges
# instead of everything above 1 clipping to the same saturated color.
COLOR_RANGE_OVERRIDE = {
    "deepseek-v2": (None, None),
}

# Vertex guarantee: force-include a representative slice of each of the
# top-K out(v) vertices' own outgoing edges (see module docstring).
TOP_K_VERTICES = 5
VERTEX_MASS_FRAC = 0.5
VERTEX_MAX_EDGES = 30


def plot_one(model: str, task: str, out_dir: Path, edge_q: float,
             max_edges: int) -> None:
    dag_path = RESULTS / "dags" / task / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        print(f"{model}: MISSING {dag_path}, skipping")
        return

    # weights_only=False: the dag dict holds plain-Python entries (model
    # string, moe_layers list, dataset name) alongside the tensors.
    dag = torch.load(dag_path, map_location="cpu", weights_only=False)

    has_shared = SHARED_TARGET in dag

    W_e, einfo = sparsify_edges(dag[TARGET], edge_q=edge_q, edge_floor_frac=0.0)
    print(f"[EDGE] {model}: edges_kept={einfo['n_edges_kept']}/"
          f"{einfo['n_edges_total']}, t_edge={einfo['t_edge']:.4g}")

    if model in LARGE_MODELS:
        cap = MAX_EDGES_OVERRIDE.get(model, max_edges)
        W_e, sinfo = subsample_edges(W_e, max_edges=cap, seed=0, n_top=TOP_K_GUARANTEED)
        print(f"[SAMPLE] {model}: sampled {sinfo['n_edges_sampled']}/"
              f"{sinfo['n_edges_before_sample']} edges down to cap={cap} "
              f"(top={sinfo['n_top_kept']}, random={sinfo['n_random_fill']})")

    W_shared = None
    if has_shared:
        W_shared, shinfo = sparsify_shared_edges(dag[SHARED_TARGET], edge_q=edge_q, edge_floor_frac=0.0)
        print(f"[SHARED] {model}: edges_kept={shinfo['n_edges_kept']}/"
              f"{shinfo['n_edges_total']}, t_edge={shinfo['t_edge']:.4g}")
        if model in LARGE_MODELS:
            cap = MAX_EDGES_OVERRIDE.get(model, max_edges)
            W_shared, shsinfo = subsample_edges(W_shared, max_edges=cap, seed=0, n_top=TOP_K_GUARANTEED)
            print(f"[SHARED SAMPLE] {model}: sampled {shsinfo['n_edges_sampled']}/"
                  f"{shsinfo['n_edges_before_sample']} edges down to cap={cap} "
                  f"(top={shsinfo['n_top_kept']}, random={shsinfo['n_random_fill']})")

    # ---- Vertex guarantee: force-include a representative slice of the
    # top-out(v) vertices' own edges (see module docstring), regardless of
    # the quantile/subsample selection above.
    guarantee_mask, guarantee_mask_shared, ginfo = guarantee_top_vertices(
        dag, TARGET, shared_target=SHARED_TARGET if has_shared else None,
        top_k=TOP_K_VERTICES, mass_frac=VERTEX_MASS_FRAC, max_edges=VERTEX_MAX_EDGES,
    )
    for v in ginfo["vertices"]:
        print(f"[GUARANTEE] {model}: {v['vertex']} out(v)={v['out']:.4g} -> "
              f"kept {v['kept']} edges ({v['mass_frac_achieved']*100:.0f}% of its mass)")

    quantile_mask = W_e.abs() > 1e-9
    guarantee_only_mask = guarantee_mask & ~quantile_mask
    if int(guarantee_only_mask.sum()):
        print(f"[GUARANTEE] {model}: +{int(guarantee_only_mask.sum())} edges "
              f"added beyond quantile/subsample selection")
    W_e = torch.where(guarantee_mask | quantile_mask, dag[TARGET], torch.zeros_like(dag[TARGET]))
    dag["_vis_edge"] = W_e

    shared_target = None
    guarantee_only_mask_shared = None
    if has_shared:
        quantile_mask_shared = W_shared.abs() > 1e-9
        guarantee_only_mask_shared = guarantee_mask_shared & ~quantile_mask_shared
        if int(guarantee_only_mask_shared.sum()):
            print(f"[GUARANTEE] {model}: +{int(guarantee_only_mask_shared.sum())} "
                  f"shared edges added beyond quantile/subsample selection")
        W_shared = torch.where(guarantee_mask_shared | quantile_mask_shared,
                               dag[SHARED_TARGET], torch.zeros_like(dag[SHARED_TARGET]))
        dag["_vis_edge_shared"] = W_shared
        shared_target = "_vis_edge_shared"

    g = thresholding_routing_graph(dag, "_vis_edge", 1e-9,
                                   shared_target=shared_target, shared_threshold=1e-9,
                                   guarantee_mask=guarantee_only_mask,
                                   guarantee_mask_shared=guarantee_only_mask_shared)

    color_vmin, color_vmax = COLOR_RANGE_OVERRIDE.get(model, (0.0, 1.0))
    show_enhanced_layered_graph(
        g, quantile=edge_q,
        target=f"{TARGET}/EDGE-first",
        model=model, model_display=MODEL_DISPLAY.get(model, model),
        dataset=task, n_prompts=dag["n_prompts"],
        layer_labels=dag["moe_layers"],
        color_vmin=color_vmin, color_vmax=color_vmax,
        save_path=out_dir / f"{model}_{task}_q{edge_q}.pdf",
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
