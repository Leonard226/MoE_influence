from pathlib import Path

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx

def update_topk_per_sender(top_weight, top_prompt, top_pos, top_token,
                           sender_j, weight, prompt_idx, pos, token_id,
                           n_experts, k_top, max_per_j):
    """For one layer, merge new events into a per-sender-j top-K-by-weight buffer.

    All buffers are updated in place; on entry, empty slots have `top_weight = -1`
    (any value < 0 works; routing weights are in [0, 1]).

    top_weight: [n_experts, k_top] float32 buffer of top-k weights per sender j
    top_prompt: [n_experts, k_top] int32   global prompt index of each top event
    top_pos:    [n_experts, k_top] int16   token position within the prompt
    top_token:  [n_experts, k_top] int32   token id at that position
    sender_j:   [n_events]  long           selected expert per event
    weight:     [n_events]  float32        routing weight per event
    prompt_idx: [n_events]  int32
    pos:        [n_events]  int16
    token_id:   [n_events]  int32
    max_per_j: upper bound on events per j in this call (e.g. bt = bsz*n_tok,
        since for a given token expert j can be in top-K at most once).
    """
    import torch
    n_events = sender_j.shape[0]
    if n_events == 0:
        return
    device = sender_j.device

    # Rank-within-j-group for each event via stable sort + cumcount.
    sort_idx = torch.argsort(sender_j, stable=True)
    sorted_j = sender_j[sort_idx]
    new_block = torch.empty_like(sorted_j, dtype=torch.bool)
    new_block[0] = True
    new_block[1:] = sorted_j[1:] != sorted_j[:-1]
    block_start = torch.where(new_block)[0]
    block_id = new_block.long().cumsum(0) - 1
    rank_sorted = torch.arange(n_events, device=device) - block_start[block_id]
    rank_orig = torch.empty_like(rank_sorted)
    rank_orig[sort_idx] = rank_sorted

    # Scatter events into [n_experts, max_per_j] padded candidate tensors.
    cand_w   = torch.full((n_experts, max_per_j), -1.0, dtype=top_weight.dtype, device=device)
    cand_p   = torch.zeros((n_experts, max_per_j), dtype=top_prompt.dtype, device=device)
    cand_pos = torch.zeros((n_experts, max_per_j), dtype=top_pos.dtype,    device=device)
    cand_t   = torch.zeros((n_experts, max_per_j), dtype=top_token.dtype,  device=device)
    cand_w  [sender_j, rank_orig] = weight
    cand_p  [sender_j, rank_orig] = prompt_idx
    cand_pos[sender_j, rank_orig] = pos
    cand_t  [sender_j, rank_orig] = token_id

    # Concat existing buffer + new candidates, take per-row top-K.
    combined_w   = torch.cat([top_weight, cand_w  ], dim=1)
    combined_p   = torch.cat([top_prompt, cand_p  ], dim=1)
    combined_pos = torch.cat([top_pos,    cand_pos], dim=1)
    combined_t   = torch.cat([top_token,  cand_t  ], dim=1)
    topk = combined_w.topk(k_top, dim=1)
    top_weight.copy_(topk.values)
    top_prompt.copy_(torch.gather(combined_p,   1, topk.indices))
    top_pos.copy_(   torch.gather(combined_pos, 1, topk.indices))
    top_token.copy_( torch.gather(combined_t,   1, topk.indices))


def sparsify_super_vertex(W, vertex_q: float = 0.995,
                          vertex_floor_frac: float = 0.4,
                          edge_floor_frac: float = 0.1):
    """Vertex-first sparsification (SE double-criterion + per-vertex edge floor).

    Stage 1: identify super-vertices by out-strength. A vertex (c, j) is super
        iff out_strength[c, j] > max(P_q(out_strength), vertex_floor_frac * max).
    Stage 2: for each super-vertex, keep its outgoing edges with magnitude
        >= edge_floor_frac * (that vertex's own max outgoing edge). The SE max/10
        floor applied per-sender (not globally) guarantees every super-vertex
        contributes at least one visible edge.

    Args:
        W: [L, N, L, N] edge tensor (sender_layer, sender_expert, recv_layer, recv_expert).
        vertex_q: percentile used for the vertex-level SE criterion.
        vertex_floor_frac: fraction of max out-strength used as the vertex floor.
        edge_floor_frac: fraction of each super-vertex's max outgoing edge.

    Returns:
        W_filtered: same shape as W, with non-surviving entries zeroed.
        super_mask: [L, N] bool, True for super-vertices.
        info: dict of diagnostic stats.
    """
    import torch
    L, N = W.shape[0], W.shape[1]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_abs = torch.abs(W.float())

    # (1) Super-vertex set.
    out_strength = (W_abs * fwd.float()).sum(dim=(2, 3))                  # [L, N]
    os_vals = out_strength[out_strength > 1e-9].cpu().numpy()
    t_vertex = max(float(np.quantile(os_vals, vertex_q)),
                   float(vertex_floor_frac * os_vals.max()))
    super_mask = out_strength > t_vertex                                  # [L, N] bool

    # (2) Per-vertex edge floor.
    W_super = W * super_mask.unsqueeze(-1).unsqueeze(-1)
    W_super_abs = torch.abs(W_super.float()) * fwd.float()
    per_sender_max = W_super_abs.flatten(2).max(dim=-1).values            # [L, N]
    per_sender_thr = per_sender_max * edge_floor_frac
    keep_mask = (W_super_abs >= per_sender_thr.unsqueeze(-1).unsqueeze(-1)) \
                & fwd & (W_super_abs > 1e-9)
    W_filtered = torch.where(keep_mask, W_super, torch.zeros_like(W_super))

    info = {
        "n_super": int(super_mask.sum().item()),
        "n_edges_kept": int(keep_mask.sum().item()),
        "t_vertex": t_vertex,
        "per_sender_max_min": float(per_sender_max[super_mask].min().item()) if super_mask.any() else 0.0,
        "per_sender_max_max": float(per_sender_max[super_mask].max().item()) if super_mask.any() else 0.0,
    }
    return W_filtered, super_mask, info


def filter_to_paths(g, min_length: int = 2):
    """Return a copy of `g` with only edges that participate in a directed
    path of length >= `min_length` edges.

    Use this on the edge-first sparsified graph to suppress isolated single
    edges and surface true circuits (chains of sequential routing decisions).

    Algorithm (assumes `g` is a DAG, which our routing graphs are by
    construction — only forward edges, sender_layer < receiver_layer):
      1. Topological sort.
      2. in_path[v]  = longest path (in edges) ending at v.
      3. out_path[v] = longest path (in edges) starting at v.
      4. Edge u->v is part of a length-L path iff
            in_path[u] + 1 + out_path[v] >= min_length.

    Preserves all vertex attributes (e.g. 'is_super', 'layer').

    Returns:
        (g_filtered, max_path_len) where max_path_len is the length (in edges)
        of the longest path in the *original* graph. Use it as a sanity check
        when picking min_length — a min_length above max_path_len drops every
        edge.
    """
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    n = g.vcount()
    topo = g.topological_sorting()  # forward topological order

    in_path = [0] * n
    for v in topo:
        for u in g.predecessors(v):
            cand = in_path[u] + 1
            if cand > in_path[v]:
                in_path[v] = cand

    out_path = [0] * n
    for v in reversed(topo):
        for w in g.successors(v):
            cand = out_path[w] + 1
            if cand > out_path[v]:
                out_path[v] = cand

    edges_to_keep = [
        e.index for e in g.es
        if in_path[e.source] + 1 + out_path[e.target] >= min_length
    ]
    max_path_len = max(in_path) if in_path else 0
    return g.subgraph_edges(edges_to_keep, delete_vertices=False), max_path_len


def sparsify_edges(W, edge_q: float = 0.9999, edge_floor_frac: float = 0.1):
    """Edge-first sparsification (global SE criterion on edge magnitudes).

    Keep edge iff |W| >= max(P_q(|forward edges|), edge_floor_frac * max(|forward edges|)).
    No per-vertex consideration; the strongest edges anywhere in the graph
    survive. Anchors on connections rather than nodes — naturally surfaces
    chains/cascades.

    Args:
        W: [L, N, L, N] edge tensor.
        edge_q: percentile used for the edge-level SE criterion.
        edge_floor_frac: fraction of global max used as the floor.

    Returns:
        W_filtered: same shape as W, with non-surviving entries zeroed.
        info: dict of diagnostic stats.
    """
    import torch
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_abs = torch.abs(W.float())

    edge_vals = W_abs[fwd]
    edge_vals = edge_vals[edge_vals > 1e-9].cpu().numpy()
    t_edge = max(float(np.quantile(edge_vals, edge_q)),
                 float(edge_floor_frac * edge_vals.max()))

    keep_mask = (W_abs >= t_edge) & fwd
    W_filtered = torch.where(keep_mask, W, torch.zeros_like(W))

    info = {
        "n_edges_total": int(edge_vals.size),
        "n_edges_kept": int(keep_mask.sum().item()),
        "t_edge": t_edge,
        "edge_max": float(edge_vals.max()),
    }
    return W_filtered, info


def sparsify_shared_edges(W, edge_q: float = 0.9999, edge_floor_frac: float = 0.1):
    """Edge-first sparsification for shared-expert edges (see sparsify_edges).

    Same criterion as sparsify_edges, but for the shared-expert edge tensor,
    which has no sender-expert axis (one shared-expert vertex per layer, not
    N_EXPERTS-many).

    Args:
        W: [L, L, N] tensor -- sender layer x receiver layer x receiver expert.
        edge_q, edge_floor_frac: see sparsify_edges.

    Returns:
        W_filtered: same shape as W, with non-surviving entries zeroed.
        info: dict of diagnostic stats.
    """
    import torch
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1)
    r_idx = torch.arange(L).view(1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_abs = torch.abs(W.float())

    edge_vals = W_abs[fwd]
    edge_vals = edge_vals[edge_vals > 1e-9].cpu().numpy()
    t_edge = max(float(np.quantile(edge_vals, edge_q)),
                 float(edge_floor_frac * edge_vals.max()))

    keep_mask = (W_abs >= t_edge) & fwd
    W_filtered = torch.where(keep_mask, W, torch.zeros_like(W))

    info = {
        "n_edges_total": int(edge_vals.size),
        "n_edges_kept": int(keep_mask.sum().item()),
        "t_edge": t_edge,
        "edge_max": float(edge_vals.max()),
    }
    return W_filtered, info


def subsample_edges(W_filtered, max_edges: int, seed: int = 0):
    """If sparsify_edges left more than max_edges surviving entries, keep a
    uniform random sample of exactly max_edges of them (fixed seed,
    reproducible) and zero the rest. Legibility-only: the quantile threshold
    itself already decided which edges qualify, this just controls how many
    of the (already equally-qualifying) survivors get drawn.

    Works for any tensor rank (4D W_softmax or 3D W_softmax_shared).

    Args:
        W_filtered: output of sparsify_edges/sparsify_shared_edges (same
            shape, non-surviving entries already zeroed).
        max_edges: cap on the number of nonzero entries to keep.
        seed: RNG seed for the sample, for reproducible figures.

    Returns:
        W_sampled: same shape as W_filtered.
        info: dict with n_edges_before_sample / n_edges_sampled.
    """
    import torch
    nz = torch.nonzero(W_filtered, as_tuple=False)
    n = nz.shape[0]
    if n <= max_edges:
        return W_filtered, {"n_edges_before_sample": n, "n_edges_sampled": n}

    g = torch.Generator().manual_seed(seed)
    keep = nz[torch.randperm(n, generator=g)[:max_edges]]
    idx = tuple(keep[:, i] for i in range(keep.shape[1]))
    W_sampled = torch.zeros_like(W_filtered)
    W_sampled[idx] = W_filtered[idx]
    return W_sampled, {"n_edges_before_sample": n, "n_edges_sampled": max_edges}


def get_thresholds(dag: dict, target: str, quantiles: list) -> list:
    import torch

    matrix = dag[target]
    N_LAYERS = matrix.shape[0]

    # Create a mask for forward edges (Layer S < Layer R)
    # This ensures we don't include invalid backward connections in our distribution
    s_idx = torch.arange(N_LAYERS).view(-1, 1, 1, 1)
    r_idx = torch.arange(N_LAYERS).view(1, 1, -1, 1)
    mask = (s_idx < r_idx).expand_as(matrix)    
    # Flatten valid weights
    valid_weights = torch.abs(matrix[mask].float())
    valid_weights = valid_weights[valid_weights > 1e-9]

    # torch.quantile errors out above ~16M elements (Qwen3-235B-A22B has ~72M
    # forward edges). numpy.quantile has no such cap.
    thresholds = np.quantile(valid_weights.cpu().numpy(), quantiles)

    # Return a dictionary mapping quantile -> threshold
    return dict(zip(quantiles, thresholds.tolist()))


def thresholding_routing_graph(dag: dict, target: str, threshold: float,
                               shared_target: str | None = None,
                               shared_threshold: float | None = None) -> ig.Graph:
    """Build the igraph DAG from a (sparsified) W_softmax tensor.

    shared_target: key into `dag` for the shared-expert edge tensor
        ([L, L, N] -- no sender-expert axis, one shared-expert vertex per
        layer). If given (and present in `dag`), one extra vertex is added
        per layer at expert-slot index N_EXPERTS (the slot right after the
        last regular expert), tagged `is_shared=True`, with outgoing-only
        edges thresholded at `shared_threshold` (defaults to `threshold`).
    """
    import numpy as np
    # Get the 4D matrix (Shape: [16, 64, 16, 64])
    matrix = dag[target]
    N_LAYERS, N_EXPERTS = matrix.shape[0], matrix.shape[1]

    has_shared = shared_target is not None and shared_target in dag
    N_SLOTS = N_EXPERTS + 1 if has_shared else N_EXPERTS
    N_NODES = N_LAYERS * N_SLOTS

    # Find where the weights are above the threshold
    s_layers, s_exps, r_layers, r_exps = np.where(np.abs(matrix) > threshold)

    # Convert those coordinates into Vertex IDs
    senders = s_layers * N_SLOTS + s_exps
    receivers = r_layers * N_SLOTS + r_exps

    # Extract the weights for these specific edges
    weights = matrix[s_layers, s_exps, r_layers, r_exps]
    edges = list(zip(senders.tolist(), receivers.tolist()))
    edge_weights = weights.tolist()

    if has_shared:
        shared_matrix = dag[shared_target]
        s_thresh = threshold if shared_threshold is None else shared_threshold
        sl_layers, sr_layers, sr_exps = np.where(np.abs(shared_matrix) > s_thresh)
        shared_senders = sl_layers * N_SLOTS + N_EXPERTS  # this layer's shared slot
        shared_receivers = sr_layers * N_SLOTS + sr_exps
        shared_weights = shared_matrix[sl_layers, sr_layers, sr_exps]
        edges += list(zip(shared_senders.tolist(), shared_receivers.tolist()))
        edge_weights += shared_weights.tolist()

    # Build the graph
    g = ig.Graph(directed=True, n=N_NODES)

    # zip pairs them up: [(s1, r1), (s2, r2), ...]
    g.add_edges(edges)

    # Assign the weights and metadata
    g.es["weight"] = edge_weights
    g.vs["layer"] = [v // N_SLOTS for v in range(N_NODES)]
    g.vs["expert"] = [v % N_SLOTS for v in range(N_NODES)]
    if has_shared:
        g.vs["is_shared"] = [(v % N_SLOTS) == N_EXPERTS for v in range(N_NODES)]

    return g


def show_enhanced_layered_graph(g, quantile: float, target: str, model: str, dataset: str, n_prompts: int,
                                 model_display: str | None = None,
                                 layer_labels: list | None = None,
                                 color_vmin: float | None = None,
                                 color_vmax: float | None = None,
                                 save_path: str | Path | None = None) -> None:
    """Layered DAG visualization. Reads N_LAYERS / N_EXPERTS from the graph's
    `layer` vertex attribute (set by thresholding_routing_graph / dag_to_igraph).

    quantile: cosmetic only -- a number printed in the plot title's "Threshold:"
        line. Has no effect on what gets drawn. Vestigial; pass anything informative.
    model_display: name shown in the plot title (natural casing, e.g.
        "Phi-3.5-MoE"). Defaults to `model` as-is if not given.
    layer_labels: optional mapping from internal DAG layer index (0..N_LAYERS-1)
        to the model's actual layer number. Use this when the DAG skips dense
        layers (e.g. DeepSeek-V2-Lite has dense layer 0, so internal M0 == model
        layer 1). If None, internal indices are used as-is. Pass dag["moe_layers"].
    color_vmin, color_vmax: fix the colorbar to this absolute range so the same
        edge magnitude looks the same shade across different models. For P_flip
        pass (0.0, 1.0). If None, defaults to the per-graph min/max magnitude
        (legacy behavior; makes cross-graph comparison harder).
    save_path: if given, the figure is also written here (format inferred from
        the extension, e.g. .pdf) before being shown. Parent directories are
        created if needed. If None (default), the figure is only displayed.

    Shared-expert vertices (graph's "is_shared" attribute, set by
    thresholding_routing_graph) render as an extra row/column past the last
    regular expert -- one per layer -- filled light green instead of the
    usual white/pale-blue receiver/sender scheme.
    """
    edge_list = g.get_edgelist()
    if not edge_list:
        print("No edges found to plot!")
        return

    # Get signed values for the title
    raw_weights = g.es["weight"]
    max_w = max(raw_weights)
    min_w = min(raw_weights)

    # Absolute values for visual scaling
    abs_weights = [abs(w) for w in raw_weights]
    max_mag, min_mag = max(abs_weights), min(abs_weights)

    # --- SPARSITY CALCULATIONS ---
    N_LAYERS = max(g.vs["layer"]) + 1
    N_SLOTS = g.vcount() // N_LAYERS  # per-layer width; includes the shared slot, if any
    has_is_shared = "is_shared" in g.vertex_attributes()
    N_EXPERTS = N_SLOTS - 1 if has_is_shared else N_SLOTS  # regular (routed) experts only
    # Map internal DAG layer index -> model layer number used for display.
    if layer_labels is None:
        layer_labels = list(range(N_LAYERS))
    TOTAL_POSSIBLE_NODES = N_LAYERS * N_EXPERTS
    # Max possible edges in a layered DAG (Layer i to Layer >i)
    TOTAL_POSSIBLE_EDGES = sum(N_EXPERTS * ((N_LAYERS - 1 - i) * N_EXPERTS) for i in range(N_LAYERS - 1))

    # Active = has an incident edge OR is flagged as a super-expert. The latter
    # ensures super-experts that had all their outgoing edges filtered by the
    # per-edge threshold still get rendered (as isolated gold nodes).
    has_is_super = "is_super" in g.vertex_attributes()
    active_node_indices = [
        v.index for v in g.vs
        if v.degree() > 0 or (has_is_super and v["is_super"])
    ]
    n_nodes_used = len(active_node_indices)
    n_edges_used = g.ecount()

    node_sparsity = (n_nodes_used / TOTAL_POSSIBLE_NODES) * 100
    edge_sparsity = (n_edges_used / TOTAL_POSSIBLE_EDGES) * 100

    # 1. Build NetworkX Graph
    G = nx.DiGraph()
    pos, labels = {}, {}

    # Orientation is chosen by shape: DEEP models (more layers than experts)
    # are drawn left-to-right (layers on x-axis); WIDE models (more experts
    # than layers) are drawn top-to-bottom (experts on x-axis). Keeps every
    # plot roughly landscape-ish rather than exploding one dimension.
    is_deep = N_LAYERS > N_EXPERTS
    X_SPACING, Y_SPACING = 1000, 300
    for node_idx in active_node_indices:
        layer, expert_idx = node_idx // N_SLOTS, node_idx % N_SLOTS
        if is_deep:
            pos[node_idx] = (layer * X_SPACING, -expert_idx * Y_SPACING)
        else:
            pos[node_idx] = (expert_idx * X_SPACING, -layer * Y_SPACING)
        if has_is_shared and g.vs[node_idx]["is_shared"]:
            labels[node_idx] = f"L{layer_labels[layer]}\nS"
        else:
            labels[node_idx] = f"L{layer_labels[layer]}\nE{expert_idx}"
        G.add_node(node_idx)

    # --- COLOR LOGIC ---
    if target.upper() in ["AVG"]:
        cmap = plt.cm.RdBu
        # Signed colormap centered at 0. Caller-provided color_vmax overrides
        # the per-graph symmetric range so cross-model plots share a scale.
        color_lim = color_vmax if color_vmax is not None else max(abs(max_w), abs(min_w))
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-color_lim, vmax=color_lim)
        cbar_label = "Inhibition vs Promotion"
    else:
        colors_array = plt.cm.Reds(np.linspace(0.35, 1.0, 256))
        cmap = mcolors.LinearSegmentedColormap.from_list('IntenseReds', colors_array)
        cmin = color_vmin if color_vmin is not None else min_mag
        cmax = color_vmax if color_vmax is not None else max_mag
        norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
        cbar_label = "Edge weight magnitude |W|"

    # Width normalization: use the same fixed range as color when the caller
    # provides one, so an edge of magnitude X renders at the same thickness in
    # every model. Falls back to per-graph min/max for backward compatibility.
    width_min = color_vmin if color_vmin is not None else min_mag
    width_max = color_vmax if color_vmax is not None else max_mag

    edge_colors, edge_widths = [], []
    for e in g.es:
        u, v = e.source, e.target
        G.add_edge(u, v)
        w = e["weight"]

        val_for_color = w if target.upper() in ["AVG"] else abs(w)
        edge_colors.append(cmap(norm(val_for_color)))

        w_norm = (abs(w) - width_min) / (width_max - width_min + 1e-9)
        w_norm = max(0.0, min(1.0, w_norm))  # clamp in case |w| sits outside [vmin, vmax]
        edge_widths.append(1.2 + (w_norm * 4.3))

    # --- DRAWING ---
    # Circle sizes in points^2 (matplotlib scatter marker-area convention).
    # Defined up-front so the figure sizing below can derive the required
    # per-unit spacing directly from the diameter.
    NODE_SIZE = 700
    SUPER_NODE_SIZE = 950

    # Node diameter in inches. Marker area = pi * r^2 (points^2), so
    # diameter_pt = 2 * sqrt(area / pi), and 1 pt = 1/72 inch.
    _max_marker_area = SUPER_NODE_SIZE if has_is_super else NODE_SIZE
    _node_diameter_in = 2 * np.sqrt(_max_marker_area / np.pi) / 72

    # Center-to-center spacing (inches) between rendered nodes. Adjacent
    # experts get the tighter 1.00x spacing (nodes just touch); adjacent
    # layers get 1.03x for a hairline visible gap. Which axis gets which
    # multiplier follows the orientation chosen above.
    _spacing_layer_in = _node_diameter_in * 1.03
    _spacing_expert_in = _node_diameter_in * 1.00

    # Deep -> x is layers, y is experts. Wide -> x is experts, y is layers.
    # Margins reserve space for title (top), x/y-axis labels + ticks, and
    # the colorbar (right).
    MARGIN_W, MARGIN_H = 4.5, 3.5
    if is_deep:
        fig_w = max(6.0, N_LAYERS * _spacing_layer_in + MARGIN_W)
        fig_h = max(5.0, N_SLOTS * _spacing_expert_in + MARGIN_H)
    else:
        fig_w = max(6.0, N_SLOTS * _spacing_expert_in + MARGIN_W)
        fig_h = max(5.0, N_LAYERS * _spacing_layer_in + MARGIN_H)
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    _model_str = model_display if model_display is not None else model
    title_str = (
        f"Model: {_model_str} | Task: {dataset} ({n_prompts} prompts) | "
        f"Threshold: {quantile} | max_W: {max_w:.2f} | min_W: {min_w:.2f} | "
        f"Nodes: {n_nodes_used}/{TOTAL_POSSIBLE_NODES} ({node_sparsity:.2f}%) | "
        f"Edges: {n_edges_used}/{TOTAL_POSSIBLE_EDGES} ({edge_sparsity:.2f}%)"
    )
    plt.title(title_str, fontsize=13, pad=18)

    # Keep the `node_size` passed to draw_networkx_edges in sync with the
    # NODE_SIZE we use to draw the circles so arrow endpoints terminate on
    # the actual circle boundary rather than in empty space or inside the
    # disk. (NODE_SIZE / SUPER_NODE_SIZE were set up above so figure sizing
    # can derive the required per-unit spacing from the circle diameter.)
    # `node_size=NODE_SIZE` tells networkx where the node boundary sits so
    # arrows terminate at the circle edge (not its centre). `min_*_margin`
    # is the extra gap added on top; keep it near zero so arrowheads land
    # right on the boundary rather than floating in space. `arrowsize` is
    # the arrowhead length in points — smaller now that circles are smaller.
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.85,
                           arrows=True, arrowsize=11, arrowstyle='-|>',
                           connectionstyle="arc3,rad=0.0", ax=ax, node_size=NODE_SIZE,
                           min_source_margin=1, min_target_margin=1)

    # Sender vs receiver split: nodes with any outgoing edge get a soft fill
    # tint so the reader can pick out senders at a glance; pure receivers
    # (only incoming edges) stay plain white. Kept intentionally shy — no red
    # / orange, no gold — so the fill hue doesn't compete with the reds
    # colormap on the edges.
    SENDER_FILL = "#c8dcef"   # pale sky blue, slightly darker
    RECEIVER_FILL = "white"
    SHARED_FILL = "#b8e6b8"   # light green
    sender_set = {v.index for v in g.vs if g.degree(v.index, mode="out") > 0}

    # Shared-expert vertices (sender-only, no routing weight) are drawn light
    # green and pulled out of the super/sender/receiver split below entirely.
    shared_active = [n for n in active_node_indices if has_is_shared and g.vs[n]["is_shared"]]
    remaining_active = [n for n in active_node_indices if n not in set(shared_active)]

    # Split active nodes by super-expert status if the "is_super" vertex
    # attribute is present (set by the caller before calling this function).
    # Super-experts are drawn larger with a gold fill and red border so they
    # stand out from receiver-only nodes (which would only appear because they
    # receive an edge from some super-expert).
    if has_is_super:
        super_active = [n for n in remaining_active if g.vs[n]["is_super"]]
        other_active = [n for n in remaining_active if not g.vs[n]["is_super"]]
        # Still apply the sender-tint to the non-super majority.
        other_senders = [n for n in other_active if n in sender_set]
        other_receivers = [n for n in other_active if n not in sender_set]
        nx.draw_networkx_nodes(G, pos, nodelist=other_receivers, node_size=NODE_SIZE,
                               node_color=RECEIVER_FILL, edgecolors='black', linewidths=1.2, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=other_senders, node_size=NODE_SIZE,
                               node_color=SENDER_FILL, edgecolors='black', linewidths=1.2, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=super_active, node_size=SUPER_NODE_SIZE,
                               node_color='gold', edgecolors='red', linewidths=2.5, ax=ax)
    else:
        senders = [n for n in remaining_active if n in sender_set]
        receivers = [n for n in remaining_active if n not in sender_set]
        nx.draw_networkx_nodes(G, pos, nodelist=receivers, node_size=NODE_SIZE,
                               node_color=RECEIVER_FILL, edgecolors='black', linewidths=1.2, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=senders, node_size=NODE_SIZE,
                               node_color=SENDER_FILL, edgecolors='black', linewidths=1.2, ax=ax)
    if shared_active:
        nx.draw_networkx_nodes(G, pos, nodelist=shared_active, node_size=NODE_SIZE,
                               node_color=SHARED_FILL, edgecolors='black', linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

    # --- AXIS & COLORBAR ---
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(1 * X_SPACING))
    ax.yaxis.set_major_locator(MultipleLocator(1 * Y_SPACING))
    if is_deep:
        # DEEP orientation: x-axis carries layers (positive, left-to-right),
        # y-axis carries experts (expert 0 at top, indices increasing down).
        def _x_tick(x, _p, _ll=layer_labels):
            idx = int(round(x / X_SPACING))
            return f"{_ll[idx]}" if 0 <= idx < len(_ll) else ""
        def _y_tick(y, _p, _N=N_EXPERTS, _shared=has_is_shared):
            if y > 1e-9:      # padding above expert 0
                return ""
            idx = int(round(-y / Y_SPACING))
            if _shared and idx == _N:
                return "S"
            return f"{idx}" if 0 <= idx < _N else ""
    else:
        # WIDE orientation: x-axis carries experts (positive, left-to-right),
        # y-axis carries layers (layer 0 at top, indices increasing down).
        def _x_tick(x, _p, _N=N_EXPERTS, _shared=has_is_shared):
            idx = int(round(x / X_SPACING))
            if _shared and idx == _N:
                return "S"
            return f"{idx}" if 0 <= idx < _N else ""
        def _y_tick(y, _p, _ll=layer_labels):
            if y > 1e-9:      # padding above layer 0
                return ""
            idx = int(round(-y / Y_SPACING))
            return f"{_ll[idx]}" if 0 <= idx < len(_ll) else ""
    ax.xaxis.set_major_formatter(FuncFormatter(_x_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_y_tick))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Fixed-width colorbar via axes_grid1: 0.3 inches wide regardless of
    # figure size, height locked to the main axes. Unlike inset_axes this
    # cooperates with tight_layout — the divider shares the space with `ax`,
    # so the colorbar shrinks/grows in lockstep with the plotting area.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.15)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=13)

    for spine in ax.spines.values():
        spine.set_visible(True)
    # NB: after plt.colorbar(sm, cax=cax), the "current axes" is the colorbar's
    # axes, so plt.xlim / plt.ylim / plt.xlabel / plt.ylabel / plt.grid would
    # silently modify the colorbar instead of the main plot. Always address
    # `ax` directly here.
    ax.grid(True, linestyle='--', alpha=0.15)

    # Always span the full architectural grid: experts 0..N_EXPERTS-1 on x,
    # layers 0..N_LAYERS-1 on y. Empty rows (layers with no active node) are
    # intentionally left blank rather than cropped — this makes layer position
    # readable across models and prevents the viz from misrepresenting where
    # super-experts sit in the stack.
    # 0.5-unit padding on each side of the outermost node keeps whitespace
    # symmetric, and the top padding is kept below one full Y_SPACING so
    # the tick locator never places a tick at +Y_SPACING (which would
    # otherwise mirror as a bogus "1" above whichever axis carries index 0).
    if is_deep:
        ax.set_xlim(-X_SPACING * 0.5, (N_LAYERS - 1) * X_SPACING + X_SPACING * 0.5)
        ax.set_ylim(-(N_SLOTS - 1) * Y_SPACING - Y_SPACING * 0.5, Y_SPACING * 0.5)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Experts")
    else:
        ax.set_xlim(-X_SPACING * 0.5, (N_SLOTS - 1) * X_SPACING + X_SPACING * 0.5)
        ax.set_ylim(-(N_LAYERS - 1) * Y_SPACING - Y_SPACING * 0.5, Y_SPACING * 0.5)
        ax.set_xlabel("Experts")
        ax.set_ylabel("Layers")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def layer_pair_mass(W, n_buckets: int = 8) -> np.ndarray:
    """Layer-pair mass distribution of an edge-weight tensor.

    For each edge (s, j, r, n), accumulates |W[s, j, r, n]| into the bucket
    determined by relative depth (s/L, r/L). The result is a 2D probability
    distribution on an n_buckets x n_buckets grid indexed by relative
    sender and receiver layer.

    Size-invariant (relative depth absorbs the layer-count mismatch between
    OLMoE and DeepSeek-V2-Lite) and within-layer permutation-invariant (we
    sum over expert dims). Caller should pre-sparsify W if the metric should
    reflect the thresholded graph rather than the full dense tensor.

    Args:
        W: weight tensor of shape [L, E, L, E] (torch or numpy).
        n_buckets: grid resolution per axis.

    Returns:
        ndarray of shape [n_buckets, n_buckets] summing to 1. Returns a
        uniform distribution if the input has no nonzero entries.
    """
    if hasattr(W, "cpu"):
        W = W.cpu().numpy()
    W = np.asarray(W, dtype=np.float64)
    L = W.shape[0]
    layer_mass = np.abs(W).sum(axis=(1, 3))                          # [L, L]
    bucket = np.minimum((np.arange(L) * n_buckets) // L, n_buckets - 1)
    M = np.zeros((n_buckets, n_buckets))
    for s in range(L):
        for r in range(L):
            M[bucket[s], bucket[r]] += layer_mass[s, r]
    total = M.sum()
    if total == 0:
        return np.full((n_buckets, n_buckets), 1.0 / (n_buckets * n_buckets))
    return M / total


def lpm_similarity(M1: np.ndarray, M2: np.ndarray, metric: str = "cosine") -> float:
    """Compare two layer-pair mass distributions.

    Args:
        M1, M2: ndarrays of shape [K, K], each summing to 1.
        metric: 'cosine' (cosine of flattened distributions; 1 = identical
            shape) or 'tv' (total-variation similarity, 1 - 0.5 * sum|p - q|;
            1 = identical distributions, 0 = disjoint support).

    Returns:
        Similarity in [0, 1].
    """
    p, q = M1.flatten(), M2.flatten()
    if metric == "cosine":
        n1, n2 = np.linalg.norm(p), np.linalg.norm(q)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float((p @ q) / (n1 * n2))
    if metric == "tv":
        return 1.0 - 0.5 * float(np.abs(p - q).sum())
    raise ValueError(f"Unknown metric: {metric!r} (expected 'cosine' or 'tv')")
