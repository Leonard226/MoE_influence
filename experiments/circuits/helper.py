import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FuncFormatter
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


def thresholding_routing_graph(dag: dict, target: str, threshold: float) -> ig.Graph:
    import numpy as np
    # Get the 4D matrix (Shape: [16, 64, 16, 64])
    matrix = dag[target]
    N_LAYERS, N_EXPERTS = matrix.shape[0], matrix.shape[1]
    N_NODES = N_LAYERS * N_EXPERTS

    # Find where the weights are above the threshold
    s_layers, s_exps, r_layers, r_exps = np.where(np.abs(matrix) > threshold) 

    # Convert those coordinates into Vertex IDs
    senders = s_layers * N_EXPERTS + s_exps
    receivers = r_layers * N_EXPERTS + r_exps

    # Extract the weights for these specific edges
    weights = matrix[s_layers, s_exps, r_layers, r_exps]

    # Build the graph
    g = ig.Graph(directed=True, n=N_NODES)
    
    # zip pairs them up: [(s1, r1), (s2, r2), ...]
    edges = list(zip(senders.tolist(), receivers.tolist()))
    g.add_edges(edges)
    
    # Assign the weights and metadata
    g.es["weight"] = weights.tolist()
    g.vs["layer"] = [v // N_EXPERTS for v in range(N_NODES)]
    g.vs["expert"] = [v % N_EXPERTS for v in range(N_NODES)]

    return g


def show_enhanced_layered_graph(g, quantile: float, target: str, model: str, dataset: str, n_prompts: int,
                                 layer_labels: list | None = None) -> None:
    """Layered DAG visualization. Reads N_LAYERS / N_EXPERTS from the graph's
    `layer` vertex attribute (set by thresholding_routing_graph / dag_to_igraph).

    layer_labels: optional mapping from internal DAG layer index (0..N_LAYERS-1)
        to the model's actual layer number. Use this when the DAG skips dense
        layers (e.g. DeepSeek-V2-Lite has dense layer 0, so internal M0 == model
        layer 1). If None, internal indices are used as-is. Pass dag["moe_layers"].
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
    N_EXPERTS = g.vcount() // N_LAYERS
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

    X_SPACING, Y_SPACING = 1000, 300
    for node_idx in active_node_indices:
        layer, expert_idx = node_idx // N_EXPERTS, node_idx % N_EXPERTS
        pos[node_idx] = (expert_idx * X_SPACING, -layer * Y_SPACING)
        labels[node_idx] = f"M{layer_labels[layer]}\nE{expert_idx}"
        G.add_node(node_idx)

    # --- COLOR LOGIC ---
    if target.upper() in ["AVG"]:
        cmap = plt.cm.RdBu
        color_lim = max(abs(max_w), abs(min_w))
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-color_lim, vmax=color_lim)
        cbar_label = "Inhibition vs Promotion"
    else:
        colors_array = plt.cm.Reds(np.linspace(0.35, 1.0, 256))
        cmap = mcolors.LinearSegmentedColormap.from_list('IntenseReds', colors_array)
        norm = mcolors.Normalize(vmin=min_mag, vmax=max_mag)
        cbar_label = "Weight Magnitude |w|"

    edge_colors, edge_widths = [], []
    for e in g.es:
        u, v = e.source, e.target
        G.add_edge(u, v)
        w = e["weight"]

        val_for_color = w if target.upper() in ["AVG"] else abs(w)
        edge_colors.append(cmap(norm(val_for_color)))

        w_norm = (abs(w) - min_mag) / (max_mag - min_mag + 1e-9)
        edge_widths.append(1.2 + (w_norm * 4.3))

    # --- DRAWING ---
    plt.figure(figsize=(25, 13))
    ax = plt.gca()

    title_str = (
        f"{model.upper()} Expert Routing DAG\n"
        f"Metric: {target} | Task: {dataset} ({n_prompts} prompts)\n"
        f"Threshold: {quantile} | max_w: {max_w:.2f} | min_w: {min_w:.2f}\n"
        f"Nodes: {n_nodes_used}/{TOTAL_POSSIBLE_NODES} ({node_sparsity:.2f}%) | "
        f"Edges: {n_edges_used}/{TOTAL_POSSIBLE_EDGES} ({edge_sparsity:.4f}%)"
    )
    plt.title(title_str, fontsize=20, fontweight='bold', pad=25)

    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.85,
                           arrows=True, arrowsize=18, arrowstyle='-|>',
                           connectionstyle="arc3,rad=0.05", ax=ax, node_size=1100,
                           min_source_margin=15, min_target_margin=18)

    # Split active nodes by super-expert status if the "is_super" vertex
    # attribute is present (set by the caller before calling this function).
    # Super-experts are drawn larger with a gold fill and red border so they
    # stand out from receiver-only nodes (which would only appear because they
    # receive an edge from some super-expert).
    if has_is_super:
        super_active = [n for n in active_node_indices if g.vs[n]["is_super"]]
        other_active = [n for n in active_node_indices if not g.vs[n]["is_super"]]
        nx.draw_networkx_nodes(G, pos, nodelist=other_active, node_size=1000,
                               node_color='white', edgecolors='black', linewidths=1.2, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=super_active, node_size=1400,
                               node_color='gold', edgecolors='red', linewidths=2.5, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='white',
                               edgecolors='black', linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

    # --- AXIS & COLORBAR ---
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(5 * X_SPACING))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(x / X_SPACING))}"))
    ax.yaxis.set_major_locator(MultipleLocator(1 * Y_SPACING))
    def _layer_tick(x, _p, _ll=layer_labels):
        idx = int(round(abs(x) / Y_SPACING))
        return f"{_ll[idx]}" if 0 <= idx < len(_ll) else ""
    ax.yaxis.set_major_formatter(FuncFormatter(_layer_tick))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(cbar_label, fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.grid(True, linestyle='--', alpha=0.15)

    # Always span the full architectural grid: experts 0..N_EXPERTS-1 on x,
    # layers 0..N_LAYERS-1 on y. Empty rows (layers with no active node) are
    # intentionally left blank rather than cropped — this makes layer position
    # readable across models and prevents the viz from misrepresenting where
    # super-experts sit in the stack.
    plt.xlim(-X_SPACING * 1.5, (N_EXPERTS - 1) * X_SPACING + X_SPACING * 1.5)
    plt.ylim(-(N_LAYERS - 1) * Y_SPACING - Y_SPACING, Y_SPACING)

    plt.xlabel("Experts e")
    plt.ylabel("Layers l")
    plt.tight_layout()
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
