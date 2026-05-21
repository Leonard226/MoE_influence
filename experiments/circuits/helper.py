import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FuncFormatter
import networkx as nx

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

    # Calculate all quantiles at once
    q_tensor = torch.tensor(quantiles, device=valid_weights.device)
    thresholds = torch.quantile(valid_weights, q_tensor)
    
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


def show_enhanced_layered_graph(g, quantile: float, target: str, model: str, dataset: str, n_prompts: int) -> None:
    """Layered DAG visualization. Reads N_LAYERS / N_EXPERTS from the graph's
    `layer` vertex attribute (set by thresholding_routing_graph / dag_to_igraph)."""
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
    TOTAL_POSSIBLE_NODES = N_LAYERS * N_EXPERTS
    # Max possible edges in a layered DAG (Layer i to Layer >i)
    TOTAL_POSSIBLE_EDGES = sum(N_EXPERTS * ((N_LAYERS - 1 - i) * N_EXPERTS) for i in range(N_LAYERS - 1))

    active_node_indices = [v.index for v in g.vs if v.degree() > 0]
    n_nodes_used = len(active_node_indices)
    n_edges_used = g.ecount()

    node_sparsity = (n_nodes_used / TOTAL_POSSIBLE_NODES) * 100
    edge_sparsity = (n_edges_used / TOTAL_POSSIBLE_EDGES) * 100

    # 1. Build NetworkX Graph
    G = nx.DiGraph()
    pos, labels = {}, {}

    X_SPACING, Y_SPACING = 250, 150
    for node_idx in active_node_indices:
        layer, expert_idx = node_idx // N_EXPERTS, node_idx % N_EXPERTS
        pos[node_idx] = (expert_idx * X_SPACING, -layer * Y_SPACING)
        labels[node_idx] = f"M{layer}\nE{expert_idx}"
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

    nx.draw_networkx_nodes(G, pos, node_size=1100, node_color='white', edgecolors='black', linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

    # --- AXIS & COLORBAR ---
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(5 * X_SPACING))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(x / X_SPACING))}"))
    ax.yaxis.set_major_locator(MultipleLocator(1 * Y_SPACING))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(round(abs(x) / Y_SPACING))}"))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(cbar_label, fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.grid(True, linestyle='--', alpha=0.15)

    all_x, all_y = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    if all_x and all_y:
        plt.xlim(min(all_x) - (X_SPACING * 1.5), max(all_x) + (X_SPACING * 1.5))
        plt.ylim(min(all_y) - Y_SPACING, max(all_y) + Y_SPACING)

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
