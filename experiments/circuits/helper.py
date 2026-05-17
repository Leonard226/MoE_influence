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


def show_enhanced_layered_graph(g, quantile: float, target: str, dataset: str, n_prompts: int) -> None:
    """Layered DAG visualization. NOTE: hardcoded to OLMoE (16 layers x 64 experts).
    Parameterize N_LAYERS / N_EXPERTS for use with other models."""
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
    N_LAYERS, N_EXPERTS = 16, 64
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
    if target.upper() in ["AVG", "AARV"]:
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

        val_for_color = w if target.upper() in ["AVG", "AARV"] else abs(w)
        edge_colors.append(cmap(norm(val_for_color)))

        w_norm = (abs(w) - min_mag) / (max_mag - min_mag + 1e-9)
        edge_widths.append(1.2 + (w_norm * 4.3))

    # --- DRAWING ---
    plt.figure(figsize=(25, 13))
    ax = plt.gca()

    title_str = (
        f"MoE Routing DAG on {n_prompts} prompts of {dataset} dataset\n"
        f"Metric: {target}\n"
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

    plt.xlabel("Layers")
    plt.ylabel("Experts")
    plt.tight_layout()
    plt.show()
