import igraph as ig

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
