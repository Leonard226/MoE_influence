"""Graph-analysis utilities for the OLMoE expert routing DAG.

The DAG is stored as edge weight tensors of shape [L, E, L, E] (e.g. APS,
ANS, mean, V_tok, AARV). This module wraps it as an igraph Graph and exposes:

    - dag_to_igraph     : sparsify a weight tensor and convert to a directed igraph.
    - extract_communities : run Leiden (RB configuration) at tunable resolution.

Plus helpers expert_id / expert_name to map between (layer, expert) and a flat
node index used by igraph.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import igraph as ig
    import leidenalg
except ImportError as e:
    raise ImportError(
        "graph_utils.py needs python-igraph and leidenalg.\n"
        "    pip install python-igraph leidenalg"
    ) from e


# ---- OLMoE constants (mirror build_dag.py) -----------------------------------
N_LAYERS = 16
N_EXPERTS = 64
N_NODES = N_LAYERS * N_EXPERTS  # 1024


# ---- Node-id helpers ---------------------------------------------------------
def expert_id(layer: int, expert: int) -> int:
    return layer * N_EXPERTS + expert


def expert_name(node: int) -> str:
    layer = node // N_EXPERTS
    expert = node % N_EXPERTS
    return f"M{layer}E{expert}"


def expert_layer(node: int) -> int:
    return node // N_EXPERTS


# ---- Internal utility --------------------------------------------------------
def _flatten_dag_weight(dag: dict, weight_key: str) -> np.ndarray:
    """Flatten the [L, E, L, E] tensor to [N_NODES, N_NODES] (sender, receiver)."""
    W4 = dag[weight_key]
    if isinstance(W4, torch.Tensor):
        W4 = W4.cpu().numpy()
    assert W4.shape == (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS), W4.shape
    # Index as W2[c*E + j, l*E + n] = W4[c, j, l, n]
    return W4.reshape(N_NODES, N_NODES).astype(np.float64)


# ---- 1. Sparsify and build the graph ----------------------------------------
def dag_to_igraph(
    dag: dict,
    weight_key: str = "APS",
    sparsification: str = "per_sender_topk",
    K: int = 5,
    quantile: float = 0.99,
    use_abs: bool | None = None,
) -> ig.Graph:
    """Build a directed igraph from the DAG tensor, sparsified to high-weight edges.

    Args:
        dag: dict loaded from `dag_*.pt` (must have the requested weight_key).
        weight_key: which edge-weight tensor to use ('APS', 'ANS', 'mean', 'V_tok', 'AARV').
        sparsification: 'per_sender_topk' (each sender keeps its K largest outgoing
            edges) or 'global_quantile' (keep edges above the given quantile).
        K: top-K parameter for per-sender mode.
        quantile: quantile parameter for global mode (0.99 keeps top 1% of nonzero).
        use_abs: rank by |weight| if True, by raw value if False. Default None
            picks |weight| for weight_keys where sign is meaningful (ANS, mean)
            and raw value for non-negative ones (APS, V_tok, AARV).

    Returns:
        Directed igraph with N_NODES vertices and 'weight' edge attribute (signed
        original weight, even if ranking used abs).
    """
    W = _flatten_dag_weight(dag, weight_key)
    abs_W = np.abs(W)

    # Default abs/raw choice based on weight semantics.
    if use_abs is None:
        use_abs = weight_key in {"ANS", "mean"}
    rank_W = abs_W if use_abs else W

    # Build edge list under the chosen sparsification.
    if sparsification == "per_sender_topk":
        # For each row, take indices of K largest entries (by rank_W).
        # Excludes self-loops (DAG forbids them anyway by construction).
        K_eff = min(K, N_NODES - 1)
        # argpartition is O(n) per row; argsort of the partition gives the order.
        part = np.argpartition(-rank_W, K_eff, axis=1)[:, :K_eff]   # [N, K]
        # Filter rows where the top-K weights are all zero.
        rows = np.repeat(np.arange(N_NODES), K_eff)
        cols = part.reshape(-1)
        weights_signed = W[rows, cols]
        weights_rank = rank_W[rows, cols]
        keep = weights_rank > 0  # drop zero weights (e.g., l <= c entries)
        rows, cols, weights_signed = rows[keep], cols[keep], weights_signed[keep]
    elif sparsification == "global_quantile":
        nz = rank_W[rank_W > 0]
        if nz.size == 0:
            raise ValueError(f"No nonzero edges for weight_key={weight_key!r}.")
        threshold = float(np.quantile(nz, quantile))
        mask = rank_W >= threshold
        rows, cols = np.nonzero(mask)
        weights_signed = W[rows, cols]
    else:
        raise ValueError(f"Unknown sparsification: {sparsification!r}")

    # Build igraph.
    g = ig.Graph(directed=True, n=N_NODES)
    g.add_edges(list(zip(rows.tolist(), cols.tolist())))
    g.es["weight"] = weights_signed.tolist()
    g.vs["layer"] = [v // N_EXPERTS for v in range(N_NODES)]
    g.vs["expert"] = [v % N_EXPERTS for v in range(N_NODES)]
    g.vs["name"] = [expert_name(v) for v in range(N_NODES)]
    return g


# ---- 2. Community detection -------------------------------------------------
def extract_communities(
    g: ig.Graph,
    resolution: float = 5.0,
    n_iterations: int = 10,
    seed: int = 0,
) -> list[set[int]]:
    """Leiden communities at the given resolution. Higher resolution = smaller
    communities. Communities are returned as sets of node indices, sorted from
    largest to smallest. Edge weights must be non-negative; if some are
    negative the function takes absolute values.
    """
    weights = np.asarray(g.es["weight"], dtype=np.float64)
    if (weights < 0).any():
        # Use abs for leidenalg (it requires non-negative).
        weights = np.abs(weights)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights.tolist(),
        n_iterations=n_iterations,
        resolution_parameter=resolution,
        seed=seed,
    )
    membership = np.asarray(partition.membership)
    communities = []
    for cid in range(len(partition)):
        members = set(int(v) for v in np.where(membership == cid)[0])
        communities.append(members)
    communities.sort(key=len, reverse=True)
    return communities


