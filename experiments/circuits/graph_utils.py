"""Graph-analysis utilities for the OLMoE expert routing DAG.

The DAG is stored as edge weight tensors of shape [L, E, L, E] (e.g. APS,
ANS, mean, V_tok, AARV). This module wraps it as an igraph Graph and exposes
the graph-algorithm primitives used to discover circuit candidates:

    - dag_to_igraph     : sparsify a weight tensor and convert to a directed igraph.
    - extract_chains    : enumerate high-weight directed paths of given length.
    - extract_communities : run Leiden (RB configuration) at tunable resolution.
    - extract_fan_outs  : find senders with concentrated outgoing edges.
    - differential_edges: edges whose weight differs sharply between two DAGs.

Plus helpers expert_id / expert_name to map between (layer, expert) and a flat
node index used by igraph.
"""
from __future__ import annotations

import math
from typing import Iterable

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


# ---- 2. Chain enumeration ----------------------------------------------------
def extract_chains(
    g: ig.Graph,
    min_length: int = 2,
    max_length: int = 4,
    weighted_score: str = "min",
    max_paths: int = 100_000_000_000,
) -> list[dict]:
    """Enumerate directed paths of edge-length in [min_length, max_length].

    Args:
        g: directed igraph (already sparsified — we walk only edges that exist).
        min_length: minimum number of edges in a returned path.
        max_length: maximum number of edges (paths cap at max_length+1 nodes).
        weighted_score: how to score a chain by its edges --- 'min' (chain
            strength = weakest link), 'sum' (total weight), 'mean' (average).
        max_paths: safety cap; stop enumerating once this many paths are found.

    Returns:
        List of {nodes, weights, score} dicts, sorted by score descending.
        Each `nodes` is a list of node indices; `weights` is the per-edge weights;
        `score` is the aggregated chain strength.
    """
    if min_length < 1 or max_length < min_length:
        raise ValueError(f"Bad length bounds: [{min_length}, {max_length}].")

    out_neigh = [list(g.successors(v)) for v in range(g.vcount())]
    edge_weight = {}
    for e in g.es:
        edge_weight[(e.source, e.target)] = e["weight"]

    paths = []
    visited_in_path = set()

    def aggregate(weights):
        if weighted_score == "min":
            return min(abs(w) for w in weights)
        if weighted_score == "sum":
            return sum(weights)
        if weighted_score == "mean":
            return sum(weights) / len(weights)
        raise ValueError(weighted_score)

    for start in range(g.vcount()):
        if not out_neigh[start]:
            continue
        # DFS up to max_length edges.
        stack = [(start, [start], [])]
        while stack:
            if len(paths) >= max_paths:
                return _sort_chains(paths)
            node, path, weights = stack.pop()
            n_edges = len(weights)
            if n_edges >= min_length:
                paths.append({
                    "nodes": path[:],
                    "weights": weights[:],
                    "score": aggregate(weights),
                })
            if n_edges >= max_length:
                continue
            for nb in out_neigh[node]:
                if nb not in path:
                    w = edge_weight[(node, nb)]
                    stack.append((nb, path + [nb], weights + [w]))

    return _sort_chains(paths)


def _sort_chains(paths):
    paths.sort(key=lambda p: p["score"], reverse=True)
    return paths


# ---- 3. Community detection -------------------------------------------------
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


# ---- 4. Fan-out detection ---------------------------------------------------
def extract_fan_outs(
    dag: dict,
    weight_key: str = "APS",
    n_targets: int = 8,
    top_n_senders: int = 20,
    use_abs: bool | None = None,
) -> list[dict]:
    """Find senders with concentrated outgoing edge weights (broadcasters).

    Concentration = fraction of the sender's total outgoing |weight| that is
    captured by its top-`n_targets` receivers. High concentration means the
    sender disperses to a specific small set of downstream receivers.

    Returns:
        List of {sender, sender_name, targets, target_names, concentration,
                 total_outgoing_weight} dicts, sorted by concentration.
    """
    W = _flatten_dag_weight(dag, weight_key)
    abs_W = np.abs(W) if (use_abs is None and weight_key in {"ANS", "mean"}) or use_abs else W
    abs_W = np.maximum(abs_W, 0.0)  # safety

    out = []
    for src in range(N_NODES):
        row = abs_W[src]
        total = row.sum()
        if total < 1e-12:
            continue
        # Top-K receivers by absolute weight.
        order = np.argsort(-row)
        topk = order[:n_targets]
        topk_mass = row[topk].sum()
        concentration = float(topk_mass / total)
        out.append({
            "sender": int(src),
            "sender_name": expert_name(int(src)),
            "targets": [int(t) for t in topk if row[t] > 0],
            "target_names": [expert_name(int(t)) for t in topk if row[t] > 0],
            "concentration": concentration,
            "total_outgoing_weight": float(total),
        })

    # Rank by concentration, prefer senders that are actually active.
    out.sort(key=lambda r: (r["concentration"], r["total_outgoing_weight"]), reverse=True)
    return out[:top_n_senders]


# ---- 5. Differential edges (cross-task circuit candidates) ------------------
def differential_edges(
    dag_a: dict,
    dag_b: dict,
    weight_key: str = "AARV",
    n_edges: int = 100,
    min_max_weight: float = 0.1,
) -> list[dict]:
    """Edges whose weight differs sharply between two DAGs.

    Returned edges are ranked by relative difference,
        ratio = |w_a - w_b| / max(|w_a|, |w_b|),
    filtered to require max(|w_a|, |w_b|) >= min_max_weight (avoids amplifying
    noise where both weights are near zero).

    Returns:
        List of {sender, sender_name, receiver, receiver_name, w_a, w_b,
                 ratio} dicts, sorted by ratio descending.
    """
    W_a = _flatten_dag_weight(dag_a, weight_key)
    W_b = _flatten_dag_weight(dag_b, weight_key)
    if W_a.shape != W_b.shape:
        raise ValueError(f"Shape mismatch: {W_a.shape} vs {W_b.shape}.")

    abs_a, abs_b = np.abs(W_a), np.abs(W_b)
    max_abs = np.maximum(abs_a, abs_b)
    diff = np.abs(W_a - W_b)
    eps = 1e-12
    ratio = diff / np.maximum(max_abs, eps)

    # Filter out edges where both weights are negligible.
    mask = max_abs >= min_max_weight
    flat_ratio = np.where(mask, ratio, -np.inf)
    flat_idx = np.argsort(-flat_ratio.reshape(-1))[:n_edges]

    out = []
    for fi in flat_idx:
        if flat_ratio.reshape(-1)[fi] == -np.inf:
            break
        src = int(fi // N_NODES)
        dst = int(fi % N_NODES)
        out.append({
            "sender": src,
            "sender_name": expert_name(src),
            "receiver": dst,
            "receiver_name": expert_name(dst),
            "w_a": float(W_a[src, dst]),
            "w_b": float(W_b[src, dst]),
            "ratio": float(ratio[src, dst]),
        })
    return out


# ---- Convenience: chain → expert-name string ---------------------------------
def chain_str(chain: dict) -> str:
    return " → ".join(expert_name(v) for v in chain["nodes"])
