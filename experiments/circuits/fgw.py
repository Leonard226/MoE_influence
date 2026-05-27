"""Fused Gromov-Wasserstein similarity between MoE routing DAGs.

Math reference: 6a154a47401c9f4881c67a3f/main.tex Section 3.

A routing DAG G = (V, E, W) is summarised as a triple (C, F, mass):
  - C    : [n, n]   intra-graph structural cost matrix
  - F    : [n, D]   per-vertex feature matrix, D = 4 + N_CLASSES
  - mass : [n]      probability mass over vertices (sums to 1)

FGW_alpha(G_1, G_2)^2 is the minimum cost of a transport plan T from mass_1 to
mass_2 that simultaneously
    (1) matches similar feature vectors    (Wasserstein term, weight 1 - alpha)
    (2) preserves intra-graph distances    (Gromov-Wasserstein term, weight alpha).

The similarity score is S_alpha = exp(-FGW_alpha^2) in (0, 1].

Computation requires the POT library (`pip install POT`).
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Token classification (coarse scheme, main.tex Sec 3.6).
# ---------------------------------------------------------------------------

TOKEN_CLASSES = [
    "punctuation",
    "bos_sink",
    "digit",
    "determiner",
    "content",
    "other",
]
N_CLASSES = len(TOKEN_CLASSES)
_CLASS_IDX = {name: i for i, name in enumerate(TOKEN_CLASSES)}

_DETERMINERS = {
    "a", "an", "the",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "every", "no", "each", "all", "both", "few", "many",
}


def classify_token(s: str) -> int:
    """Map a decoded token string to a TOKEN_CLASSES index."""
    sl = s.lower().strip()

    # Special tokens: angle-bracket-enclosed, BOS/EOS markers, full-width pipes
    # (DeepSeek), underscored sentence markers, etc.
    if len(sl) >= 3 and sl.startswith("<") and sl.endswith(">"):
        return _CLASS_IDX["bos_sink"]
    if ("begin" in sl and "sentence" in sl) or "endoftext" in sl \
            or "startoftext" in sl or "im_start" in sl or "im_end" in sl:
        return _CLASS_IDX["bos_sink"]
    if "｜" in s and ("begin" in sl or "end" in sl):
        return _CLASS_IDX["bos_sink"]

    stripped = s.strip()
    if not stripped or all(not c.isalnum() for c in stripped):
        return _CLASS_IDX["punctuation"]

    if stripped.lower() in _DETERMINERS:
        return _CLASS_IDX["determiner"]

    if all(c.isdigit() or c in ".,-+$%" for c in stripped):
        return _CLASS_IDX["digit"]

    if any(c.isalpha() for c in stripped):
        return _CLASS_IDX["content"]

    return _CLASS_IDX["other"]


def compute_class_histogram(top_token: torch.Tensor, top_weight: torch.Tensor,
                            tokenizer) -> torch.Tensor:
    """For each vertex (l, n), return a histogram over TOKEN_CLASSES from the
    top-B routed tokens. Empty buffers fall back to all-mass-on-"other".

    Args:
        top_token: [L, N, B] int  -- token ids
        top_weight: [L, N, B] float -- routing weights (sentinel <0 = unused slot)
        tokenizer: HF tokenizer (must support `decode([id])`)

    Returns:
        [L, N, N_CLASSES] tensor, each row sums to 1.
    """
    L, N, B = top_token.shape
    hist = torch.zeros(L, N, N_CLASSES, dtype=torch.float32)
    other_idx = _CLASS_IDX["other"]

    # Cache token-id -> class to amortise tokenizer.decode + classify_token.
    tok2class: Dict[int, int] = {}

    for l in range(L):
        for n in range(N):
            mask = top_weight[l, n] > 0
            if not mask.any():
                hist[l, n, other_idx] = 1.0
                continue
            ids = top_token[l, n][mask].tolist()
            counts = torch.zeros(N_CLASSES, dtype=torch.float32)
            for tid in ids:
                cls = tok2class.get(tid)
                if cls is None:
                    cls = classify_token(tokenizer.decode([tid]))
                    tok2class[tid] = cls
                counts[cls] += 1
            hist[l, n] = counts / counts.sum()
    return hist


# ---------------------------------------------------------------------------
# Structural cost: weighted shortest path with edge cost (1 - W).
# ---------------------------------------------------------------------------

def _shortest_path_costs(W_fwd: torch.Tensor, L: int, N: int,
                         edge_threshold: float = 0.0) -> np.ndarray:
    """All-pairs weighted shortest-path distances on the forward DAG.

    Edge cost = 1 - W(e) in [0, 1] (high W -> short edge).
    Edges with W < edge_threshold are dropped (computational sparsification).
    Output is symmetrised: for each unordered pair (u, v) we set
    d(u, v) = d(v, u) = the (unique) forward-direction distance.
    Unreachable pairs receive distance (L - 1) (the depth diameter).

    Returns: [n_verts, n_verts] symmetric distance matrix (float64).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n_verts = L * N

    # Build sparse cost matrix: index a forward edge (c, j) -> (l, n) by
    # (c*N + j) -> (l*N + n).
    W_abs = W_fwd.abs().cpu()
    mask = W_abs > edge_threshold
    if not mask.any():
        return np.zeros((n_verts, n_verts), dtype=np.float64)

    s_l, s_n, t_l, t_n = mask.nonzero(as_tuple=True)
    s_v = (s_l * N + s_n).numpy()
    t_v = (t_l * N + t_n).numpy()
    costs = (1.0 - W_abs[mask]).numpy()

    cost_mat = csr_matrix((costs, (s_v, t_v)), shape=(n_verts, n_verts))
    dist = shortest_path(cost_mat, directed=True, method="D")  # [V, V], inf if unreachable

    # Symmetrise (forward DAG: at most one direction is finite).
    dist_sym = np.minimum(dist, dist.T)
    # Replace remaining inf (truly disconnected pairs) with L - 1, the maximum
    # possible weighted-path cost on a forward DAG with L layers.
    diameter = float(max(L - 1, 1))
    dist_sym[np.isinf(dist_sym)] = diameter
    np.fill_diagonal(dist_sym, 0.0)
    return dist_sym.astype(np.float64)


# ---------------------------------------------------------------------------
# Triple construction: (C, F, mass) from a DAG.
# ---------------------------------------------------------------------------

def build_triple(dag: Dict[str, Any],
                 tokenizer=None,
                 *,
                 beta: float = 0.5,
                 edge_threshold: float = 0.0,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Construct the FGW triple (C, F, mass) for one routing DAG.

    Args:
        dag: dict produced by build_dag.py; expects keys P_add, P_rem,
            n_tokens_selected, top_token, top_weight, moe_layers.
        tokenizer: HuggingFace tokenizer for the model (decodes token ids in
            top_token). If None, the token-class histogram is set to a uniform
            "other"-only distribution (P6 contributes nothing).
        beta: mixing weight for the structural cost:
            C = beta * |depth_u - depth_v| + (1 - beta) * d_path / (L - 1).
            beta = 1 skips the (expensive) shortest-path computation.
        edge_threshold: drop edges with W < threshold before computing
            shortest paths. 0.0 = use the dense graph. See main.tex
            "Graph thresholding".

    Returns:
        C    : [n, n] np.float64  -- structural cost
        F    : [n, D] np.float64  -- features [depth, out~, in~, load, class_0..class_{T-1}]
        mass : [n]    np.float64  -- probability mass, sums to 1
        meta : dict with L, N, n_verts, D
    """
    W = (dag["P_add"] + dag["P_rem"]).float()
    L, N = W.shape[0], W.shape[1]
    n_verts = L * N

    # Forward mask in case build_dag stored anything in the backward triangle.
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_fwd = W * fwd.float()

    # --- Features ---
    # 1. relative depth
    layer_idx = torch.arange(L).view(L, 1).expand(L, N).float()
    depth = layer_idx / max(L - 1, 1)  # [L, N], in [0, 1]

    # 2. out-strength (sender side)
    out_strength = W_fwd.sum(dim=(2, 3))  # [L, N]
    out_max = out_strength.max().clamp(min=1e-12)
    out_norm = out_strength / out_max

    # 3. in-strength (receiver side)
    in_strength = W_fwd.sum(dim=(0, 1))  # [L, N]
    in_max = in_strength.max().clamp(min=1e-12)
    in_norm = in_strength / in_max

    # 4. layer-relative token load
    n_tok = dag["n_tokens_selected"].float()  # [L, N]
    layer_mean = n_tok.mean(dim=1, keepdim=True).clamp(min=1e-12)  # [L, 1]
    load = n_tok / layer_mean  # [L, N], ~1 = average

    # 5. token-class histogram
    if tokenizer is not None and "top_token" in dag and "top_weight" in dag:
        class_hist = compute_class_histogram(dag["top_token"], dag["top_weight"], tokenizer)
    else:
        class_hist = torch.zeros(L, N, N_CLASSES, dtype=torch.float32)
        class_hist[..., _CLASS_IDX["other"]] = 1.0

    F = torch.cat([
        depth.unsqueeze(-1),        # [L, N, 1]
        out_norm.unsqueeze(-1),
        in_norm.unsqueeze(-1),
        load.unsqueeze(-1),
        class_hist,                  # [L, N, N_CLASSES]
    ], dim=-1).reshape(n_verts, -1).double().numpy()

    # --- Mass ---
    out_flat = out_strength.reshape(-1).cpu().numpy().astype(np.float64)
    eps = 1e-6 * (out_flat.max() if out_flat.max() > 0 else 1.0)
    mass = out_flat + eps
    mass = mass / mass.sum()

    # --- Structural cost C ---
    depth_flat = depth.reshape(-1).cpu().numpy().astype(np.float64)  # [n]
    C_depth = np.abs(depth_flat[:, None] - depth_flat[None, :])      # [n, n]

    if beta < 1.0:
        C_path_raw = _shortest_path_costs(W_fwd, L, N, edge_threshold=edge_threshold)
        C_path = C_path_raw / max(L - 1, 1)  # normalise to [0, 1]
    else:
        C_path = np.zeros_like(C_depth)

    C = beta * C_depth + (1.0 - beta) * C_path

    meta = {"L": L, "N": N, "n_verts": n_verts, "D": F.shape[1],
            "beta": beta, "edge_threshold": edge_threshold}
    return C.astype(np.float64), F, mass, meta


# ---------------------------------------------------------------------------
# FGW distance and similarity.
# ---------------------------------------------------------------------------

def fgw_distance(triple1: Tuple, triple2: Tuple, *,
                 alpha: float = 0.5,
                 n_init: int = 10,
                 max_iter: int = 200,
                 seed: int = 0,
                 ) -> Tuple[float, np.ndarray]:
    """Compute FGW_alpha(G1, G2)^2 by best-of-n_init random restarts.

    Args:
        triple1, triple2: (C, F, mass, [meta]) tuples from build_triple.
        alpha: in [0, 1]. 0 = pure Wasserstein on features; 1 = pure
            Gromov-Wasserstein on structure; 0.5 = balanced.
        n_init: number of random initialisations (FGW is non-convex; we keep
            the best of n_init runs).
        max_iter: max iterations per run for the block-coordinate solver.
        seed: rng seed for the initial transport plans.

    Returns:
        dist_sq: float -- FGW_alpha^2 = min_T L_alpha(T).
        T_best:  [n1, n2] np.float64 -- corresponding transport plan.
    """
    import ot  # POT; lazy import so module-level imports work without it.

    C1, F1, mass1 = triple1[0], triple1[1], triple1[2]
    C2, F2, mass2 = triple2[0], triple2[1], triple2[2]

    # Feature cost: M[i, k] = ||F1[i] - F2[k]||_2^2.
    M = ot.dist(F1, F2, metric="sqeuclidean")

    rng = np.random.default_rng(seed)
    best_dist = float("inf")
    best_T = None

    for run in range(n_init):
        if run == 0:
            G0 = None  # POT's default warm start (outer-product of marginals)
        else:
            # Random transport plan respecting the marginals (Sinkhorn-like).
            G0 = rng.random((len(mass1), len(mass2)))
            G0 = G0 * (mass1[:, None] / G0.sum(axis=1, keepdims=True))
            G0 = G0 * (mass2[None, :] / G0.sum(axis=0, keepdims=True).clip(min=1e-30))

        try:
            T, log = ot.gromov.fused_gromov_wasserstein(
                M, C1, C2, mass1, mass2,
                alpha=alpha, loss_fun="square_loss",
                log=True, max_iter=max_iter,
                G0=G0,
            )
            dist_sq = float(log["fgw_dist"])
        except Exception:
            continue

        if dist_sq < best_dist:
            best_dist = dist_sq
            best_T = T

    if best_T is None:
        raise RuntimeError("FGW failed on every random init")
    return best_dist, best_T


def fgw_similarity(triple1: Tuple, triple2: Tuple, *,
                   alpha: float = 0.5,
                   **kwargs) -> Tuple[float, np.ndarray]:
    """Similarity S_alpha(G1, G2) = exp(-FGW_alpha^2), in (0, 1].

    Returns (S, T) where S is the similarity and T the optimal transport plan.
    """
    dist_sq, T = fgw_distance(triple1, triple2, alpha=alpha, **kwargs)
    return float(np.exp(-dist_sq)), T