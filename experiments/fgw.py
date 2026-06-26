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
# Token classification (context-aware POS tags on raw text; main.tex Sec 3.6).
#
# We tag each prompt with spaCy and look up the POS for each model (sub)token
# via the HuggingFace tokenizer's offset_mapping. This is model-agnostic (no
# Ġ/##/▁ heuristics) and resolves homonyms via sentence context (e.g. "run"
# as noun vs. verb).
# ---------------------------------------------------------------------------

TOKEN_CLASSES = ["content", "functional", "punctuation", "numeric", "special"]
N_CLASSES = len(TOKEN_CLASSES)
_CLASS_IDX = {name: i for i, name in enumerate(TOKEN_CLASSES)}

# spaCy UPOS -> macro class. INTJ ("oh", "wow") goes to content (semantic);
# PART ("to", "n't") and AUX ("is", "would") to functional; X / SPACE to special.
_UPOS_TO_CLASS = {
    "NOUN":  "content",
    "PROPN": "content",
    "VERB":  "content",
    "ADJ":   "content",
    "ADV":   "content",
    "INTJ":  "content",
    "DET":   "functional",
    "PRON":  "functional",
    "ADP":   "functional",
    "AUX":   "functional",
    "CCONJ": "functional",
    "SCONJ": "functional",
    "CONJ":  "functional",
    "PART":  "functional",
    "PUNCT": "punctuation",
    "SYM":   "punctuation",
    "NUM":   "numeric",
    "X":     "special",
    "SPACE": "special",
}


def build_token_classification(
    prompts,
    tokenizer,
    *,
    max_length: int = 32,
    spacy_model: str = "en_core_web_sm",
    verbose: bool = False,
) -> Dict[Tuple[int, int], int]:
    """Build a (prompt_idx, position) -> class_idx lookup for the given prompts.

    For each prompt we
        1. tokenize with the model tokenizer (returning character offsets),
        2. POS-tag the raw text with spaCy,
        3. for each model (sub)token, look up the spaCy POS covering its char
           span and map UPOS -> macro class.

    Model special tokens (BOS / EOS / pad / ...) and empty-span offsets are
    classified as "special". Whitespace-only subwords (rare) fall back to
    "functional".

    Args:
        prompts: list of raw strings, indexed by prompt_idx -- typically the
            same list `build_dag.py` consumed (re-call the dataset helper with
            matching args to reproduce it exactly).
        tokenizer: HuggingFace fast tokenizer (must support
            `return_offsets_mapping`). Use the model's own tokenizer: the
            position indices depend on it.
        max_length: truncation length matching DAG-build time (default 32).
        spacy_model: spaCy model name. `en_core_web_sm` is fast and adequate
            for macro-class binning. Run once:
                python -m spacy download en_core_web_sm

    Returns:
        dict[(prompt_idx, position) -> int (index into TOKEN_CLASSES)].
    """
    import spacy
    nlp = spacy.load(spacy_model, disable=["parser", "ner", "lemmatizer"])

    special_ids = {int(i) for i in tokenizer.all_special_ids}
    out: Dict[Tuple[int, int], int] = {}
    n_unknown = 0

    for pi, prompt in enumerate(prompts):
        enc = tokenizer(
            prompt,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        doc = nlp(prompt)

        for pos, (tid, (start, end)) in enumerate(zip(ids, offsets)):
            # Specials: tokenizer's declared special set, or empty char span.
            if int(tid) in special_ids or end <= start:
                out[(pi, pos)] = _CLASS_IDX["special"]
                continue

            tok_text = prompt[start:end]
            if not tok_text.strip():
                # Pure-whitespace subword (rare). Treat as functional glue.
                out[(pi, pos)] = _CLASS_IDX["functional"]
                continue

            # `alignment_mode="expand"` snaps to whole-word boundaries so
            # subword pieces inherit the POS of the parent word.
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is not None and len(span) >= 1:
                upos = span[0].pos_
            else:
                # Fallback: midpoint scan over spaCy tokens.
                mid = (start + end) // 2
                upos = "X"
                for st in doc:
                    if st.idx <= mid < st.idx + len(st.text):
                        upos = st.pos_
                        break

            cls_name = _UPOS_TO_CLASS.get(upos)
            if cls_name is None:
                n_unknown += 1
                cls_name = "content" if any(c.isalpha() for c in tok_text) else "special"
            out[(pi, pos)] = _CLASS_IDX[cls_name]

        if verbose and (pi + 1) % 500 == 0:
            print(f"  classified {pi + 1}/{len(prompts)} prompts", flush=True)

    if verbose and n_unknown:
        print(f"  {n_unknown} tokens had unmapped UPOS tags (used fallback)", flush=True)
    return out


def compute_class_histogram(
    top_weight: torch.Tensor,
    top_prompt: torch.Tensor,
    top_pos: torch.Tensor,
    classification: Dict[Tuple[int, int], int],
) -> torch.Tensor:
    """For each vertex (l, n), histogram over TOKEN_CLASSES from its top-B
    routed events. Empty buckets fall back to all-mass-on-"special".

    Args:
        top_weight: [L, N, B] float -- routing weight (sentinel <0 = unused slot).
        top_prompt: [L, N, B] int   -- global prompt index.
        top_pos:    [L, N, B] int   -- position within the prompt.
        classification: dict from build_token_classification.

    Returns:
        [L, N, N_CLASSES] tensor, each row sums to 1.
    """
    L, N, _ = top_weight.shape
    hist = torch.zeros(L, N, N_CLASSES, dtype=torch.float32)
    special_idx = _CLASS_IDX["special"]

    for l in range(L):
        for n in range(N):
            mask = top_weight[l, n] > 0
            if not mask.any():
                hist[l, n, special_idx] = 1.0
                continue
            prompts = top_prompt[l, n][mask].tolist()
            positions = top_pos[l, n][mask].tolist()
            counts = torch.zeros(N_CLASSES, dtype=torch.float32)
            for pi, po in zip(prompts, positions):
                counts[classification.get((int(pi), int(po)), special_idx)] += 1
            hist[l, n] = counts / counts.sum()
    return hist


# ---------------------------------------------------------------------------
# Structural cost: weighted shortest path with edge cost (1 - W).
# ---------------------------------------------------------------------------

def _local_costs(W_fwd: torch.Tensor, L: int, N: int,
                 edge_threshold: float = 0.0) -> np.ndarray:
    """Direct-edge structural cost on the forward DAG (no path aggregation).

    Justification: W(u -> v) is defined as the first-order counterfactual
    influence of ablating u on v's router output (main.tex §3, "edge weight
    definition"). The paper explicitly states this is NOT propagated through
    intermediate MoE layers, so multi-hop products / sums / hitting times /
    effective resistances do not correspond to any well-defined composite
    influence. The only locally-justified pairwise structural quantity is the
    direct edge magnitude itself.

    C_local(u, v) = 1 - |W_uv|       for forward edges with |W| > threshold
                  = 1                otherwise (no direct measured influence)

    Symmetrised by direction-mirroring (forward DAG: at most one direction
    per unordered pair has a non-zero entry).

    Returns: [n_verts, n_verts] symmetric float64 in [0, 1], zero on diagonal.
    """
    n_verts = L * N
    W_abs = W_fwd.abs().cpu().numpy().reshape(n_verts, n_verts).astype(np.float64)

    C = np.ones((n_verts, n_verts), dtype=np.float64)
    mask = W_abs > edge_threshold
    C[mask] = 1.0 - W_abs[mask]

    # At most one of (u, v), (v, u) has a non-zero W (forward-only DAG); the
    # other side is still at 1. min gives the forward-direction value.
    C = np.minimum(C, C.T)
    np.fill_diagonal(C, 0.0)
    return C


def _conn_costs(W_fwd: torch.Tensor, L: int, N: int,
                edge_threshold: float = 0.0,
                eps: float = 1e-12,
                gamma: float = 1.0) -> np.ndarray:
    """Katz path-sum structural cost on the forward DAG.

    Phi^gamma(u, v) := sum over forward paths p from u to v of
                       gamma^(|p|-1) * prod_{e in p} |W_e|
                    =  (W (I - gamma W_sparse)^{-1})_{uv}

    The gamma in (0, 1] is a per-hop discount factor:
        gamma = 1    -> unweighted Neumann path-sum (every length equal).
        gamma < 1    -> longer paths contribute exponentially less.
    The motivation for gamma < 1 is that without a discount the combinatorial
    growth of path count with depth-distance can let "many weak long paths"
    overwhelm "a few strong short paths", inverting the structural-coupling
    intuition (near pairs end up looking less coupled than far pairs). A
    discount gamma * b * w_bar < 1 (where b is branching, w_bar mean edge
    weight) tames this so per-hop typical contribution shrinks geometrically.

    Phi^gamma is still a *topological descriptor* of pairwise coupling:
    large = many short strong-edged paths; small = few/weak paths;
    zero = unreachable.

    Computation: since our vertex indexing is (layer * N + n) and edges go
    strictly forward (sender_layer < receiver_layer), W is upper-triangular
    and (I - gamma W_sparse) is upper-triangular with unit diagonal. The
    Neumann series terminates exactly (W nilpotent on a DAG), and the
    all-pairs solution is a single triangular back-substitution
    (O(V^3 / 2) FLOPs, BLAS-accelerated). Note: solving against W on the
    right gives W (I - gamma W)^{-1}, which excludes the zero-step identity
    term (so Phi^gamma diagonal is 0 by construction). For gamma = 1 the
    off-diagonal entries are identical to the previous (I - W)^{-1} formula.

    Transform Phi^gamma -> cost in [0, 1]:
        C(u, v) = clip( -log(max(Phi^gamma, eps)) / -log(eps), 0, 1 )

    Behaviour:
        Phi^gamma = 0  (unreachable)              -> C = 1   (max cost)
        Phi^gamma small (loosely coupled)         -> C close to 1
        Phi^gamma = 1  (e.g. direct strong edge)  -> C = 0
        Phi^gamma > 1  (super-connected)          -> C = 0   (clipped)
        diagonal                                  -> C = 0

    Direction-mirrored: for any unordered pair (u, v), at most one of
    Phi(u,v), Phi(v,u) is non-zero on a strict forward DAG.

    Returns: [n_verts, n_verts] symmetric float64 in [0, 1], zero on diagonal.
    """
    import scipy.linalg

    n_verts = L * N
    W_abs = W_fwd.abs().cpu().numpy().reshape(n_verts, n_verts).astype(np.float64)

    # Q-sparsify (matches the rest of the sweep).
    W_sparse = np.where(W_abs > edge_threshold, W_abs, 0.0)

    # Solve (I - gamma W_sparse) X = W_sparse via triangular back-sub.
    # Result X = W (I - gamma W)^{-1} = sum_{k>=1} gamma^{k-1} W^k off-diagonal.
    A = np.eye(n_verts, dtype=np.float64) - gamma * W_sparse
    Phi = scipy.linalg.solve_triangular(A, W_sparse, lower=False)

    # Forward DAG: only one of (u,v), (v,u) has Phi > 0; max picks the
    # forward-direction value and yields a symmetric coupling matrix.
    Phi = np.maximum(Phi, Phi.T)

    # -log transform with clamped floor; normalise by -log(eps) to [0, 1].
    log_floor = -np.log(eps)
    C = -np.log(np.clip(Phi, eps, None)) / log_floor
    C = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(C, 0.0)
    return C


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
                 classification: Optional[Dict[Tuple[int, int], int]] = None,
                 *,
                 edge_threshold: float = 0.0,
                 edge_tensor: str = "W_softmax",
                 act_norm_method: str = "rank",
                 load_norm_method: str = "raw",
                 structural_mode: str = "path",
                 gamma: float = 1.0,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Construct the FGW triple (C, F, mass) for one routing DAG.

    The structural cost C is set directly to the chosen structural-mode
    descriptor (no depth mixing). The depth feature lives only in F, where
    the Wasserstein channel of FGW penalises cross-depth matches without
    double-encoding depth in C. (This corresponds to the legacy β=0 setting;
    β has been retired.)

    Args:
        dag: dict produced by build_dag.py; expects keys n_tokens_selected,
            top_weight, top_prompt, top_pos, moe_layers, plus the edge tensor
            selected via `edge_tensor` (default "W_softmax").
        classification: dict[(prompt_idx, position) -> class_idx] from
            build_token_classification. If None, the token-class histogram is
            set to all-mass-on-"special" (P6 contributes nothing).
        edge_threshold: drop edges with W < threshold before computing
            shortest paths. 0.0 = use the dense graph. See main.tex
            "Graph thresholding".
        edge_tensor: which 4D tensor in `dag` to use as the edge weight
            W[c, j, l, n]. Default "W_softmax" (the softmax-mass perturbation
            primary edge weight; cf. main.tex §2). Alternative options stored
            in the DAG: "W_softmax_var", "W_softmax_signed", "APS", "ANS".
            (The previous AARV/P_add/P_rem alternative edge weights were
            removed when switching to Approach 2 pairwise-isolated ablation;
            they live in the archived `approach1-frozen` branch.)

    Returns:
        C    : [n, n] np.float64  -- structural cost
        F    : [n, D] np.float64  -- features [depth, out~, in~, load, act~, class_0..class_{T-1}]
        mass : [n]    np.float64  -- probability mass, sums to 1
        meta : dict with L, N, n_verts, D
    """
    if edge_tensor not in dag:
        raise KeyError(
            f"edge_tensor={edge_tensor!r} not in dag (available keys: {sorted(dag)})"
        )
    W = dag[edge_tensor].float()
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
    #
    # Two normalisation choices:
    #   "raw"     : load(v) = n_tok(v) / mean_in_layer(n_tok). Mean = 1 per
    #                layer, range up to N. (Legacy; structurally heavier-weighted
    #                than other features in Euclidean L_2 because N varies from
    #                8 (Mixtral) to 128+ (Qwen3-235B). Audit at 2026-06-15
    #                showed 6/8 models have heavy-tailed load distributions.)
    #   "log_max" : per-layer log-max compression
    #                load(v) = log(1 + load_raw(v)) / log(1 + max_n' load_raw(ell, n'))
    #               Range [0, 1] per layer, scales comparably across N.
    n_tok = dag["n_tokens_selected"].float()  # [L, N]
    layer_mean = n_tok.mean(dim=1, keepdim=True).clamp(min=1e-12)  # [L, 1]
    load_raw = n_tok / layer_mean  # [L, N], mean = 1 per layer
    if load_norm_method == "raw":
        load = load_raw
    elif load_norm_method == "log_max":
        log_load = torch.log1p(load_raw)  # [L, N]
        log_max = torch.log1p(load_raw.max(dim=1, keepdim=True).values).clamp(min=1e-12)  # [L, 1]
        load = log_load / log_max  # [L, N], range [0, 1]
    else:
        raise ValueError(
            f"Unknown load_norm_method={load_norm_method!r}; "
            "expected 'raw' or 'log_max'."
        )

    # 5. activation magnitude (Su et al. super-expert feature).
    # Raw `act` spans 3-5 orders of magnitude within a graph and another
    # 100-10000x across models -- raw values would dominate the Wasserstein
    # term and break cross-model comparison.
    #
    # Two normalisation choices:
    #   "rank"     : within-graph rank in [0, 1]. Erases distribution shape:
    #                two graphs with very different act distributions look
    #                identical to the metric. (Legacy default.)
    #   "log_max"  : log(1 + act) / log(1 + max(act)). Preserves dynamic range
    #                and super-expert dominance while spreading the bulk of
    #                the distribution across [0, 1]. Motivated by the per-model
    #                Zipf curves (see main.tex, fig:act-distribution).
    act = dag["act"].float().reshape(-1)            # [n_verts]
    if act_norm_method == "rank":
        ranks = act.argsort().argsort().float()
        act_norm = (ranks / max(float(ranks.max()), 1.0)).reshape(L, N)
    elif act_norm_method == "log_max":
        log_act = torch.log1p(act)
        log_max = torch.log1p(act.max()).clamp(min=1e-12)
        act_norm = (log_act / log_max).reshape(L, N)
    else:
        raise ValueError(
            f"Unknown act_norm_method={act_norm_method!r}; "
            "expected 'rank' or 'log_max'."
        )

    # 6. token-class histogram
    if classification is not None and "top_prompt" in dag and "top_pos" in dag:
        class_hist = compute_class_histogram(
            dag["top_weight"], dag["top_prompt"], dag["top_pos"], classification)
    else:
        class_hist = torch.zeros(L, N, N_CLASSES, dtype=torch.float32)
        class_hist[..., _CLASS_IDX["special"]] = 1.0

    F = torch.cat([
        depth.unsqueeze(-1),        # [L, N, 1]
        out_norm.unsqueeze(-1),
        in_norm.unsqueeze(-1),
        load.unsqueeze(-1),
        act_norm.unsqueeze(-1),      # NEW: rank-normalised Su et al. act
        class_hist,                  # [L, N, N_CLASSES]
    ], dim=-1).reshape(n_verts, -1).double().numpy()

    # --- Mass ---
    out_flat = out_strength.reshape(-1).cpu().numpy().astype(np.float64)
    eps = 1e-6 * (out_flat.max() if out_flat.max() > 0 else 1.0)
    mass = out_flat + eps
    mass = mass / mass.sum()

    # --- Structural cost C ---
    if structural_mode == "path":
        C_struct_raw = _shortest_path_costs(W_fwd, L, N, edge_threshold=edge_threshold)
        C = C_struct_raw / max(L - 1, 1)  # normalise to [0, 1]
    elif structural_mode == "local":
        # Already in [0, 1]: 1 - |W| on direct edges, 1 elsewhere.
        C = _local_costs(W_fwd, L, N, edge_threshold=edge_threshold)
    elif structural_mode == "conn":
        # Already in [0, 1]: -log(Phi^gamma) / -log(eps), clipped.
        C = _conn_costs(W_fwd, L, N, edge_threshold=edge_threshold,
                        gamma=gamma)
    else:
        raise ValueError(
            f"Unknown structural_mode={structural_mode!r}; "
            "expected 'path', 'local', or 'conn'."
        )

    meta = {"L": L, "N": N, "n_verts": n_verts, "D": F.shape[1],
            "edge_threshold": edge_threshold,
            "structural_mode": structural_mode, "gamma": gamma}
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