"""Per-model architecture variants for the score decomposition (main.tex,
"Architecture-specific variations"). Each axis of variation is a small,
independently-checkable function; a model's MODELS registry entry selects
which ones apply to it. Models that need extra config (SparseMixer's
jitter_eps, DeepSeek's group_limit) get a functools.partial with that config
baked in, so every call site sees the same plain signature.

Default (RMSNorm, renormalized top-K softmax, no scaling, no group limit) is
what all four "clean" models (Mixtral x2, Qwen3 x2) use unmodified.
"""
import torch


# ---------------------------------------------------------------------------
# Frozen normalization denominator: 1 / RMS(x) or 1 / sigma(x), computed once
# per (token, receiver layer) from the real, observed residual-stream state.
# ---------------------------------------------------------------------------
def norm_denom_rmsnorm(after_res1, eps):
    """after_res1: [..., d_e]. Returns [...] = 1 / RMS(x)."""
    return torch.rsqrt(after_res1.pow(2).mean(dim=-1) + eps)


def norm_denom_layernorm(after_res1, eps):
    """after_res1: [..., d_e]. Returns [...] = 1 / sigma(x), population variance."""
    mu = after_res1.mean(dim=-1, keepdim=True)
    var = (after_res1 - mu).pow(2).mean(dim=-1)
    return torch.rsqrt(var + eps)


# ---------------------------------------------------------------------------
# Linearized normalization: distributes a sender's individual contribution e
# through the (frozen-denominator) normalization. gamma: [d_e]. denom_inv:
# [..., 1]-broadcastable, from norm_denom_* above. Bias (LayerNorm's beta) is
# NOT included here — it's sender-independent and already present, exactly,
# in orig_score via the real forward-pass hook; it only needs to be omitted
# from the per-sender marginal contribution, not added back anywhere.
# ---------------------------------------------------------------------------
def ln_bar_rmsnorm(e, gamma, denom_inv):
    """e: [..., d_e]. Pure scaling, no mean-centering."""
    return e * gamma * denom_inv


def ln_bar_layernorm(e, gamma, denom_inv):
    """e: [..., d_e]. Mean-centered per e (linear in e), then scaled."""
    e_centered = e - e.mean(dim=-1, keepdim=True)
    return e_centered * gamma * denom_inv


# ---------------------------------------------------------------------------
# p_fn: scores [..., N_EXPERTS] -> p [..., N_EXPERTS], zero outside the
# selected set. This is the single source of truth for "what weight would
# this model assign", used identically for p_orig and for the per-pair
# perturbed reconstruction — there is no separate recomputation path.
# ---------------------------------------------------------------------------
def p_renorm_topk_softmax(scores, top_k):
    """Default: full softmax, top-K, renormalize the top-K to sum to 1."""
    softmax_full = torch.softmax(scores, dim=-1)
    topk_vals, topk_idx = torch.topk(softmax_full, top_k, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    p = torch.zeros_like(softmax_full)
    p.scatter_(-1, topk_idx, topk_vals)
    return p


def _group_eligibility_mask(softmax_full, n_group, topk_group):
    """softmax_full: [..., N]. Returns bool [..., N], True where the expert's
    group is among the topk_group groups (by per-group max probability)."""
    *lead, N = softmax_full.shape
    group_scores = softmax_full.view(*lead, n_group, N // n_group).max(dim=-1).values  # [..., n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1)[1]                       # [..., topk_group]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(-1, group_idx, 1)
    return group_mask.unsqueeze(-1).expand(*lead, n_group, N // n_group).reshape(*lead, N).bool()


def p_no_renorm_softmax(scores, top_k, group_limit=None):
    """OLMoE, DeepSeek: full softmax, top-K, no renormalization (doesn't sum
    to 1). group_limit, if set, is (n_group, topk_group): DeepSeek-V2's
    group-then-expert eligibility filter, applied to the softmax
    probabilities (NOT the raw logits) before top-K selection — matching
    MoEGate.forward's tmp_scores = scores.masked_fill(~group_mask, 0.0).
    """
    softmax_full = torch.softmax(scores, dim=-1)
    if group_limit is not None:
        n_group, topk_group = group_limit
        mask = _group_eligibility_mask(softmax_full, n_group, topk_group)
        softmax_full = softmax_full.masked_fill(~mask, 0.0)
    topk_vals, topk_idx = torch.topk(softmax_full, top_k, dim=-1)
    p = torch.zeros_like(softmax_full)
    p.scatter_(-1, topk_idx, topk_vals)
    return p


def p_sparsemixer(scores, top_k, jitter_eps):
    """Phi-3.5-MoE. Two sequential masked softmaxes (main.tex's M_1, M_2),
    eval-mode path only (argmax selection, no stochastic jitter) — matches
    sparsemixer()'s `training=False` branch in modeling_phimoe_customized.py
    exactly. top_k must be 2.
    """
    if top_k != 2:
        raise ValueError("SparseMixer only supports top_k=2")

    # Stage 1: n_1* = argmax, survivor set M_1 (relative distance from max).
    max1, n1 = scores.max(dim=-1, keepdim=True)
    factor1 = scores.abs().clamp(min=max1)
    mask1 = ((max1 - scores) / factor1) > (2 * jitter_eps)
    omega1 = torch.softmax(scores.masked_fill(mask1, float("-inf")), dim=-1)
    p1 = omega1.gather(-1, n1)

    # Stage 2: mask out n_1*, repeat on the remainder for n_2*, M_2. The
    # threshold factor/comparison use the ORIGINAL scores (not stage-2-
    # masked), matching upstream; position n_1* is already -inf regardless.
    scores2 = scores.scatter(-1, n1, float("-inf"))
    max2, n2 = scores2.max(dim=-1, keepdim=True)
    factor2 = scores.abs().clamp(min=max2)
    mask2 = ((max2 - scores) / factor2) > (2 * jitter_eps)
    omega2 = torch.softmax(scores2.masked_fill(mask2, float("-inf")), dim=-1)
    p2 = omega2.gather(-1, n2)

    p = torch.zeros_like(scores)
    p.scatter_(-1, n1, p1)
    p.scatter_(-1, n2, p2)
    return p
