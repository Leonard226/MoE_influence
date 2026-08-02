"""Shared score-decomposition / softmax-mass-perturbation core.

Used by both build_dag.py (single-node) and build_dag_multinode.py
(multi-node) so the W_softmax computation is defined exactly once. See
build_dag.py's module docstring for the definition of W_softmax and the
pairwise-isolated-ablation mechanism.

Architecture-specific behavior (main.tex, "Architecture-specific
variations") is injected via three per-model functions from
experiments/routing_variants.py — norm_denom_fn, ln_bar_fn, p_fn — plus a
scalar routing_scale. p_orig and p_pert are computed by the SAME p_fn call
(p_orig from orig_score, p_pert from the per-pair perturbed score), so a
model's routing rule is encoded exactly once, not twice.

Shared experts (main.tex, "Shared experts"): unlike top-K senders, a shared
expert has no incoming edges (its output doesn't depend on routing) and its
outgoing edges are unconditional — active for every token, not just tokens
that selected it. So its edges use the same ln_bar_fn/p_fn machinery as
regular senders, but with the shared expert's raw (unweighted) output in
place of a routing-weighted omega_S, and a plain sum over ALL tokens instead
of an index_add_ gated by selection. Optional: models without shared experts
pass shared_expert_output=None and this block is skipped entirely.
"""
import torch

from experiments.helper import update_topk_per_sender


def accumulate_wsm(
    after_res1, after_norm2, selected, weighted_out,   # [bsz, L, n_tok, ...] hooks, MoE layers only
    G_recv, gamma_recv, EPS,
    N_LAYERS, N_EXPERTS, TOP_K, D_E, device,
    input_ids, prompt_offset,
    wsm_accum, n_tokens_selected,
    top_weight, top_prompt, top_pos, top_token, K_TOP_TOKENS,
    norm_denom_fn, ln_bar_fn, p_fn, routing_scale=1.0,
    shared_expert_output=None, wsm_shared_accum=None, n_tokens_total=None,
):
    """Update wsm_accum / n_tokens_selected / top-K token buffers for one batch.

    All accumulator arguments are mutated in place; nothing is returned.

    shared_expert_output: [bsz, L, n_tok, d_e] hook (MoE layers only), or None
    for models with no shared experts. When given, wsm_shared_accum and
    n_tokens_total must also be given (see module docstring).
    """
    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # 1 / (frozen normalization denominator), per token, per receiver layer R.
    denom_inv = norm_denom_fn(after_res1.float(), EPS).permute(0, 2, 1).reshape(bt, N_LAYERS)  # [bt, L_recv]

    # Original assignment scores at every receiver layer.
    after_norm2_r = (after_norm2.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E))   # [bt, L, d_e]
    orig_score = torch.einsum("lnd,bld->bln", G_recv, after_norm2_r)       # [bt, L, N_EXPERTS]

    # Sender-side reshapes: [bt, S, k, ...]
    omega = (weighted_out.float().permute(0, 2, 1, 3, 4).reshape(bt, N_LAYERS, TOP_K, D_E))   # [bt, S, k, d_e]
    sel = (selected.long().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, TOP_K))                  # [bt, S, k]

    # p^{l,n}(i) at every layer, via the model's own p_fn — zero outside the
    # selected set. Used both as sender S's own routing weight (top-K token
    # buffer, gathered at the real hook-sourced sel) and as receiver R's
    # p_orig(n) in the perturbation delta below.
    p_orig_full = p_fn(orig_score, TOP_K) * routing_scale               # [bt, L, N_EXPERTS]

    # Per-event auxiliary indices (layer-independent), used by the top-K buffer update.
    event_bt   = torch.arange(bt, device=device).repeat_interleave(TOP_K)                     # [bt*top_k]
    event_bsz  = event_bt // n_tok
    event_pos  = (event_bt % n_tok).to(torch.int16)                                           # [bt*top_k]
    prompt_indices = torch.arange(prompt_offset, prompt_offset + bsz, dtype=torch.int32, device=device)  # [bsz]
    event_prompt   = prompt_indices[event_bsz]                                                # [bt*top_k] int32
    event_token    = input_ids.flatten()[event_bt].to(torch.int32)                            # [bt*top_k] int32

    for S in range(N_LAYERS):
        sel_S = sel[:, S, :]                                    # [bt, top_k]
        n_tokens_selected[S] += torch.bincount(sel_S.flatten(), minlength=N_EXPERTS)

        # Update top-K-by-routing-weight token buffer for sender (S, j).
        sender_weight = torch.gather(p_orig_full[:, S, :], dim=-1, index=sel_S)  # [bt, top_k]
        update_topk_per_sender(
            top_weight[S], top_prompt[S], top_pos[S], top_token[S],
            sel_S.flatten(), sender_weight.flatten(),
            event_prompt, event_pos, event_token,
            N_EXPERTS, K_TOP_TOKENS, max_per_j=bt,
        )

        if S == N_LAYERS - 1:
            continue
        omega_S = omega[:, S, :, :]                             # [bt, top_k, d_e]

        for R in range(S + 1, N_LAYERS):
            # ln_bar^R(omega_S) — linearized normalization, per architecture's ln_bar_fn.
            ln_bar = ln_bar_fn(omega_S, gamma_recv[R].view(1, 1, D_E), denom_inv[:, R].view(bt, 1, 1))  # [bt, k, d_e]
            # scores[bt, k, n] = g^{R,n} · ln_bar  — per-edge sub-score
            #   S(g^{R,n}, e^{S,j_k}_{out,i})  with j_k = sel_S[bt, k].
            scores = torch.einsum("ed,bkd->bke", G_recv[R], ln_bar)  # [bt, k, N_EXPERTS]

            sel_flat = sel_S.flatten()

            # ---- Softmax-mass perturbation (per-pair isolated ablation) ----
            # For each edge (sender k, receiver n) we ablate sender k's contribution
            # to n ONLY (not to other receivers at the same layer). Each edge has its
            # OWN perturbed score vector; p_fn is re-run on it from scratch.
            #
            #   pert_score_pair[b, k, n, m] = orig_score[b, R, m]                  if m ≠ n
            #                               = orig_score[b, R, n] - scores[b, k, n]  if m == n
            #
            # After p_fn we read off the diagonal (m == n) to obtain p_pert_{k→n}(n).
            #
            # Memory: pert_score_pair, p_pert_full each take [bt, TOP_K, N, N]
            # float32 = ~520 MB at (bt=1000, K=8, N=128). Built, consumed, `del`'d
            # within this iteration.
            N = N_EXPERTS
            diag_idx = torch.arange(N, device=device)

            # Build per-pair perturbed scores.
            pert_score_pair = (orig_score[:, R, :]
                               .view(bt, 1, 1, N)
                               .expand(bt, TOP_K, N, N)
                               .clone())                                           # [bt, K, N, N]
            # Subtract scores[b, k, n] from the diagonal in m: pert_score_pair[b, k, n, n].
            pert_score_pair[:, :, diag_idx, diag_idx] = (
                pert_score_pair[:, :, diag_idx, diag_idx] - scores
            )

            p_pert_full = p_fn(pert_score_pair, TOP_K) * routing_scale             # [bt, K, N, N]
            del pert_score_pair

            # p_pert_n[b, k, n] = p_pert under the (k → n) per-pair perturbation,
            # at receiver index n itself (the diagonal of the [N, N] slab).
            p_pert_n = p_pert_full[:, :, diag_idx, diag_idx]                       # [bt, K, N]
            del p_pert_full

            # p_orig(n) at receiver R, zero outside top-K. Same for every sender k → broadcast over K.
            p_orig_R = p_orig_full[:, R, :].unsqueeze(1).expand(bt, TOP_K, N_EXPERTS)  # [bt, K, N]

            # delta_{c,j,l,n} = p_orig(n) - p_pert_{c→n}(n) ∈ [-1, 1]. wsm = E[|delta|].
            delta = p_orig_R - p_pert_n                                            # [bt, K, N]
            wsm_accum[S, :, R, :].index_add_(0, sel_flat, delta.abs().flatten(0, 1))

            del ln_bar, scores
            del p_orig_R, p_pert_n, delta

    # ---- Shared-expert edges: unconditional, every layer -> every downstream receiver ----
    if shared_expert_output is not None:
        n_tokens_total += bt
        e_shared = (shared_expert_output.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E))  # [bt, L, d_e]

        for ell in range(N_LAYERS):
            if ell == N_LAYERS - 1:
                continue
            e_shared_ell = e_shared[:, ell, :]                      # [bt, d_e]

            for R in range(ell + 1, N_LAYERS):
                ln_bar = ln_bar_fn(e_shared_ell, gamma_recv[R].view(1, D_E), denom_inv[:, R].view(bt, 1))  # [bt, d_e]
                scores = torch.einsum("ed,bd->be", G_recv[R], ln_bar)  # [bt, N_EXPERTS]

                N = N_EXPERTS
                diag_idx = torch.arange(N, device=device)

                pert_score = (orig_score[:, R, :]
                             .view(bt, 1, N)
                             .expand(bt, N, N)
                             .clone())                               # [bt, N, N]
                pert_score[:, diag_idx, diag_idx] = (
                    pert_score[:, diag_idx, diag_idx] - scores
                )

                p_pert_full = p_fn(pert_score, TOP_K) * routing_scale  # [bt, N, N]
                del pert_score

                p_pert_n = p_pert_full[:, diag_idx, diag_idx]        # [bt, N]
                del p_pert_full

                p_orig_R = p_orig_full[:, R, :]                      # [bt, N]

                # No selection to condition on — shared experts are active for
                # every token, so this is a plain sum over all bt tokens.
                delta = p_orig_R - p_pert_n                          # [bt, N]
                wsm_shared_accum[ell, R, :] += delta.abs().sum(dim=0)

                del ln_bar, scores, p_orig_R, p_pert_n, delta
