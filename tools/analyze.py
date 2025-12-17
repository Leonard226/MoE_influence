import torch
import einops
def rmsnorm_breakdown_batch(vector, components, model, mode="default", variance_epsilon=1e-05, device="cuda:0"):
    n_prompts_B, n_layers, n_tokens, n_dim = vector.shape
    variance = vector.to(device).pow(2).mean(-1, keepdim=True) # shape: [n_prompts_B, n_layers, n_tokens, 1]
    rsqrt = torch.rsqrt(variance + variance_epsilon) # shape: [n_prompts_B, n_layers, n_tokens, 1]
    weight = torch.stack([model.model.layers[layer_id].post_attention_layernorm.weight.data.to(device) for layer_id in range(n_layers)], dim=0) # shape: [n_layers, n_dim]
    weight = weight.repeat(n_prompts_B * n_tokens, 1, 1).reshape(n_prompts_B, n_tokens, n_layers, n_dim).permute(0, 2, 1, 3) # shape: [n_prompts_B, n_layers, n_tokens, n_dim]
    if mode == "TAM": # P: prompt, R: receiving layer, S: sending layer, T: token, D: n_dim
        breakdowns = [torch.einsum("PRTD,PSTD->PRSTD", rsqrt * weight, i.to(device)) for i in components]
    elif mode == "H": # P: prompt, R: receiving layer, S: sending layer, Q: q_token, K: k_token, H: head, D: n_dim
        breakdowns = [torch.einsum("PRQD,PSQKHD->PRSQKHD", rsqrt * weight, i.to(device)) for i in components]
    elif mode == "H_simplified": # unused in this file
        n_heads = components[0].shape[4]
        print(n_heads)
        tmp_results = torch.zeros((n_prompts_B, n_layers, n_tokens, n_tokens, n_heads, n_dim))
        for j in range(n_layers):
            tmp_results[:, j, ...] = torch.einsum("PQD,PQKHD->PQKHD", (rsqrt * weight)[:, j, ...], components[0][:, j, ...].to(device))
        breakdowns = [tmp_results]
    elif mode == "H_agnostic":
        breakdowns = [torch.einsum("PRQD,PSQHD->PRSQHD", rsqrt * weight, i.to(device)) for i in components]
    elif mode == "E": # P: prompt, R: receiving layer, S: sending layer, T: token, D: n_dim, E: n_experts
        breakdowns = [torch.einsum("PRTD,PSTED->PRSTED", rsqrt * weight, i.to(device)) for i in components]
    return breakdowns

def decompose_attn_out_helper_batch(v, pattern, n_layers, model):
    """ Decompose the attention output into the shape of [n_prompts_B, n_layers, n_tokens, n_tokens, n_heads, n_dim] (PSQKHD)
    Reference: https://github.com/facebookresearch/llm-transparency-tool/blob/f1340f0757b959c75c139f7aa91aef16eddced67/llm_transparency_tool/models/tlens_model.py#L287
    """
    # P: prompt, S: sending_layer, H: head, K: key_pos, A: n_head_dim, Q: query_pos, D: n_dim (hidden state dim)
    z = torch.einsum("PSHKA,PSHQK->PSQKHA", v, pattern)
    W_O = torch.stack([model.model.layers[layer_id].self_attn.o_proj.weight for layer_id in range(n_layers)])
    W_O = einops.rearrange(W_O, "n_layers d_model (index d_head)->n_layers index d_head d_model", index=16) # n_heads = 16
    decomposed_attn = torch.einsum("PSQKHA,SHAD->PSQKHD", z, W_O.float())
    return decomposed_attn