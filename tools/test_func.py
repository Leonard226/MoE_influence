import torch
import einops
from fancy_einsum import einsum

def decomposed_attn_batch_for_test(v, pattern, model):
    """Decompose the attention output into the shape of [q, k, n_heads, n_dim1]
    Reference: https://github.com/facebookresearch/llm-transparency-tool/blob/f1340f0757b959c75c139f7aa91aef16eddced67/llm_transparency_tool/models/tlens_model.py#L287
    :param1 v: value matrix of attention layer | shape:[n_heads, n_tokens, dim_head]
    :param2 pattern: the matrix Q(K^T) | shape:[n_heads, q=n_tokens, k=n_tokens]
    :param3 layer_id: which layer? should be an int.
    :param4 model: assigned model
    :return: the decomposition result
    """
    # v.shape [n_prompts, n_layers, n_heads, n_tokens, dim_head]
    # pattern.shape [n_prompts, n_layers, n_heads=16, q=n_tokens, k=n_tokens]
    # z.shape [n_prompts, n_layers, q=n_tokens, k=n_tokens, n_heads=16, dim_head=128]
    # W_O.shape [n_layers, n_dim1=2048, n_dim2=2048] -> [n_heads=16, dim_head=128, n_dim1]
    # decomposed_attn.shape  [n_prompts, n_layers, q, k, n_heads, n_dim1]
    n_layers = v.shape[1]
    v = v.permute(0, 1, 3, 2, 4)
    z = einsum(
        "prompt layer key_pos head d_head, "
        "prompt layer head query_pos key_pos -> "
        "prompt layer query_pos key_pos head d_head",
        v,
        pattern,
    )
    W_O = torch.stack([model.model.layers[layer_id].self_attn.o_proj.weight for layer_id in range(n_layers)], dim=0) # olmoe
    W_O = einops.rearrange(
    W_O,
    "layer d_model (index d_head)->layer index d_head d_model",
    index=16,
    )
    decomposed_attn = einsum(
        "prompt layer pos key_pos head d_head, "
        "layer head d_head d_model -> "
        "prompt layer pos key_pos head d_model",
        z,
        W_O,
    )
    return decomposed_attn