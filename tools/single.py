import torch
from tools.verbose import rmsnorm_breakdown, decompose_attn_out_helper

def decompose_TAM_single(prompt_ls, model, tokenizer, router_weight_ls, top_n):
    """ Decomposition: single prompt, token(T), attn_out(A), and moe_out(M).
        Note that top_n can vary - top_k or n_experts or other ranges.
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])

    layer_input = hook_dict["hook_layer_input"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    attn_output = hook_dict["hook_attn_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    mlp_output = hook_dict["hook_mlp_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]

    n_layers = len(router_weight_ls)
    n_experts = router_weight_ls[1].shape[0]

    token_score = torch.zeros((n_tokens, n_layers, n_experts, 1))
    token_cumulative_score = torch.zeros((n_tokens, n_layers, 1))
    token_abs_cumulative_score = torch.zeros((n_tokens, n_layers, 1))

    attn_score = torch.zeros((n_tokens, n_layers, n_experts, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    attn_cumulative_score = torch.zeros((n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    attn_abs_cumulative_score = torch.zeros((n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer

    moe_score = torch.zeros((n_tokens, n_layers, n_experts, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    moe_cumulative_score = torch.zeros((n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    moe_abs_cumulative_score = torch.zeros((n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer

    for L in range(n_layers):
        router_weight_vectors = router_weight_ls[L] # shape: [n_experts, n_dim]
        for T in range(n_tokens):
            ## decomposition: moe_in = token + (multiple) attn_out + (multiple) moe_out
            token_components = layer_input[0, T].reshape(1, -1) # res_in of Layer 0
            attn_components = attn_output[:(L+1), T] # attn_out of Layer 0~L
            moe_components = mlp_output[:L, T] # moe_out of Layer 0~(L-1)
            
            token_rmsnorm, attn_rmsnorm, moe_rmsnorm = rmsnorm_breakdown(after_res1[L, T], [token_components, attn_components, moe_components], L, model)
            ## NOTE: shape: token_rmsnorm [1, n_dim]; attn_rmsnorm [L+1, n_dim]; moe_rmsnorm [L, n_dim]

            ## score: original_score = token_score + (multiple) attn_out_score + (multiple) moe_out_score
            original_score = torch.matmul(router_weight_vectors, after_norm2[L, T])
            top_n_experts = torch.argsort(original_score, descending=True)[:top_n]

            token_score[T, L] = torch.matmul(router_weight_vectors, token_rmsnorm.T)  # shape: [n_experts, 1]
            attn_score[T, L, :, :(L+1)] = torch.matmul(router_weight_vectors, attn_rmsnorm.T)  # shape: [n_experts, L+1]
            moe_score[T, L, :, :L] = torch.matmul(router_weight_vectors, moe_rmsnorm.T)  # shape: [n_experts, L]
            
            token_cumulative_score[T, L] = torch.sum(token_score[T, L, top_n_experts, :], dim=0) # sum t
            token_abs_cumulative_score[T, L] = torch.sum(torch.abs(token_score[T, L, top_n_experts, :]), dim=0) # sum (abs t)
            attn_cumulative_score[T, L] = torch.sum(attn_score[T, L, top_n_experts, :], dim=0) # sum a
            attn_abs_cumulative_score[T, L] = torch.sum(torch.abs(attn_score[T, L, top_n_experts, :]), dim=0) # sum (abs a)
            moe_cumulative_score[T, L] = torch.sum(moe_score[T, L, top_n_experts, :], dim=0) # sum m
            moe_abs_cumulative_score[T, L] = torch.sum(torch.abs(moe_score[T, L, top_n_experts, :]), dim=0) # sum (abs m)
            
    return token_score, attn_score, moe_score, moe_cumulative_score, attn_cumulative_score, token_cumulative_score, moe_abs_cumulative_score, attn_abs_cumulative_score, token_abs_cumulative_score

def decompose_H_single(prompt_ls, model, tokenizer, router_weight_ls, top_n):
    """ Decomposition: single prompt, head(H).
        Note that top_n can vary - top_k or n_experts or other ranges.
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])

    attn_v = hook_dict["hook_v"].squeeze(0) # shape: [n_layers, n_heads, n_tokens, head_dim]
    attn_weights = hook_dict["hook_attn_weights"].squeeze(0) # shape: [n_layers, n_heads, n_tokens, n_tokens] # first n_tokens -> queries , second n_tokens -> keys
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]

    n_layers = len(router_weight_ls)
    n_experts, n_dim = router_weight_ls[1].shape
    n_heads = attn_v.shape[1]

    expert_ids_by_rank = torch.zeros((n_layers, n_tokens, n_experts), dtype=torch.int)

    for send_L in range(0, n_layers): # sending layer
        decomposed_attn_out = decompose_attn_out_helper(attn_v[send_L], attn_weights[send_L], send_L, model) # shape: [n_tokens, n_tokens, n_heads, n_dim] first n_tokens: queries, second n_tokens: keys
        for recv_L in range(send_L, n_layers): # receiving layer
            router_weight_vectors = router_weight_ls[recv_L]
            for T in range(n_tokens): # examined token (query token)
                ## decomposition
                decomposed_attn_rmsnorm = rmsnorm_breakdown(after_res1[recv_L, T], [decomposed_attn_out[T].reshape(-1, n_dim)], recv_L, model)[0] # shape: [n_tokens * n_heads, n_dim]
                
                ## score
                decomposed_attn_score = torch.matmul(router_weight_vectors, decomposed_attn_rmsnorm.T) # shape: [n_experts, n_tokens * n_heads]
                
                ## expert selection
                original_score = torch.matmul(router_weight_vectors, after_norm2[recv_L, T])
                expert_ids_by_rank[recv_L, T] = torch.argsort(original_score, descending=True)
                top_n_experts = expert_ids_by_rank[recv_L, T][:top_n] # torch.argsort(original_score, descending=True)[:top_n]