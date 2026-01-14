import torch
import einops
from tools.plot import tril_drawer_TAM, matrix_drawer_H_token_head, scatter_drawer_H_expert, scatter_drawer_H_head, matrix_drawer_H_with_sum, M_drawer, matrix_attn_weight_verbose, matrix_attn_weight_comparison_verbose
from tqdm import tqdm
def rmsnorm_breakdown(vector, components, layer_id, model, variance_epsilon=1e-05):
    """ Apply RMSNorm on the components (Single prompt)
    :param1 vector: the input of a RMSNorm
    :param2 components: components of the "vector". Note that their sum should be equal to the "vector".
    :param3 layer_id: which layer? Should be an integer.
    :param4 model: assigned model
    :param5 variance_epsilon: term for numerical stability
    :return: a list of RMSNorm-ed components
    """
    variance = vector.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + variance_epsilon)
    weight = model.model.layers[layer_id].post_attention_layernorm.weight.data
    breakdowns = [weight * (i * rsqrt) for i in components]
    return breakdowns

def logit_diff(logits, token_id1, token_id2):
    """ Logit difference, metric for patching (Single prompt) """
    return logits[token_id1] - logits[token_id2]

def prob(logits, token_id):
    """ Probability, metric for patching (Single prompt) """
    probs = torch.softmax(logits, dim=0)
    return probs[token_id]

def decompose_TAM_verbose(prompt_ls, model, tokenizer, router_weight_ls, top_n, output_dir):
    """ Decomposition: single prompt, token(T), attn_out(A), and moe_out(M).
        Note that top_n can vary - top_k or n_experts or other ranges.
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]

    layer_input = hook_dict["hook_layer_input"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    attn_output = hook_dict["hook_attn_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    mlp_output = hook_dict["hook_mlp_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    print(tokens_str)
    print(layer_input.shape, attn_output.shape, after_res1.shape, after_norm2.shape, mlp_output.shape)

    n_layers = len(router_weight_ls)
    n_experts = router_weight_ls[1].shape[0] # DeepSeekMoE layer 0 is not MoE, so here we preset it to 1

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
            # if T != (n_tokens - 1):
            #     continue
            print("layer{}, token{}: {}".format(L, T, tokens_str[T]))

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
            
            ## contribution of token: negative -> green; non-negative -> blue
            # out_str = ""
            # for n in top_n_experts:
            #     cur_score = token_score[T, L, n]
            #     if cur_score >= 0: # blue
            #         out_str += "\033[1m\033[44m{: .2f} ".format(cur_score.item())
            #     else: # green
            #         out_str += "\033[1m\033[42m{: .2f} ".format(cur_score.item())
            # out_str +="\033[0m"
            # print(out_str)
            
            ## contribution of attn_out: negative -> green; non-negative -> blue
            # send_L = 2 # just an example, you can select other layers
            # if L < send_L:
            #     continue # cannot contribute to previous layers
            # out_str = ""
            # for n in top_n_experts:
            #     cur_score = attn_score[T, L, n, send_L]
            #     if cur_score >= 0: # blue # check the contribution of attn Layer 2 on other layers
            #         out_str += "\033[1m\033[44m{: .2f} ".format(cur_score.item())
            #     else: # green
            #         out_str += "\033[1m\033[42m{: .2f} ".format(cur_score.item())
            # out_str +="\033[0m"
            # print(out_str)
            
            ## contribution of moe_out: negative -> green; non-negative -> blue
            # send_L = 2 # just an example, you can select other layers
            # if L <= send_L:
            #     continue # cannot contribute to current and previous layers
            # out_str = ""
            # for n in top_n_experts:
            #     cur_score = moe_score[T, L, n, send_L]
            #     if cur_score >= 0: # blue # check the contribution of attn Layer 2 on other layers
            #         out_str += "\033[1m\033[44m{: .2f} ".format(cur_score.item())
            #     else: # green
            #         out_str += "\033[1m\033[42m{: .2f} ".format(cur_score.item())
            # out_str +="\033[0m"
            # print(out_str)

            ## checker
            # print("CHECK: shape:", token_rmsnorm.shape, attn_rmsnorm.shape, moe_rmsnorm.shape, token_score.shape, attn_score.shape, moe_score.shape)
            # component_sum = torch.sum(attn_rmsnorm, 0) + torch.sum(moe_rmsnorm, 0) + token_rmsnorm
            # print("CHECK: input:", component_sum[:, :3], after_norm2[L, T, :3]) # component_sum MUST be equal to the original rmsnorm (precision error is acceptable)
            # component_score_sum = torch.sum(moe_score[T, L, 0]) + torch.sum(attn_score[T, L, 0]) + token_score[T, L, 0]
            # print("CHECK: score:", component_score_sum, original_score[0]) # component_score_sum MUST be equal to the original score (precision error is acceptable)

            ## print
            # print("T", token_score[T, L, top_n_experts]) # scores assigned by tokens
            # print("A", attn_score[T, L, top_n_experts]) # scores assigned by attn_out
            # print("M", moe_score[T, L, top_n_experts]) # scores assigned by moe_out
            # print("A + M", attn_score[T, L, top_n_experts] + moe_score[T, L, top_n_experts]) # scores assigned by previous layers to current layer
            # print("last A", attn_score[T, L, top_n_experts, L]) # score assigned by attn_out in the current layer

            # print("sum (abs T) i.e., abs T", torch.sum(torch.abs(token_score[T, L, top_n_experts]))) # sum of absolute scores assigned by the input token
            # print("sum (abs A)", torch.sum(torch.abs(attn_score[T, L, top_n_experts]))) # sum of absolute scores assigned by attn_out (Layer ~L)
            # print("sum (abs M)", torch.sum(torch.abs(moe_score[T, L, top_n_experts]))) # sum of absolute scores assigned by moe_out (Layer ~(L-1))

            # print("abs A", torch.sum(torch.abs(attn_score[T, L, top_n_experts]),dim=0)) # absolute scores assigned by attn_out in each layer (Layer ~L)
            # print("abs M", torch.sum(torch.abs(moe_score[T, L, top_n_experts]),dim=0)) # absolute scores assigned by moe_out in each layer (Layer ~(L-1))
            # print("abs (A+M)", torch.sum(torch.abs(attn_score[T, L, top_n_experts, :L] + moe_score[T, L, top_n_experts, :L]), dim=0)) # absolute scores assigned by each layer (Layer ~(L-1))
            # print("abs (last A)", torch.abs(attn_score[T, L, top_n_experts, L])) # absolute scores assigned by attn_out in the current layer
            # print("L2-norm A", torch.norm(attn_rmsnorm, p=2, dim=1)) # L2-norm of attn_rmsnorm
            # print("L2-norm M", torch.norm(moe_rmsnorm, p=2, dim=1)) # L2-norm of moe_rmsnorm
    
    ## Example: score var; abs cumulative score; cumulative score; positive cumulative score; negative cumulative score
    T = 9
    attn_score_var = torch.var(attn_score, dim=2, correction=0)
    moe_score_var = torch.var(moe_score, dim=2, correction=0)
    token_score_var = torch.var(token_score, dim=2, correction=0)
    # moe score var
    tril_drawer_TAM(moe_score_var[T], "moe_score_var_T{}".format(T), output_dir, (11, 11), diagonal=0, add_patch=[], title="moe_score_var_T{}".format(T), xlabel="MoE Layer (Sending)", ylabel="MoE Layer (Receiving)", is_variance=True)
    # attn score var
    tril_drawer_TAM(attn_score_var[T], "attn_score_var_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="attn_score_var_T{}".format(T), xlabel="Attention Layer (Sending)", ylabel="MoE Layer (Receiving)", is_variance=True)
    # moe absolute cumulative score
    tril_drawer_TAM(moe_abs_cumulative_score[T], "moe_abs_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=0, add_patch=[], title="moe_abs_cumulative_score_T{}".format(T), xlabel="MoE Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # moe cumulative score
    tril_drawer_TAM(moe_cumulative_score[T], "moe_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=0, add_patch=[], title="moe_cumulative_score_T{}".format(T), xlabel="MoE Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # attn absolute cumulative score
    tril_drawer_TAM(attn_abs_cumulative_score[T], "attn_abs_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="attn_abs_cumulative_score_T{}".format(T), xlabel="Attention Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # attn cumulative score
    tril_drawer_TAM(attn_cumulative_score[T], "attn_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="attn_cumulative_score_T{}".format(T), xlabel="Attention Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # moe positive cumulative score
    tril_drawer_TAM((moe_cumulative_score[T] + moe_abs_cumulative_score[T]) / 2, "moe_positive_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=0, add_patch=[], title="moe_positive_cumulative_score_T{}".format(T), xlabel="MoE Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # moe negative cumulative score
    tril_drawer_TAM((moe_cumulative_score[T] - moe_abs_cumulative_score[T]) / 2, "moe_negative_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=0, add_patch=[], title="moe_negative_cumulative_score_T{}".format(T), xlabel="MoE Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # attn positive cumulative score
    tril_drawer_TAM((attn_cumulative_score[T] + attn_abs_cumulative_score[T]) / 2, "attn_positive_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="attn_positive_cumulative_score_T{}".format(T), xlabel="Attention Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # attn negative cumulative score
    tril_drawer_TAM((attn_cumulative_score[T] - attn_abs_cumulative_score[T]) / 2, "attn_negative_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="attn_negative_cumulative_score_T{}".format(T), xlabel="Attention Layer (Sending)", ylabel="MoE Layer (Receiving)")
    # attn cumulative score plus moe cumulative score
    tril_drawer_TAM((attn_cumulative_score[T] + moe_cumulative_score[T]), "layer_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="layer_cumulative_score_T{}".format(T), xlabel="Attention + MoE Layer (Sending)", ylabel="MoE Layer (Receiving)")
    
    # token
    tril_drawer_TAM(token_score_var[T], "token_score_var_T{}".format(T), output_dir, (11, 9), diagonal=1, add_patch=[], title="token_score_var_T{}".format(T), xlabel="Token (Sending)", ylabel="MoE Layer (Receiving)", is_variance=True)
    tril_drawer_TAM(token_abs_cumulative_score[T], "token_abs_cumulative_score_T{}".format(T), output_dir, (11, 7), diagonal=1, add_patch=[], title="token_abs_cumulative_score_T{}".format(T), xlabel="Token (Sending)", ylabel="MoE Layer (Receiving)")
    tril_drawer_TAM(token_cumulative_score[T], "token_cumulative_score_T{}".format(T), output_dir, (11, 7), diagonal=1, add_patch=[], title="token_cumulative_score_T{}".format(T), xlabel="Token (Sending)", ylabel="MoE Layer (Receiving)")
    tril_drawer_TAM((token_cumulative_score[T] + token_abs_cumulative_score[T]) / 2, "token_positive_cumulative_score_T{}".format(T), output_dir, (11, 7), diagonal=1, add_patch=[], title="token_positive_cumulative_score_T{}".format(T), xlabel="Token (Sending)", ylabel="MoE Layer (Receiving)")
    tril_drawer_TAM((token_cumulative_score[T] - token_abs_cumulative_score[T]) / 2, "token_negative_cumulative_score_T{}".format(T), output_dir, (11, 7), diagonal=1, add_patch=[], title="token_negative_cumulative_score_T{}".format(T), xlabel="Token (Sending)", ylabel="MoE Layer (Receiving)")

    return token_score, attn_score, moe_score, moe_cumulative_score, attn_cumulative_score, token_cumulative_score, moe_abs_cumulative_score, attn_abs_cumulative_score, token_abs_cumulative_score

def decompose_attn_out_helper(v, pattern, layer_id, model):
    """ Decompose the attention output into the shape of [q, k, n_heads, n_dim1] Reference: https://github.com/facebookresearch/llm-transparency-tool/blob/f1340f0757b959c75c139f7aa91aef16eddced67/llm_transparency_tool/models/tlens_model.py#L287
    :param1 v: value matrix of attention layer | shape:[n_heads, n_tokens, dim_head]
    :param2 pattern: the matrix Q(K^T) | shape:[n_heads, q=n_tokens, k=n_tokens]
    :param3 layer_id: which layer? should be an int.
    :param4 model: assigned model
    :return: the decomposition result
    """
    # OLMoE:
    # v.shape [n_heads=16, n_tokens, dim_head=128]
    # pattern.shape [n_heads=16, q=n_tokens, k=n_tokens]
    # z.shape [q=n_tokens, k=n_tokens, n_heads=16, dim_head=128]
    # W_O.shape [dim_model_1=2048, dim_model_2=2048] -> [n_heads=16, dim_head=128, dim_model_1] (dim_model_2 is decomposed)
    # decomposed_attn.shape  [q, k, n_heads, dim_model]
    # K: key_pos, H: head Q: query_pos, A: dim_head, D: dim_model (hidden state dim)
    z = torch.einsum("HKA,HQK->QKHA", v, pattern)
    W_O = model.model.layers[layer_id].self_attn.o_proj.weight
    n_heads = v.shape[0]
    W_O = einops.rearrange(W_O, "d_model (index d_head)->index d_head d_model", index=n_heads)
    decomposed_attn = torch.einsum("QKHA,HAD->QKHD", z, W_O)
    return decomposed_attn

def decompose_H_verbose(prompt_ls, model, tokenizer, router_weight_ls, top_n, output_dir, draw_mode=[], cached_experts=None):
    """ Decomposition: single prompt, head(H).
        Note that top_n can vary - top_k or n_experts or other ranges.
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]

    attn_v = hook_dict["hook_v"].squeeze(0) # shape: [n_layers, n_heads, n_tokens, head_dim]
    attn_weights = hook_dict["hook_attn_weights"].squeeze(0) # shape: [n_layers, n_heads, n_tokens, n_tokens] # first n_tokens -> queries , second n_tokens -> keys
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    attn_output = hook_dict["hook_attn_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim] # for check if the decomposition is implemented correctly
    print(attn_v.shape, attn_weights.shape, after_res1.shape, after_norm2.shape)

    n_layers = len(router_weight_ls)
    n_experts, n_dim = router_weight_ls[1].shape
    n_heads = attn_v.shape[1]

    expert_ids_by_rank = torch.zeros((n_layers, n_tokens, n_experts), dtype=torch.int)

    for send_L in range(0, n_layers): # sending layer
        decomposed_attn_out = decompose_attn_out_helper(attn_v[send_L], attn_weights[send_L], send_L, model) # shape: [n_tokens, n_tokens, n_heads, n_dim] first n_tokens: queries, second n_tokens: keys
        for recv_L in range(send_L, n_layers): # receiving layer
            router_weight_vectors = router_weight_ls[recv_L]
            for T in range(n_tokens): # examined token (query token)
                if T != 9 or recv_L != 3 or send_L != 1:
                    continue

                print("send_L{} recv_L{} T{}".format(send_L, recv_L, T))

                ## decomposition
                decomposed_attn_rmsnorm = rmsnorm_breakdown(after_res1[recv_L, T], [decomposed_attn_out[T].reshape(-1, n_dim)], recv_L, model)[0] # shape: [n_tokens * n_heads, n_dim]
                
                ## score
                decomposed_attn_score = torch.matmul(router_weight_vectors, decomposed_attn_rmsnorm.T) # shape: [n_experts, n_tokens * n_heads]
                
                ## expert selection
                original_score = torch.matmul(router_weight_vectors, after_norm2[recv_L, T])
                expert_ids_by_rank[recv_L, T] = torch.argsort(original_score, descending=True)
                top_n_experts = expert_ids_by_rank[recv_L, T][:top_n] # torch.argsort(original_score, descending=True)[:top_n]
                print(recv_L, T, top_n_experts)

                if 1 in draw_mode: # Matrix 1: x: Token y: Head
                    matrix_drawer_H_token_head(torch.sum(decomposed_attn_score[top_n_experts, :], 0).reshape(n_tokens, n_heads), "Mode1_TokenHead_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, (11, 11), [], tokens_str, "T{}_A{}M{} (Token x Head)".format(T, send_L, recv_L))
                    if cached_experts is not None:
                        cached_selected_experts = cached_experts[recv_L, T, :top_n]
                        matrix_drawer_H_token_head(torch.sum(decomposed_attn_score[cached_selected_experts, :], 0).reshape(n_tokens, n_heads), "Mode1_TokenHead_cached_experts_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, (11, 11), [], tokens_str, "cached_experts_T{}_A{}M{} (Token x Head)".format(T, send_L, recv_L))
                if 2 in draw_mode: # Scatter 1 or 2: x: Expert, y: Score Each experts is assigned scores by 16 heads (seems useless)
                    expert_head_score = torch.sum(decomposed_attn_score.reshape(n_experts, n_tokens, n_heads), 1) # default order # shape: [n_experts, n_heads]
                    # ordered_expert_head_score = expert_head_score[expert_ids_by_rank[recv_L, T]] # sort by original_router_result
                    # ordered_expert_head_score_2 = expert_head_score[torch.argsort(expert_head_score.sum(axis=1))] # sort by attn_out_score of experts
                    scatter_drawer_H_expert(expert_head_score, 1, "Mode2_Expert_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, title="T{}_A{}M{}".format(T, send_L, recv_L))
                if 3 in draw_mode: # Scatter 3 x:Head, y:Score , All tokens
                    expert_head_score = torch.sum(decomposed_attn_score.reshape(n_experts, n_tokens, n_heads), 1)
                    scatter_drawer_H_head(expert_head_score, "Mode3_Head_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, title="T{}_A{}M{}".format(T, send_L, recv_L))
                if 4 in draw_mode: # Scatter 4 # same as scatter 3, but focus on only a specified key token
                    checked_k_token =  9
                    one_token_das = decomposed_attn_score.reshape(n_experts, n_tokens, n_heads)[:, checked_k_token, :].squeeze(1)
                    scatter_drawer_H_head(one_token_das, "Mode4_HeadOneToken_T{}_A{}k{}->M{}".format(T, send_L, checked_k_token, recv_L), output_dir, title="T{}_A{}k{}M{}".format(T, send_L, checked_k_token, recv_L))
                if 5 in draw_mode: # Matrix 2: x: Token, y: Selected experts  [token_level (0)]
                    token_level = torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n, n_tokens, n_heads), 2)
                    matrix_drawer_H_with_sum(token_level, "Mode5_ExpertToken_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, (13, 13), tokens_str, title="T{}_A{}M{} (Token x Expert)".format(T, send_L, recv_L), xlabel="Token", ylabel="Selected Expert")
                    # out_str = ""
                    # for k in range(top_n):
                    #     out_str +="Expert {}: ".format(k)
                    #     for t in range(n_tokens):
                    #         out_str += "{: .2f} ".format(token_level[k, t].item())
                    #     out_str += "\n"
                    # ## print sum of scores, assigned by each token
                    # net_token_score = torch.sum(token_level, dim=0)
                    # out_str += "Sum      :"
                    # for t in range(n_tokens):
                    #     out_str += "\033[42m{: .2f} ".format(net_token_score[t].item())
                    # out_str +="\033[0m"
                    # print(out_str)
                if 6 in draw_mode: # Matrix 3: x: Heads, y: Selected experts  [head_level (1)]
                    head_level = torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n, n_tokens, n_heads), 1)
                    matrix_drawer_H_with_sum(head_level, "Mode6_HeadExpert_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, (13, 13), None, title="T{}_A{}M{} (Head x Expert)".format(T, send_L, recv_L), xlabel="Head", ylabel="Selected Expert")
                    # out_str = ""
                    # for k in range(top_n):
                    #     out_str +="Expert {}: ".format(k)
                    #     for h in range(n_heads):
                    #         out_str += "{: .2f} ".format(head_level[k, h].item())
                    #     out_str += "\n"
                    # ## print sum of scores, assigned by each head (3)
                    # net_head_score = torch.sum(head_level, dim=0)
                    # out_str += "Sum      :"
                    # for h in range(n_heads):
                    #     out_str += "\033[42m{: .2f} ".format(net_head_score[h].item())
                    # out_str +="\033[0m"
                    # print(out_str)
                if 7 in draw_mode: # Matrix 1: x: Token y: Head (Positive/Negative separated) NOTE: not averaged
                    pos = (decomposed_attn_score + torch.abs(decomposed_attn_score)).div(2).sum(0).reshape(n_tokens, n_heads) # all experts
                    neg = (decomposed_attn_score - torch.abs(decomposed_attn_score)).div(2).sum(0).reshape(n_tokens, n_heads) # all experts
                    # pos = (decomposed_attn_score[top_n_experts, :] + torch.abs(decomposed_attn_score[top_n_experts, :])).div(2).sum(0).reshape(n_tokens, n_heads) # selected experts only
                    # neg = (decomposed_attn_score[top_n_experts, :] - torch.abs(decomposed_attn_score[top_n_experts, :])).div(2).sum(0).reshape(n_tokens, n_heads) # selected experts only

                    matrix_drawer_H_token_head(pos, "Mode7_TokenHeadPositive_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, (13, 13), [], tokens_str, "T{}_A{}M{} (Token x Head, positive)".format(T, send_L, recv_L))
                    matrix_drawer_H_token_head(neg, "Mode7_TokenHeadNegative_T{}_A{}M{}".format(T, send_L, recv_L), output_dir, (13, 13), [], tokens_str, "T{}_A{}M{} (Token x Head, negative)".format(T, send_L, recv_L))
                
                ## zero patching, check the influence on routing decisions caused by a specified key token and one of its head through the corruption
                # zero_patched_token, zero_patched_head = 3, 4 ## NOTE: zero_patched_head is in the SENDING layer
                # zero_patching_router_result = torch.matmul(router_weight_vectors, after_norm2[recv_L, T] - decomposed_attn_rmsnorm.reshape(n_tokens, n_heads, n_dim)[zero_patched_token, zero_patched_head])
                # zero_patching_top_n_experts = torch.argsort(zero_patching_router_result, descending=True)[:top_n]
                # print("Original routing decisions:", torch.sort(original_score, descending=True)) # original routing decisions
                # print("Corrupted routing decisions:", torch.sort(zero_patching_router_result, descending=True)) # corrupted routing decisions (after zero patching)
                
                ## check the score assigned by each head
                # print("Score assigned to top_n:", torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n * n_tokens, n_heads), 0)) # on selected experts 
                # print("Score assigned to all experts:", torch.sum(decomposed_attn_score.reshape(n_experts * n_tokens, n_heads), 0)) # on all experts
                
                ## check the positive score and negative score assigned by each head
                ## sum_score = decomposed_attn_score.reshape(n_experts, n_tokens, n_heads).sum(0).sum(0) # just for check if the impolementation is correct
                # tmp_das = decomposed_attn_score.reshape(n_experts * n_tokens, n_heads)
                # sum_score = torch.sum(tmp_das, dim=0)
                # sum_abs_score = torch.sum(torch.abs(tmp_das), dim=0)
                # pos_score = (sum_score + sum_abs_score) / 2
                # neg_score = (sum_score - sum_abs_score) / 2
                # print(sum_score)
                # print(pos_score)
                # print(neg_score)
                
                ## check the score assigned by a specified head on a specified key token
                # which_key_token, which_head = 3, 8
                # decomposed_attn_score_specified_token_head = torch.matmul(router_weight_vectors, decomposed_attn_rmsnorm.reshape(n_tokens, n_heads, n_dim)[which_key_token, which_head, :])
                # print(torch.sum(decomposed_attn_score_specified_token_head[top_n_experts])) # on selected experts
                # print(torch.sum(decomposed_attn_score_specified_token_head)) # on all experts

                ## checker
                # attn_rmsnorm = rmsnorm_breakdown(after_res1[recv_L, T], [attn_output[send_L, T]], recv_L, model)[0]
                # print("CHECK: input:", decomposed_attn_rmsnorm.sum(0)[:3], attn_rmsnorm[:3]) # component_sum MUST be equal to the original rmsnorm (precision error is acceptable)
                # print("CHECK: input consistency:", torch.allclose(decomposed_attn_rmsnorm.sum(0), attn_rmsnorm, atol=1e-5)) ## NOTE: we do not guarantee the precision error is smaller than the threshold
                # attn_score = torch.matmul(router_weight_vectors, attn_rmsnorm)
                # print("CHECK: score:", torch.sum(decomposed_attn_score,dim=1)[:3], attn_score[:3]) # the sum of decomposed_attn_score MUST be equal to attn_score (precision error is acceptable)

                # print(decomposed_attn_score.shape)
                # print(decomposed_attn_score[top_n_experts, :].reshape(top_n, n_tokens, n_heads).shape)
                # print(torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n, n_tokens, n_heads), 2)) # token-level (0)
                # print(torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n, n_tokens, n_heads), 1)) # head-level (1)
                # print(torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n, n_tokens * n_heads), 1)) # attn scores assigned to experts (2)
                # print(torch.sum(decomposed_attn_score[top_n_experts, :].reshape(top_n * n_tokens, n_heads), 0)) # sum of scores, assigned by each head (3)
                print(torch.sum(decomposed_attn_score[top_n_experts, :], 0).reshape(n_tokens, n_heads)) # token-head-level (4) [draw mode 1]; also for test batch implementation
                ## observe the norms
                # print("key=Token 0, norm of projected head output:", torch.norm(decomposed_attn_rmsnorm.reshape(n_tokens, n_heads, n_dim)[0], p=2, dim=-1))
                # print("key=Token 9, norm of projected head output:", torch.norm(decomposed_attn_rmsnorm.reshape(n_tokens, n_heads, n_dim)[9], p=2, dim=-1))
                ## observe positive rate (For Heads in Layer 13), (seems useless)
                # results: (OLMoE, prompt_maryjohnjohn) A13H1=13, A13H2=43, A13H5=13; (OLMoE, prompt_davidmiketom) A13H1=38, A13H2=16, A13H5=42 
                # print("key=Token 9, positive score rate, A13H1:", torch.sum(decomposed_attn_score.reshape(n_experts, n_tokens, n_heads)[:, 9, 1] > 0))
                # print("key=Token 9, positive score rate, A13H2:", torch.sum(decomposed_attn_score.reshape(n_experts, n_tokens, n_heads)[:, 9, 2] > 0))
                # print("key=Token 9, positive score rate, A13H5:", torch.sum(decomposed_attn_score.reshape(n_experts, n_tokens, n_heads)[:, 9, 5] > 0))
                # print("key=Token 9, positive score rate:", (decomposed_attn_score > 0).sum(dim=0).reshape(n_tokens, n_heads)[9, :])

    return expert_ids_by_rank

def decompose_M_verbose(prompt_ls, model, tokenizer, router_weight_ls, top_k, output_dir):
    """ Decomposition: single prompt, experts (M).
        Note that top_k is fixed.
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]

    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    expert_weighted_outputs = hook_dict["hook_expert_weighted_outputs"].squeeze(0) # shape: [n_layers, n_tokens, original_top_k, n_dim]
    # print(after_res1.shape, after_norm2.shape, expert_weighted_outputs.shape)

    n_layers = len(router_weight_ls)
    n_experts = router_weight_ls[1].shape[0]

    for send_L in range(0, n_layers): # sending layer
        for recv_L in range(send_L, n_layers): # receiving layer
            router_weight_vectors = router_weight_ls[recv_L]
            for T in range(n_tokens): # examined token
                if T != 9 or recv_L != 3 or send_L != 1:
                    continue
                # if T != 13:
                #     continue

                ## decomposition
                expert_outs = expert_weighted_outputs[send_L, T] # shape: [original_top_k, n_dim]
                expert_out_rmsnorm = rmsnorm_breakdown(after_res1[recv_L,T], [expert_outs], recv_L, model)[0] # shape: [original_top_k, n_dim]

                ## score
                decomposed_expert_out_score = torch.matmul(router_weight_vectors, expert_out_rmsnorm.T) # shape: [n_experts, original_top_k]

                ## expert selection
                original_score = torch.matmul(router_weight_vectors, after_norm2[recv_L, T])
                original_top_k_experts = torch.argsort(original_score, descending=True)[:top_k]
                # print("layer{}, token{}: {}".format(recv_L, T, original_top_k_experts))

                ## checker
                # print("CHECK: shape:", expert_out_rmsnorm.shape, decomposed_expert_out_score.shape)
                
                M_drawer(decomposed_expert_out_score, original_top_k_experts, "score_assignment_T{}send_M{}recv_M{}".format(T, send_L, recv_L), output_dir, "score_assignment_T{}send_M{}recv_M{}".format(T, send_L, recv_L))

def attn_weights_verbose(prompt_ls, model, tokenizer, output_dir):
    """ attention weights. (pre-softmax & post-softmax) """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]
    attn_weights = hook_dict["hook_attn_weights"].squeeze(0) # shape: [n_layers, n_heads, Q_n_tokens, K_n_tokens]
    attn_weights_before_softmax = hook_dict["hook_attn_weights_before_softmax"].squeeze(0) # shape: [n_layers, n_heads, Q_n_tokens, K_n_tokens]
    n_layers, n_heads, _, _ = attn_weights.shape
    print(attn_weights.shape, attn_weights_before_softmax.shape)

    ## examples
    L, H, T = 1, 4, 9
    matrix_drawer_H_token_head(attn_weights[L, H], "attn_weights_L{}H{}".format(L, H), output_dir, (13, 13), [], tokens_str, "attn_weights_L{}H{}".format(L, H), xlabel="Key Token", ylabel="Query Token")
    
    matrix_attn_weight_verbose(attn_weights[L, :, T, :(T + 1)], "attn_weights_L{}T{}".format(L, T), output_dir, figsize=(13, 13), title="attn_weights_L{}T{}".format(L, T), xlabel="Key Token", ylabel="Head")
    matrix_attn_weight_verbose(attn_weights_before_softmax[L, :, T, :(T + 1)], "attn_weights_before_softmax_L{}T{}".format(L, T), output_dir, figsize=(13, 13), title="attn_weights_before_softmax_L{}T{}".format(L, T), xlabel="Key Token", ylabel="Head")

def attn_weights_comparison_verbose(prompt_ls, model, tokenizer, output_dir):
    """ same as function 'attn_weights_verbose', but demonstrate two prompts simultaneously for comparison
    len(prompt_ls) == 2
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]
    attn_weights = hook_dict["hook_attn_weights"] # shape: [2, n_layers, n_heads, Q_n_tokens, K_n_tokens]

    ## example
    matrix_attn_weight_comparison_verbose(attn_weights[0], attn_weights[1], tokens_str, output_dir)

def attn_weights_score_comparison_verbose(prompt_ls, model, tokenizer, router_weight_ls, top_n, output_dir):
    """ Find the correlation between attention weights and the expert scores assigned by output of attention heads. """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt")
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]
    attn_v = hook_dict["hook_v"].squeeze(0) # shape: [n_layers, n_heads, n_tokens, head_dim]
    attn_weights = hook_dict["hook_attn_weights"].squeeze(0) # shape: [n_layers, n_heads, n_tokens, n_tokens] # first n_tokens -> queries , second n_tokens -> keys
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    n_layers, n_heads, n_tokens, _ = attn_weights.shape
    n_experts, n_dim = router_weight_ls[1].shape
    
    decomposed_score_collect = torch.zeros((n_layers, n_layers, n_tokens, n_tokens, n_heads, n_experts)) # [recv_L, send_L, Q, K, H, E]
    decomposed_score_topk_sum_collect = torch.zeros((n_layers, n_layers, n_tokens, n_tokens, n_heads))

    for send_L in tqdm(range(0, n_layers)): # sending layer
        decomposed_attn_out = decompose_attn_out_helper(attn_v[send_L], attn_weights[send_L], send_L, model) # shape: [n_tokens, n_tokens, n_heads, n_dim] first n_tokens: queries, second n_tokens: keys
        for recv_L in range(send_L, n_layers): # receiving layer
            router_weight_vectors = router_weight_ls[recv_L]
            for T in range(n_tokens): # examined token (query token)
                # print("send_L{} recv_L{} T{}".format(send_L, recv_L, T))

                ## decomposition
                decomposed_attn_out_rmsnorm = rmsnorm_breakdown(after_res1[recv_L, T], [decomposed_attn_out[T].reshape(-1, n_dim)], recv_L, model)[0] # shape: [n_tokens * n_heads, n_dim]
                
                ## score
                decomposed_attn_out_score = torch.matmul(router_weight_vectors, decomposed_attn_out_rmsnorm.T) # [n_experts, n_tokens * n_heads]
                decomposed_score_collect[recv_L, send_L, T] = decomposed_attn_out_score.reshape(n_experts, n_tokens, n_heads).permute(1, 2, 0)
                
                original_score = torch.matmul(router_weight_vectors, after_norm2[recv_L, T])
                top_n_experts = torch.argsort(original_score, descending=True)[:top_n]
                decomposed_score_topk_sum_collect[recv_L, send_L, T] = decomposed_attn_out_score[top_n_experts].sum(0).reshape(n_tokens, n_heads)
    
    corr_mat = torch.zeros((n_layers, n_tokens))
    for L in range(n_layers): # sending layer
        for T in range(n_tokens): # query
            # print(torch.corrcoef(torch.stack((attn_weights[L, :, T].flatten(), decomposed_score_topk_sum_collect[L, L, T].flatten()))))
            corr_mat[L, T] = torch.corrcoef(torch.stack((attn_weights[L, :, T].flatten(), decomposed_score_topk_sum_collect[L, L, T].flatten())))[0,1] # each shape is [K * H]

    matrix_drawer_H_token_head(corr_mat.T, "corr_mat", output_dir, figsize=(13,13), add_patch=[], token_ls=tokens_str, title="correlation between attn_weights and the sum of scores of top_K expert", xlabel="Layer", ylabel="Token", need_description=True)
    return