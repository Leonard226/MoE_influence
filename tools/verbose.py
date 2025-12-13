import torch
from tools.plot import tril_drawer_TAM

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
