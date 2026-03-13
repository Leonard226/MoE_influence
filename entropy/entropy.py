import torch
from tqdm import tqdm
# 当前的实现可能有问题：我将Answer的内容也作为prompt输入了，可能不应该这样做。
# 思考输入/输出的norm... 输入的norm不相同的话，合理吗?
# from tools.misc import project_to_logits
def entropy_from_logits(logits, dim=-1):
    log_p = torch.log_softmax(logits, dim)
    p = torch.softmax(logits, dim)
    return -(p * log_p).sum(dim)
def prob_from_logits(logits, dim=-1):
    return torch.softmax(logits, dim)

def project_to_logits(vector, final_var, model, isqwen=False):
    """ Compute the logits (for function 'decompose_XA_verbose')
    :param1 vector: output of the layer (or a vector with the same shape, e.g., attn_out, head_out, ...)
    :param2 final_var: the variance of final layer output
    :param3 model: assigned model
    :return: the logits
    """
    ## NOTE: we respect the original implementation of the projection in the model code and reuse it here.
    ##       But note that the coefficient - RMS(.) actually does not matter as this is a scalar multiplication and we can skip it.
    # vector = vector * torch.rsqrt(vector.pow(2).mean(-1, keepdim=True) + 1e-05) # rsqrt(vector), alternative
    # print(vector.shape, final_var.shape, model.model.norm.weight.data.shape)

    vector = vector * torch.rsqrt(final_var + 1e-05) # rsqrt(final_var), actually we can skip this operation
    
    vector_rmsn = vector * model.model.norm.weight.data
    if isqwen:
        vector_logits = model.lm_head(vector_rmsn.half()).data # for qwen
    else:
        vector_logits = model.lm_head(vector_rmsn).data # for olmoe
    return vector_logits


def find_entropy(prompt_ls, model, tokenizer, router_weight_ls, max_token_per_prompt, bsz=100, isqwen=False):
    batch_token = tokenizer(prompt_ls, return_tensors="pt", max_length=max_token_per_prompt, padding=True, truncation=True)
    n_prompts, max_n_tokens = batch_token["attention_mask"].shape
    print('max_n_tokens', max_n_tokens)
    token_ls = None

    # for bt in batch_token["input_ids"]:
    #     print("token_id:", bt)
    #     print("decode:", [tokenizer.decode(x) for x in bt])
    #     return_token = [tokenizer.decode(x) for x in bt]
    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, _ = router_weight_vectors.shape
    
    token_entropy = torch.zeros((n_prompts, max_n_tokens))
    top_k_entropy = torch.zeros((n_prompts, max_n_tokens, n_layers))
    top_k_weight = torch.zeros((n_prompts, max_n_tokens, 8, n_layers))
    all_expert_entropy = torch.zeros((n_prompts, max_n_tokens, n_layers))
    moe_input_entropy = torch.zeros((n_prompts, n_layers, max_n_tokens))
    rank_all_experts = torch.empty((n_prompts, max_n_tokens, n_experts, n_layers))
    for B in tqdm(range(0, n_prompts, bsz)):
        model_outputs, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        cur_bsz = after_norm2.shape[0]
        
        ## token_entropy
        prediction = model_outputs[0] # [batch_size, n_tokens, vocab_size]
        token_entropy[B:B+bsz] = entropy_from_logits(prediction)
        # print('pred entropy\n', token_entropy[B:B+bsz])

        ## top_k_entropy
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors.float(), after_norm2)
        
        top_n_scores, expert_ids = torch.sort(original_score, dim=2, descending=True)
        rank_all_experts[B:B+bsz] = expert_ids
        top_k_entropy[B:B+bsz] = entropy_from_logits(top_n_scores[:, :, :8, :], dim=2)
        top_k_weight[B:B+bsz] = prob_from_logits(top_n_scores[:, :, :8, :], dim=2)
        # print("\033[31m entropy maximum=", torch.log(torch.tensor([8])), "\033[0m")
        # for L in range(n_layers):
        #     print(top_k_entropy[0, :, L])

        ## all expert entropy
        # print("\033[31m entropy maximum=", torch.log(torch.tensor([64])), "\033[0m")
        all_expert_entropy[B:B+bsz] = entropy_from_logits(original_score, dim=2)
        # for L in range(n_layers):
        #     print(all_expert_entropy[0, :, L])

        ## moe input entropy
        final_var = hook_dict["hook_layer_output"][0, -1, :, :].pow(2).mean(-1, keepdim=True)
        ptl = project_to_logits(after_norm2, final_var, model, isqwen)
        moe_input_entropy[B:B+bsz] = entropy_from_logits(ptl, dim=-1)
        # for L in range(n_layers):
        #     print(moe_input_entropy[0, L])
        
        ## checker
        # for i in range(14):
        #     smax = torch.softmax(prediction[0,i],dim=0)
        #     log_smax = torch.log(smax)
        #     print(-torch.sum(torch.mul(smax, log_smax)))

        ## checker
        # layer_output = hook_dict["hook_layer_output"]
        # final_var = hook_dict["hook_layer_output"][0, -1, :, :].pow(2).mean(-1, keepdim=True)
        # tmp_logits = torch.zeros((14, 50304))
        # for k in range(14):
        #     tmp_logits[k] = project_to_logits(layer_output[0, -1, k], final_var[k], model)
        # print(entropy_from_logits(tmp_logits, dim=1))        
        
        ## checker
        # for L in range(n_layers):
        #     print(L, entropy_from_logits(top_n_scores[0,:,:8,L], dim=1))        
        
        ## checker
        # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        # final_var = hook_dict["hook_layer_output"][0, -1, :, :].pow(2).mean(-1, keepdim=True)
        
        # for j in range(n_layers):
        #     tmp_logits = torch.zeros((max_n_tokens, 50304)) # 50304 for olmoe; 151936 for qwen
        #     for k in range(max_n_tokens):
        #         tmp_logits[k] = project_to_logits(after_norm2[0, j, k], final_var[k], model)
        #     # torch.set_printoptions(sci_mode=False)
        #     # print("input of  moe layer {}, entropy\n".format(j), entropy_from_logits(tmp_logits, dim=1))
        #     moe_input_entropy[0, j] = entropy_from_logits(tmp_logits, dim=1)
        #     print(j, moe_input_entropy[0,j])

    # return token_entropy[0], top_k_entropy[0], all_expert_entropy[0], moe_input_entropy[0], return_token
    return token_entropy, top_k_entropy, batch_token["input_ids"], rank_all_experts, top_k_weight