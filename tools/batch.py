import torch
from tqdm import tqdm
from tools.plot import tril_drawer_TAM
from tools.analyze import rmsnorm_breakdown_batch, decompose_attn_out_helper_batch

def decompose_TAM_batch(prompt_ls, model, tokenizer, router_weight_ls, bsz=100, max_token_per_prompt=32, output_dir=None):
    """ Decomposition: multiple prompts, token(T), attn_out(A), and moe_out(M).
        This implementation is primarily to check if the score computation is implemented correctly.
        NOTE: this function collects all the experts. (n_experts)
    """
    batch_token = tokenizer(prompt_ls, return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True)
    n_tokens_ls = torch.sum(batch_token["attention_mask"], dim=1) # a list showing the number of tokens of each prompt
    n_prompts, max_n_tokens = batch_token["attention_mask"].shape
    
    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, n_dim = router_weight_vectors.shape
    token_score_collect = torch.zeros((max_n_tokens, n_layers, n_experts, 1))
    moe_out_score_collect = torch.zeros((max_n_tokens, n_layers, n_experts, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    attn_out_score_collect = torch.zeros((max_n_tokens, n_layers, n_experts, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer

    ## Deprecated, no longer used
    # token_score_breakdown_batch = torch.zeros((n_prompts, max_n_tokens, top_n, n_layers, 1))
    # moe_cumulative_score_batch = torch.zeros((n_prompts, max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    # moe_abs_cumulative_score_batch = torch.zeros((n_prompts, max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    # attn_cumulative_score_batch = torch.zeros((n_prompts, max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    # attn_abs_cumulative_score_batch = torch.zeros((n_prompts, max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    
    for B in tqdm(range(0, n_prompts, bsz)):
        _, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        layer_input = hook_dict["hook_layer_input"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        attn_output = hook_dict["hook_attn_output"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        after_res1 = hook_dict["hook_after_res1"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        mlp_output = hook_dict["hook_mlp_output"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        n_prompts_B = layer_input.shape[0]
        
        token_components = layer_input[:, 0, :, :].unsqueeze(1) # res_in of Layer 0
        token_rmsnorm, attn_out_rmsnorm, moe_out_rmsnorm = rmsnorm_breakdown_batch(after_res1, [token_components, attn_output, mlp_output], model, mode="TAM")
        ## NOTE: token_rmsnorm shape: [n_prompts_B, n_layers, 1, max_n_tokens, n_dim]
        ## NOTE: moe_out_rmsnorm, attn_out_rmsnorm shape: [n_prompts_B, n_layers, n_layers, max_n_tokens, n_dim] # first n_layers -> recv_layer , second n_layers -> send_layer
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors, after_norm2) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers]
        # top_n_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :top_n, :]
        token_score = torch.einsum("RED,PRSTD->PTERS", router_weight_vectors, token_rmsnorm)
        attn_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors, attn_out_rmsnorm), diagonal=0)
        moe_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors, moe_out_rmsnorm), diagonal=-1) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers, n_layers] first n_layers -> recv_layer , second n_layers -> send_layer
        token_score_collect += token_score.sum(dim=0).transpose(1,2)
        attn_out_score_collect += attn_out_score.sum(dim=0).transpose(1,2)
        moe_out_score_collect += moe_out_score.sum(dim=0).transpose(1,2)
        
        ## equivalent to
        # check_original_score = torch.einsum("ijk,ikmn->ijmn", router_weight_vectors, after_norm2.permute(1,3,2,0)) # shape: [n_layers, n_experts, max_n_tokens, n_prompts_B]
        # check_original_score = check_original_score.permute(3,2,1,0) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers]
        # check_token_score = torch.einsum("ijk,ikpqr->ijpqr", router_weight_vectors, token_rmsnorm.permute(1,4,3,2,0)) # shape: [n_layers, n_experts, max_n_tokens, 1, n_prompts_B]
        # check_token_score = check_token_score.permute(4,2,1,0,3) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers, 1]
        # check_attn_out_score = (torch.einsum("ijk,ikpqr->ijpqr", router_weight_vectors, attn_out_rmsnorm.permute(1,4,3,2,0)))
        # check_attn_out_score = torch.tril(check_attn_out_score.permute(4,2,1,0,3), diagonal=0)
        # check_moe_out_score = torch.einsum("ijk,ikpqr->ijpqr", router_weight_vectors, moe_out_rmsnorm.permute(1,4,3,2,0)) # shape: [n_layers, n_experts, max_n_tokens, n_layers, n_prompts_B] first n_layers -> recv_layer , second n_layers -> send_layer
        # check_moe_out_score = torch.tril(check_moe_out_score.permute(4,2,1,0,3), diagonal=-1) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers, n_layers] first n_layers -> recv_layer , second n_layers -> send_layer
        # print("CHECK: original_score:", original_score[0, 1, 2, :3], check_original_score[0, 1, 2, :3])
        # print("CHECK: token_score:", token_score[0, 1, 2, :3, 0], check_token_score[0, 1, 2, :3, 0])
        # print("CHECK attn_out_score:", attn_out_score[0, 1, 2, :5, 1], check_attn_out_score[0, 1, 2, :5, 1])
        # print("CHECK moe_out_score:", moe_out_score[0, 1, 2, :5, 1], check_moe_out_score[0, 1, 2, :5, 1])

        ## Code for check if implemented correctly
        # print("CHECK: shape:", layer_input.shape, attn_output.shape, after_res1.shape, after_norm2.shape, mlp_output.shape)
        # print("CHECK: shape:", token_score.shape, moe_out_score.shape, attn_out_score.shape, token_rmsnorm.shape, attn_out_rmsnorm.shape, moe_out_rmsnorm.shape)
        # print("CHECK: score:", original_score[0,13,0,5], token_score[0,13,0,5,0] + torch.sum(attn_out_score[0,13,0,5,:]) + torch.sum(moe_out_score[0,13,0,5,:]))
        # test_original_score = torch.einsum("PTRED,PRTD->PTRE", router_weight_vectors.repeat(n_prompts_B * max_n_tokens, 1, 1, 1).reshape(n_prompts_B, max_n_tokens, n_layers, n_experts, n_dim), after_norm2).transpose(2,3)
        # print("CHECK: score:", test_original_score[0,13,0,5])

        ## for examining cumulative scores
        # print(token_score.shape, top_n_experts.unsqueeze(-1).shape)
        # token_score_breakdown_batch[B:B+bsz] = torch.gather(token_score, dim=2, index=top_n_experts.long().unsqueeze(-1))
        # moe_cumulative_score_batch[B:B+bsz] = torch.sum(torch.gather(moe_out_score, dim=2, index=top_n_experts.long().unsqueeze(-1).repeat(1,1,1,1,n_layers)), dim=2)
        # attn_cumulative_score_batch[B:B+bsz] = torch.sum(torch.gather(attn_out_score, dim=2, index=top_n_experts.long().unsqueeze(-1).repeat(1,1,1,1,n_layers)), dim=2)
        # moe_abs_cumulative_score_batch[B:B+bsz] = torch.sum(torch.abs(torch.gather(moe_out_score, dim=2, index=top_n_experts.long().unsqueeze(-1).repeat(1,1,1,1,n_layers))), dim=2)
        # attn_abs_cumulative_score_batch[B:B+bsz] = torch.sum(torch.abs(torch.gather(attn_out_score, dim=2, index=top_n_experts.long().unsqueeze(-1).repeat(1,1,1,1,n_layers))), dim=2)
        # print("CHECK: score:", top_n_experts[0, 1, 0, 3],  token_score[0, 1, top_n_experts[0, 1, 0, 3], 3, 0], token_score_breakdown_batch[0, 1, 0, 3, 0])
        # print("CHECK: score:", top_n_experts[0, 1, 0, 3],  attn_out_score[0, 1, top_n_experts[0, 1, :, 3], 3, 0].sum(), attn_cumulative_score_batch[0, 1, 3, 0])
        # print("CHECK: score:", top_n_experts[0, 1, 0, 3],  moe_out_score[0, 1, top_n_experts[0, 1, :, 3], 3, 0].sum(), moe_cumulative_score_batch[0, 1, 3, 0])
    
    ## all experts
    T = 9
    tril_drawer_TAM(moe_out_score_collect[T].sum(1).div(n_prompts), "moe_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=0, add_patch=[], title="moe_cumulative_score_T{}".format(T), xlabel="MoE Layer (Sending)", ylabel="MoE Layer (Receiving)")
    tril_drawer_TAM(attn_out_score_collect[T].sum(1).div(n_prompts), "attn_cumulative_score_T{}".format(T), output_dir, (11, 11), diagonal=1, add_patch=[], title="attn_cumulative_score_T{}".format(T), xlabel="Attention Layer (Sending)", ylabel="MoE Layer (Receiving)")
    return

def decompose_H_batch(prompt_dict_ls, model, tokenizer, router_weight_ls, top_n, n_heads, bsz):
    """ Decomposition: multiple prompts,  head(H).
        Note that top_n can vary - top_k or n_experts or other ranges.
        NOTE: this function consumes a lot of memory. Avoid using this function directly.
        NOTE: this function is designed for IOI task. The task-agnostic version is in analyze.py.
    """
    prompt_ls = [i["text"] for i in prompt_dict_ls]
    batch_token = tokenizer(prompt_ls, return_tensors="pt", padding=True)
    n_prompts = len(prompt_ls)
    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, _ = router_weight_vectors.shape
    
    # for duplicate token heads
    # q_token_position_ls = torch.tensor([i["S_token_pos"][1] for i in prompt_dict_ls])
    # k_token_position_ls = torch.tensor([i["S_token_pos"][0] for i in prompt_dict_ls])

    # for name mover heads
    q_token_position_ls = torch.tensor([i["END_token_pos"] for i in prompt_dict_ls])
    k_token_position_ls = torch.tensor([i["IO_token_pos"] for i in prompt_dict_ls])

    # collector = torch.zeros((n_prompts, n_heads))
    # collector_expert_scores = torch.zeros((n_prompts, n_heads, n_experts))

    for B in tqdm(range(0, n_prompts, bsz)):
        _, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        attn_v = hook_dict["hook_v"] # shape: [n_prompts_B, n_layers, n_heads, max_n_tokens, head_dim]
        attn_weights = hook_dict["hook_attn_weights"] # shape: [n_prompts_B, n_layers, n_heads, n_tokens, n_tokens] # first n_tokens -> queries , second n_tokens -> keys
        after_res1 = hook_dict["hook_after_res1"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        n_prompts_B = after_res1.shape[0]
        decomposed_attn_out = decompose_attn_out_helper_batch(attn_v, attn_weights, n_layers, model) # shape: [n_prompts_B, n_layers, n_tokens, n_tokens, n_heads, n_dim] first n_tokens: queries, second n_tokens: keys
        head_rmsnorm = rmsnorm_breakdown_batch(after_res1, [decomposed_attn_out], model, mode="H")[0]
        ## NOTE: head_rmsnorm shape: [n_prompts_B, n_layers, n_layers, max_n_tokens, max_n_tokens, n_heads, n_dim] # first n_layers -> recv_layer , second n_layers -> send_layer
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors, after_norm2) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers]
        top_n_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :top_n, :]
        head_score = torch.tril(torch.einsum("RED,PRSQKHD->PQKHERS", router_weight_vectors, head_rmsnorm), diagonal=0)
        
        ## checker
        # print(top_n_experts.shape, top_n_experts[0, 9, :, 3]) # for test (compare with decompose_H_verbose)
        # print(torch.sum(head_score[0, 9, 3, 4, top_n_experts[0, 9, :, 3], 3, 1])) # for test (compare with decompose_H_verbose)
        ## use demo
        # print(head_score[torch.arange(n_prompts_B), q_token_position_ls[B:B+n_prompts_B], k_token_position_ls[B:B+n_prompts_B], :, :, 3, 1].shape)
        # collector[B:B+bsz] = torch.sum(head_score[torch.arange(n_prompts_B), q_token_position_ls[B:B+n_prompts_B], k_token_position_ls[B:B+n_prompts_B], :, :, 3, 1], dim=2) # duplicate token heads
        # collector[B:B+bsz] = torch.sum(head_score[torch.arange(n_prompts_B), q_token_position_ls[B:B+n_prompts_B], k_token_position_ls[B:B+n_prompts_B], :, :, 13, 13], dim=2) # name mover heads
        # collector_expert_scores[B:B+bsz] = head_score[torch.arange(n_prompts_B), q_token_position_ls[B:B+n_prompts_B], k_token_position_ls[B:B+n_prompts_B], :, :, 13, 13]
        
    # return collector, collector_expert_scores