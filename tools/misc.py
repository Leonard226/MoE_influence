## NOTE: The following code was not used in the experiments in the paper but is provided for reference.

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tools.verbose import rmsnorm_breakdown

def layer_print(model):
    """ Print layer info of the assigned model. """
    for k, v in model.state_dict().items():
        print(k, v.shape)
    print(model.config)

def run_template(prompt_ls, model, tokenizer):
    """ A simple template to show how to obtain some basic info. """
    batch_token = tokenizer(prompt_ls, return_tensors="pt", padding=True)
    model_outputs, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])

    ## Code for checking
    print("prompts:", tokenizer.batch_decode(batch_token["input_ids"]))
    for bt in batch_token["input_ids"]:
        print("token_id:", bt)
        print("decode:", [tokenizer.decode(x) for x in bt])
    prediction = model_outputs[0] # [batch_size, n_tokens, vocab_size]
    predicted_top10 = torch.argsort(prediction[0, -1], descending=True)[:10] # 0=first prompt, -1=last token
    predicted_text = [tokenizer.decode(x) for x in predicted_top10]
    print("top10 predicted_text of the first prompt at the last token:", predicted_text)

    return batch_token, model_outputs, hook_dict

def matrix_drawer(data, name, output_dir, cmap_set="RdBu", title="", xlabel="", ylabel=""):
    """ A simple template for visualizing a matrix. """
    data = data.detach().cpu().numpy() # 2-D data
    plt.figure(figsize=(11,11))
    plt.imshow(data, cmap=cmap_set)
    for r in range(data.shape[1]):
        for c in range(data.shape[0]):
            plt.text(r, c, np.round(data[c, r], 2), fontsize=10, horizontalalignment="center", verticalalignment="center")
    plt.colorbar()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png")
    plt.close("all")

def scatter_drawer(data, name, output_dir, title="", xlabel="", ylabel=""):
    """ A simple template for visualizing a scatter plot. """
    data = data.detach().cpu().numpy() # 2-D data
    n_dim1, n_dim2 = data.shape
    xs = [i for i in range(n_dim1) for _ in range(n_dim2)]
    ys = data.reshape(-1)
    plt.scatter(xs, ys, alpha=0.3, s=5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png")
    plt.close("all")

def cosine_similarity(inputs_vectors, router_weight_vectors):
    """ Compute the cosine similarity between the vectors (for function 'decompose_XA_verbose')
    :param1 inputs_vectors | shape:[num1, n_dim]
    :param2 router_weight_vectors | shape:[num2, n_dim]
    :return: a result matrix
    """
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos_sim(inputs_vectors, router_weight_vectors)

def project_to_logits(vector, final_var, model):
    """ Compute the logits (for function 'decompose_XA_verbose')
    :param1 vector: output of the layer (or a vector with the same shape, e.g., attn_out, head_out, ...)
    :param2 final_var: the variance of final layer output
    :param3 model: assigned model
    :return: the logits
    """
    ## NOTE: we respect the original implementation of the projection in the model code and reuse it here.
    ##       But note that the coefficient - RMS(.) actually does not matter as this is a scalar multiplication and we can skip it.
    # vector = vector * torch.rsqrt(vector.pow(2).mean(-1, keepdim=True) + 1e-05) # rsqrt(vector), alternative
    vector = vector * torch.rsqrt(final_var + 1e-05) # rsqrt(final_var), actually we can skip this operation
    vector_rmsn = vector * model.model.norm.weight.data
    vector_logits = model.lm_head(vector_rmsn).data
    return vector_logits

def colored_print(h_expert, r_score, a_score, top_n, mode=1):
    """ Print results for function 'decompose_XA_verbose'
    :param1 h_expert: selected experts' IDs
    :param2 r_score: scores assigned by residual stream inputs (all experts)
    :param3 a_score: scores assigned by attention layer outputs (all experts)
    :param4 top_n: (int)
    :param5 mode: 1, 2, 3, 4, 5, 6 (int) read the comments below for the related info
    """
    if mode == 1: # print the IDs of selected experts: if the SCORE assigned by res_in is GREATER than the SCORE assigned by attn_out, then blue; otherwise green
        out_str = ""
        for i in range(top_n):
            if r_score[h_expert[i]] > a_score[h_expert[i]]: # blue
                out_str += "\033[1m\033[44m{:2d} ".format(h_expert[i].item())
            else: # green
                out_str += "\033[1m\033[42m{:2d} ".format(h_expert[i].item())
        out_str +="\033[0m"
        print(out_str)
    elif mode == 2: # print the IDs of selected experts: if the RANK assigned by res_in is NOT LOWER than the RANK assigned by attn_out, then blue; otherwise green
        out_str = ""
        sorted_r_score_id = torch.argsort(r_score, descending=True).detach().cpu().tolist()
        sorted_a_score_id = torch.argsort(a_score, descending=True).detach().cpu().tolist()
        for i in range(top_n):
            if sorted_r_score_id.index(h_expert[i]) <= sorted_a_score_id.index(h_expert[i]): # blue
                out_str += "\033[1m\033[44m{:2d} ".format(h_expert[i].item())
            else: # green
                out_str += "\033[1m\033[42m{:2d} ".format(h_expert[i].item())
        out_str +="\033[0m"
        print(out_str)
    elif mode == 3: # print the scores assigned by original input / res_in / attn_out
        out_str = ""
        for i in range(top_n):
            out_str += "\033[1m{: .2f} ".format((r_score[h_expert[i]]+a_score[h_expert[i]]).item())
        print(out_str)
        out_str = ""
        for i in range(top_n):
            out_str += "\033[1m\033[44m{: .2f} ".format(r_score[h_expert[i]].item())
        print(out_str)
        out_str =""
        for i in range(top_n):
            out_str += "\033[1m\033[42m{: .2f} ".format(a_score[h_expert[i]].item())
        out_str +="\033[0m\033[K"
        print(out_str)
    elif mode == 4: # print res_in_score minus attn_out_score, i.e., the difference between the scores: if the SCORE assigned by res_in is GREATER than the SCORE assigned by attn_out, then blue; otherwise green
        out_str = ""
        for i in range(top_n):
            if r_score[h_expert[i]] > a_score[h_expert[i]]: # blue
                out_str += "\033[1m\033[44m{: .2f} ".format((r_score[h_expert[i]] - a_score[h_expert[i]]).item())
            else: # green
                out_str += "\033[1m\033[42m{: .2f} ".format((r_score[h_expert[i]] - a_score[h_expert[i]]).item())
        out_str +="\033[0m"
        print(out_str)
    elif mode == 5:
        h_expert = h_expert.detach().cpu().tolist()
        r_expert = torch.argsort(r_score, descending=True)[:top_n].detach().cpu().tolist()
        a_expert = torch.argsort(a_score, descending=True)[:top_n].detach().cpu().tolist()

        str = "x~ ["
        r_expert_selected_recorder = [False] * top_n
        a_expert_selected_recorder = [False] * top_n
        for j in range(top_n): # if the experts selected by original input are selected by res_in only, then blue; by attn_out only, then green; by both, then red; by neither, then default color
            if h_expert[j] in r_expert and h_expert[j] in a_expert: # red
                str += "\033[1m\033[41m{:2d} ".format(h_expert[j])
                r_expert_selected_recorder[r_expert.index(h_expert[j])] = True
                a_expert_selected_recorder[a_expert.index(h_expert[j])] = True
            elif h_expert[j] in r_expert: # blue
                str += "\033[1m\033[44m{:2d} ".format(h_expert[j])
                r_expert_selected_recorder[r_expert.index(h_expert[j])] = True
            elif h_expert[j] in a_expert: # green
                str += "\033[1m\033[42m{:2d} ".format(h_expert[j])
                a_expert_selected_recorder[a_expert.index(h_expert[j])] = True
            else: # default color
                str += "\033[0m{:2d} ".format(h_expert[j])
        str += "\033[0m]"
        str += "\nr  ["
        for j in range(top_n): # if the experts selected by res_in are also selected by the original input, then blue
            if r_expert_selected_recorder[j]:
                str += "\033[1m\033[44m{:2d} ".format(r_expert[j])
            else:
                str += "\033[0m{:2d} ".format(r_expert[j])
        str += "\033[0m]"
        str += "\na  ["
        for j in range(top_n): # if the experts selected by attn_out are also selected by the original input, then green
            if a_expert_selected_recorder[j]:
                str += "\033[1m\033[42m{:2d} ".format(a_expert[j])
            else:
                str += "\033[0m{:2d} ".format(a_expert[j])
        str += "\033[0m]"
        print(str)
    elif mode == 6:
        h_expert = h_expert.detach().cpu().tolist()
        r_expert = torch.argsort(r_score, descending=True)[:top_n].detach().cpu().tolist()
        a_expert = torch.argsort(a_score, descending=True)[:top_n].detach().cpu().tolist()

        str = "x~ ["
        r_expert_selected_recorder = [False] * top_n
        a_expert_selected_recorder = [False] * top_n
        for j in range(top_n): # if the experts selected by original input are selected by res_in, then blue; if not selected by res_in but by attn_out, then green; otherwise, default color
            if h_expert[j] in r_expert:
                str += "\033[1m\033[44m{:2d} ".format(h_expert[j])
                r_expert_selected_recorder[r_expert.index(h_expert[j])] = True
            elif h_expert[j] in a_expert:
                str += "\033[1m\033[42m{:2d} ".format(h_expert[j])
                a_expert_selected_recorder[a_expert.index(h_expert[j])] = True
            else:
                str += "\033[0m{:2d} ".format(h_expert[j])
        str += "\033[0m]"
        str += "\nr  ["
        for j in range(top_n): # if the experts selected by res_in are also selected by the original input, then blue
            if r_expert_selected_recorder[j]:
                str += "\033[1m\033[44m{:2d} ".format(r_expert[j])
            else:
                str += "\033[0m{:2d} ".format(r_expert[j])
        str += "\033[0m]"
        str += "\na  ["
        for j in range(top_n): # if the experts selected by attn_out are also selected by the original input, then green
            if a_expert_selected_recorder[j]:
                str += "\033[1m\033[42m{:2d} ".format(a_expert[j])
            else:
                str += "\033[0m{:2d} ".format(a_expert[j])
        str += "\033[0m]"
        print(str)

def decompose_XA_verbose(prompt_ls, model, tokenizer, router_weight_ls, top_n, output_dir, mode=1):
    """ Decomposition: one sample, layer_input(X) and attn_out(A).
        For mode info, please refer to function 'colored_print'.  
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt", padding=True)
    model_outputs, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])

    ## check if 'project_to_logits' is implemented correctly
    # prediction = model_outputs[0] # [batch_size, n_tokens, vocab_size]
    # predicted_top10 = torch.argsort(prediction[0, -1], descending=True)[:10] # 0=first prompt, -1=last token
    # predicted_text = [tokenizer.decode(x) for x in predicted_top10]
    # print("top10 predicted_text of the first prompt at the last token:", predicted_text)

    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])
    tokens_str = [tokenizer.decode(x) for x in batch_token["input_ids"][0]]

    layer_input = hook_dict["hook_layer_input"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    attn_output = hook_dict["hook_attn_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    final_var = hook_dict["hook_layer_output"][0, -1, -1].pow(2).mean(-1, keepdim=True) # last layer, last token; final_var is a 1D tensor with one elememt
    layer_output = hook_dict["hook_layer_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    selected_experts = hook_dict["hook_selected_experts"].squeeze(0) # shape: [n_layers, n_tokens, default_top_k]
    print(tokens_str)
    print(layer_input.shape, attn_output.shape, after_res1.shape, after_norm2.shape, layer_output.shape, selected_experts.shape)

    n_layers = len(router_weight_ls)
    n_experts = router_weight_ls[1].shape[0]

    original_experts = torch.zeros((n_tokens, n_layers, n_experts), dtype=torch.int32) ## NOTE: experts selected by original score
    res_in_score = torch.zeros((n_tokens, n_layers, n_experts))
    attn_out_score = torch.zeros((n_tokens, n_layers, n_experts))

    for L in range(n_layers):
        router_weight_vectors = router_weight_ls[L] # shape: [n_experts, n_dim]
        for T in range(n_tokens):
            if T != (n_tokens - 1): # just a filter, can be removed
                continue
            print("layer{}, token{}: {}".format(L, T, tokens_str[T]))
            print("selected experts: ", selected_experts[L, T])
            
            ## decomposition: moe_in = res_in + attn_out [This is the core of the function.]
            res_in_rmsnorm, attn_out_rmsnorm = rmsnorm_breakdown(after_res1[L, T], [layer_input[L, T], attn_output[L, T]], L, model)

            ## score: original_score = res_in_score + attn_out_score
            res_in_score[T, L] = torch.matmul(router_weight_vectors, res_in_rmsnorm)
            attn_out_score[T, L] = torch.matmul(router_weight_vectors, attn_out_rmsnorm)
            original_score = torch.matmul(router_weight_vectors, after_norm2[L, T])
            
            ## expert selection
            original_experts[T, L] = torch.argsort(original_score, descending=True)
            experts_selected_by_res_in = torch.argsort(res_in_score[T, L], descending=True)
            experts_selected_by_attn_out = torch.argsort(attn_out_score[T, L], descending=True)

            ## print the selected experts
            colored_print(original_experts[T, L, :top_n], res_in_score[T, L], attn_out_score[T, L], top_n, mode=mode)

            ## print the current predictions
            # final_logits = project_to_logits(layer_output[L, T], final_var[0], model)
            # logit_lens_predicted_top10 = torch.argsort(final_logits, dim=-1, descending=True)[:10]
            # logit_lens_predicted_text = [tokenizer.decode(x) for x in logit_lens_predicted_top10]
            # print("top10 predictions:", logit_lens_predicted_text)

            ## print L2-norm
            # print("l2norm_layer_in {: .2f}".format(torch.norm(after_norm2[L, T]).item()))
            # print("l2norm_res_in_rmsnorm {: .2f}".format(torch.norm(res_in_rmsnorm).item()))
            # print("l2norm_attn_out_rmsnorm {: .2f}".format(torch.norm(attn_out_rmsnorm).item()))
            
            ## decomposition of attn_out vector (l2norm, angle)
            # cosine_theta = torch.dot(res_in_rmsnorm, attn_out_rmsnorm) / (torch.norm(res_in_rmsnorm) * torch.norm(attn_out_rmsnorm))
            # l2norm_attn_out = torch.norm(attn_out_rmsnorm)
            # print("l2norm_attn_out, parallel_component(cos): {: .3f}".format((l2norm_attn_out * cosine_theta).item()))
            # print("l2norm_attn_out, orthogonal_component(sin): {: .3f}".format((l2norm_attn_out * torch.sqrt(1 - cosine_theta * cosine_theta)).item()))
            # print("angle:", torch.rad2deg(torch.acos(cosine_theta)))

            ## scores of selected experts (if determined by MoE_in / res_in / attn_out)
            # print("original_score:", original_score[original_experts[T, L, :top_n]])
            # print("res_in_score:", res_in_score[T, L][experts_selected_by_res_in[:top_n]])
            # print("attn_out_score:", attn_out_score[T, L][experts_selected_by_attn_out[:top_n]])

            ## cosine similarity between the examined components and routing vectors
            # cos_sim_rwv_res_in = cosine_similarity(res_in_rmsnorm.unsqueeze(0), router_weight_vectors[original_experts[T, L]]) # shape: [n_experts]
            # cos_sim_rwv_attn_out = cosine_similarity(attn_out_rmsnorm.unsqueeze(0), router_weight_vectors[original_experts[T, L]]) # shape: [n_experts]
            # print("cos_sim of router_weight_vectors and res_in:", cos_sim_rwv_res_in)
            # print("cos_sim of router_weight_vectors and attn_out:", cos_sim_rwv_attn_out)

            ## checker
            # print("CHECK: input", after_norm2[L, T, :3], res_in_rmsnorm[:3] + attn_out_rmsnorm[:3]) # the sum of each column of component_rmsnorm MUST be equal to the original rmsnorm (precision error is acceptable)
            # print("CHECK: score:", original_score[:3], res_in_score[T, L, :3] + attn_out_score[T, L, :3]) # original_score MUST be equal to res_in_score + attn_out_score (precision error is acceptable)
            # print("CHECK: cos_sim of router_weight_vectors and res_in [0]:", torch.dot(res_in_rmsnorm, router_weight_vectors[original_experts[T, L, 0]]) / torch.sqrt(torch.sum(torch.pow(res_in_rmsnorm, 2))) / torch.sqrt(torch.sum(torch.pow(router_weight_vectors[original_experts[T, L, 0]], 2)))) # cosine similarity
            # print(router_weight_vectors.shape, res_in_rmsnorm.shape, attn_out_rmsnorm.shape, after_norm2.shape, original_score.shape)
    
    T = 13
    L = 0
    
    scatter_drawer_XA_layer_verbose(attn_out_score[T], original_experts[T, :, :top_n], "scatter_drawer_XA_layer_verbose__attn_out_score_T{}".format(T), output_dir, title="Score of experts assigned by attention layer output (a), Token {}".format(T))
    scatter_drawer_XA_layer_verbose(res_in_score[T], original_experts[T, :, :top_n], "scatter_drawer_XA_layer_verbose__res_in_score_T{}".format(T), output_dir, title="Score of experts assigned by residual stream input (r), Token {}".format(T))
    scatter_drawer_XA_layer_verbose(attn_out_score[T] + res_in_score[T], original_experts[T, :, :top_n], "scatter_drawer_XA_layer_verbose__original_score_T{}".format(T), output_dir, title="Score of experts assigned by original score, Token {}".format(T))
    scatter_drawer_XA_expert_verbose(res_in_score[T, L, original_experts[T, L]], attn_out_score[T, L, original_experts[T, L]], name="scatter_drawer_XA_expert_verbose__T{}L{}".format(T, L), output_dir=output_dir, title="Score of experts assigned by a / r, Token {} Layer {}".format(T, L))
    line_drawer_XA_ratio_verbose(res_in_score[T], attn_out_score[T], absolute=True, name="line_drawer_XA_ratio_verbose__absolute_T{}".format(T), output_dir=output_dir, title="Ratio: |r|>=|a| (Absolute value), Token {}".format(T))
    line_drawer_XA_ratio_verbose(res_in_score[T], attn_out_score[T], absolute=False, name="line_drawer_XA_ratio_verbose__original_T{}".format(T), output_dir=output_dir, title="Ratio: r>=a (Original value), Token {}".format(T))

    return res_in_score, attn_out_score, original_experts

def scatter_drawer_XA_layer_verbose(data, experts, name, output_dir, title=""):
    """ Scatter the scores of experts. Highlight the selected experts. x axis->Layer y axis->Score 
    :param1 data: scores of ALL experts (shape: [n_layers, n_experts])
    :param2 experts: selected experts (shape:[n_layers, top_n])
    :param3 name: name of the saved file
    :param4 output_dir: directory of the saved file
    :param5 title: title of the figure
    """
    data = data.detach().cpu().numpy() # shape: [n_layers, n_experts]
    experts = experts.detach().cpu().numpy().astype(np.int32) # shape: [n_layers, top_n]
    n_layers, n_experts = data.shape
    _, top_n = experts.shape
    selected_experts_data =[]
    selected_experts_layer = [i for i in range(n_layers) for _ in range(top_n)]
    unselected_experts_data = []
    unselected_experts_layer = [i for i in range(n_layers) for _ in range(n_experts - top_n)]
    experts_num = [i for i in range(n_experts)]
    for L in range(n_layers):
        selected_experts_data.extend(data[L, experts[L]])
        unselected_experts_data.extend(data[L, np.setdiff1d(experts_num, experts[L], True)])
    
    plt.grid()
    plt.scatter(unselected_experts_layer, unselected_experts_data, alpha=0.3, s=5, c="k", label="unselected experts")
    plt.scatter(selected_experts_layer, selected_experts_data, alpha=0.7, s=5, c="r", label="selected experts", marker="X")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(output_dir + name + ".png") # .pdf
    plt.close("all")

def scatter_drawer_XA_expert_verbose(res_in_score, attn_out_score, name, output_dir, title=""):
    """ Scatter the scores of experts of a specific layer. x axis->Rank of Expert y axis->Score 
    :param1 res_in_score: scores of ALL experts assigned by res_in (shape: [n_experts])
    :param2 attn_out_score: scores of ALL experts assigned by attn_out (shape: [n_experts])
    :param3 name: name of the saved file
    :param4 output_dir: directory of the saved file
    :param5 title: title of the figure
    """
    res_in_score = res_in_score.detach().cpu().numpy() # shape: [n_experts]
    attn_out_score = attn_out_score.detach().cpu().numpy() # shape: [n_experts]
    n_experts = res_in_score.shape[0]
    xs = [i for i in range(n_experts)]
    
    plt.grid()
    plt.scatter(xs, res_in_score, c="b", s=5, label="res_in_score")
    plt.scatter(xs, attn_out_score, c="g", s=5, label="attn_out_score")
    plt.title(title)
    plt.xlabel("Expert score rank")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(output_dir + name + ".png") # .pdf
    plt.close("all")

def line_drawer_XA_ratio_verbose(res_in_score, attn_out_score, absolute, name, output_dir, title=""):
    """ Ratio of res_in_score >= attn_out_score or |res_in_score| >= |attn_out_score|. x axis->Layer y axis->Ratio 
    :param1 res_in_score: scores of ALL experts assigned by res_in (shape: [n_layers, n_experts])
    :param2 attn_out_score: scores of ALL experts assigned by attn_out (shape: [n_layers, n_experts])
    :param3 absolute: if True, use absolute value; if False, use original value
    :param4 name: name of the saved file
    :param5 output_dir: directory of the saved file
    :param6 title: title of the figure
    """
    ## res_in_score, attn_out_score shape: [n_layers, n_experts]
    n_layers, n_experts = res_in_score.shape
    if absolute: # True or False
        res_in_score = torch.abs(res_in_score)
        attn_out_score = torch.abs(attn_out_score)
    xs = [i for i in range(n_layers)]
    ys = [torch.count_nonzero(torch.ge(res_in_score[i], attn_out_score[i])).item()/n_experts for i in range(n_layers)]
    
    plt.grid()
    plt.plot(xs, ys, c="r", marker="o")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Ratio")
    plt.savefig(output_dir + name + ".png") # .pdf
    plt.close("all")

def decompose_XA_single(prompt_ls, model, tokenizer, router_weight_ls):
    """ decomposition: one sample, layer_input(X) and attn_out(A).
        A simplified version of  'decompose_XA_verbose'. (Remove the verbose output)
    """
    ## run
    batch_token = tokenizer(prompt_ls, return_tensors="pt", padding=True)
    _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
    ## collect info
    n_tokens = torch.sum(batch_token["attention_mask"])

    layer_input = hook_dict["hook_layer_input"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    attn_output = hook_dict["hook_attn_output"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_res1 = hook_dict["hook_after_res1"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]
    after_norm2 = hook_dict["hook_after_norm2"].squeeze(0) # shape: [n_layers, n_tokens, n_dim]

    n_layers = len(router_weight_ls)
    n_experts = router_weight_ls[1].shape[0]

    original_experts = torch.zeros((n_tokens, n_layers, n_experts), dtype=torch.int32) ## NOTE: experts selected by original score
    res_in_score = torch.zeros((n_tokens, n_layers, n_experts))
    attn_out_score = torch.zeros((n_tokens, n_layers, n_experts))

    for L in range(n_layers):
        router_weight_vectors = router_weight_ls[L] # shape: [n_experts, n_dim]
        for T in range(n_tokens):
            ## decomposition: moe_in = res_in + attn_out [This is the core of the function.]
            res_in_rmsnorm, attn_out_rmsnorm = rmsnorm_breakdown(after_res1[L, T], [layer_input[L, T], attn_output[L, T]], L, model)
            
            ## score: original_score = res_in_score + attn_out_score
            res_in_score[T, L] = torch.matmul(router_weight_vectors, res_in_rmsnorm)
            attn_out_score[T, L] = torch.matmul(router_weight_vectors, attn_out_rmsnorm)
            original_score = torch.matmul(router_weight_vectors, after_norm2[L, T])
            
            ## expert selection
            original_experts[T, L] = torch.argsort(original_score, descending=True)
            # experts_selected_by_res_in = torch.argsort(res_in_score[T, L], descending=True)
            # experts_selected_by_attn_out = torch.argsort(attn_out_score[T, L], descending=True)
   
    return res_in_score, attn_out_score, original_experts

def G_matrix_analysis(router_weight_ls):
    """ compute (1/n (\sum_i {g_i g_i^T})) - \bar{g} \bar{g}^T """
    ## version 1
    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, n_dim = router_weight_vectors.shape

    mean_g = router_weight_vectors.mean(dim=1)
    g_gT = router_weight_vectors.unsqueeze(-1) @ router_weight_vectors.unsqueeze(-2) # shape: [n_layers, n_experts, n_dim, n_dim]
    mean_g_mean_gT = mean_g.unsqueeze(-1) @ mean_g.unsqueeze(-2)
    G = g_gT.sum(dim=1).div(n_experts) - mean_g_mean_gT
    G_eigs = torch.linalg.eigvalsh(G)
    G_eigs_mins = G_eigs[:, 0]
    G_eigs_maxs = G_eigs[:, -1]
    print(G_eigs_mins)
    print(G_eigs_maxs)
    ## checker
    # tmp = torch.outer(router_weight_vectors[0, 2], router_weight_vectors[0, 2])
    # print(g_gT[0, 2, 1, :3], tmp[1, :3])

    ## version 2, a different implementation with a similar result (maxs are the same, but mins vary due to the computational error)
    # router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    # n_layers, n_experts, n_dim = router_weight_vectors.shape

    # mean_g = router_weight_vectors.mean(dim=1, keepdim=True)
    # gi_minus_mean_g = router_weight_vectors - mean_g

    # G = gi_minus_mean_g.transpose(1, 2) @ gi_minus_mean_g / n_experts
    # G_eigs = torch.linalg.eigvalsh(G)
    # print(G_eigs[:, 0]) # mins
    # print(G_eigs[:, -1]) # maxs

## NOTE: the following code is unchecked
# from tqdm import tqdm
# import plotly.express as px

# def activation_patching(prompt_dict_ls_ORIG, prompt_dict_ls_NEW, model, tokenizer, send_info, recv_info, output_dir, n_layers, n_heads, bsz=20):
#     def logit_diff_batch(logits, token_id1_ls, token_id2_ls):
#         """Logit difference (Multiple prompts)
#         :param1 logits: shape: [n_prompts, vocab_size]
#         :param2 token_id1_ls: shape: [n_prompts]
#         :param3 token_id2_ls: shape: [n_prompts]
#         :return: shape: [n_prompts]
#         """
#         return logits[torch.arange(logits.shape[0]), token_id1_ls] - logits[torch.arange(logits.shape[0]), token_id2_ls]
#     ## prepration
#     prompt_ls_ORIG = [ i["text"] for i in prompt_dict_ls_ORIG ]
#     prompt_ls_NEW = [ i["text"] for i in prompt_dict_ls_NEW ]
#     io_token_id_ls_ORIG = torch.tensor([ i["IO_token_id"][0] for i in prompt_dict_ls_ORIG ], dtype=torch.int) ## NOTE: possible variant: ["IO_token_id"] for GPT2
#     s1_token_id_ls_ORIG = torch.tensor([ i["S_token_id"][0] for i in prompt_dict_ls_ORIG ], dtype=torch.int) ## NOTE: possible variant: ["S1_token_id"] for GPT2
#     end_token_pos_ls_ORIG = torch.tensor([ i["END_token_pos"] for i in prompt_dict_ls_ORIG ], dtype=torch.int)
#     recv_type = recv_info["type"]
#     print(recv_type)
#     ## metrics
#     n_prompts = len(prompt_dict_ls_ORIG)
#     logit_diff_matrix = torch.zeros((n_prompts, n_layers, n_heads, len(recv_type)))
#     logit_diff_ORIG = torch.zeros((n_prompts))
#     prob_io_name_matrix = torch.zeros((n_prompts, n_layers, n_heads, len(recv_type)))
#     prob_io_name_ORIG = torch.zeros((n_prompts))
#     prob_s_name_matrix = torch.zeros((n_prompts, n_layers, n_heads, len(recv_type)))
#     prob_s_name_ORIG = torch.zeros((n_prompts))
#     for B in tqdm(range(0, n_prompts, bsz)):
#         ## tokenization
#         batch_token_ORIG = tokenizer(prompt_ls_ORIG[B:B+bsz], return_tensors="pt", padding=True)
#         batch_token_NEW = tokenizer(prompt_ls_NEW[B:B+bsz], return_tensors="pt", padding=True)
#         ## token positions, token id's
#         cur_bsz = len(batch_token_ORIG["input_ids"]) # current batch size
#         send_token_pos_ls = torch.tensor(send_info["token_pos_ls"][B:B+cur_bsz]) # positions of sending tokens
#         recv_token_pos_ls = torch.tensor(recv_info["token_pos_ls"][B:B+cur_bsz]) # positions of receiving tokens
#         io_name_id_ls = io_token_id_ls_ORIG[B:B+cur_bsz] # id's of IO tokens
#         s1_name_id_ls = s1_token_id_ls_ORIG[B:B+cur_bsz] # id's of S1 tokens
#         ## input
#         model_outputs_ORIG, hook_dict_ORIG = model(input_ids=batch_token_ORIG["input_ids"], attention_mask=batch_token_ORIG["attention_mask"]) # forward pass B
#         _, hook_dict_NEW = model(input_ids=batch_token_NEW["input_ids"], attention_mask=batch_token_NEW["attention_mask"]) # forward pass A
#         ## prediction
#         prediction_ORIG = model_outputs_ORIG[0][torch.arange(cur_bsz), end_token_pos_ls_ORIG[B:B+cur_bsz]]
#         ## metrics
#         logit_diff_ORIG[B:B+bsz] = logit_diff_batch(prediction_ORIG, io_name_id_ls, s1_name_id_ls)
#         # prob_io_ORIG[B:B+bsz] = prob(prediction_ORIG, io_name_id_ls)
#         # prob_s_ORIG[B:B+bsz] = prob(prediction_ORIG, s_name_id_ls)
#         ## preparation
#         # q2_CORR = hook_dict_CORR["hook_q2"]
#         # k2_CORR = hook_dict_CORR["hook_k2"]
#         ## Now, patching # patch_pos_q, patch_pos_k are used in fine grain patch
#         for L in range(0, n_layers):
#             for H in range(0, n_heads):
#                 for cur_patch_type, counter in zip(recv_type, range(len(recv_type))):
#                     patch_ls = None
#                     match cur_patch_type:
#                         case "q":
#                             patch_ls = [["q_head", L, H, recv_token_pos_ls, hook_dict_NEW["hook_q"][torch.arange(cur_bsz), L, H, recv_token_pos_ls]]]
#                         case "k":
#                             patch_ls = [["k_head", L, H, recv_token_pos_ls, hook_dict_NEW["hook_k"][torch.arange(cur_bsz), L, H, recv_token_pos_ls]]]
#                         case "v":
#                             patch_ls = [["v_head", L, H, recv_token_pos_ls, hook_dict_NEW["hook_v"][torch.arange(cur_bsz), L, H, recv_token_pos_ls]]]
#                         case "o":
#                             # patch_ls = [["before_matmul_wo_activation_patching", L, H, recv_token_pos_ls, hook_dict_NEW["hook_before_matmul_wo"][torch.arange(cur_bsz), L, H, recv_token_pos_ls]]] # NOTE: for OLMoE
#                             patch_ls = [["before_matmul_wo_activation_patching", L, H, recv_token_pos_ls, hook_dict_NEW["hook_before_matmul_wo"][torch.arange(cur_bsz), L, recv_token_pos_ls, H]]] # NOTE: for GPT2
#                         case "Q": # "q2->one_token"
#                             pass
#                         case "K": # "k2->one_token"
#                             pass
#                         case "V": # "v->one_token"
#                             pass
#                     model_outputs_P, _ = model(input_ids=batch_token_ORIG["input_ids"], attention_mask=batch_token_ORIG["attention_mask"], patching=patch_ls)
#                     prediction_P = model_outputs_P[0][torch.arange(cur_bsz), end_token_pos_ls_ORIG[B:B+cur_bsz]]
#                     logit_diff_matrix[B:B+cur_bsz, L, H, counter] = logit_diff_batch(prediction_P, io_name_id_ls, s1_name_id_ls)
#                     ## TODO: prob...
#     logit_diff_ORIG = logit_diff_ORIG.unsqueeze(1).repeat(1, n_layers * n_heads * len(recv_type)).reshape(n_prompts, n_layers, n_heads, len(recv_type))
#     logit_diff_normalized_matrix = torch.div(logit_diff_matrix - logit_diff_ORIG, logit_diff_ORIG).mean(0)
#     for cur_recv_type, counter in zip(recv_type, range(len(recv_type))):
#         fig = px.imshow(logit_diff_normalized_matrix[:,:,counter].detach().cpu().numpy(), color_continuous_scale="RdBu", color_continuous_midpoint=0,)
#         fig.update_xaxes(side="top")
#         fig.show()

# def check_prob(prompt):
#     ## TODO: fix this function
#     attn_v, attn_weights, final_var = run_regular(prompt, 'check_attn_prob')
#     which_layer = 13
#     decomposed_attn_output = decomposed_attn(attn_v[which_layer], attn_weights[which_layer], which_layer)
#     tmp = torch.sum(decomposed_attn_output, dim=1)
#     for j in range(n_heads):
#         bsvalues = get_final_logits(tmp[-1,j], final_var[0])
#         #print(bsvalues.shape)
#         # print(io_name_id, s_name_id) # Mary: 6393 John: 2516
#         print('head:{}'.format(j))
#         print('io_bsvalue:{: .2e} s_bsvalue:{: .2e}'.format(bsvalues[2516].item(), bsvalues[6393].item()))
#         print('io_prob:{: .2e} s_prob:{: .2e}'.format(prob(bsvalues, 2516).item(), prob(bsvalues, 6393).item()))