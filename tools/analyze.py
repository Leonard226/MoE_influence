import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
from tqdm import tqdm
import nltk
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

np.random.seed(42) # if you want reproducibility for tsne figures

# If you use nltk for the first time, you may need these codes
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

def matrix_drawer_patch(data, name, output_dir, figsize=(13, 13), add_patch=[], title="", xlabel="Head", ylabel="Layer", need_description=False):
    """ matrix for path patching """
    data = data.detach().cpu().numpy()
    np.save(output_dir + name + ".npy", data)
    plt.figure(figsize=figsize)
    
    vmin, vmax = data.min(), data.max()
    if vmin < 0 and vmax > 0:
        normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap_type = "RdBu"
    elif vmax <= 0:
        normalize = mcolors.Normalize(vmin=vmin, vmax=0)
        cmap_type = "Reds_r"
    else: # vmin >= 0
        normalize = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap_type = "Blues"
    
    with sns.axes_style("white"):
        ax = sns.heatmap(data, square=True, annot=True, fmt=".2f", cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.5, "pad": 0.08, "aspect": 5, "ticks": [-0.2, 0, 0.2, 0.4]}, linewidth=.5)
        # for grid in add_patch:
        #     ax.add_patch(Rectangle((grid[0], grid[1]), 1, 1, fill=False, edgecolor="blue", lw=3))
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_title("Logit diff.\nvariation", fontsize=30, pad=5)
    # cbar.set_ticklabels(["-20%", "0%", "20%", "40%"])
    plt.xticks([k + 0.5 for k in range(0, data.shape[1], 5)], [str(k) for k in range(0, data.shape[1], 5)], fontsize=30)
    plt.yticks([k + 0.5 for k in range(0, data.shape[0], 5)], [str(k) for k in range(0, data.shape[0], 5)], fontsize=30, rotation=0)
    if need_description:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(output_dir + name + ".pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close("all")

def tril_drawer_tam_analyze(data, name, output_dir, figsize=(11,11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="", need_lognorm=False, need_description=True, tick_mode=None, cbar_label=None, need_no_annotations=True, demo_now=False):
    """ lower triangular matrix. for moe->moe, diagnoal=0; for attn->moe, diagonal=1. """
    data = data.detach().cpu().numpy()
    
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask, k=diagonal)] = True
    
    vmin, vmax = data.min(), data.max()
    if need_lognorm: # vmin >= 0
        normalize = mcolors.Normalize(vmin=0, vmax=vmax) ## FIXME: temporary
        # normalize = mcolors.LogNorm(vmin=data[data>0].min(), vmax=vmax)
        cmap_type = "Greens"
    elif vmin < 0 and vmax > 0:
        normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap_type = "RdBu"
    elif vmax <= 0:
        normalize = mcolors.Normalize(vmin=vmin, vmax=0)
        cmap_type = "Reds_r"
    else: # vmin >= 0
        normalize = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap_type = "Blues"

    plt.figure(figsize=figsize)
    with sns.axes_style("white"):
        # deepseek/mixtral annot size:7; qwen annot size: 5
        if need_lognorm:
            fmt_setting = ".1e"
        else:
            fmt_setting = ".2f"
        ax = sns.heatmap(data, mask=mask, square=True, annot=True, annot_kws={"size": 7}, fmt=fmt_setting, cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.5}, linewidth=.5)
        for grid in add_patch:
            ax.add_patch(Rectangle((grid[0], grid[1]), 1, 1, fill=False, edgecolor="blue", lw=3))
        cbar = ax.collections[0].colorbar
        # cbar_min, cbar_max = cbar.mappable.get_clim()
        cbar.set_ticks([vmin, vmax])
        cbar.ax.tick_params(labelsize=40)
        cbar.set_label(cbar_label, fontsize=40)
        
    if need_description:
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        
        if tick_mode == "T":
            plt.xticks([0], [""])
            plt.yticks([k + 0.5 for k in range(0, data.shape[0], 5)], [str(k) for k in range(0, data.shape[0], 5)])
        elif tick_mode in ["A", "M"]:
            plt.xticks([k + 0.5 for k in range(0, data.shape[1], 5)], [str(k) for k in range(0, data.shape[1], 5)])
            plt.yticks([k + 0.5 for k in range(0, data.shape[0], 5)], [str(k) for k in range(0, data.shape[0], 5)])
    np.save(output_dir + "_" + name + ".npy", data)
    plt.savefig(output_dir + name + ".png", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(output_dir + name + ".pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close("all")

    if need_no_annotations:
        plt.figure(figsize=figsize)
        with sns.axes_style("white"):
            ax = sns.heatmap(data, mask=mask, square=True, annot=False, fmt=".2f", cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.5}, linewidth=.5)
            for grid in add_patch:
                ax.add_patch(Rectangle((grid[0], grid[1]), 1, 1, fill=False, edgecolor="blue", lw=3))
            cbar = ax.collections[0].colorbar
            # cbar_min, cbar_max = cbar.mappable.get_clim()
            cbar.set_ticks([vmin, vmax])
            cbar.ax.tick_params(labelsize=40)
            cbar.set_label(cbar_label, fontsize=40)

        if need_description:
            plt.title(title, fontsize=20)
            plt.xlabel(xlabel, fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            
            if tick_mode == "T":
                plt.xticks([0], [""])
                plt.yticks([k + 0.5 for k in range(0, data.shape[0], 5)], [str(k) for k in range(0, data.shape[0], 5)])
            elif tick_mode in ["A", "M"]:
                plt.xticks([k + 0.5 for k in range(0, data.shape[1], 5)], [str(k) for k in range(0, data.shape[1], 5)])
                plt.yticks([k + 0.5 for k in range(0, data.shape[0], 5)], [str(k) for k in range(0, data.shape[0], 5)])

        plt.savefig(output_dir + name + "no_annotations"+".png", bbox_inches="tight", pad_inches=0)
        plt.savefig(output_dir + name + "no_annotations"+".pdf", bbox_inches="tight", pad_inches=0)
        plt.close("all")

    if demo_now:
        mask = np.triu(np.ones_like(data, dtype=bool), k=diagonal)
        data = np.where(mask, np.nan, data)
        if need_lognorm:
            fig = px.imshow(np.log10(data + 1e-9), color_continuous_scale=cmap_type, title=title, labels=dict(x=xlabel, y=ylabel), range_color=[-4.5, 0])
        else:
            fig = px.imshow(data, color_continuous_scale=cmap_type, title=title, labels=dict(x=xlabel, y=ylabel))
        fig.update_xaxes(side="top")
        fig.show()

def logit_diff_batch(logits, token_id1_ls, token_id2_ls):
    """ Logit difference (Multiple prompts)
    :param1 logits: shape: [n_prompts, vocab_size]
    :param2 token_id1_ls: shape: [n_prompts]
    :param3 token_id2_ls: shape: [n_prompts]
    :return: shape: [n_prompts]
    """
    return logits[torch.arange(logits.shape[0]), token_id1_ls] - logits[torch.arange(logits.shape[0]), token_id2_ls]

def prob_batch(logits, token_id_ls):
    """ Probability
    :param1 logits: shape: [n_prompts, vocab_size]
    :param2 token_id_ls: shape: [n_prompts]
    :return: shape: [n_prompts]
    """
    probs = torch.softmax(logits, dim=1)
    return probs[torch.arange(probs.shape[0]), token_id_ls]

def rmsnorm_breakdown_batch(vector, components, model, mode="default", variance_epsilon=1e-05, device="cuda:0"):
    """
    Break down the input into components.
    Original RMSNorm(x) = x * gamma * rsqrt(x)
    Contribution of a component c: ContributionRMSNorm(c, x) = c * gamma * rsqrt(x)
    :param vector: the vector to be decomposed. NOTE: the decomposition is linear. the vector is the input of an MoE layer.
    :param components: the components of the input "vector". NOTE: the components contribute to the numerator.
    :param model: assigned model
    :param mode: type of decomposition
    :param variance_epsilon: just adopt the setting in the original implementation
    :param device: you can assign another gpu to relieve the burden
    """
    n_prompts_B, n_layers, n_tokens, n_dim = vector.shape
    variance = vector.to(device).pow(2).mean(-1, keepdim=True) # shape: [n_prompts_B, n_layers, n_tokens, 1]
    rsqrt = torch.rsqrt(variance + variance_epsilon) # shape: [n_prompts_B, n_layers, n_tokens, 1]
    weight = torch.stack([model.model.layers[layer_id].post_attention_layernorm.weight.data.to(device) for layer_id in range(n_layers)], dim=0) # shape: [n_layers, n_dim]
    weight = weight.view(1, n_layers, 1, n_dim).expand(*vector.shape) # shape: [n_prompts_B, n_layers, n_tokens, n_dim]
    
    if mode == "TAM": # P: prompt, R: receiving layer, S: sending layer, T: token, D: n_dim
        breakdowns = [torch.einsum("PRTD,PSTD->PRSTD", rsqrt * weight, i.to(device)) for i in components]
    elif mode == "H": # P: prompt, R: receiving layer, S: sending layer, Q: q_token, K: k_token, H: head, D: n_dim
        breakdowns = [torch.einsum("PRQD,PSQKHD->PRSQKHD", rsqrt * weight, i.to(device)) for i in components]
    elif mode == "H_simplified": # sending layer and receiving layer are the same layer; unused in this file. TODO: check this branch
        n_heads = components[0].shape[4]
        tmp_results = torch.zeros((n_prompts_B, n_layers, n_tokens, n_tokens, n_heads, n_dim))
        for j in range(n_layers):
            tmp_results[:, j, ...] = torch.einsum("PQD,PQKHD->PQKHD", (rsqrt * weight)[:, j, ...], components[0][:, j, ...].to(device))
        breakdowns = [tmp_results]
        # simplified version, unchecked TODO: check it
        # rsqrt_mul_weight = rsqrt * weight
        # tmp_results_test = rsqrt_mul_weight.unsqueeze(3).unsqueeze(4) * components[0].to(device)
        # breakdowns_test = [tmp_results]
        # print(torch.allclose(breakdowns[0], breakdowns_test[0]))
    elif mode == "H_agnostic":
        breakdowns = [torch.einsum("PRQD,PSQHD->PRSQHD", rsqrt * weight, i.to(device)) for i in components]
    elif mode == "E": # P: prompt, R: receiving layer, S: sending layer, T: token, D: n_dim, E: n_experts
        breakdowns = [torch.einsum("PRTD,PSTED->PRSTED", rsqrt * weight, i.to(device)) for i in components]
    # else: # plain implementation, no longer used
    #     breakdowns = [weight * (i * rsqrt) for i in components]
    return breakdowns

def diff_breakdown_batch(vector, components, model, mode="default", variance_epsilon=1e-05, device="cuda:0"):
    """
    Using RMSNorm(x) - RMSNorm(x-c) to compute the contribution of a component c
    
    :param vector: the vector to be decomposed. NOTE: the decomposition is NOT linear. the vector is the input of an MoE layer.
    :param components: the components of the input "vector".
    :param model: assigned model
    :param mode: type of decomposition
    :param variance_epsilon: just adopt the setting in the original implementation
    :param device: you can assign another gpu to relieve the burden
    """
    n_prompts_B, n_layers, n_tokens, n_dim = vector.shape
    variance = vector.pow(2).mean(-1, keepdim=True) # shape: [n_prompts_B, n_layers, n_tokens, 1]
    rsqrt = torch.rsqrt(variance + variance_epsilon) # shape: [n_prompts_B, n_layers, n_tokens, 1]
    weight = torch.stack([model.model.layers[layer_id].post_attention_layernorm.weight.data for layer_id in range(n_layers)], dim=0) # shape: [n_layers, n_dim]
    weight = weight.view(1, n_layers, 1, n_dim).expand(*vector.shape) # shape: [n_prompts_B, n_layers, n_tokens, n_dim]
    original_rmsnorm = rsqrt * weight * vector

    if mode == "default":
        new_vector = vector.unsqueeze(2) - components.unsqueeze(1) # new_vector = x - c, shape: [n_prompts_B, n_layers, n_sending_layers, n_tokens, n_dim]
        new_variance = new_vector.pow(2).mean(-1, keepdim=True)
        new_rsqrt = torch.rsqrt(new_variance + variance_epsilon) # shape: [n_prompts_B, n_layers, n_sending_layers, n_tokens, 1]
        new_rmsnorm = new_rsqrt * weight.unsqueeze(2) * new_vector
        diff_breakdown_rmsnorm = original_rmsnorm.unsqueeze(2) - new_rmsnorm # shape: [n_prompts_B, n_layers, n_sending_layers, n_tokens, n_dim]
        
    elif mode == "E": # TODO: check this
        cur_n_experts = components.shape[3]
        diff_breakdown_rmsnorm = torch.zeros((n_prompts_B, n_layers, n_layers, n_tokens, cur_n_experts, n_dim)) # PRSTED
        for L in range(n_layers):
            tmp_vector = vector[:, L, :, :].unsqueeze(2).unsqueeze(1).expand(n_prompts_B, L + 1, n_tokens, cur_n_experts, n_dim) - components[:, :(L+1), ...]
            tmp_variance = tmp_vector.to(device).pow(2).mean(-1, keepdim=True)
            tmp_rsqrt = torch.rsqrt(tmp_variance + variance_epsilon)
            tmp_rmsnorm = torch.einsum("PRTED,PRTED->PRTED", tmp_rsqrt * (weight[:, L, :, :].unsqueeze(2).unsqueeze(1).expand(n_prompts_B, L + 1, n_tokens, cur_n_experts, n_dim)), tmp_vector)
            diff_breakdown_rmsnorm[:, L, :(L+1)]= original_rmsnorm[:, L, ...].unsqueeze(2).unsqueeze(1).expand(n_prompts_B, L + 1, n_tokens, cur_n_experts, n_dim) - tmp_rmsnorm

    elif mode == "H_agnostic": # TODO: check this
        # vector shape: [n_prompts_B, n_layers, n_tokens, n_dim] (PRTD)
        # components shape: [n_prompts_B, n_layers, n_tokens, n_heads, n_dim] (output of heads) (PSTHD)
        n_heads = components.shape[3]
        diff_breakdown_rmsnorm = torch.zeros((n_prompts_B, n_layers, n_layers, n_tokens, n_heads, n_dim)) # PRSTHD
        for L in range(n_layers):
            tmp_vector = vector[:, L, :, :].unsqueeze(2).unsqueeze(1).expand(n_prompts_B, L + 1, n_tokens, n_heads, n_dim) - components[:, :(L+1), ...]
            tmp_variance = tmp_vector.to(device).pow(2).mean(-1, keepdim=True)
            tmp_rsqrt = torch.rsqrt(tmp_variance + variance_epsilon)
            tmp_rmsnorm = torch.einsum("PRTHD,PRTHD->PRTHD", tmp_rsqrt * (weight[:, L, :, :].unsqueeze(2).unsqueeze(1)).expand(n_prompts_B, L + 1, n_tokens, n_heads, n_dim), tmp_vector)
            diff_breakdown_rmsnorm[:, L, :(L+1)]= original_rmsnorm[:, L, ...].unsqueeze(2).unsqueeze(1).expand(n_prompts_B, L + 1, n_tokens, n_heads, n_dim) - tmp_rmsnorm
        
    # return original_rmsnorm
    ## check if the implementation is consistent with the previous one
    # rigor_breakdown_rmsnorm = torch.empty((n_prompts_B, n_layers, 1, n_tokens, n_dim))
    # for L in range(n_layers):
    #     tmp_vector = vector[:, L, :, :].unsqueeze(1) - components
    #     tmp_variance = tmp_vector.to(device).pow(2).mean(-1, keepdim=True)
    #     tmp_rsqrt = torch.rsqrt(tmp_variance + variance_epsilon)
    #     # print('bbb',tmp_rsqrt.shape, weight[:,L,:,:].unsqueeze(1).shape, (tmp_rsqrt*(weight[:,L,:,:].unsqueeze(1))).shape, tmp_vector.shape)
    #     tmp_rmsnorm = torch.einsum("PRTD,PRTD->PRTD", tmp_rsqrt*(weight[:,L,:,:].unsqueeze(1)), tmp_vector)
    #     # print('ccc', original_rmsnorm[:,L,...].unsqueeze(1).shape, tmp_rmsnorm.shape)
    #     rigor_breakdown_rmsnorm[:,L,...]= original_rmsnorm[:,L,...].unsqueeze(1) - tmp_rmsnorm
    # print("consistent", torch.allclose(rigor_breakdown_rmsnorm, diff_breakdown_rmsnorm))
    # exit()
    return diff_breakdown_rmsnorm

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

def path_patching(prompt_dict_ls_ORIG, prompt_dict_ls_NEW, model, tokenizer, send_info, recv_info, output_dir, n_layers, n_heads, bsz=20, demo_now=False):
    """ Figure *a (Appendix)
        path patching: For more info, check https://github.com/redwoodresearch/Easy-Transformer 
        example (GPT-2):
        prompt_orig = "After the lunch, Matthew and Andrew went to the hospital. Matthew gave a basketball to Andrew" # Matthew: 9308, Andrew: 6858
        prompt_new = "After the lunch, Erin and Tiffany went to the hospital. Jesse gave a basketball to Tiffany" # Erin: 28894, Tiffany: 40928, Jesse: 18033
        prompt_dict_ls_ORIG = [{"text":prompt_orig, "IO_token_id":[6858], "S_token_id":[9308], "S2_token_id":[9308], "END_token_pos":16}] * 10
        prompt_dict_ls_NEW = [{"text":prompt_new, "IO_token_id":[40928], "S_token_id":[28894], "S2_token_id":[18033], "END_token_pos":16}] * 10
        send_info = {"token_pos_ls":([16]*10)} # example: [ i[END_token_pos] for i in prompt_dict_ls_ORIG ]
        recv_info = {"type":"l","token_pos_ls":([16]*10)}
        # send_info = {"token_pos_ls":([12]*10)}
        # recv_info = {"type":"qkv","token_pos_ls":([12]*10),"head_pos":[(7, 3), (7, 9), (8, 6), (8, 10)]}
    """

    ## preparation
    prompt_ls_ORIG = [ i["text"] for i in prompt_dict_ls_ORIG ]
    prompt_ls_NEW = [ i["text"] for i in prompt_dict_ls_NEW ]
    io_token_id_ls_ORIG = torch.tensor([ i["IO_token_id"][0] for i in prompt_dict_ls_ORIG ], dtype=torch.int)
    s1_token_id_ls_ORIG = torch.tensor([ i["S_token_id"][0] for i in prompt_dict_ls_ORIG ], dtype=torch.int)
    end_token_pos_ls_ORIG = torch.tensor([ i["END_token_pos"] for i in prompt_dict_ls_ORIG ], dtype=torch.int)
    recv_type = recv_info["type"]

    ## metrics
    n_prompts = len(prompt_dict_ls_ORIG)
    logit_diff_matrix = torch.zeros((n_prompts, n_layers, n_heads, len(recv_type)))
    logit_diff_ORIG = torch.zeros((n_prompts))
    prob_io_name_matrix = torch.zeros((n_prompts, n_layers, n_heads, len(recv_type)))
    prob_io_name_ORIG = torch.zeros((n_prompts))

    ## Now, process X_ORIG (forward pass B), X_NEW (forward pass A)
    for B in tqdm(range(0, n_prompts, bsz)):
        ## tokenization
        batch_token_ORIG = tokenizer(prompt_ls_ORIG[B:B+bsz], return_tensors="pt", padding=True)
        batch_token_NEW = tokenizer(prompt_ls_NEW[B:B+bsz], return_tensors="pt", padding=True)
        ## token positions, token id's
        cur_bsz = len(batch_token_ORIG["input_ids"]) # current batch size
        send_token_pos_ls = torch.tensor(send_info["token_pos_ls"][B:B+cur_bsz]) # positions of sending tokens
        recv_token_pos_ls = torch.tensor(recv_info["token_pos_ls"][B:B+cur_bsz]) # positions of receiving tokens
        io_name_id_ls = io_token_id_ls_ORIG[B:B+cur_bsz] # id's of IO tokens
        s1_name_id_ls = s1_token_id_ls_ORIG[B:B+cur_bsz] # id's of S1 tokens
        # print(prompt_ls_ORIG[0], send_token_pos_ls[0], recv_token_pos_ls[0], io_name_id_ls[0], s1_name_id_ls[0], batch_token_ORIG["input_ids"][0]) # for check
        ## input
        model_outputs_ORIG, hook_dict_ORIG = model(input_ids=batch_token_ORIG["input_ids"], attention_mask=batch_token_ORIG["attention_mask"]) # forward pass B
        _, hook_dict_NEW = model(input_ids=batch_token_NEW["input_ids"], attention_mask=batch_token_NEW["attention_mask"]) # forward pass A
        ## prediction
        prediction_ORIG = model_outputs_ORIG[0][torch.arange(cur_bsz), end_token_pos_ls_ORIG[B:B+cur_bsz]]
        ## metrics
        logit_diff_ORIG[B:B+bsz] = logit_diff_batch(prediction_ORIG, io_name_id_ls, s1_name_id_ls)
        prob_io_name_ORIG[B:B+bsz] = prob_batch(prediction_ORIG, io_name_id_ls)
        ## preparation
        # attn_out_ORIG = hook_dict_ORIG["hook_attn_output"] # shape: [cur_bsz, n_layers, max_n_tokens, n_dim]
        q_ORIG = hook_dict_ORIG["hook_q"] # shape: [cur_bsz, n_layers, n_heads, max_n_tokens, n_head_dim]
        k_ORIG = hook_dict_ORIG["hook_k"] # shape: [cur_bsz, n_layers, n_heads, max_n_tokens, n_head_dim]
        v_ORIG = hook_dict_ORIG["hook_v"] # shape: [cur_bsz, n_layers, n_heads, max_n_tokens, n_head_dim]
        before_matmul_wo_ORIG = hook_dict_ORIG["hook_before_matmul_wo"] # shape: [cur_bsz, n_layers, n_heads, max_n_tokens, n_head_dim] (OLMoE) / [cur_bsz, n_layers, max_n_tokens, n_heads, n_head_dim] (GPT2)
        before_matmul_wo_NEW = hook_dict_NEW["hook_before_matmul_wo"]
        # print(q_ORIG.shape, k_ORIG.shape, v_ORIG.shape, before_matmul_wo_ORIG.shape)

        ## Now, patching (forward pass C)
        patch_C = []
        # freeze all heads
        for L in range(0, n_layers):# min_recv_layer_id):
            patch_C.append(["q", L, q_ORIG[:, L]])
            patch_C.append(["k", L, k_ORIG[:, L]])
            patch_C.append(["v", L, v_ORIG[:, L]])
        # replace sender (heads) one by one
        for send_L in range(0, n_layers): # TODO: no need to loop if send_L >= deepest recv_L, may modify here to save time
            for send_H in range(0, n_heads):
                ## construct the patch
                bmw_C = before_matmul_wo_ORIG[:,send_L].detach().clone()
                bmw_C[torch.arange(cur_bsz), send_H, send_token_pos_ls] = before_matmul_wo_NEW[torch.arange(cur_bsz), send_L, send_H, send_token_pos_ls] # NOTE: for OLMoE
                # NOTE: If need patching 2 tokens at the same time, add it here manually
                # bmw_C[torch.arange(cur_bsz), send_token_pos_ls, send_H] = before_matmul_wo_NEW[torch.arange(cur_bsz), send_L, send_token_pos_ls, send_H] # NOTE: for GPT2

                patch_C_extra = [["before_matmul_wo", send_L, send_H, bmw_C]]
                model_outputs_C, hook_dict_C = model(input_ids=batch_token_ORIG["input_ids"], attention_mask=batch_token_ORIG["attention_mask"], patching=(patch_C + patch_C_extra))

                prediction_C = model_outputs_C[0][torch.arange(cur_bsz), end_token_pos_ls_ORIG[B:B+cur_bsz]] # for check
                # print("C", send_L, send_H, logit_diff_batch(prediction_C, io_name_id_ls, s1_name_id_ls)) # for check
                for counter, cur_recv_type in enumerate(recv_type):
                    if "l" == cur_recv_type: # last resid_post, so forward pass D is not necessary
                        prediction_C = model_outputs_C[0][torch.arange(cur_bsz), end_token_pos_ls_ORIG[B:B+cur_bsz]]
                        logit_diff_matrix[B:B+cur_bsz, send_L, send_H, counter] = logit_diff_batch(prediction_C, io_name_id_ls, s1_name_id_ls)
                        prob_io_name_matrix[B:B+cur_bsz, send_L, send_H, counter] = prob_batch(prediction_C, io_name_id_ls)
                    else: # "q"/"k"/"v"
                        ## Now, patching (forward pass D)
                        patch_D = []
                        for d_L, d_H in recv_info["head_pos"]:
                            match cur_recv_type: # can be compacted as one statement ... cur_recv_type + "_head" 
                                case "q":
                                    patch_D.append(["q_head", d_L, d_H, recv_token_pos_ls, hook_dict_C["hook_q"][torch.arange(cur_bsz), d_L, d_H, recv_token_pos_ls]])
                                case "k":
                                    patch_D.append(["k_head", d_L, d_H, recv_token_pos_ls, hook_dict_C["hook_k"][torch.arange(cur_bsz), d_L, d_H, recv_token_pos_ls]])
                                case "v":
                                    patch_D.append(["v_head", d_L, d_H, recv_token_pos_ls, hook_dict_C["hook_v"][torch.arange(cur_bsz), d_L, d_H, recv_token_pos_ls]])
                        model_outputs_D, _ = model(input_ids=batch_token_ORIG["input_ids"], attention_mask=batch_token_ORIG["attention_mask"], patching=patch_D)
                        prediction_D = model_outputs_D[0][torch.arange(cur_bsz), end_token_pos_ls_ORIG[B:B+cur_bsz]]
                        # print("D", send_L, send_H, logit_diff_batch(prediction_D, io_name_id_ls, s1_name_id_ls)) # for check
                        logit_diff_matrix[B:B+cur_bsz, send_L, send_H, counter] = logit_diff_batch(prediction_D, io_name_id_ls, s1_name_id_ls)
                        prob_io_name_matrix[B:B+cur_bsz, send_L, send_H, counter] = prob_batch(prediction_D, io_name_id_ls)
    
    logit_diff_ORIG = logit_diff_ORIG.unsqueeze(1).repeat(1, n_layers * n_heads * len(recv_type)).reshape(n_prompts, n_layers, n_heads, len(recv_type))
    logit_diff_normalized_matrix = torch.div(logit_diff_matrix - logit_diff_ORIG, logit_diff_ORIG).mean(0)
    # print(logit_diff_normalized_matrix)

    prob_io_name_ORIG =prob_io_name_ORIG.unsqueeze(1).repeat(1, n_layers * n_heads * len(recv_type)).reshape(n_prompts, n_layers, n_heads, len(recv_type))
    prob_io_name_diff_matrix = (prob_io_name_matrix - prob_io_name_ORIG).mean(0)
    
    for counter, cur_recv_type in enumerate(recv_type):
        if output_dir is not None:
            matrix_drawer_patch(logit_diff_normalized_matrix[:, :, counter], "logit_diffs_normalized_{}".format(cur_recv_type), output_dir, title="logit_diffs_normalized_{}".format(cur_recv_type))
        if demo_now:
            fig = px.imshow(logit_diff_normalized_matrix[:, :, counter].detach().cpu().numpy(), color_continuous_scale="RdBu", color_continuous_midpoint=0, title="logit_diffs_normalized_{}".format(cur_recv_type), labels=dict(x="Head", y="Layer", color="Logit diff. variation"))
            fig.update_xaxes(side="top")
            fig.show()
            # fig = px.imshow(prob_io_name_diff_matrix[:, :, counter].detach().cpu().numpy(), color_continuous_scale="RdBu", color_continuous_midpoint=0, title="prob_io_name_diff_{}".format(cur_recv_type), labels=dict(x="Head", y="Layer", color="Prob diff"))
            # fig.update_xaxes(side="top")
            # fig.show()

def pos_tagging(prompt_ls, tokenizer, max_token_per_prompt, dataset_sz=-1):
    """ Add part-of-speech tag to each token. 
        If dataset_sz=-1, then it will try to use all the given prompts.
        TODO: may use spaCy to replace nltk?
    """
    # reference: https://github.com/slavpetrov/universal-pos-tags/blob/master/en-ptb.map
    pos_swap_map = {".":'.', "(":'.', ")":'.', ":":'.', "''":'.', "EX":'DET', "JJS":'ADJ', "WRB":'ADV', "VBG":'VERB', "VBP":'VERB', "NN":'NOUN', "SYM":'X', "VB":'VERB', "UH":'X', "NNPS":'NOUN', "NNP":'NOUN', "``":'.', "$":'.', "NNS":'NOUN', "JJR":'ADJ', "MD":'VERB', "RP":'PRT', "VBD":'VERB', "DT":'DET', "POS":'PRT', "RBR":'ADV', ",":'.', "VBZ":'VERB', "PDT":'DET', "VBN":'VERB', "WP$":'PRON', "WDT":'DET', "WP":'PRON', "PRP$":'PRON', "CD":'NUM', "IN":'ADP', "#":'.', "CC":'CONJ', "RB":'ADV', "FW":'X', "RBS":'ADV', "PRP":'PRON', "LS":'X', "JJ":'ADJ', "TO":'PRT'} 
    # prompt_pos_ls = []
    # for prompt in prompt_ls:
    #     pos_tagging = nltk.pos_tag(nltk.tokenize.word_tokenize(prompt))
    #     prompt_pos_ls.append([[i[0], pos_swap_map[i[1]]] for i in pos_tagging])
    token_pos_ls = []
    saved_prompt_id = []
    # bad_token_dict = dict()
    # bad_token_counter = 0

    encodings = tokenizer(prompt_ls, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True) # padding=False may be removed?
    tokenized_words = [nltk.tokenize.word_tokenize(p) for p in prompt_ls]
    pos_tags = nltk.pos_tag_sents(tokenized_words)
   
    for i, cur_prompt in enumerate(prompt_ls):
        tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][i])
        offsets = encodings["offset_mapping"][i]
        pos_tagging = pos_tags[i]
        ## find the intervals of the words given by nltk
        cursor = 0
        word_info = []
        for cur_word, cur_pos in pos_tagging: # robust enough?
            word_begin = cur_prompt.find(cur_word, cursor)
            word_end = word_begin + len(cur_word)
            word_info.append((word_begin, word_end, cur_pos, cur_word))
            cursor = word_end
        # print(word_info)

        cur_token_pos_ls = []
        word_idx = 0
        for tok, (tok_begin, tok_end) in zip(tokens, offsets):
            tok_pos = None
            
            while word_idx < len(word_info) and tok_begin >= word_info[word_idx][1]: # word_info[word_idx][1] is the next position after the last character of a word
                word_idx += 1
            if word_idx < len(word_info):
                word_begin, word_end, word_pos, _ = word_info[word_idx]
                # if tok_end <= word_end and ((tok_begin >= word_begin) or (tok_begin == (word_begin - 1) and word_begin > 0 and cur_prompt[word_begin - 1] == " ")): # alternative
                if tok_end <= word_end and ((tok_begin >= word_begin) or (tok_begin == (word_begin - 1) and cur_prompt[word_begin - 1] == " ")): # may be other characters instead of " ", but we disregard those characters for simplicity
                    tok_pos = pos_swap_map[word_pos]
                
            if tok_pos is None:
                tok_pos = "?" # failure
                # print("{} [{}]".format(i,tok), "tag undetermined")
                # bad_token_dict[tok] = bad_token_dict.get(tok, 0) + 1
                # bad_token_counter += 1
                
            cur_token_pos_ls.append((tok, tok_pos))

        if len(cur_token_pos_ls) == max_token_per_prompt:
            saved_prompt_id.append(i)
        else:
            print("Not long enough: id {} len {}".format(i, len(cur_token_pos_ls)))
        token_pos_ls.append(cur_token_pos_ls)

        if dataset_sz != -1 and len(saved_prompt_id) >= dataset_sz:
            break

    # print(bad_token_counter)
    # print(sorted(bad_token_dict))
    print("len of token_pos_ls", len(token_pos_ls))
    # batch_token = tokenizer([prompt_ls[j] for j in saved_prompt_id], return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True)
    batch_token = {k: v[saved_prompt_id] for k, v in encodings.items() if k != "offset_mapping"}
    # return saved_prompt_id, batch_token, [token_pos_ls[i] for i in saved_prompt_id]
    return batch_token, token_pos_ls

def decompose_token_tsne(prompt_ls, model, tokenizer, router_weight_ls, output_dir, bsz=50, max_token_per_prompt=32, dataset_sz=1000, demo_now=False):
    """ Figures 2a, 2b """
    batch_token, token_pos_ls = pos_tagging(prompt_ls, tokenizer, max_token_per_prompt, dataset_sz) # get the POS of tokens
    n_prompts, max_n_tokens = batch_token["attention_mask"].shape
    print("num of prompts: {}".format(n_prompts))

    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, n_dim = router_weight_vectors.shape

    token_score_collect = torch.zeros((n_prompts, max_n_tokens, n_experts, n_layers, 1), device="cpu") # n_layers -> recv_layer
    token_embedding_collect = torch.zeros((n_prompts, max_n_tokens, n_dim), device="cpu")

    for B in tqdm(range(0, n_prompts, bsz)):
        _, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        layer_input = hook_dict["hook_layer_input"] # shape: [n_prompts, n_layers, max_n_tokens, n_dim]
        after_res1 = hook_dict["hook_after_res1"] # shape: [n_prompts, n_layers, max_n_tokens, n_dim]
        token_components = layer_input[:, 0, :, :].unsqueeze(1) # res_in of Layer 0
        token_rmsnorm = rmsnorm_breakdown_batch(after_res1, [token_components], model, mode="TAM")[0]
        # token_rmsnorm = diff_breakdown_batch(after_res1, token_components, model, mode="default") # NOTE: decomposition based on difference
        
        ## NOTE: token_rmsnorm shape: [n_prompts, n_layers, 1, max_n_tokens, n_dim]
        token_score = torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), token_rmsnorm)
        token_score_collect[B:B+bsz, :, :, :, :] = token_score
        token_embedding_collect[B:B+bsz, :, :] = layer_input[:, 0, :, :]

        # for check
        # print(layer_input.shape, after_res1.shape, token_rmsnorm.shape, batch_token["attention_mask"][B:B+bsz].sum(1))
    
    ## part-of-speech (Figure 1a)
    token_score_plot = token_score_collect.reshape(n_prompts * max_n_tokens, -1)
    T_tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30, max_iter=800).fit_transform(token_score_plot.detach().numpy())
    tsne_min, tsne_max = T_tsne.min(0), T_tsne.max(0)
    tsne_norm = (T_tsne - tsne_min) / (tsne_max - tsne_min)

    plt.figure(figsize=(20, 20))

    POS_colors = {"VERB":'grey', "NOUN":'red', "PRON":'peru',"ADJ":'orange', "ADV":'yellowgreen', "ADP":'lightgreen', "CONJ":'green', "DET":'aqua', "NUM":'blue', "PRT":'steelblue', "X":'purple', ".":'pink', "?":'yellow'}
    pos_ls = []
    color_ls = []
    alpha_ls = [] # we want to filter out the token with undetermined pos ("?")
    for k in token_pos_ls:
        # pos_ls.extend([i[1] for i in k]) # used for px.scatter
        pos_ls.extend(k)
        color_ls.extend([POS_colors[i[1]] for i in k])
        alpha_ls.extend([1 if i[1] != "?" else 0 for i in k])
    print("len of pos_ls:", len(pos_ls))
    
    np.save(output_dir + "tsne_norm" + ".npy", tsne_norm)
    np.save(output_dir + "color_ls" + ".npy", color_ls)
    np.save(output_dir + "alpha_ls" + ".npy", alpha_ls)
    fig, ax = plt.subplots()
    ax.scatter(tsne_norm[:, 0], tsne_norm[:, 1], s=5, c=color_ls, alpha=alpha_ls)
    content_patches = [mpatches.Patch(color="red", label="Noun"), 
                       mpatches.Patch(color="orange", label="Adjective"),
                       mpatches.Patch(color="grey", label="Verb"),
                       mpatches.Patch(color="yellowgreen", label="Adverb"),
                       mpatches.Patch(color="blue", label="Number")
                       ]

    function_patches = [mpatches.Patch(color="green", label="Conjunction"),
                        mpatches.Patch(color="aqua", label="Determiner"),
                        mpatches.Patch(color="lightgreen", label="Adposition"),
                        mpatches.Patch(color="pink", label="Punctuation"),
                        mpatches.Patch(color="peru", label="Pronoun"),
                        mpatches.Patch(color="steelblue", label="Particle")]

    other_patches = [mpatches.Patch(color="purple", label="Foreign word, typo, abbr.")]

    legend_content = ax.legend(handles=content_patches, title="Content words", loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend_content)
    legend_function = ax.legend(handles=function_patches, title="Function words", loc="upper left", bbox_to_anchor=(1.05, 0.6))
    ax.add_artist(legend_function)
    legend_other = ax.legend(handles=other_patches, title="Other words", loc="upper left", bbox_to_anchor=(0.85, 0.15))
    ax.add_artist(legend_other)

    plt.axis("off")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(output_dir + "token_POS_distribution" + ".png") # ".pdf"
    plt.close("all")
    
    ## position (Figure 2b)
    ## We do not use this version.
    # tsne_norm2 = tsne_norm.reshape(n_prompts, max_n_tokens, -1)
    # print(tsne_norm2.shape)
    # selection = [0, 1, 5, 10]
    # plt.figure(figsize=(9, 9))
    # position_colors = ["red","black", "grey", "blue"]
    # for k, (position, color) in enumerate(zip(selection, position_colors)):
    #     plt.scatter(tsne_norm2[:, k, 0], tsne_norm2[:, k, 1], s=5, c=color, label=str(position))
    # plt.legend()
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(output_dir + "token_position_distribution" + ".png") # ".pdf"
    # plt.close("all")
    
    selection = [0, 1, 5, 10, 15, 20]
    position_colors = ["red", "black", "grey", "blue", "purple", "aqua"]
    token_score_plot2 = token_score_collect[:, selection, :, :].reshape(n_prompts * len(selection), -1)
    T_tsne2 = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30, max_iter=800).fit_transform(token_score_plot2.detach().numpy())
    tsne_min2, tsne_max2 = T_tsne2.min(0), T_tsne2.max(0)
    tsne_norm2 = (T_tsne2 - tsne_min2) / (tsne_max2 - tsne_min2)
    np.save(output_dir + "tsne_norm2" + ".npy", tsne_norm2)
    
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    tsne_norm2 = tsne_norm2.reshape(n_prompts, len(selection), -1)
    for k, (position, color) in enumerate(zip(selection, position_colors)):
        ax2.scatter(tsne_norm2[:, k, 0], tsne_norm2[:, k, 1], s=5, c=color, label=str(position))

    # ax2.legend(title="Position of token", loc="upper left", bbox_to_anchor=(1.05, 1)) # just change the legend
    
    legend_patches = [mpatches.Patch(color=position_colors[k], label=str(selection[k])) for k in range(len(selection))] 
    legend_position = ax2.legend(handles=legend_patches, title="Position of token", loc="upper left", bbox_to_anchor=(1.05, 1))
    ax2.add_artist(legend_position)
    
    ax2.axis("off")
    fig2.tight_layout(rect=[0, 0, 0.75, 1])
    fig2.savefig(output_dir + "token_position_distribution" + ".png") # ".pdf"
    plt.close("all")
    
    ## embedding (for comparison, not used in the paper)
    token_embedding_plot = token_embedding_collect.reshape(n_prompts * max_n_tokens, -1)
    T_tsne3 = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=100, max_iter=1200).fit_transform(token_embedding_plot.detach().cpu().numpy())
    tsne_min3, tsne_max3 = T_tsne3.min(0), T_tsne3.max(0)
    tsne_norm3 = (T_tsne3 - tsne_min3) / (tsne_max3 - tsne_min3)

    if demo_now:
        fig = px.scatter(x=tsne_norm[:, 0], y=tsne_norm[:, 1], color=pos_ls, title="token_POS_distribution (Figure 2a)")
        fig.show()
        fig = px.scatter(x=tsne_norm3[:, 0], y=tsne_norm3[:, 1], color=pos_ls, title="token_embedding_POS_distribution")
        fig.show()
        fig = px.scatter(x=tsne_norm2[:, :, 0].reshape(-1), y=tsne_norm2[:, :, 1].reshape(-1), color=["position" + str(i) for i in selection] * n_prompts, title="token_position_distribution (Figure 2b)")
        fig.show()
    return

def decompose_TAM_tril(prompt_ls, model, tokenizer, router_weight_ls, output_dir, top_n, bsz=100, max_token_per_prompt=32, demo_now=False):
    """ Decomposition: multiple prompts, token(T), attn_out(A), and moe_out(M).
        Note that top_n can vary - top_k or n_experts or other ranges.
        Figures 2c and 3.
    """
    batch_token = tokenizer(prompt_ls, return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True) # should not use padding in principle
    n_prompts, max_n_tokens = batch_token["attention_mask"].shape
    
    router_weight_vectors = torch.stack(router_weight_ls, dim=0)#.float() # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, _ = router_weight_vectors.shape

    token_score_var_collect = torch.zeros((max_n_tokens, n_layers, 1))
    attn_out_score_var_collect = torch.zeros((max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    moe_out_score_var_collect = torch.zeros((max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer

    token_score_topn_sum_collect = torch.zeros((max_n_tokens, n_layers, 1))
    moe_out_score_topn_sum_collect = torch.zeros((max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    attn_out_score_topn_sum_collect = torch.zeros((max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    token_score_topn_abs_sum_collect = torch.zeros((max_n_tokens, n_layers, 1))
    moe_out_score_topn_abs_sum_collect = torch.zeros((max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    attn_out_score_topn_abs_sum_collect = torch.zeros((max_n_tokens, n_layers, n_layers)) # first n_layers -> recv_layer , second n_layers -> send_layer
    
    expert_score_collect = torch.zeros((n_prompts, max_n_tokens, n_experts, n_layers))
    
    ## NOTE: extra info
    token_rmsnorm_norm = torch.zeros((max_n_tokens, n_layers, 1)) # TRS
    moe_out_rmsnorm_norm = torch.zeros((max_n_tokens, n_layers, n_layers))
    attn_out_rmsnorm_norm = torch.zeros((max_n_tokens, n_layers, n_layers))
    token_rmsnorm2_norm = torch.zeros((max_n_tokens, n_layers, 1))
    moe_out_rmsnorm2_norm = torch.zeros((max_n_tokens, n_layers, n_layers))
    attn_out_rmsnorm2_norm = torch.zeros((max_n_tokens, n_layers, n_layers))

    token_norm = torch.zeros((max_n_tokens, 1))
    moe_out_norm = torch.zeros((max_n_tokens, n_layers))
    attn_out_norm = torch.zeros((max_n_tokens, n_layers))
    token_id_collect = torch.zeros(len(prompt_ls))
    m1e9_count = 0
    m1e18_count = 0

    def transfer(data):
        out = []
        for r in data:
            tmp = []
            for c in r:
                if c < 16:
                    tmp.append('A'+str(c))
                elif c < 32:
                    tmp.append('M'+str(c-16))
                else:
                    tmp.append('T')
            out.append(tmp)
        print(out)
        # print(data)

    for B in tqdm(range(0, n_prompts, bsz)):
        ## NOTE: for checking predicted tokens, commented by default
        # model_outputs, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        # prediction = model_outputs[0] # [batch_size, n_tokens, vocab_size]
        # predicted_top1 = torch.argsort(prediction[:, -1], descending=True)[:, :1]
        # token_id_collect[B:B + predicted_top1.shape[0]] = predicted_top1[:, 0]
        # for i in range(bsz):
        #     predicted_token_id = torch.argsort(prediction[i, -1], descending=True)[:1] # 0=first prompt, -1=last token
        #     predicted_text = [tokenizer.decode(x) for x in predicted_token_id]
        #     print(predicted_text, predicted_token_id)
        # continue

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
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors.float(), after_norm2) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers] # .float() for qwen
        top_n_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :top_n, :]
        token_score = torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), token_rmsnorm)
        attn_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), attn_out_rmsnorm), diagonal=0)
        moe_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), moe_out_rmsnorm), diagonal=-1) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers, n_layers] first n_layers -> recv_layer , second n_layers -> send_layer
        
        ## NOTE: additional experiment 1 (causal intervention experiment)
        token_rmsnorm2 = diff_breakdown_batch(after_res1, token_components, model, mode="default") # NOTE: decomposition based on difference
        attn_out_rmsnorm2 = diff_breakdown_batch(after_res1, attn_output, model, mode="default") # NOTE: decomposition based on difference
        moe_out_rmsnorm2 = diff_breakdown_batch(after_res1, mlp_output, model, mode="default") # NOTE: decomposition based on difference
        # token_score = torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), token_rmsnorm2)
        # attn_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), attn_out_rmsnorm2), diagonal=0)
        # moe_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), moe_out_rmsnorm2), diagonal=-1)
        
        ######
        ## NOTE: additional experiment 2 (remove a high-variance expert and observe)
        top_k = 8
        L, E = 4, 14  # options: L4E14, L2E30
        tmp_mlp_output = mlp_output.clone()
        mask = (top_n_experts[:, :, :top_k, L] == E).unsqueeze(-1)
        expert_weighted_outputs = hook_dict["hook_expert_weighted_outputs"] # shape: [n_prompts_B, n_layers, max_n_tokens, original_top_k, n_dim]
        tmp_mlp_output[:, L] -= (expert_weighted_outputs[:, L] * mask).sum(dim=2)
        tmp_moe_out_rmsnorm = rmsnorm_breakdown_batch(after_res1, [tmp_mlp_output], model, mode="TAM")[0]
        tmp_moe_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), tmp_moe_out_rmsnorm), diagonal=-1)
        m1e9_count += torch.nonzero(top_n_experts[:, :, :top_k, 1] == 9).shape[0]
        m1e18_count += torch.nonzero(top_n_experts[:, :, :top_k, 1] == 18).shape[0]
        # OBSOLETE implementation (Do not use)
        # top_k_experts_indices = torch.nonzero(top_n_experts[:, :, :top_k, L] == E)
        # expert_weighted_outputs = hook_dict["hook_expert_weighted_outputs"] # shape: [n_prompts_B, n_layers, max_n_tokens, original_top_k, n_dim]
        # tmp_mlp_output2 = mlp_output.clone()
        # for p, t, e in top_k_experts_indices:
        #     tmp_mlp_output2[p, L, t] -= expert_weighted_outputs[p, L, t, e] # remove the influence of expert (L, E)
        # print(torch.allclose(tmp_mlp_output, tmp_mlp_output2))
        ######
        ## variance (will be averaged)
        token_score_var_collect += token_score.var(dim=2).sum(dim=0)
        attn_out_score_var_collect += attn_out_score.var(dim=2).sum(dim=0)
        moe_out_score_var_collect += moe_out_score.var(dim=2).sum(dim=0)

        ## check the components with largest variance for each token
        # compact_vars = torch.cat([attn_out_score.var(dim=2), moe_out_score.var(dim=2), token_score.var(dim=2)], dim=-1)
        # compact_vars_argsort = torch.argsort(compact_vars, descending=True, dim=-1)
        # print(compact_vars_argsort.shape)
        # print(tokenizer.decode(batch_token["input_ids"][B]))
        # for k in range(10):
        #     print(k, tokenizer.decode(batch_token["input_ids"][B, k]))
        #     print(transfer(compact_vars_argsort[B,k,:,:5].detach().cpu().numpy()))
        # exit()

        ## top-n
        ind_token = top_n_experts.long().unsqueeze(-1)
        ind_other = top_n_experts.long().unsqueeze(-1).expand(-1, -1, -1, -1, n_layers).contiguous()
        token_score_topn_sum_collect += torch.gather(token_score, dim=2, index=ind_token).sum(dim=(0, 2))
        moe_out_score_topn_sum_collect += torch.gather(moe_out_score, dim=2, index=ind_other).sum(dim=(0, 2))
        attn_out_score_topn_sum_collect += torch.gather(attn_out_score, dim=2, index=ind_other).sum(dim=(0, 2))
        token_score_topn_abs_sum_collect += torch.gather(token_score, dim=2, index=ind_token).abs().sum(dim=(0, 2))
        moe_out_score_topn_abs_sum_collect += torch.gather(moe_out_score, dim=2, index=ind_other).abs().sum(dim=(0, 2))
        attn_out_score_topn_abs_sum_collect += torch.gather(attn_out_score, dim=2, index=ind_other).abs().sum(dim=(0, 2))

        ## expert score collection
        expert_score_collect[B:B+bsz] = original_score
        
        ## NOTE: check if implemented correctly
        # tmp_token_score_topn = torch.gather(token_score, dim=2, index=top_n_experts.long().unsqueeze(-1))
        # print(token_score[0,1,:,2,0])
        # print(top_n_experts[0,1,:,2])
        # print(tmp_token_score_topn[0,1,:,2,0]) # sort token_score by top_n_experts
        # tmp_moe_out_score_topn = torch.gather(moe_out_score, dim=2, index=top_n_experts.long().unsqueeze(-1).repeat(1,1,1,1,n_layers))
        # print(moe_out_score[0,1,:,3,2])
        # print(top_n_experts[0,1,:,3])
        # print(tmp_moe_out_score_topn[0,1,:,3,2])
        # exit()

        ## NOTE: check the angle between component and the input vector of RMSNorm, commented by default
        # for x in range(n_layers):
        #     cos_sim_token = torch.dot(mlp_output[1, 6, 2], after_res1[1, x, 2])/ (torch.norm(mlp_output[1, 6, 2]) * torch.norm(after_res1[1, x, 2]))
        #     print(cos_sim_token, torch.norm(mlp_output[1, 6, 2]), torch.norm(after_res1[1, x, 2]), x)
        # cos_sim_attn_out = torch.nn.functional.cosine_similarity(attn_output, after_res1, dim=-1)
        # print(cos_sim_attn_out.shape, attn_output.shape, cos_sim_attn_out.shape)
        # exit()

        ## NOTE: for check
        # tmp_after_norm2 = diff_breakdown_batch(after_res1, after_res1, model, mode="default", variance_epsilon=1e-05, device="cuda:0")
        # print(tmp_after_norm2[0,0,0,:3], after_norm2[0,0,0,:3])
        # tmp_after_norm2 = diff_breakdown_batch(after_res1, attn_output, model, mode="default", variance_epsilon=1e-05, device="cuda:0")
        # tmp_attn_out_score = torch.tril(torch.einsum("RED,PRSTD->PTERS", router_weight_vectors.float(), tmp_after_norm2), diagonal=0)
        # print(tmp_attn_out_score[0,0,0,:,0], attn_out_score[0,0,0,:,0])
        
        token_rmsnorm_norm += (torch.norm(token_rmsnorm, p=2, dim=-1)).sum(0).permute(2,0,1)
        moe_out_rmsnorm_norm += torch.tril((torch.norm(moe_out_rmsnorm, p=2, dim=-1)).sum(0).permute(2,0,1), diagonal=-1)
        attn_out_rmsnorm_norm += torch.tril((torch.norm(attn_out_rmsnorm, p=2, dim=-1)).sum(0).permute(2,0,1), diagonal=0)
        token_rmsnorm2_norm += (torch.norm(token_rmsnorm2, p=2, dim=-1)).sum(0).permute(2,0,1)
        moe_out_rmsnorm2_norm += torch.tril((torch.norm(moe_out_rmsnorm2, p=2, dim=-1)).sum(0).permute(2,0,1), diagonal=-1)
        attn_out_rmsnorm2_norm += torch.tril((torch.norm(attn_out_rmsnorm2, p=2, dim=-1)).sum(0).permute(2,0,1), diagonal=0)

        attn_out_norm += (torch.norm(attn_output, p=2, dim=-1)).sum(0).permute(1,0)
        moe_out_norm += (torch.norm(mlp_output, p=2, dim=-1)).sum(0).permute(1,0)
        token_norm += (torch.norm(token_components, p=2, dim=-1)).sum(0).permute(1,0)

    token_vars = token_score_var_collect[1:].mean(0).div(n_prompts) # drop leading token
    attn_vars = attn_out_score_var_collect[1:].mean(0).div(n_prompts) # drop leading token
    moe_vars = moe_out_score_var_collect[1:].mean(0).div(n_prompts) # drop leading token
    
    compact_vars = torch.cat([attn_vars, moe_vars, token_vars], dim=1)
    compact_vars_argsort = torch.argsort(compact_vars, descending=True, dim=1)
    
    transfer(compact_vars_argsort[:,:5].detach().cpu().numpy()) # retrieve the sending layers with largest variance of assigned scores for each receiving layers

    ## var, pos/neg score of tokens (w/o Token 0)
    tril_drawer_tam_analyze(token_score_var_collect[1:].mean(0).div(n_prompts), name="token_var_without_T0", output_dir=output_dir, figsize=(3, 11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="Receiving MoE Layer", need_lognorm=True, need_description=True, tick_mode="T", cbar_label="Variance", demo_now=demo_now)
    tril_drawer_tam_analyze((token_score_topn_sum_collect[1:] + token_score_topn_abs_sum_collect[1:]).div(2 * top_n).mean(0).div(n_prompts), name="token_avg_positive_without_T0", output_dir=output_dir, figsize=(3, 11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="Receiving MoE Layer", need_description=True, tick_mode="T", cbar_label="Average Positive Score", demo_now=demo_now)
    tril_drawer_tam_analyze((token_score_topn_sum_collect[1:] - token_score_topn_abs_sum_collect[1:]).div(2 * top_n).mean(0).div(n_prompts), name="token_avg_negative_without_T0", output_dir=output_dir, figsize=(3, 11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="Receiving MoE Layer", need_description=True, tick_mode="T", cbar_label="Average Negative Score", demo_now=demo_now)
    
    ## var, pos/neg of attn and moe (w/o Token 0)
    tril_drawer_tam_analyze(attn_out_score_var_collect[1:].mean(0).div(n_prompts), name="attn_var_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Variance of scores assigned by attention layers", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_lognorm=True, need_description=True, tick_mode="A", cbar_label="Variance", demo_now=demo_now)
    tril_drawer_tam_analyze(moe_out_score_var_collect[1:].mean(0).div(n_prompts), name="moe_var_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Variance of scores assigned by MoE layers", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_lognorm=True, need_description=True, tick_mode="M", cbar_label="Variance", demo_now=demo_now)
    tril_drawer_tam_analyze((attn_out_score_topn_sum_collect[1:] + attn_out_score_topn_abs_sum_collect[1:]).div(2 * top_n).mean(0).div(n_prompts), name="attn_avg_positive_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Average Positive Scores assigned by attention layers", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="A", cbar_label="Average Positive Score", demo_now=demo_now)
    tril_drawer_tam_analyze((attn_out_score_topn_sum_collect[1:] - attn_out_score_topn_abs_sum_collect[1:]).div(2 * top_n).mean(0).div(n_prompts), name="attn_avg_negative_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Average Negative Scores assigned by attention layers", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="A", cbar_label="Average Negative Score", demo_now=demo_now)
    tril_drawer_tam_analyze((moe_out_score_topn_sum_collect[1:] + moe_out_score_topn_abs_sum_collect[1:]).div(2 * top_n).mean(0).div(n_prompts), name="moe_avg_positive_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Average Positive Scores assigned by MoE layers", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="M", cbar_label="Average Positive Score", demo_now=demo_now)
    tril_drawer_tam_analyze((moe_out_score_topn_sum_collect[1:] - moe_out_score_topn_abs_sum_collect[1:]).div(2 * top_n).mean(0).div(n_prompts), name="moe_avg_negative_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Average Negative Scores assigned by MoE layers", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="M", cbar_label="Average Negative Score", demo_now=demo_now)
    
    ## var, pos/neg scores of tokens (Token 0 only)
    tril_drawer_tam_analyze(token_score_var_collect[0].div(n_prompts), name="token_var_T0", output_dir=output_dir, figsize=(3, 11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="Receiving MoE Layer", need_lognorm=True, need_description=True, tick_mode="T", cbar_label="Variance", demo_now=False)
    tril_drawer_tam_analyze((token_score_topn_sum_collect[0] + token_score_topn_abs_sum_collect[0]).div(2 * top_n).div(n_prompts), name="token_avg_positive_T0", output_dir=output_dir, figsize=(3, 11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="Receiving MoE Layer", need_description=True, tick_mode="T", cbar_label="Average Positive Score", demo_now=False)
    tril_drawer_tam_analyze((token_score_topn_sum_collect[0] - token_score_topn_abs_sum_collect[0]).div(2 * top_n).div(n_prompts), name="token_avg_negative_T0", output_dir=output_dir, figsize=(3, 11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="Receiving MoE Layer", need_description=True, tick_mode="T", cbar_label="Average Negative Score", demo_now=False)

    ## var, pos/neg of attn and moe (Token 0 only)
    tril_drawer_tam_analyze(attn_out_score_var_collect[0].div(n_prompts), name="attn_var_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Variance of scores assigned by attention layers (Token 0 only)", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_lognorm=True, need_description=True, tick_mode="A", cbar_label="Variance", demo_now=False)
    tril_drawer_tam_analyze(moe_out_score_var_collect[0].div(n_prompts), name="moe_var_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Variance of scores assigned by MoE layers (Token 0 only)", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_lognorm=True, need_description=True, tick_mode="M", cbar_label="Variance", demo_now=False)
    tril_drawer_tam_analyze((attn_out_score_topn_sum_collect[0] + attn_out_score_topn_abs_sum_collect[0]).div(2 * top_n).div(n_prompts), name="attn_avg_positive_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Average Positive Scores assigned by attention layers (Token 0 only)", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="A", cbar_label="Average Positive Score", demo_now=False)
    tril_drawer_tam_analyze((attn_out_score_topn_sum_collect[0] - attn_out_score_topn_abs_sum_collect[0]).div(2 * top_n).div(n_prompts), name="attn_avg_negative_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Average Negative Scores assigned by attention layers (Token 0 only)", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="A", cbar_label="Average Negative Score", demo_now=False)
    tril_drawer_tam_analyze((moe_out_score_topn_sum_collect[0] + moe_out_score_topn_abs_sum_collect[0]).div(2 * top_n).div(n_prompts), name="moe_avg_positive_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Average Positive Scores assigned by MoE layers (Token 0 only)", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="M", cbar_label="Average Positive Score", demo_now=False)
    tril_drawer_tam_analyze((moe_out_score_topn_sum_collect[0] - moe_out_score_topn_abs_sum_collect[0]).div(2 * top_n).div(n_prompts), name="moe_avg_negative_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Average Negative Scores assigned by MoE layers (Token 0 only)", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="M", cbar_label="Average Negative Score", demo_now=False)
    
    ## var, mean of the scores
    print("score var: ", expert_score_collect.reshape(-1, n_experts, n_layers).var(dim=1).mean(0)) # variance of expert scores in each layer
    print("score mean: ", expert_score_collect.reshape(-1, n_layers).mean(0)) # mean of expert scores in each layer
    
    ## norm
    tril_drawer_tam_analyze((moe_out_rmsnorm_norm[1:]).mean(0).div(n_prompts), name="old_moe_norm_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="Old Average Norm of components from MoE layers", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="M", cbar_label="Norm", demo_now=demo_now)
    tril_drawer_tam_analyze((attn_out_rmsnorm_norm[1:]).mean(0).div(n_prompts), name="old_attn_norm_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="Old Average Norm of components from attention layers", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="A", cbar_label="Norm", demo_now=demo_now)
    tril_drawer_tam_analyze((moe_out_rmsnorm2_norm[1:]).mean(0).div(n_prompts), name="new_moe_norm_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=0, add_patch=[], title="New Average Norm of components from MoE layers", xlabel="Sending MoE Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="M", cbar_label="Norm", demo_now=demo_now)
    tril_drawer_tam_analyze((attn_out_rmsnorm2_norm[1:]).mean(0).div(n_prompts), name="new_attn_norm_without_T0", output_dir=output_dir, figsize=(11, 11), diagonal=1, add_patch=[], title="New Average Norm of components from attention layers", xlabel="Sending Attention Layer", ylabel="Receiving MoE Layer", need_description=True, tick_mode="A", cbar_label="Norm", demo_now=demo_now)
    
    print("moe out norm", moe_out_norm[1:].mean(0).div(n_prompts))
    print("attn out norm", attn_out_norm[1:].mean(0).div(n_prompts))
    plt.scatter([i for i in range(n_layers)], moe_out_norm[1:].mean(0).div(n_prompts).detach().cpu().numpy(), s=5, c='b', label="L2-norm, MoE layer output")
    plt.scatter([i for i in range(n_layers)], attn_out_norm[1:].mean(0).div(n_prompts).detach().cpu().numpy(), s=5, c='g', label="L2-norm, attention layer output")
    plt.grid()
    plt.title("Average L2 norm of layer output")
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("L2 norm")
    np.save(output_dir + "l2_norm_moe_output" + ".npy", moe_out_norm[1:].mean(0).div(n_prompts).detach().cpu().numpy())
    np.save(output_dir + "l2_norm_attn_output" + ".npy", attn_out_norm[1:].mean(0).div(n_prompts).detach().cpu().numpy())
    plt.savefig(output_dir + "l2_norm" + ".png") # ".pdf"
    plt.close("all")

    print("hit counters", m1e9_count, m1e18_count)
    return

def decompose_IOI_map_score(prompt_dict_ls_1, prompt_dict_ls_2, model, tokenizer, router_weight_ls, output_dir, n_heads, top_n, bsz, demo_now=False):
    """ 1. score assigned by attention layer output. Apply simplified implementation. (q/k/h)
        2. attention map (after softmax)
        Note that top_n can vary - top_k or n_experts or other ranges.
        Figure * b~g. (appendix)
    """
    n_prompts = len(prompt_dict_ls_1)
    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, _, _ = router_weight_vectors.shape

    def helper(prompt_dict_ls):
        prompt_ls = [i["text"] for i in prompt_dict_ls]
        token_pos_ls_dict = {"END": torch.tensor([i["END_token_pos"] for i in prompt_dict_ls]), 
                             "S2": torch.tensor([i["S_token_pos"][1] for i in prompt_dict_ls]), 
                             "S1+1": torch.tensor([i["S1+1_token_pos"] for i in prompt_dict_ls]), 
                             "S1": torch.tensor([i["S_token_pos"][0] for i in prompt_dict_ls]), 
                             "IO": torch.tensor([i["IO_token_pos"] for i in prompt_dict_ls])}
        q_token_pos_ls = ["END", "END", "END", "END", "END", "S2", "S2", "S2", "S1+1"]
        k_token_pos_ls = ["END", "S2", "S1+1", "S1", "IO", "S1+1", "S1", "IO", "S1"]
        score_var_collect = torch.zeros((len(q_token_pos_ls), n_heads, n_layers, n_layers)) # first n_layers -> recv, second n_layers -> send
        score_avg_collect = torch.zeros((len(q_token_pos_ls), n_heads, n_layers, n_layers)) # first n_layers -> recv, second n_layers -> send
        attn_map_collect = torch.zeros((len(q_token_pos_ls), n_layers, n_heads)) # first n_layers -> recv, second n_layers -> send
        # attn_map_with_norm_collect = torch.zeros((len(q_token_pos_ls), n_layers, n_heads)) # softmax(q*k)^2 * ||v||^2
        
        for B in tqdm(range(0, n_prompts, bsz)):
            batch_token = tokenizer(prompt_ls[B:B+bsz], return_tensors="pt", padding=True)
            _, hook_dict = model(input_ids=batch_token["input_ids"], attention_mask=batch_token["attention_mask"])
            attn_v = hook_dict["hook_v"] # shape: [n_prompts_B, n_layers, n_heads, max_n_tokens, head_dim]
            attn_weights = hook_dict["hook_attn_weights"] # shape: [n_prompts_B, n_layers, n_heads, n_tokens, n_tokens] # first n_tokens -> queries, second n_tokens -> keys
            after_res1 = hook_dict["hook_after_res1"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
            after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
            decomposed_attn_out = decompose_attn_out_helper_batch(attn_v, attn_weights, n_layers, model) # shape: [n_prompts_B, n_layers, n_tokens, n_tokens, n_heads, n_dim] first n_tokens -> queries, second n_tokens -> keys
            
            n_prompts_B = attn_weights.shape[0]
            for j in range(len(q_token_pos_ls)): ## NOTE: we only draw S2 -> END, S1 -> END, and IO -> END in the paper
                cur_q_token_pos_ls = token_pos_ls_dict[q_token_pos_ls[j]]
                cur_k_token_pos_ls = token_pos_ls_dict[k_token_pos_ls[j]]
                simplified_decomposed_attn_out = decomposed_attn_out[torch.arange(n_prompts_B), :, cur_q_token_pos_ls[B:B+n_prompts_B], cur_k_token_pos_ls[B:B+n_prompts_B]] # shape: [n_prompts_B, n_layers, n_heads, n_dim]
                simplified_decomposed_attn_out = simplified_decomposed_attn_out.unsqueeze(2).unsqueeze(2)  # shape: [n_prompts_B, n_layers, 1, 1, n_heads, n_dim]
                head_rmsnorm = rmsnorm_breakdown_batch(after_res1[torch.arange(n_prompts_B), :, cur_q_token_pos_ls[B:B+n_prompts_B]].unsqueeze(2), [simplified_decomposed_attn_out], model, mode="H")[0]
                # TODO: head_rmsnorm2 by diff_breakdown_batch
                original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors, after_norm2) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers]
                top_n_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :top_n, :]
                head_score = torch.tril(torch.einsum("RED,PRSQKHD->PQKHERS", router_weight_vectors, head_rmsnorm.to(router_weight_vectors.device)), diagonal=0)

                ## CHECK 1
                # print(head_score.shape, top_n_experts.shape)
                # print("CHECK: before top-n:", head_score[4, 0, 0, 3, :, 3, 2])

                ## if top-n is needed
                ind = top_n_experts[torch.arange(n_prompts_B), cur_q_token_pos_ls[B:B+n_prompts_B]].view(n_prompts_B, 1, 1, 1, top_n, n_layers, 1).expand(n_prompts_B, 1, 1, n_heads, top_n, n_layers, n_layers)
                head_score = torch.gather(head_score, dim=4, index=ind)

                ## CHECK 2
                # print(top_n_experts[4, cur_q_token_pos_ls[B+4], :, 3])
                # print("CHECK: after top-n:", head_score[4, 0, 0, 3, :, 3, 2])
                
                ##
                score_var_collect[j] += head_score[:, 0, 0].var(dim=2).sum(0)
                score_avg_collect[j] += head_score[:, 0, 0].mean(dim=2).sum(0) ## NOTE: not used in the paper
                attn_map_collect[j] += attn_weights[torch.arange(n_prompts_B), :, :, cur_q_token_pos_ls[B:B+n_prompts_B], cur_k_token_pos_ls[B:B+n_prompts_B]].sum(0)
                ## unused
                # attn_map_with_norm_collect[j] += simplified_decomposed_attn_out[torch.arange(n_prompts_B), :, 0, 0, :].norm(dim=-1).sum(0)
        
        score_var_collect = score_var_collect.div(n_prompts)
        score_avg_collect = score_avg_collect.div(n_prompts)
        attn_map_collect = attn_map_collect.div(n_prompts)
        # attn_map_with_norm_collect = attn_map_with_norm_collect.div(n_prompts)
        # return score_var_collect, score_avg_collect, attn_map_collect, attn_map_with_norm_collect
        return score_var_collect, _, attn_map_collect, _
    
    def drawer(data, name): # for avg or var of scores (Ax->My), NOT used in the paper
        map_name = ["END->END", "END->S2", "END->S1+1", "END->S1", "END->IO", "S2->S1+1", "S2->S1", "S2->IO", "S1+1->S1"]
        for i in range(len(map_name)):
            for L in range(n_layers):
                plt.figure(figsize=(12,12))
                to_draw = data[i, :, L, :(L+1)].permute(1,0).detach().cpu().numpy() # data shape: [num_maps, n_heads, n_layers, n_layers] # first n_layers -> recv, second n_layers -> send
                if to_draw.min() < 0:
                    sns.heatmap(to_draw, square=True, annot=True, fmt=".0e", annot_kws={"size":9}, cmap="RdBu", cbar_kws={"shrink": 0.8}, linewidth=1, norm=mcolors.TwoSlopeNorm(vcenter = 0, vmin=to_draw.min(), vmax=to_draw.max()))
                else:
                    sns.heatmap(to_draw, square=True, annot=True, fmt=".0e", annot_kws={"size":9}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, norm=mcolors.LogNorm(vmin=to_draw.min(), vmax=to_draw.max()))
                plt.title("attention_score_batch_" + name + "_" + str(i) + "_" + map_name[i] + "to layer" + str(L))
                plt.savefig(output_dir + "attention_score_batch_" + name + "_" + str(i) + "_" + map_name[i] + "to_L" + str(L) + ".png")
                plt.close("all")

    def drawer2(data): # for var of scores (Figures * b~d), used in the paper
        # 1=S2, 3=S1, 4=IO
        # mats = [data[4, :, 15, :16].permute(1, 0).detach().cpu().numpy(), data[3, :, 15, :16].permute(1, 0).detach().cpu().numpy(), data[1, :, 15, :16].permute(1, 0).detach().cpu().numpy()] # IO, S1, S2 # -> last MoE layer
        mats = [data[4, :, 13:16, 13].permute(1, 0).detach().cpu().numpy(), data[3, :, 13:16, 13].permute(1, 0).detach().cpu().numpy(), data[1, :, 13:16, 13].permute(1, 0).detach().cpu().numpy()] # IO, S1, S2 -> MoE layer in the same block
        fig, axes = plt.subplots(1, 3, figsize=(25, 15), sharex=False, sharey=False)
        g11 = sns.heatmap(mats[0], square=True, annot=True, fmt=".1e", annot_kws={"size":5}, cmap="Oranges", cbar_kws={"shrink": 0.5}, linewidth=1, cbar=False, ax=axes[0])
        g12 = sns.heatmap(mats[1], square=True, annot=True, fmt=".1e", annot_kws={"size":5}, cmap="Oranges", cbar_kws={"shrink": 0.5}, linewidth=1, cbar=False, ax=axes[1])
        g13 = sns.heatmap(mats[2], square=True, annot=True, fmt=".1e", annot_kws={"size":5}, cmap="Oranges", cbar_kws={"shrink": 0.5}, linewidth=1, cbar=False, ax=axes[2])
        im = axes[-1].collections[0]
        fig.colorbar(im, ax=axes, location="right", shrink=0.5)
        np.save(output_dir + "qk_our_score_END_IO" + ".npy", mats[0])
        np.save(output_dir + "qk_our_score_END_S1" + ".npy", mats[1])
        np.save(output_dir + "qk_our_score_END_S2" + ".npy", mats[2])
        plt.savefig(output_dir + "qk_our_score" + ".png", bbox_inches="tight", pad_inches=0.01)
        plt.savefig(output_dir + "qk_our_score" + ".pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close("all")
        if demo_now:
            fig = px.imshow(mats[0], color_continuous_scale="Oranges", title="var IO->END", labels=dict(x="Head", y="Layer"))
            fig.show()
            fig = px.imshow(mats[1], color_continuous_scale="Oranges", title="var S1->END", labels=dict(x="Head", y="Layer"))
            fig.show()
            fig = px.imshow(mats[2], color_continuous_scale="Oranges", title="var S2->END", labels=dict(x="Head", y="Layer"))
            fig.show()
        
    def drawer3(map_collect): # for attn map (Figures * e~g), used in the paper
        # 1=S2, 3=S1, 4=IO
        map_collect = map_collect.detach().cpu().numpy()
        mats = [map_collect[4], map_collect[3], map_collect[1]]
        fig, axes = plt.subplots(1, 3, figsize=(25, 15), sharex=False, sharey=False)
        g11 = sns.heatmap(mats[0], square=True, annot=True, fmt=".1e", annot_kws={"size":5}, cmap="Blues", cbar_kws={"shrink": 0.5}, linewidth=1, cbar=False, ax=axes[0])
        g12 = sns.heatmap(mats[1], square=True, annot=True, fmt=".1e", annot_kws={"size":5}, cmap="Blues", cbar_kws={"shrink": 0.5}, linewidth=1, cbar=False, ax=axes[1])
        g13 = sns.heatmap(mats[2], square=True, annot=True, fmt=".1e", annot_kws={"size":5}, cmap="Blues", cbar_kws={"shrink": 0.5}, linewidth=1, cbar=False, ax=axes[2])
        im = axes[-1].collections[0]
        fig.colorbar(im, ax=axes, location="right", shrink=0.5)
        np.save(output_dir + "attn_map_END_IO" + ".npy", mats[0])
        np.save(output_dir + "attn_map_END_S1" + ".npy", mats[1])
        np.save(output_dir + "attn_map_END_S2" + ".npy", mats[2])
        plt.savefig(output_dir + "attn_map" + ".png", bbox_inches="tight", pad_inches=0.01)
        plt.savefig(output_dir + "attn_map" + ".pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close("all")
        if demo_now:
            fig = px.imshow(mats[0], color_continuous_scale="Blues", title="attn map IO->END", labels=dict(x="Head", y="Layer"))
            fig.show()
            fig = px.imshow(mats[1], color_continuous_scale="Blues", title="attn map S1->END", labels=dict(x="Head", y="Layer"))
            fig.show()
            fig = px.imshow(mats[2], color_continuous_scale="Blues", title="attn map S2->END", labels=dict(x="Head", y="Layer"))
            fig.show()

    def drawer4(map_collect1, map_collect2): # for attn map (compare two maps), not used in the paper
        map_diff = (map_collect1 - map_collect2).permute(1, 2, 0).detach().cpu().numpy()
        for L in range(n_layers):
            plt.figure(figsize=(12,12))
            if map_diff[L].min() < 0:
                ax = sns.heatmap(map_diff[L], square=True, annot=True, fmt=".3f", annot_kws={"size":9}, cmap="RdBu", cbar_kws={"shrink": 0.8}, linewidth=1, norm=mcolors.TwoSlopeNorm(vcenter = 0, vmin=map_diff[L].min(), vmax=map_diff[L].max()))
            else:
                ax = sns.heatmap(map_diff[L], square=True, annot=True, fmt=".3f", annot_kws={"size":9}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, norm=mcolors.Normalize(vmin=0, vmax=map_diff[L].max()))
            ax.set_xticklabels(labels=["END->END", "END->S2", "END->S1+1", "END->S1", "END->IO", "S2->S1+1", "S2->S1", "S2->IO", "S1+1->S1"], rotation=45)

            plt.title("attention_map_batch_layer_" + str(L))
            plt.savefig(output_dir + "attention_map_batch_layer_" + str(L) + ".png")
            plt.close("all")
    
    # score_var_collect_1, score_avg_collect_1, attn_map_collect_1, attn_map_with_norm_collect_1 = helper(prompt_dict_ls_1)
    
    # score_var_collect_2, score_avg_collect_2, attn_map_collect_2, _ = helper(prompt_dict_ls_2)
    score_var_collect_1, _, attn_map_collect_1, _ = helper(prompt_dict_ls_1)

    # drawer(score_avg_collect_1, "avg")
    
    # drawer(score_var_collect_1 - score_var_collect_2, "var_diff")
    # drawer(score_var_collect_1, "var")
    
    drawer2(score_var_collect_1)
    
    drawer3(attn_map_collect_1)
    # drawer4(attn_map_collect_1, attn_map_collect_2)
    # drawer3(attn_map_with_norm_collect_1)

def H_agnostic_matrix_drawer(data, name, output_dir, add_patch=[], title="", xlabel="Head", ylabel="Layer", need_description=False, need_lognorm=False, cbar_label="Variance", data_type="", demo_now=False):
    data = data.detach().cpu().numpy()
    
    vmin, vmax = data.min(), data.max()
    if data_type == "Variance":
        normalize = mcolors.LogNorm(vmin=data[data>0].min(), vmax=vmax) # vmin=1e-4 (for plotting)
        cmap_type = "Greens"
    elif need_lognorm: # vmin >= 0
        normalize = mcolors.LogNorm(vmin=data[data>0].min(), vmax=vmax)
        cmap_type = "Greens"
    elif vmin < 0 and vmax > 0:
        normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap_type = "RdBu"
    elif vmax <= 0:
        normalize = mcolors.Normalize(vmin=vmin, vmax=0)
        cmap_type = "Reds_r"
    else: # vmin >= 0
        normalize = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap_type = "Blues"

    plt.figure(figsize=(13, 13))
    with sns.axes_style("white"):
        if need_lognorm or data_type == "Variance":
            fmt_setting = ".1e"
        else:
            fmt_setting = ".2f"
        ax = sns.heatmap(data, square=True, annot=True, fmt=fmt_setting, cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.5}, linewidth=.5) # annot=False
        for grid in add_patch:
            ax.add_patch(Rectangle((grid[0], grid[1]), 1, 1, fill=False, edgecolor="blue", lw=3))
        cbar = ax.collections[0].colorbar
        # cbar_min, cbar_max = cbar.mappable.get_clim()
        # cbar.set_ticks([vmin, vmax])
        cbar.ax.tick_params(labelsize=40)
        cbar.set_label(cbar_label, fontsize=40)

    plt.xticks([k + 0.5 for k in range(0, data.shape[1], 5)], [str(k) for k in range(0, data.shape[1], 5)])
    plt.yticks([k + 0.5 for k in range(0, data.shape[0], 5)], [str(k) for k in range(0, data.shape[0], 5)], rotation=0)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if need_description:
        plt.title(title)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel(ylabel, fontsize=30)
    np.save(output_dir + name + ".npy", data)
    plt.savefig(output_dir + name + ".png", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(output_dir + name + ".pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close("all")
    if demo_now:
        fig = px.imshow(data, color_continuous_scale=cmap_type, title=name, labels=dict(x=xlabel, y=ylabel))
        fig.update_xaxes(side="top")
        fig.show()

def decompose_H_agnostic(prompt_ls, model, tokenizer, router_weight_ls, output_dir, n_heads, top_n, bsz, max_token_per_prompt, demo_now=False):
    """ Figures 3 """ 
    batch_token = tokenizer(prompt_ls, return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True) # should not use padding in principle
    n_prompts, max_n_tokens = batch_token["attention_mask"].shape

    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, _, _ = router_weight_vectors.shape

    device = "cuda:0"
    head_var_collect = torch.zeros((max_n_tokens, n_layers, n_heads, n_layers), device=device) # first n_layers -> recv_layer , second n_layers -> send_layer
    head_topn_sum_collect = torch.zeros((max_n_tokens, n_layers, n_heads, n_layers), device=device) # first n_layers -> recv_layer , second n_layers -> send_layer
    head_topn_abs_sum_collect = torch.zeros((max_n_tokens, n_layers, n_heads, n_layers), device=device) # first n_layers -> recv_layer , second n_layers -> send_layer
    head_norm = torch.zeros((max_n_tokens, n_layers, n_heads)) # n_layers -> send_layer

    for B in tqdm(range(0, n_prompts, bsz)):
        _, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        after_res1 = hook_dict["hook_after_res1"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        before_matmul_wo = hook_dict["hook_before_matmul_wo"] # shape: [n_prompts_B, n_layers, n_heads, max_n_tokens, n_head_dim]
        
        ## simplified: Ax->Mx (only attend the MoE layer in the same block)
        W_O = torch.stack([model.model.layers[layer_id].self_attn.o_proj.weight for layer_id in range(n_layers)])
        W_O = einops.rearrange(W_O, "n_layers d_model (index d_head)->n_layers index d_head d_model", index=16) # index = n_dim // n_head_dim = n_heads = 16
        decomposed_attn_out = torch.einsum("PSHQA,SHAD->PSQHD", before_matmul_wo, W_O)
        head_rmsnorm = rmsnorm_breakdown_batch(after_res1, [decomposed_attn_out], model, mode="H_agnostic", device=device)[0]
        
        # head_rmsnorm2 = diff_breakdown_batch(after_res1, decomposed_attn_out, model, mode="H_agnostic", variance_epsilon=1e-05, device=device)  # NOTE: decomposition based on difference
        
        ## NOTE: head_rmsnorm shape: [n_prompts_B, n_layers, n_layers, max_n_tokens, n_heads, n_dim] # first n_layers -> recv_layer , second n_layers -> send_layer
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors, after_norm2) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers]
        top_n_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :top_n, :].to(device)
        head_score = torch.tril(torch.einsum("RED,PRSTHD->PTEHRS", router_weight_vectors.to(device), head_rmsnorm), diagonal=0)
        head_score = head_score.permute(0,1,2,4,3,5) # PTERHS
        head_var_collect += head_score.var(dim=2).sum(dim=0)
        head_topn_sum_collect += torch.gather(head_score, dim=2, index=top_n_experts.long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, n_heads, n_layers)).sum(dim=(0, 2))
        head_topn_abs_sum_collect += torch.gather(head_score, dim=2, index=top_n_experts.long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, n_heads, n_layers)).abs().sum(dim=(0, 2))
        head_norm += torch.norm(decomposed_attn_out, p=2, dim=-1).sum(0).permute(1,0,2) # QSH
        ## Code for check if implemented correctly
        # print(head_score.shape, top_n_experts.shape, top_n_experts.long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, n_heads, n_layers).shape)
        # check_head_score = torch.einsum("RED,PRSTHD->PTEHRS", router_weight_vectors.to(device), head_rmsnorm)
        # print("CHECK: score:", check_head_score[0,1,2,3,5,4], torch.dot(router_weight_vectors[5,2].to(device), head_rmsnorm[0,5,4,1,3]))
    
    ## variance
    H_agnostic_matrix_drawer(head_var_collect[1:].mean(0).div(n_prompts)[torch.arange(n_layers, device=head_var_collect.device), :, torch.arange(n_layers, device=head_var_collect.device)], name="H_agnostic_var_without_T0", output_dir=output_dir, need_lognorm=False, cbar_label="Variance", data_type="Variance", demo_now=demo_now, need_description=True)
    H_agnostic_matrix_drawer(head_var_collect[0, torch.arange(n_layers, device=head_var_collect.device), :, torch.arange(n_layers, device=head_var_collect.device)].div(n_prompts), name="H_agnostic_var_T0", output_dir=output_dir, need_lognorm=False, cbar_label="Variance", data_type="Variance", demo_now=False, need_description=True)
    
    ## avg positive, avg negative of top n, and norm
    H_agnostic_matrix_drawer((head_topn_sum_collect[1:] + head_topn_abs_sum_collect[1:]).mean(0).div(2 * n_prompts * top_n)[torch.arange(n_layers,device=head_var_collect.device),:, torch.arange(n_layers,device=head_var_collect.device)], name="H_agnostic_positive_without_T0", output_dir=output_dir, cbar_label="APS", demo_now=demo_now, need_description=True)
    H_agnostic_matrix_drawer((head_topn_sum_collect[1:] - head_topn_abs_sum_collect[1:]).mean(0).div(2 * n_prompts * top_n)[torch.arange(n_layers,device=head_var_collect.device),:, torch.arange(n_layers,device=head_var_collect.device)], name="H_agnostic_negative_without_T0", output_dir=output_dir, cbar_label="ANS", demo_now=demo_now, need_description=True)
    H_agnostic_matrix_drawer(head_norm[1:].mean(0).div(n_prompts), name="H_agnostic_norm", output_dir=output_dir, demo_now=False, need_description=True, cbar_label="L2-norm")
    
    ## compare avg positive and avg negative
    avg_pos = (head_topn_sum_collect[1:] + head_topn_abs_sum_collect[1:]).mean(0).div(2 * n_prompts * top_n)[torch.arange(n_layers,device=head_var_collect.device),:, torch.arange(n_layers,device=head_var_collect.device)]
    avg_neg = (head_topn_sum_collect[1:] - head_topn_abs_sum_collect[1:]).mean(0).div(2 * n_prompts * top_n)[torch.arange(n_layers,device=head_var_collect.device),:, torch.arange(n_layers,device=head_var_collect.device)]
    compare_pos_neg = avg_pos - avg_neg.abs()
    compare_pos_neg = compare_pos_neg.detach().cpu().numpy()
    if demo_now:
        fig = px.imshow(compare_pos_neg, color_continuous_scale="RdBu", color_continuous_midpoint=0, title="pos-abs(neg)", labels=dict(x="Head", y="Layer"))
        fig.update_xaxes(side="top")
        fig.show()    
    return

def decompose_E(prompt_ls, model, tokenizer, router_weight_ls, output_dir, top_k, bsz, max_token_per_prompt, model_id, demo_now=False):
    """ Figures 5 """
    batch_token = tokenizer(prompt_ls, return_tensors="pt", max_length=max_token_per_prompt, padding=False, truncation=True) # should not use padding in principle
    n_prompts, _ = batch_token["attention_mask"].shape

    router_weight_vectors = torch.stack(router_weight_ls, dim=0) # shape: [n_layers, n_experts, n_dim]
    n_layers, n_experts, _ = router_weight_vectors.shape

    occur_counter = torch.zeros((n_layers, n_experts)) # count how many times the experts are selected
    score_variance = torch.zeros((n_layers, n_experts, n_layers)) # first n_layers -> send_layer, second n_experts -> recv_layer
    norm_recorder = torch.zeros((n_layers, n_experts)) # norm of weighted expert output
    norm_projected_recorder = torch.zeros((n_experts)) # only for experiment: M1->M2
    projected_variance_recorder = torch.zeros((n_experts)) # only for experiment: M1->M2
    norm_projected_recorder_counter = torch.zeros((n_experts)) # only for experiment: M1->M2
    top_k_change = torch.zeros((n_layers, n_experts, n_layers)) # first n_layers -> send_layer, second n_experts -> recv_layer
    top_k_change_counter =  torch.zeros((n_layers, n_experts, n_layers)) # first n_layers -> send_layer, second n_experts -> recv_layer
    top_64_change = torch.zeros((n_layers, n_experts, n_layers)) # OLMoE has 64 experts each layer, first n_layers -> send_layer, second n_experts -> recv_layer

    def expert_counter(mat):
        mat_tmp = mat.permute(1,0) + n_experts * torch.arange(n_layers).unsqueeze(1) # give each expert a temporary id: cur_layer * n_experts + old_expert_id
        counter = torch.bincount(mat_tmp.flatten(), minlength=(n_experts * n_layers)).reshape(n_layers, n_experts)
        # check, take OLMoE as an example
        # print("CHECK:", mat.permute(1,0)[3,:3])
        # print("CHECK:", (mat.permute(1,0) + n_experts * torch.arange(n_layers).unsqueeze(1))[3, :3])
        # print("CHECK:", (mat_tmp == 128).sum(), counter[2,0]) # should be equal
        return counter

    def add_by_index_map(send_expert_id, vars, score_variance_mat):
        """ sum up the variance """
        n_all_tokens, n_send_layers, n_recv_layers = vars.shape
        
        i_idx = torch.arange(n_send_layers, device=send_expert_id.device).unsqueeze(0).expand(n_all_tokens, n_send_layers) # shape: [n_all_tokens, n_send_layers]
        j_idx = send_expert_id  # shape: [n_all_tokens, n_send_layers]

        i_idx = i_idx.reshape(-1).to(score_variance_mat.device) # sending layer id
        j_idx = j_idx.reshape(-1).to(score_variance_mat.device) # sending expert id
        vars_mat = vars.reshape(-1, n_recv_layers).to(score_variance_mat.device)

        score_variance_mat.index_put_((i_idx, j_idx), vars_mat, accumulate=True)
        # return score_variance_mat

    def norm_add_by_index_map(send_expert_id, norms, norm_mat):
        # shape: send_expert_id: [P * T * K, S]; norms: [P, S, T, K]; norm_mat [S, E]
        norm_mat.scatter_add_(1, send_expert_id.permute(1,0), norms.permute(0,2,3,1).reshape(-1, n_layers).permute(1,0))

        ## for check
        # print(send_expert_id.permute(1,0).shape, norms.permute(0,2,3,1).reshape(-1, n_layers).permute(1,0).shape, norm_mat.shape)
        # print(send_expert_id.shape, norms.shape)
        # tmp_ids = torch.where(send_expert_id[:,0] == 63)[0]
        # print(norms.permute(0,2,3,1).reshape(-1, n_layers)[tmp_ids,0].sum())
        # print(norm_mat[0, 63])
        
    for B in tqdm(range(0, n_prompts, bsz)):
        _, hook_dict = model(input_ids=batch_token["input_ids"][B:B+bsz], attention_mask=batch_token["attention_mask"][B:B+bsz])
        after_res1 = hook_dict["hook_after_res1"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        after_norm2 = hook_dict["hook_after_norm2"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        expert_weighted_outputs = hook_dict["hook_expert_weighted_outputs"] # shape: [n_prompts_B, n_layers, max_n_tokens, original_top_k, n_dim]
        
        ffn_out_rmsnorm = rmsnorm_breakdown_batch(after_res1, [expert_weighted_outputs], model, mode="E")[0]
        ffn_out_rmsnorm2 = diff_breakdown_batch(after_res1, expert_weighted_outputs, model, mode="E")
        ## check if rmsnorm_breakdown_batch(mode="E") is implemented correctly
        # mlp_output = hook_dict["hook_mlp_output"] # shape: [n_prompts_B, n_layers, max_n_tokens, n_dim]
        # print(mlp_output[0,1,2,:3], expert_weighted_outputs.sum(3)[0,1,2,:3]) # should be equal
        # tmp_rmsnorm = rmsnorm_breakdown_batch(after_res1, [mlp_output], model, mode="TAM")[0] # PRSTD
        # print(ffn_out_rmsnorm[0,2,1,2,:].sum(0)[:3], tmp_rmsnorm[0,2,1,2,:3]) # should be equal; ffn_out_rmsnorm: PRSTED
        # exit()

        ## NOTE: ffn_out_rmsnorm shape: [n_prompts_B, n_layers, n_layers, max_n_tokens, original_top_k, n_dim] # first n_layers -> recv_layer , second n_layers -> send_layer
        original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors.float(), after_norm2.float()) # shape: [n_prompts_B, max_n_tokens, n_experts, n_layers] # .float() for qwen
        original_top_k_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :top_k, :]
        # NOTE: diagonal=-1; we now use K instead of E to denote the dimension in PRSTED.
        ffn_out_score = torch.tril(torch.einsum("RED,PRSTKD->PTEKRS", router_weight_vectors.to(ffn_out_rmsnorm.device).float(), ffn_out_rmsnorm.float()), diagonal=-1) # K = original_top_k, i.e., selected experts of send_L # .float() for qwen
        send_expert_id = original_top_k_experts.reshape(-1, n_layers)
        # print(len(torch.nonzero(send_expert_id[:,1]==2)) + occur_counter[1,2]) ## for check
        occur_counter += expert_counter(send_expert_id) # assume that no padding tokens
        # print(occur_counter[1,2]) ## for check
        
        tmp_vars = ffn_out_score.reshape(-1, n_experts, top_k, n_layers, n_layers).var(dim=1).reshape(-1, n_layers, n_layers).permute(0, 2, 1) # after permutation: -1, send_layers, recv_layers
        add_by_index_map(send_expert_id, tmp_vars, score_variance) # this is an in-place operation

        ## NOTE: check if add_by_index_map implemented correctly
        # inds = torch.nonzero(original_top_k_experts[:,:,:,1]==2)
        # tmp1 = ffn_out_score[inds[:,0],inds[:,1],:,inds[:,2], 3, 1] # recv=3, send=1E2
        # print(tmp1.var(dim=1).sum(0)) # recv=3, send=M1E2
        # print(score_variance[1,2,3])
        # exit()
        
        ## NOTE: checking the norm of high-influence
        tmp_norm = torch.norm(expert_weighted_outputs, p=2, dim=-1)
        norm_add_by_index_map(send_expert_id, tmp_norm, norm_recorder)
        n_prompts_B = expert_weighted_outputs.shape[0]
        ## FIXME: (1) may remove this (just for check the tokens), only for OLMoE
        # for i in range(n_prompts_B):
        #     for k in range(max_token_per_prompt):
        #         ## check the tokens related to the specific experts (as long as the experts are selected)
        #         # if 9 in original_top_k_experts[i, k, :, 1]:
        #         #     print(tokenizer.decode(batch_token["input_ids"][B + i, k]))
        #         # if 30 in original_top_k_experts[i, k, :, 2]:
        #         #     print(tokenizer.decode(batch_token["input_ids"][B + i, k]))
        #         # if 18 in original_top_k_experts[i, k, :, 1]:
        #         #     print(tokenizer.decode(batch_token["input_ids"][B + i, k]))
        #         for m in range(top_k):
        #             if 9 == original_top_k_experts[i,k,m,1]:
        #                 print(9, m, tokenizer.decode(batch_token["input_ids"][B+i,k]), torch.norm(expert_weighted_outputs[i,1,k,:],dim=-1), original_top_k_experts[i,k,:,1])
        #             # if 27 == original_top_k_experts[i,k,m,1]:
        #             #     print(27, m, tokenizer.decode(batch_token["input_ids"][B+i,k]), torch.norm(expert_weighted_outputs[i,1,:,m]))
        #             # elif 9 == original_top_k_experts[i,k,m,1]:
        #             #     print(9, m, tokenizer.decode(batch_token["input_ids"][B+i,k]), torch.norm(expert_weighted_outputs[i,1,k,m]))
        #             # elif 18 == original_top_k_experts[i,k,m,1]:
        #             #     print(18, m, tokenizer.decode(batch_token["input_ids"][B+i,k]), torch.norm(expert_weighted_outputs[i,1,k,m]))
        
        ## FIXME: (2) temporary addition, only for OLMoE
        top_64_experts = torch.argsort(original_score, dim=2, descending=True)[:, :, :64, :]
        for s in [1]:#range(16): # sending layer
            for r in [2]:#range(s+1, 16): # receiving layer
                for s_e in range(64): # experts in sending layer
                    cur_send_expert_indices = torch.nonzero(top_64_experts[:, :, :8, s] == s_e)
                    if len(cur_send_expert_indices) == 0: # this expert is not selected in the sending layer
                        continue
                    tmp_score = original_score.clone() # shape: PTER

                    for p, t, s_e_pos in cur_send_expert_indices:
                        tmp_score[p, t, :, r] -= ffn_out_score[p, t, :, s_e_pos, r, s] # 对接收层中的所有专家移去send_expert的贡献
                        norm_projected_recorder[s_e] += torch.norm(ffn_out_rmsnorm[p, r, s, t, s_e_pos], p=2, dim=-1)
                        projected_variance_recorder[s_e] += torch.matmul(router_weight_vectors[r, :], ffn_out_rmsnorm[p, r, s, t, s_e_pos]).var()
                        norm_projected_recorder_counter[s_e] += 1
                        tmp_score_top = torch.argsort(tmp_score[p, t, :, r], descending=True)
                        original_score_top = top_64_experts[p, t, :, r]
                        pos_in_b = torch.empty(64, dtype=torch.long)
                        pos_in_b[tmp_score_top] = torch.arange(64) # pos_in_b[x] denotes the rank of Expert x after the perturbation
                        pos_in_a = torch.arange(64)
                        diff = torch.abs(pos_in_a - pos_in_b[original_score_top]) # the rank shift caused by the perturbations
                        top_k_change[s, s_e, r] += diff[:8].sum()
                        top_64_change[s, s_e, r] += diff.sum()
                        # print(tmp_score_top)
                        # print(original_score_top)
                        # print(diff)
                        # print("\n\n")

        ## CHECK
        # print(torch.nonzero(send_expert_id[:, 0] == 0))
        # tmp_ind = torch.nonzero(send_expert_id[:, 0] == 0)
        # print(tmp_vars[tmp_ind, 0, 1].sum())
        # print(score_variance[0, 0, 1])
        
    ## CHECK    
    # print(score_variance[1, 3, 2], projected_variance_recorder[3]) # for check, M1E3->M2, should be equal
    # print(occur_counter[1, 3], score_variance[1, 3, 2] / occur_counter[1, 3])
    
    norm_recorder /= occur_counter
    score_variance /= occur_counter.reshape(n_layers, n_experts, 1).repeat(1,1,n_layers)
    
    ## CHECK
    # print(occur_counter[1, 3], norm_projected_recorder_counter[3]) # for check, should be equal 
    # print(top_k_change[1, :, 3], norm_projected_recorder_counter)
    
    def scatter_drawer(data, name):
        n_following_layers = data.shape[1]
        data = data.detach().cpu().numpy()
        plt.figure(figsize=(13, 13))
        for i in range(n_experts):
            for j in range(n_following_layers):
                plt.text(j, data[i, j], str(i), fontdict={"weight":"bold", "size":9})
        plt.xlim(0, n_layers)
        plt.ylim(-0.05, 2)
        plt.ylabel("Score variance")
        plt.grid()
        plt.title("decompose_E_" + name)
        plt.savefig(output_dir + "decompose_E_" + name + ".png")
        plt.close("all")

    ## just for finding experts with high variance
    # print(torch.sort(score_variance[0, :, 1]))
    # print(torch.sort(score_variance[1, :, 2]))
    # print(torch.sort(score_variance[2, :, 3]))
    # print(torch.sort(score_variance[3, :, 4]))
    # print(torch.sort(score_variance[2, :, 4]))
    # for i in range(n_experts):
    #     print(i,score_variance[1, i, :])

    # print(score_variance[1,9,2:])
    # print(score_variance[2,30,3:])
    # print(score_variance[4,14,5:])
    # print(score_variance[1,68,2:])

    ## plot the distribution of score variances
    for k in range(n_layers - 1):
        scatter_drawer(score_variance[k, :, k+1:], "score_variance_send_layer" + str(k))
    
    def olmoe_scatter_drawer(): # TODO: check this function
        data = score_variance.detach().cpu().numpy()
        plt.figure(figsize=(13, 13))
        others_x =[]
        others_y =[]
        for send_L in range(n_layers):
            for E in range(n_experts):
                if [send_L, E] not in [[1, 9], [2, 30], [4, 14]]:
                    others_x.extend([L for L in range(send_L+1, n_layers)])
                    others_y.extend(data[send_L,E,send_L+1:])
                elif [send_L, E] == [1, 9]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="r", marker="^", label="M1E9")
                elif [send_L, E] == [2, 30]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="g", marker="s", label="M2E30")
                elif [send_L, E] == [4, 14]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="b", marker="p", label="M4E14")
        plt.scatter(others_x, others_y, s=100, alpha=0.5, c="black", label="Other Experts")

        plt.grid()
        plt.legend(markerscale=2, fontsize=30, loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.title("decompose_E_" + name)
        print('CHECK M1->M5 score variance\n', data[1,:,5])
        np.save(output_dir  + "olmoe_scatter" + ".npy", data)
        plt.savefig(output_dir + "olmoe_scatter" + ".png", bbox_inches="tight", pad_inches=0.01)
        plt.savefig(output_dir + "olmoe_scatter" + ".pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close("all")
        if demo_now:
            x = [k for i in range(0, n_layers) for j in range(0, n_experts) for k in range(i + 1, n_layers)]
            y = [data[i,j,k] for i in range(0, n_layers) for j in range(0, n_experts) for k in range(i + 1, n_layers)]
            color = ["M{}E{}".format(i, j) for i in range(0, n_layers) for j in range(0, n_experts) for _ in range(i + 1, n_layers)]
            fig = px.scatter(x=x, y=y, color=color, title="varaince of scores (MxEy->Mz), OLMoE", labels=dict(x="Layers", y="Score Variance"))
            fig.show()

    def qwen_scatter_drawer(): # TODO: check this function
        data = score_variance.detach().cpu().numpy()
        plt.figure(figsize=(13,13))
        special_experts = [[1,68],[2,92],[3,82],[21,69],[24,111],[31,56],[33,69]]
        others_x =[]
        others_y =[]
        for send_L in range(n_layers):
            # if send_L not in [1, 2, 4]:
            #     others_x.extend([L for e in range(n_experts) for L in range(send_L+1, n_layers)])
            #     others_y.extend(data[send_L,:,send_L+1:].reshape(-1))
            # else:
            for E in range(n_experts):
                if [send_L, E] not in [[1,68],[2,92],[3,82],[21,69],[22,92], [24,111],[31,56],[33,69]]:
                    others_x.extend([L for L in range(send_L+1, n_layers)])
                    others_y.extend(data[send_L,E,send_L+1:])
                elif [send_L, E] == [1, 68]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="y", marker="^", label="M1E68")
                elif [send_L, E] == [2, 92]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="m", marker="s", label="M2E92")
                elif [send_L, E] == [3, 82]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="c", marker="p", label="M3E82")
                elif [send_L, E] == [21, 69]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="orange", marker="H", label="M21E69")
                elif [send_L, E] == [22, 92]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="yellowgreen", marker="8", label="M22E92")
                elif [send_L, E] == [24, 111]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="indigo", marker="*", label="M24E111")
                elif [send_L, E] == [31, 56]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="lightcoral", marker="P", label="M31E56")
                elif [send_L, E] == [33, 69]:
                    plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=150, c="peru", marker="X", label="M33E69")
        plt.scatter(others_x, others_y, s=100, alpha=0.5, c="black", label="Other Experts")

        plt.grid()
        # ax = plt.gca()
        # circle = plt.Circle((2, 0.2652), 0.05, color="r", fill=False, linewidth=2)
        # ax.add_patch(circle)
        plt.scatter(2, data[1, 68, 2], s=1000, facecolors="none", edgecolors="r", linewidths=4)
        plt.legend(markerscale=2, fontsize=30, loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.title("decompose_E_" + name)
        np.save(output_dir + "qwen_scatter" + ".npy", data)
        plt.savefig(output_dir + "qwen_scatter" + ".png", bbox_inches="tight", pad_inches=0.01)
        plt.savefig(output_dir + "qwen_scatter" + ".pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close("all")

    def mixtral_scatter_drawer(): # TODO: check this function
        data = score_variance.detach().cpu().numpy()
        others_x =[]
        others_y =[]
        plt.figure(figsize=(13,13))
        for send_L in range(n_layers):
            for E in range(n_experts):
                if [send_L, E] not in [[1,3]]:
                    others_x.extend([L for L in range(send_L+1, n_layers)])
                    others_y.extend(data[send_L,E,send_L+1:])
                # elif [send_L, E] == [1, 3]:
                #     plt.scatter([L for L in range(send_L+1, n_layers)], data[send_L, E, send_L+1:], s=250, c="y", marker="^", label="M1E3")
        plt.scatter(others_x, others_y, s=100, alpha=0.5, c="black", label="Other Experts")
        plt.scatter([L for L in range(2, n_layers)], data[1, 3, 2:], s=250, c="y", marker="^", label="M1E3")
        plt.scatter(n_layers - 1, data[1, 3, n_layers - 1], s=1000, facecolors="none", edgecolors="r", linewidths=4)
        plt.grid()
        plt.legend(markerscale=2, fontsize=30, loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("Receiving Layer", fontsize=30)
        plt.ylabel("Variance", fontsize=30)
        # plt.title("decompose_E_" + name)
        np.save(output_dir + "mixtral_scatter" + ".npy", data) # score variance
        plt.savefig(output_dir + "mixtral_scatter" + ".png", bbox_inches="tight", pad_inches=0.01)
        plt.savefig(output_dir + "mixtral_scatter" + ".pdf", bbox_inches="tight", pad_inches=0.01)
        plt.close("all")

    def plot_var_and_topK(norm, norm_proj, topk, topn, var, time, var_proj): # TODO: check this function
        norm_proj = norm_proj.detach().cpu().numpy()
        topk = topk.detach().cpu().numpy()
        topn = topn.detach().cpu().numpy()
        var = var.detach().cpu().numpy()
        time = time.detach().cpu().numpy()
        var_proj = var_proj.detach().cpu().numpy()
        print(norm_proj)
        norm_proj /= time
        topk /= time
        topn /= time
        for i in range(64):
            print(i, norm[i].round(2), norm_proj[i].round(2), topk[i].round(2), topn[i].round(2), var[i].round(2))
        print(norm.shape, topk.shape, topn.shape, var.shape)
        norm_proj_sortarg = np.argsort(norm_proj)
        xs = [i for i in range(64)]
        plt.scatter(xs, norm[norm_proj_sortarg]*5, color='r', label='norm(×5)')
        plt.scatter(xs, norm_proj[norm_proj_sortarg], color='yellow', label='norm_proj')
        plt.scatter(xs, topk[norm_proj_sortarg], color='g', label='topk experts change rate')
        plt.scatter(xs, topn[norm_proj_sortarg]/100, color='b', label='all experts change rate (×1/100)')
        plt.scatter(xs, var[norm_proj_sortarg]*100, color='k', label='var(×100)')
        plt.legend()
        plt.savefig(output_dir + 'var_and_top_k_.png', dpi=300, bbox_inches='tight')
        # plt.savefig(output_dir + 'var_and_top_k_.pdf', dpi=300, bbox_inches='tight')
        var_proj /= time
        print(var)
        print(var_proj)
        print("norm", "norm_proj", "topk", " all", "var_proj", " freq")
        print(np.corrcoef(np.array([norm, norm_proj, topk, topn, var, time]), rowvar=True).round(3))
        exit()
        print(np.corrcoef(np.array([norm, norm_proj, topk, topn, var, var_proj, time]), rowvar=True).round(2))
    
    if "OLMoE" in model_id:
        olmoe_scatter_drawer()
        norm_recorder = norm_recorder.detach().cpu().numpy()
        x = [i for i in range(0, n_layers) for j in range(0, n_experts)]
        y = [norm_recorder[i, j] for i in range(0, n_layers) for j in range(0, n_experts)]

        print('CHECK score variance M1->M2 \n', score_variance[1, :, 2][:3], (projected_variance_recorder/norm_projected_recorder_counter)[:3]) # should be the same as above, otherwise please check the code!
        
        np.save(output_dir + "olmoe_norm" + ".npy", norm_recorder)
        np.save(output_dir + "top_k_change_M1_to_M2" + ".npy", top_k_change[1, :, 5].detach().cpu().numpy())
        np.save(output_dir + "top_64_change_M1_to_M2" + ".npy", top_64_change[1, :, 5].detach().cpu().numpy())
        np.save(output_dir + "norm_projected_recorder" + ".npy", norm_projected_recorder.detach().cpu().numpy())
        np.save(output_dir + "norm_projected_recorder_counter" + ".npy", norm_projected_recorder_counter.detach().cpu().numpy())
        np.save(output_dir + "projected_variance_recorder" + ".npy", projected_variance_recorder.detach().cpu().numpy())
        
        plot_var_and_topK(norm_recorder[1], norm_projected_recorder, top_k_change[1, :, 10], top_64_change[1, :, 10], score_variance[1, :, 10], norm_projected_recorder_counter, projected_variance_recorder)
        plt.scatter(x, y, s=5)
        plt.title("L2-norm, OLMoE")
        plt.xlabel("Receiving Layer", fontsize=30)
        plt.ylabel("L2-Norm", fontsize=30)
        if demo_now:
            x = [i for i in range(0, n_layers) for j in range(0, n_experts)]
            y = [norm_recorder[i,j] for i in range(0, n_layers) for j in range(0, n_experts)]
            color = ["x{}y{}".format(i, j) for i in range(0, n_layers) for j in range(0, n_experts)]
            fig = px.scatter(x=x, y=y, color=color, title="L2-norm, OLMoE", labels=dict(x="Layers", y="L2-norm"))
            fig.show()
    elif "Mixtral" in model_id:
        mixtral_scatter_drawer()
    elif "Qwen" in model_id:
        qwen_scatter_drawer()
    
    return