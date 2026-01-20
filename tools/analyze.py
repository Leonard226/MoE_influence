import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
from tqdm import tqdm

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

def tril_drawer_tam(data, name, output_dir, figsize=(11,11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="", need_lognorm=False, need_description=True, tick_mode=None, cbar_label=None, need_no_annotations=True, demo_now=False):
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
        plt.savefig(output_dir + "_" + name + "no_annotations"+".pdf", bbox_inches="tight", pad_inches=0)
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
    # else: # unused, unchecked
    #     breakdowns = [weight * (i * rsqrt) for i in components]
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
