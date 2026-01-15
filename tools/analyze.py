import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px

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
    cbar.set_ticklabels(["-20%", "0%", "20%", "40%"])
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