import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import torch

def tril_drawer_TAM(data, name, output_dir, figsize=(11,11), diagonal=1, add_patch=[], title="", xlabel="", ylabel="", is_variance=False, need_description=True, tick_mode=None, cbar_label=None):
    """ lower triangular matrix. for moe->moe, diagnoal=0; for attn->moe or token->moe, diagonal=1 """
    data = data.detach().cpu().numpy()
    
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask, k=diagonal)] = True
    
    vmin, vmax = data.min(), data.max()
    if is_variance:
        normalize = mcolors.LogNorm(vmin=data[data>0].min(), vmax=vmax) # avoid zeros TODO: this code gives a warning, check it
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
        ## deepseek/mixtral annot size:7; qwen annot size: 5
        if is_variance:
            fmt_setting = ".1e"
        else:
            fmt_setting = ".2f"
        ax = sns.heatmap(data, mask=mask, square=True, annot=need_description, annot_kws={"size": 7}, fmt=fmt_setting, cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.5}, linewidth=.5)
        for grid in add_patch:
            ax.add_patch(Rectangle((grid[0], grid[1]), 1, 1, fill=False, edgecolor="blue", lw=3))
        # cbar = ax.collections[0].colorbar
        # cbar_min, cbar_max = cbar.mappable.get_clim()
        # cbar.set_ticks([vmin, vmax])
        # cbar.ax.tick_params(labelsize=40)
        # cbar.set_label(cbar_label, fontsize=40)
        
    if need_description:
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        axis_stride = 5
        if tick_mode == "T":
            plt.xticks([0], [""])
            plt.yticks([k + 0.5 for k in range(0, data.shape[0], axis_stride)], [str(k) for k in range(0, data.shape[0], axis_stride)])
        elif tick_mode == "A":
            plt.xticks([k + 0.5 for k in range(0, data.shape[1], axis_stride)], [str(k) for k in range(0, data.shape[1], axis_stride)])
            plt.yticks([k + 0.5 for k in range(0, data.shape[0], axis_stride)], [str(k) for k in range(0, data.shape[0], axis_stride)])
        elif tick_mode == "M":
            plt.xticks([k + 0.5 for k in range(0, data.shape[1], axis_stride)], [str(k) for k in range(0, data.shape[1], axis_stride)])
            plt.yticks([k + 0.5 for k in range(0, data.shape[0], axis_stride)], [str(k) for k in range(0, data.shape[0], axis_stride)])
    
    np.save(output_dir + name + ".npy", data)
    plt.savefig(output_dir + name + ".png", bbox_inches="tight", pad_inches=0) # .pdf
    plt.close("all")

def matrix_drawer_H_token_head(data, name, output_dir, figsize=(13,13), add_patch=[], token_ls=[], title="", xlabel="Head", ylabel="Token", need_description=True):
    """ (Mode 1, 7) matrix. x: Head, y: Token """
    data = data.detach().cpu().numpy()
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

    y_label_ls = [str(T) + "| " + token_ls[T] for T in range(len(token_ls))]

    with sns.axes_style("white"):
        ax = sns.heatmap(data, square=True, annot=True, fmt=".2f", cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.6}, linewidth=.5)
        ax.set_yticklabels(labels=y_label_ls, rotation=0)
        for grid in add_patch:
            ax.add_patch(Rectangle((grid[0], grid[1]), 1, 1, fill=False, edgecolor="black", lw=3))
    if need_description:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png", bbox_inches="tight", pad_inches=0) # ".pdf"
    plt.close("all")

def scatter_drawer_H_expert(data, submode, name, output_dir, title=""):
    """ (Mode 2) scatter plot. x: Expert, y: Score """
    n_experts, n_heads = data.shape
    data = data.detach().cpu().numpy()

    if submode == 1:
        xs = np.arange(n_experts)
        for H in range(n_heads):
            plt.scatter(xs, data[:, H], s=5, alpha=0.5, label=str(H))
    elif submode == 2:
        color_ls = ["k","grey","r","sandybrown","orange","gold","yellowgreen","lawngreen","g","aquamarine","cyan","dodgerblue","b","indigo","violet","pink"] # 16 heads
        for H in range(n_heads):
            for E in range(n_experts):
                plt.text(E, data[E, H], str(H), c=color_ls[H], fontdict={"weight":"bold", "size":9})
        plt.ylim(data.min() - 0.05, data.max() + 0.05)
        plt.xlim(0, n_experts)
    
    plt.grid(axis="y")
    plt.title(title)
    plt.xlabel("Expert")
    plt.ylabel("Score")
    plt.savefig(output_dir + name + ".png") # ".pdf"
    plt.close("all")

def scatter_drawer_H_head(data, name, output_dir, title=""):
    """ (Mode 3, 4) scatter plot. x: Head, y: Score """
    n_experts, n_heads = data.shape
    data = data.detach().cpu().numpy()
    x_ls = [H for H in range(n_heads) for _ in range(n_experts)]
    y_ls = [data[E, H] for H in range(n_heads) for E in range(n_experts)]
    plt.scatter(x_ls, y_ls, s=5, alpha=0.5)
    plt.ylim(data.min() - 0.05, data.max() + 0.05)
    plt.xlim(0, n_heads)
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Score")
    plt.xticks([H for H in range(n_heads)], [str(H) for H in range(n_heads)])
    plt.grid()
    plt.savefig(output_dir + name + ".png") # ".pdf"
    plt.close("all")

def matrix_drawer_H_with_sum(data, name, output_dir, figsize=(13,13), token_ls=None, title="", xlabel="", ylabel="", need_description=True):
    """ (Mode 5, 6) matrix. x: Token (Mode 5)/ Head (Mode 6) y: Selected Expert """
    
    top_n = data.shape[0]
    net_token_score = torch.sum(data, dim=0).unsqueeze(0)
    data = torch.cat((data, net_token_score), dim=0)
    data = data.detach().cpu().numpy()
    plt.figure(figsize=figsize)

    vmin, vmax = data[:top_n].min(), data[:top_n].max()
    if vmin < 0 and vmax > 0:
        normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap_type = "RdBu"
    elif vmax <= 0:
        normalize = mcolors.Normalize(vmin=vmin, vmax=0)
        cmap_type = "Reds_r"
    else: # vmin >= 0
        normalize = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap_type = "Blues"

    y_ls = [i for i in range(top_n)] + ["Sum"]
    if token_ls:
        x_label_ls = [str(T) + "| " + token_ls[T] for T in range(len(token_ls))]
    else:
        x_label_ls = np.arange(data.shape[1])
    
    with sns.axes_style("white"):
        ax = sns.heatmap(data, square=True, annot=True, fmt=".2f", cmap=cmap_type, norm=normalize, cbar_kws={"shrink": 0.8}, annot_kws={"color":"black"}, linewidth=.5)
        ax.add_patch(Rectangle((0, top_n), data.shape[1], 1, fill=True, edgecolor="blue", facecolor="lightgrey", lw=3))
        ax.set_xticklabels(labels=x_label_ls, rotation=0)
        ax.set_yticklabels(y_ls, rotation=0)
    if need_description:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png") # ".pdf"
    plt.close("all")