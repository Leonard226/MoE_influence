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