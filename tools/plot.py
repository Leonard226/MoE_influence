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

def decompose_H_comparison_batch_single_drawer(head_score, tokens_ls, n_layers, n_heads, output_dir):
    """ plot the scores assigned by output of (L, H, Q, K) to experts in another MoE layer. One prompt. head score shape: PQKHERS """
    for send_L in range(0, n_layers):
        for H in range(0, n_heads):
            for recv_L in range(send_L, n_layers):
                if send_L != 1 or recv_L != 3 or H != 4:
                    continue
                
                ## NOTE: Selected expert only version is unchecked and unused.
                # original_score = torch.einsum("RED,PRTD->PTER", router_weight_vectors, after_norm2) # [n_prompts, max_n_tokens, n_experts, n_layers]
                # original_top_k_experts = torch.argsort(original_score, dim=2, descending=True)[:,:,:top_k, :]
                # score1 = head_score[0, torch.arange(len(tokens_ls)).reshape(-1, 1).repeat(1, 8), :, H, original_top_k_experts[0, :, :, recv_L], recv_L, send_L] # selected experts only
                # pos1 = (score1 + torch.abs(score1)).div(2).sum(1)
                # neg1 = (score1 - torch.abs(score1)).div(2).sum(1)

                score1 = head_score[0, :, :, H, :, recv_L, send_L] # all experts
                pos1 = (score1 + torch.abs(score1)).div(2).sum(-1)
                neg1 = (score1 - torch.abs(score1)).div(2).sum(-1)
                fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                fig.suptitle("(All experts) attn_score_sendL{}H{}recvL{}_ioi".format(send_L, H, recv_L))
                # fig.suptitle("(Selected experts) attn_score_sendL{}H{}recvL{}_ioi".format(send_L, H, recv_L))

                mask = np.zeros_like(pos1.detach().cpu().numpy())
                mask[np.triu_indices_from(mask, k=1)] = True
                g1 = sns.heatmap(pos1.detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":11}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=ax1)
                g2 = sns.heatmap(neg1.detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":11}, cmap="Reds_r", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=ax2)
                for ax in [g1, g2]:
                    ax.add_patch(Rectangle((0, 4), 5, 1, fill=False, edgecolor="red", lw=3))
                    ax.add_patch(Rectangle((0, 9), 10, 1, fill=False, edgecolor="red", lw=3))
                    ax.add_patch(Rectangle((0, 13), 14, 1, fill=False, edgecolor="red", lw=3))
                    ax.add_patch(Rectangle((1, 4), 1, 1, fill=False, edgecolor="black", lw=5))
                    ax.add_patch(Rectangle((3, 4), 1, 1, fill=False, edgecolor="black", lw=5))
                    for k in [9, 13]:
                        ax.add_patch(Rectangle((1, k), 1, 1, fill=False, edgecolor="black", lw=5))
                        ax.add_patch(Rectangle((3, k), 1, 1, fill=False, edgecolor="black", lw=5))
                        ax.add_patch(Rectangle((9, k), 1, 1, fill=False, edgecolor="black", lw=5))
                ax1.set_title("positive")
                ax2.set_title("negative")
                ax1.set_yticklabels(tokens_ls, rotation=0)
                # plt.savefig(output_dir + "selected_attn_score_sendL{}H{}recvL{}_ioi".format(send_L, H, recv_L) + ".png")
                plt.savefig(output_dir + "attn_score_sendL{}H{}recvL{}_ioi".format(send_L, H, recv_L) + ".png")
                plt.close("all")

def decompose_H_comparison_batch_pair_drawer(head_score, tokens_ls, n_layers, n_heads, output_dir):
    """ plot the scores assigned by output of (L, H, Q, K) to experts in another MoE layer. Two prompts. head score shape: PQKHERS """
    for send_L in range(0, n_layers):
        for H in range(0, n_heads):
            for recv_L in range(send_L, n_layers):
                if send_L != 1 or recv_L != 3 or H != 4:
                    continue
                # if send_L != 13:
                #     continue
                # if not (H == 1 or H == 2 or H == 5):
                #     continue
                # if send_L != 9 or H != 14 or recv_L != 10:
                #     continue
                # if send_L != 1 or H != 8:
                #     continue

                ## NOTE: Selected expert only version is unchecked and unused. Please refer to function "decompose_H_comparison_batch_single_drawer".
                # score = head_score[:, torch.arange(len(tokens_ls)).reshape(-1, 1).repeat(1, 8), :, H, original_top_k_experts[0, :, :, recv_L], recv_L, send_L] # selected experts only
                # pos = (score + torch.abs(score)).div(2).sum(1)
                # neg = (score - torch.abs(score)).div(2).sum(1)
                # pos = pos.permute(1, 0, 2)
                # neg = neg.permute(1, 0, 2)

                score = head_score[:, :, :, H, :, recv_L, send_L] # all experts
                pos = (score + torch.abs(score)).div(2).sum(-1)
                neg = (score - torch.abs(score)).div(2).sum(-1)
                
                fig, axes = plt.subplots(2, 3, figsize=(25, 15))
                fig.suptitle("(All experts) attn_score_sendL{}H{}recvL{}_ioi_two_prompts".format(send_L, H, recv_L))
                # fig.suptitle("(Selected experts) attn_score_sendL{}H{}recvL{}_ioi_two_prompts".format(send_L, H, recv_L))

                mask = np.zeros_like(pos[0].detach().cpu().numpy())
                mask[np.triu_indices_from(mask, k=1)] = True
                diff_pos = (pos[0] - pos[1]).detach().cpu().numpy()
                diff_neg = (neg[0] - neg[1]).detach().cpu().numpy()
                print("diff_pos max/min:", diff_pos.max(), diff_pos.min())

                if diff_pos.min() < 0 and diff_pos.max() > 0:
                    g11 = sns.heatmap(diff_pos, mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="RdBu", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[0][0], norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=diff_pos.min(), vmax=diff_pos.max()))
                elif diff_pos.min() >= 0: # bad practice, just a temporal remedy
                    g11 = sns.heatmap(diff_pos, mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[0][0], norm=mcolors.Normalize(vmin=diff_pos.min(), vmax=diff_pos.max()))
                else:
                    g11 = sns.heatmap(diff_pos, mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="Reds_r", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[0][0], norm=mcolors.Normalize(vmin=diff_pos.min(), vmax=diff_pos.max()))
                # g11 = sns.heatmap(diff_pos, mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="RdBu", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[0][0], norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=diff_pos.min(), vmax=diff_pos.max()))
                g12 = sns.heatmap(pos[0].detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[0][1])
                g13 = sns.heatmap(pos[1].detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[0][2])
                g21 = sns.heatmap(diff_neg, mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="RdBu", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[1][0], norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=diff_neg.min(), vmax=diff_neg.max()))
                g22 = sns.heatmap(neg[0].detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="Reds_r", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[1][1])
                g23 = sns.heatmap(neg[1].detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":9}, cmap="Reds_r", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=axes[1][2])

                for ax in [g11, g12, g13, g21, g22, g23]:
                    ax.add_patch(Rectangle((0, 4), 5, 1, fill=False, edgecolor="red", lw=3))
                    ax.add_patch(Rectangle((0, 9), 10, 1, fill=False, edgecolor="red", lw=3))
                    ax.add_patch(Rectangle((0, 13), 14, 1, fill=False, edgecolor="red", lw=3))
                    ax.add_patch(Rectangle((1, 4), 1, 1, fill=False, edgecolor="black", lw=5))
                    ax.add_patch(Rectangle((3, 4), 1, 1, fill=False, edgecolor="black", lw=5))
                    for k in [9, 13]:
                        ax.add_patch(Rectangle((1, k), 1, 1, fill=False, edgecolor="black", lw=5))
                        ax.add_patch(Rectangle((3, k), 1, 1, fill=False, edgecolor="black", lw=5))
                        ax.add_patch(Rectangle((9, k), 1, 1, fill=False, edgecolor="black", lw=5))
                
                g11.set_yticklabels(tokens_ls, rotation=0)
                g21.set_yticklabels(tokens_ls, rotation=0)
                g11.set_title("diff_pos")
                g12.set_title("pos_ioi")
                g13.set_title("pos_abc")
                g21.set_title("diff_neg")
                g22.set_title("neg_ioi")
                g23.set_title("neg_abc")
                # plt.savefig(output_dir + "selected_attn_score_sendL{}H{}recvL{}_ioi_two_prompts".format(send_L, H, recv_L) + ".png")
                plt.savefig(output_dir + "attn_score_sendL{}H{}recvL{}_ioi_two_prompts".format(send_L, H, recv_L) + ".png")
                plt.close("all")

def decompose_H_expert_score_scatter_batch(head_score, query_position, key_position, send_L, recv_L, output_dir):
    """ scatter the scores assigned by output of (L, H, Q, K) to experts in another MoE layer. The first prompt only. head score shape: PQKHERS """
    ## scatter of head scores of specific heads
    # send_L, recv_L = 13, 13
    # score_A13H0 = head_score[0, -1, 9, 0, :, recv_L, send_L] # all experts, q=Token END, k=Token S2
    # score_A13H1 = head_score[0, -1, 9, 1, :, recv_L, send_L] # all experts, q=Token END, k=Token S2
    # score_A13H2 = head_score[0, -1, 9, 2, :, recv_L, send_L] # all experts, q=Token END, k=Token S2
    # score_A13H5 = head_score[0, -1, 9, 5, :, recv_L, send_L] # all experts, q=Token END, k=Token S2
    # score_A13H10 = head_score[0, -1, 9, 10, :, recv_L, send_L] # all experts, q=Token END, k=Token S2
    # score_A13H11 = head_score[0, -1, 9, 11, :, recv_L, send_L] # all experts, q=Token END, k=Token S2

    # plt.scatter([i for i in range(64)], score_A13H0.detach().cpu().numpy(), s=5, c="black",label="A13H0")
    # plt.scatter([i for i in range(64)], score_A13H1.detach().cpu().numpy(), s=5, c="red",label="A13H1")
    # plt.scatter([i for i in range(64)], score_A13H2.detach().cpu().numpy(), s=5, c="blue",label="A13H2")
    # plt.scatter([i for i in range(64)], score_A13H5.detach().cpu().numpy(), s=5, c="green",label="A13H5")
    # plt.scatter([i for i in range(64)], score_A13H10.detach().cpu().numpy(), s=5, c="pink",label="A13H10")
    # plt.scatter([i for i in range(64)], score_A13H11.detach().cpu().numpy(), s=5, c="cyan",label="A13H11")

    n_experts = head_score.shape[4]
    color_ls = ["k","grey","r","sandybrown","orange","gold","yellowgreen","lawngreen","g","aquamarine","cyan","dodgerblue","b","indigo","violet","pink"]
    plt.figure(figsize=(15,15))
    for H in range(16):
        score = head_score[0, query_position, key_position, H, :, recv_L, send_L]
        plt.scatter([i for i in range(n_experts)], score.detach().cpu().numpy(), s=5, c=color_ls[H], label="A{}H{}".format(send_L, H))
        print("H{}, score var:{}, score mean:{}".format(H, score.var(), score.mean()))
    plt.grid()
    plt.legend()
    
    # plt.ylim(-0.2,0.2)
    plt.savefig(output_dir + "attn_score_sendL{}recvL{}_expert_score_scatter_ioi".format(send_L, recv_L) + ".png")
    plt.close("all")

def M_drawer(data, original_top_k_experts, name, output_dir, title=""):
    """ Scatter the scores assigned by selected experts in MoE Layer x to all the experts in
    :param data: decomposed_expert_out_score, shape: [n_experts, original_top_k], the scores assigned by the original top K experts in the SENDING layer to n_experts in the receiving layer
    :param original_top_k_experts: the original top K experts in the RECEIVING layer
    """
    ## vanilla version, monochrome
    plt.grid()
    n_experts, top_k = data.shape
    xs = [i for _ in range(n_experts) for i in range(top_k)]
    ys = data.reshape(-1).detach().cpu().numpy()
    plt.scatter(xs, ys, s=5, c="b")
    plt.title(title)
    plt.xlabel("Top K experts (Sending Layer)")
    plt.ylabel("Score of experts (Receiving Layer)")
    plt.savefig(output_dir + "monochrome_" + name + ".png") # .pdf
    plt.close("all")

    ## bichrome, highlight the selected experts in the RECEIVING layer in red
    plt.grid()
    xs1 = [i for _ in range(top_k) for i in range(top_k)]
    xs2 = [i for _ in range(n_experts - top_k) for i in range(top_k)]
    ys1 = data[original_top_k_experts, :].reshape(-1).detach().cpu().numpy()
    mask = torch.ones((n_experts), dtype=torch.bool)
    mask[original_top_k_experts] = False
    ys2 = data[mask, :].reshape(-1).detach().cpu().numpy()
    plt.scatter(xs2, ys2, s=5, c="b", label="unselected experts")
    plt.scatter(xs1, ys1, s=5, c="r", label="selected experts")
    plt.xticks(ticks=np.arange(top_k), labels=data[original_top_k_experts, :].sum(0).detach().cpu().numpy().round(2))
    plt.title(title)
    plt.legend()
    plt.xlabel("Cumulative score assigned by TopK experts in SEND L. to selected experts in RECV L.")
    plt.ylabel("Score of experts (Receiving Layer)")
    plt.savefig(output_dir + "bichrome_" + name + ".png") # .pdf
    plt.close("all")

def matrix_attn_weight_verbose(data, name, output_dir, figsize=(13,13), title="", xlabel="Key Token", ylabel="Head", need_description=True):
    plt.figure(figsize=figsize)
    with sns.axes_style("white"):
        ax = sns.heatmap(data.detach().cpu().numpy(), square=True, annot=True, fmt=".2f", cmap="RdBu", cbar_kws={"shrink": 0.8}, linewidth=.5)
    if need_description:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.savefig(output_dir + name + ".png")
    plt.close("all")

def matrix_attn_weight_comparison_verbose(attn_weights_ORIG, attn_weights_NEW, tokens_str, output_dir):
    n_layers, n_heads, _, _ = attn_weights_ORIG.shape

    y_ls = [str(i) + "| " + tokens_str[i] for i in range(len(tokens_str))]
    for L in range(n_layers):
        for H in range(n_heads):
            if L != 1 or H != 4:
                continue
            data_ORIG = attn_weights_ORIG[L, H]
            data_NEW = attn_weights_NEW[L, H]
            data_diff = data_ORIG - data_NEW
            plt.figure(figsize=(40, 40))
            fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            fig.suptitle("attn_weight_L{}H{}_diff_ioi_abc".format(L, H))
            mask = np.zeros_like(data_ORIG.detach().cpu().numpy())
            mask[np.triu_indices_from(mask, k=1)] = True
            g1 = sns.heatmap(data_diff.detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":11}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=ax1, norm=mcolors.Normalize(vmin=0, vmax=1))
            g2 = sns.heatmap(data_ORIG.detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":11}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=ax2)
            g3 = sns.heatmap(data_NEW.detach().cpu().numpy(), mask=mask, square=True, annot=True, fmt=".2f", annot_kws={"size":11}, cmap="Blues", cbar_kws={"shrink": 0.8}, linewidth=1, cbar=False, ax=ax3)
            for ax in [g1, g2, g3]:
                ax.add_patch(Rectangle((0, 4), 5, 1, fill=False, edgecolor="red", lw=3))
                ax.add_patch(Rectangle((0, 9), 10, 1, fill=False, edgecolor="red", lw=3))
                ax.add_patch(Rectangle((0, 13), 14, 1, fill=False, edgecolor="red", lw=3))
                ax.add_patch(Rectangle((1, 4), 1, 1, fill=False, edgecolor="black", lw=5))
                ax.add_patch(Rectangle((3, 4), 1, 1, fill=False, edgecolor="black", lw=5))
                for k in [9, 13]:
                    ax.add_patch(Rectangle((1, k), 1, 1, fill=False, edgecolor="black", lw=5))
                    ax.add_patch(Rectangle((3, k), 1, 1, fill=False, edgecolor="black", lw=5))
                    ax.add_patch(Rectangle((9, k), 1, 1, fill=False, edgecolor="black", lw=5))

            ax1.set_title("diff")
            ax2.set_title("ioi")
            ax3.set_title("abc")
            ax1.set_yticklabels(y_ls, rotation=0)

            plt.savefig(output_dir + "attn_weight_L{}H{}_diff_ioi_abc".format(L, H) + ".png")
            plt.close("all")
