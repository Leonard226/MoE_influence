"""Tier 2 per-neuron AARV scan (paper Fig 6 analog at the neuron level).

For each named expert (c, j), ablate every one of its $d_{ffn} = 1024$ neurons
individually and compute the resulting AARV at every downstream receiving
layer. Output is a tensor `aarv[c,j][z, l]` per (target expert, neuron, recv
layer), conditioned on tokens where (c, j) is in the top-8.

Usage:
    python experiments/ablation/per_neuron_sweep.py
"""
import json
import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "ablation")
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"
torch.set_default_device(device)
torch.set_grad_enabled(False)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from transformers import AutoTokenizer
from dataset.c4_dataset import c4_dataset_helper

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS = 64
D_E = 2048
D_FFN = 1024
TOP_K = 8
EPS = 1e-5

N_PROMPTS = 5000
BSZ = 50
MAX_TOKENS = 32

# Named experts to scan.
TARGETS = [
    ("M1E9",   1,  9),
    ("M2E30",  2, 30),
    ("M4E14",  4, 14),
    ("M14E60", 14, 60),
]

print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

print(f"Loading C4 ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = c4_dataset_helper(dataset_len=N_PROMPTS, seed=None, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# Precompute A^{l,n}_{c,j,z} for each named expert.
G = torch.stack([
    model.model.layers[l].mlp.gate.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, n_experts, d_e]
gamma = torch.stack([
    model.model.layers[l].post_attention_layernorm.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, d_e]
G_tilde = G * gamma.unsqueeze(1)  # [L, n_experts, d_e]

A_per_target = {}
for _, c, j in TARGETS:
    Wd = model.model.layers[c].mlp.experts[j].down_proj.weight.detach().to(device, dtype=torch.float32)  # [d_e, d_ffn]
    A_per_target[(c, j)] = torch.einsum("lnd,dz->lnz", G_tilde, Wd)  # [L, n_experts, d_ffn]

# Accumulators: per (target, neuron z, recv layer).
aarv_accum = {(c, j): torch.zeros((D_FFN, N_LAYERS), dtype=torch.float64, device=device)
              for _, c, j in TARGETS}
token_count = {(c, j): 0 for _, c, j in TARGETS}

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches × {len(TARGETS)} ablation targets ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    _, hook_dict = model(input_ids=input_ids, attention_mask=attention_mask)

    after_res1 = hook_dict["hook_after_res1"]                # [bsz, L, n_tok, d_e]
    after_norm2 = hook_dict["hook_after_norm2"]              # [bsz, L, n_tok, d_e]
    routing_w = hook_dict["hook_routing_weights"]            # [bsz*n_tok, L, top_k]
    selected = hook_dict["hook_selected_experts"]            # [bsz, L, n_tok, top_k]
    alpha = hook_dict["hook_alpha"]                          # [bsz, L, n_tok, top_k, d_ffn]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    after_norm2_r = after_norm2.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)
    after_res1_r  = after_res1.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)
    sel_r = selected.long().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, TOP_K)
    routing_w_r = routing_w.float()
    alpha_r = alpha.float().permute(0, 2, 1, 3, 4).reshape(bt, N_LAYERS, TOP_K, D_FFN)

    rms_inv = torch.rsqrt(after_res1_r.pow(2).mean(dim=-1) + EPS)              # [bt, L]
    orig_score = torch.einsum("lnd,ild->iln", G, after_norm2_r)                 # [bt, L, n_experts]
    orig_top_k = torch.argsort(orig_score, dim=-1, descending=True)[:, :, :TOP_K]  # [bt, L, top_k]

    for name, c_target, j_target in TARGETS:
        sel_at_c = sel_r[:, c_target, :]
        slot_mask = (sel_at_c == j_target).float()
        token_active = slot_mask.sum(dim=-1) > 0.5
        n_active = int(token_active.sum().item())
        if n_active == 0:
            continue

        r_target = (routing_w_r[:, c_target, :] * slot_mask).sum(dim=-1)  # [bt]
        alpha_target = (alpha_r[:, c_target, :, :] * slot_mask.unsqueeze(-1)).sum(dim=1)  # [bt, d_ffn]
        A_target = A_per_target[(c_target, j_target)]  # [L, n_experts, d_ffn]

        for l_recv in range(c_target + 1, N_LAYERS):
            # K_iz[i, z] = (1/RMS^l_i) * r^{c,j}(i) * α^{c,j}_z(i)
            K_iz = (rms_inv[:, l_recv].unsqueeze(-1)
                    * r_target.unsqueeze(-1)
                    * alpha_target)  # [bt, d_ffn]

            # Δ[i, n, z] = K_iz[i, z] * A^{l, n}_{c,j,z}
            A_l = A_target[l_recv]                        # [n_experts, d_ffn]
            delta_inz = torch.einsum("iz,nz->inz", K_iz, A_l)  # [bt, n_experts, d_ffn]

            # Perturbed score and ranking under each z-ablation.
            pert_score = orig_score[:, l_recv, :].unsqueeze(-1) - delta_inz   # [bt, n_experts, d_ffn]
            pert_sorted = torch.argsort(pert_score, dim=1, descending=True)   # [bt, n_experts, d_ffn]
            pert_rank_of = torch.empty_like(pert_sorted)
            n_idx_expand = (torch.arange(N_EXPERTS, device=device)
                            .view(1, -1, 1)
                            .expand_as(pert_sorted))
            pert_rank_of.scatter_(dim=1, index=pert_sorted, src=n_idx_expand)

            # Original top-K at this (i, l_recv).
            orig_top_k_l = orig_top_k[:, l_recv, :]                            # [bt, top_k]
            pert_ranks_at_topk = torch.gather(
                pert_rank_of,
                dim=1,
                index=orig_top_k_l.unsqueeze(-1).expand(bt, TOP_K, D_FFN),
            )  # [bt, top_k, d_ffn]

            # |orig_rank - pert_rank|, where original ranks are 0..top_k-1.
            orig_ranks_b = (torch.arange(TOP_K, device=device)
                            .view(1, -1, 1)
                            .expand(bt, TOP_K, D_FFN))
            rank_shift = (pert_ranks_at_topk - orig_ranks_b).abs().float()    # [bt, top_k, d_ffn]
            aarv_iz = rank_shift.mean(dim=1)                                  # [bt, d_ffn]

            # Accumulate for active tokens only.
            aarv_iz_active = aarv_iz * token_active.unsqueeze(-1).float()
            aarv_accum[(c_target, j_target)][:, l_recv] += aarv_iz_active.sum(dim=0).to(torch.float64)

            del K_iz, delta_inz, pert_score, pert_sorted, pert_rank_of
            del pert_ranks_at_topk, rank_shift, aarv_iz, aarv_iz_active, n_idx_expand, orig_ranks_b

        token_count[(c_target, j_target)] += n_active

    del hook_dict, after_res1, after_norm2, routing_w, selected, alpha
    del after_norm2_r, after_res1_r, sel_r, routing_w_r, alpha_r, rms_inv
    del orig_score, orig_top_k
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# Normalize and save.
results = {}
for name, c, j in TARGETS:
    n = max(token_count[(c, j)], 1)
    results[name] = {
        "c": c, "j": j,
        "aarv": (aarv_accum[(c, j)] / n).cpu(),       # [d_ffn, L]
        "n_tokens": int(token_count[(c, j)]),
    }
torch.save(results, os.path.join(output_dir, "per_neuron_aarv.pt"))
print(f"Saved per_neuron_aarv.pt with results for {[r for r in results]}")

# Quick stdout summary: top neurons per expert at the late receiving layer (l = L-1).
print("\nTop-5 neurons by AARV at the late receiving layer:")
for name, c, j in TARGETS:
    aarv = results[name]["aarv"].numpy()
    l_late = N_LAYERS - 1
    top_z = aarv[:, l_late].argsort()[::-1][:5]
    print(f"  {name} (c={c}, j={j}, n_tokens={results[name]['n_tokens']}, l={l_late}):")
    for z in top_z:
        print(f"    z = {int(z):4d}   AARV = {aarv[z, l_late]:.3f}")
