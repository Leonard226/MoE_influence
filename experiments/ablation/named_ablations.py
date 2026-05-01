"""Tier 2 — per-neuron AARV (paper Eq. 13 extended to single neurons / coalitions).

For each ablation target (a (c, j) expert with a chosen subset of neurons z),
compute the rank shift the ablation induces in the top-K selection at every
downstream MoE layer l > c, averaged over tokens at which the parent expert is
selected in the top-8.

Score subtraction (linear in the score, not variance):

    Δ^{l,n}_{c,j,z}(i) = (1 / RMS^l_i) * r^{c,j}(i) * α^{c,j}_z(i) * A^{l,n}_{c,j,z}
    A^{l,n}_{c,j,z}     = g^{l,n} · (W_d_(:,z)^{c,j} ⊙ γ^l)

For a whole-expert ablation, sum the per-neuron Δ over all z = 1,…,d_ffn.

AARV per (token, recv layer, target):
    AARV(i, l) = mean over the original top-K experts e of |rank_orig(e) - rank_pert(e)|

Aggregated by averaging over tokens for which the parent expert was selected.

Output:
    ablation/aarv.pt          dict with per-(target, recv layer) AARV + token counts
    ablation/aarv_summary.json
"""
import json
import os
import sys
import time

import numpy as np
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

# --- Determine the rank-50 (low-variance) neuron inside M1E9 for control C1. ---
T_dyn_path = os.path.join(config["result_path"], "variance", "T_dyn.pt")
T_dyn = torch.load(T_dyn_path, map_location="cpu")  # [L, E, d_ffn, L]
neuron_score = T_dyn.sum(dim=3).numpy()             # [L, E, d_ffn]
m1e9_sorted = np.argsort(neuron_score[1, 9])[::-1]
M1E9_Z_LOW = int(m1e9_sorted[50])
del T_dyn, neuron_score, m1e9_sorted

# --- Ablation targets ---
# (label, sending_layer c, sending_expert j, list of z indices to ablate)
ABLATIONS = [
    ("A1_M1E9_whole",      1,  9, list(range(D_FFN))),
    ("A2_M1E9_z915",       1,  9, [915]),
    ("A3_M2E30_whole",     2, 30, list(range(D_FFN))),
    ("A4_M2E30_z742",      2, 30, [742]),
    ("A5_M4E14_whole",     4, 14, list(range(D_FFN))),
    ("A6_M4E14_z391",      4, 14, [391]),
    ("A7_M4E14_coalition", 4, 14, [391, 336, 956]),
    ("C1_M1E9_z_rank50",   1,  9, [M1E9_Z_LOW]),
    ("C2_M14E60_whole",   14, 60, list(range(D_FFN))),
    ("C3_M14E60_z357",    14, 60, [357]),
    ("C4_M0E0_whole",      0,  0, list(range(D_FFN))),
]
N_TARGETS = len(ABLATIONS)
print(f"{N_TARGETS} ablation targets. Control C1 ablates M1E9 z={M1E9_Z_LOW} (rank-50 inside M1E9).", flush=True)

# --- Load model + tokenizer + dataset ---
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

# --- Receiver-side weights and per-target alignment tensor A^{l,n}_{c,j,z} ---
G = torch.stack([
    model.model.layers[l].mlp.gate.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, n_experts, d_e]
gamma = torch.stack([
    model.model.layers[l].post_attention_layernorm.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, d_e]
G_tilde = G * gamma.unsqueeze(1)  # [L, n_experts, d_e]

# Precompute A_per_target[(c, j)] = einsum("lnd,dz->lnz", G_tilde, W_d^{c,j})
# Only for unique (c, j) pairs we need.
A_per_target = {}
for _, c, j, _ in ABLATIONS:
    if (c, j) in A_per_target:
        continue
    Wd = model.model.layers[c].mlp.experts[j].down_proj.weight.detach().to(device, dtype=torch.float32)  # [d_e, d_ffn]
    A_per_target[(c, j)] = torch.einsum("lnd,dz->lnz", G_tilde, Wd)  # [L, n_experts, d_ffn]

# Precompute z-index tensors for each target.
Z_PER_TARGET = [torch.tensor(z_set, device=device, dtype=torch.long) for _, _, _, z_set in ABLATIONS]

# --- Accumulators ---
aarv_accum = torch.zeros((N_TARGETS, N_LAYERS), dtype=torch.float64, device=device)
aarv_count = torch.zeros((N_TARGETS, N_LAYERS), dtype=torch.long,    device=device)

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches (bsz={BSZ}, max_tokens={MAX_TOKENS}) ...", flush=True)
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

    # Reshape to [bt, L, ...].
    after_norm2_r = after_norm2.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)  # [bt, L, d_e]
    after_res1_r  = after_res1.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)
    sel_r = selected.long().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, TOP_K)
    routing_w_r = routing_w.float()                                                    # already [bt, L, top_k]
    alpha_r = alpha.float().permute(0, 2, 1, 3, 4).reshape(bt, N_LAYERS, TOP_K, D_FFN)

    # 1 / RMS^l_i
    rms_inv = torch.rsqrt(after_res1_r.pow(2).mean(dim=-1) + EPS)  # [bt, L]

    # Original score: S^{l,n}(i) = g^{l,n} · after_norm2[i, l, :]
    orig_score = torch.einsum("lnd,ild->iln", G, after_norm2_r)  # [bt, L, n_experts]

    # Original ranking: rank_of[i, l, n] = position of expert n in descending sort.
    orig_sorted = torch.argsort(orig_score, dim=-1, descending=True)  # [bt, L, n_experts]
    orig_rank_of = torch.empty_like(orig_sorted)
    orig_rank_of.scatter_(-1, orig_sorted,
                          torch.arange(N_EXPERTS, device=device).expand_as(orig_sorted))
    orig_top_k = orig_sorted[:, :, :TOP_K]                       # [bt, L, top_k]

    for t_idx, (_, c_target, j_target, _) in enumerate(ABLATIONS):
        # Find slot of j_target in top-K at sending layer c_target.
        sel_at_c = sel_r[:, c_target, :]                         # [bt, top_k]
        slot_mask = (sel_at_c == j_target).float()               # [bt, top_k]; one-hot per row when active
        token_active = slot_mask.sum(dim=-1) > 0.5               # [bt]

        if not token_active.any():
            continue

        # r^{c,j}(i) and α^{c,j}_z(i) for active tokens.
        r_target = (routing_w_r[:, c_target, :] * slot_mask).sum(dim=-1)  # [bt]
        alpha_at_c = alpha_r[:, c_target, :, :]                  # [bt, top_k, d_ffn]
        alpha_target = (alpha_at_c * slot_mask.unsqueeze(-1)).sum(dim=1)  # [bt, d_ffn]

        # Restrict to z_set (only ablate these neurons).
        z_set_t = Z_PER_TARGET[t_idx]
        alpha_z = alpha_target.index_select(1, z_set_t)          # [bt, |z_set|]
        A_z = A_per_target[(c_target, j_target)].index_select(2, z_set_t)  # [L, n_experts, |z_set|]

        # Σ_z α^{c,j}_z(i) · A^{l,n}_{c,j,z}
        contrib = torch.einsum("iz,lnz->iln", alpha_z, A_z)      # [bt, L, n_experts]

        # Δ_score = (1/RMS^l_i) * r^{c,j}(i) * Σ_z α · A
        delta_score = rms_inv.unsqueeze(-1) * r_target.view(bt, 1, 1) * contrib

        # Active tokens only, and ablation only matters for l > c_target.
        recv_mask = (torch.arange(N_LAYERS, device=device) > c_target).float()  # [L]
        delta_score = delta_score * token_active.view(bt, 1, 1).float() * recv_mask.view(1, N_LAYERS, 1)

        # Perturbed score and ranking.
        pert_score = orig_score - delta_score
        pert_sorted = torch.argsort(pert_score, dim=-1, descending=True)
        pert_rank_of = torch.empty_like(pert_sorted)
        pert_rank_of.scatter_(-1, pert_sorted,
                              torch.arange(N_EXPERTS, device=device).expand_as(pert_sorted))

        # Rank shift over the original top-K.
        orig_ranks_at_topk = torch.gather(orig_rank_of, -1, orig_top_k)   # [bt, L, top_k]  (= 0..top_k-1)
        pert_ranks_at_topk = torch.gather(pert_rank_of, -1, orig_top_k)   # [bt, L, top_k]
        rank_shift = (pert_ranks_at_topk.float() - orig_ranks_at_topk.float()).abs()  # [bt, L, top_k]
        aarv_il = rank_shift.mean(dim=-1)                                # [bt, L]

        # Accumulate over (active tokens) × (recv layers > c_target).
        valid_mask = token_active.view(bt, 1) & (torch.arange(N_LAYERS, device=device) > c_target).view(1, N_LAYERS)  # bool [bt, L]
        aarv_accum[t_idx] += (aarv_il * valid_mask.float()).sum(dim=0).to(torch.float64)
        aarv_count[t_idx] += valid_mask.long().sum(dim=0)

        del slot_mask, alpha_at_c, alpha_target, alpha_z, A_z, contrib, delta_score
        del pert_score, pert_sorted, pert_rank_of, orig_ranks_at_topk, pert_ranks_at_topk, rank_shift, aarv_il, valid_mask

    del hook_dict, after_res1, after_norm2, routing_w, selected, alpha
    del after_norm2_r, after_res1_r, sel_r, routing_w_r, alpha_r, rms_inv
    del orig_score, orig_sorted, orig_rank_of, orig_top_k
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# --- Normalize: AARV(target, l) = sum_{i, valid} aarv_il / count_il ---
aarv = aarv_accum / aarv_count.clamp(min=1).to(torch.float64)
aarv = aarv.masked_fill(aarv_count == 0, 0.0)

# Save full data + a JSON summary.
torch.save({
    "aarv": aarv.cpu(),                                      # [n_targets, L]
    "count": aarv_count.cpu(),                               # [n_targets, L]
    "labels": [a[0] for a in ABLATIONS],
    "ablations": [{"label": a[0], "c": a[1], "j": a[2], "z_set": a[3]} for a in ABLATIONS],
    "M1E9_Z_LOW": M1E9_Z_LOW,
}, os.path.join(output_dir, "aarv.pt"))

# JSON summary: AARV summed over recv layers per target (plus per-l for the four named experts).
aarv_cpu = aarv.cpu().numpy()
count_cpu = aarv_count.cpu().numpy()
summary = {
    "M1E9_Z_LOW_for_C1": M1E9_Z_LOW,
    "totals": [],
}
for t_idx, (label, c, j, z_set) in enumerate(ABLATIONS):
    total_l = float(aarv_cpu[t_idx].sum())
    mean_l = float(aarv_cpu[t_idx, c+1:].mean()) if c < N_LAYERS - 1 else 0.0
    summary["totals"].append({
        "label": label,
        "c": c, "j": j, "n_z": len(z_set),
        "AARV_sum_over_l": total_l,
        "AARV_mean_over_downstream_l": mean_l,
        "n_active_tokens_per_l": [int(count_cpu[t_idx, l]) for l in range(N_LAYERS)],
        "AARV_per_l": [float(aarv_cpu[t_idx, l]) for l in range(N_LAYERS)],
    })
with open(os.path.join(output_dir, "aarv_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# --- Stdout summary table ---
print("==================== Tier 2 AARV summary ====================\n")
print(f"{'target':<22} {'AARV (mean over l > c)':>22}  active_tok_avg")
for t_idx, (label, c, j, z_set) in enumerate(ABLATIONS):
    if c >= N_LAYERS - 1:
        mean_aarv = 0.0
        avg_count = 0.0
    else:
        mean_aarv = float(aarv_cpu[t_idx, c + 1:].mean())
        avg_count = float(count_cpu[t_idx, c + 1:].mean())
    n_z = len(z_set)
    print(f"{label:<22} {mean_aarv:>22.4f}  {avg_count:>14.1f}  (|z|={n_z})")

print("\nFraction of expert AARV recovered by named neuron(s):")
expert_to_neuron = {
    ("A1_M1E9_whole",      "A2_M1E9_z915",       "M1E9 single neuron z=915"),
    ("A3_M2E30_whole",     "A4_M2E30_z742",      "M2E30 single neuron z=742"),
    ("A5_M4E14_whole",     "A6_M4E14_z391",      "M4E14 single neuron z=391"),
    ("A5_M4E14_whole",     "A7_M4E14_coalition", "M4E14 coalition {z=391, 336, 956}"),
    ("C2_M14E60_whole",    "C3_M14E60_z357",     "M14E60 single neuron z=357"),
    ("A1_M1E9_whole",      "C1_M1E9_z_rank50",   "M1E9 rank-50 neuron (control)"),
}
labels = [a[0] for a in ABLATIONS]
for whole, neuron, descr in expert_to_neuron:
    iw = labels.index(whole)
    inn = labels.index(neuron)
    cw = ABLATIONS[iw][1]
    if cw < N_LAYERS - 1:
        whole_aarv = aarv_cpu[iw, cw + 1:].mean()
        neuron_aarv = aarv_cpu[inn, cw + 1:].mean()
        frac = neuron_aarv / whole_aarv if whole_aarv > 0 else 0.0
    else:
        frac = 0.0
    print(f"  {descr:<40}  {frac * 100:.1f}%   "
          f"(neuron AARV {neuron_aarv:.3f} / whole AARV {whole_aarv:.3f})")

print(f"\nArtifacts written to {output_dir}")
