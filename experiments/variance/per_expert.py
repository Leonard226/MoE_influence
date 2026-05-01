"""Tier 1 — paper-faithful per-expert variance.

Implements the same metric as Li et al.'s `decompose_E` (tools/analyze.py:1124):

    score_variance[S, J, R] = avg_{i : (S,J) selected at i} Var_n S(g^{R,n}, m^{S,J}_{out,i})

where S^{R,n}_{(S,J)}(i) = (1/RMS^R_i) * router^{R,n} . (r^{S,J}(i) * e^{S,J}_out(i)) ⊙ γ^R
is the (un-decomposed) per-expert score at receiver (R, n) at token i, and the variance
is taken over the N=64 candidate receivers n at fixed (R, i).

This differs from sum_z T_dyn[S, J, z, R] (our Tier 1 proxy) by the cross-covariance
terms across the d_ffn neurons inside the expert. Run on the same C4 prompts as
variance.py for direct comparability.

Output (under config.yaml:result_path / variance/):
    score_variance_per_expert.pt        [n_layers, n_experts, n_layers] fp32
"""
import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "variance")
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
TOP_K = 8
EPS = 1e-5

N_PROMPTS = 5000
BSZ = 50
MAX_TOKENS = 32

print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# Receiver-side weights as fp32.
G_recv = torch.stack([
    model.model.layers[R].mlp.gate.weight.detach().to(device, dtype=torch.float32)
    for R in range(N_LAYERS)
])  # [L, n_experts, d_e]
gamma_recv = torch.stack([
    model.model.layers[R].post_attention_layernorm.weight.detach().to(device, dtype=torch.float32)
    for R in range(N_LAYERS)
])  # [L, d_e]

print(f"Loading C4 ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = c4_dataset_helper(dataset_len=N_PROMPTS, seed=None, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

score_var_accum = torch.zeros((N_LAYERS, N_EXPERTS, N_LAYERS), dtype=torch.float32, device=device)
count = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device)

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    _, hook_dict = model(input_ids=input_ids, attention_mask=attention_mask)

    after_res1 = hook_dict["hook_after_res1"]                       # [bsz, L, n_tok, d_e]
    selected = hook_dict["hook_selected_experts"]                   # [bsz, L, n_tok, top_k]
    weighted_out = hook_dict["hook_expert_weighted_outputs"]        # [bsz, L, n_tok, top_k, d_e]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # 1/RMS at receiver R, per token.
    rms_sq = after_res1.float().pow(2).mean(dim=-1) + EPS  # [bsz, L, n_tok]
    rms_inv = torch.rsqrt(rms_sq).permute(0, 2, 1).reshape(bt, N_LAYERS)  # [bt, L_recv]

    # Reshape sender-side quantities: [bt, L_send, top_k, ...]
    omega = (weighted_out.float()
             .permute(0, 2, 1, 3, 4)
             .reshape(bt, N_LAYERS, TOP_K, D_E))  # [bt, S, k, d_e]
    sel = (selected.long()
           .permute(0, 2, 1, 3)
           .reshape(bt, N_LAYERS, TOP_K))  # [bt, S, k]

    for S in range(N_LAYERS):
        sel_S = sel[:, S, :]                  # [bt, top_k]
        count[S] += torch.bincount(sel_S.flatten(), minlength=N_EXPERTS)

        if S == N_LAYERS - 1:
            continue
        omega_S = omega[:, S, :, :]           # [bt, top_k, d_e]

        for R in range(S + 1, N_LAYERS):
            # LN-bar^R(omega_S) = omega_S * gamma^R / RMS^R_i
            ln_bar = omega_S * gamma_recv[R].view(1, 1, D_E) * rms_inv[:, R].view(bt, 1, 1)  # [bt, k, d_e]
            # scores[bt, k, n_experts_R] = G^R . ln_bar
            scores = torch.einsum("ed,bkd->bke", G_recv[R], ln_bar)  # [bt, k, N_EXPERTS]
            # Variance over n at fixed (bt, k): N=64 receivers at layer R.
            var_n = scores.var(dim=-1, unbiased=False)  # [bt, k]

            # Accumulate into score_var_accum[S, J=sel_S, R].
            score_var_accum[S, :, R].index_add_(0, sel_S.flatten(), var_n.flatten())
            del ln_bar, scores, var_n

    del hook_dict, after_res1, selected, weighted_out, omega, sel, rms_sq, rms_inv
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# Normalize.
count_safe = count.clamp(min=1).to(torch.float32)
score_variance = score_var_accum / count_safe.view(N_LAYERS, N_EXPERTS, 1)
score_variance = score_variance.masked_fill((count == 0).view(N_LAYERS, N_EXPERTS, 1), 0.0)
score_variance_cpu = score_variance.cpu()

torch.save(score_variance_cpu, os.path.join(output_dir, "score_variance_per_expert.pt"))
print(f"Saved score_variance_per_expert.pt — shape {tuple(score_variance_cpu.shape)}\n")

# Sanity-check: compare to sum_z T_dyn (Tier 1 proxy) for the top-5 experts.
T_dyn = torch.load(os.path.join(output_dir, "T_dyn.pt"))
T_dyn_per_expert = T_dyn.sum(dim=2)  # [S, J, R]

paper_total = score_variance_cpu.sum(dim=2)
proxy_total = T_dyn_per_expert.sum(dim=2)

print("--- Top 5 experts by paper-faithful variance, with sum_z proxy comparison ---")
order = torch.argsort(paper_total.flatten(), descending=True)[:8]
for idx in order:
    S = idx.item() // N_EXPERTS
    J = idx.item() % N_EXPERTS
    paper = paper_total[S, J].item()
    proxy = proxy_total[S, J].item()
    diff = (proxy - paper) / paper * 100 if paper > 0 else 0.0
    print(f"  M{S}E{J}: paper-faithful={paper:.4f}  sum_z proxy={proxy:.4f}  proxy-vs-paper diff = {diff:+.2f}%")
