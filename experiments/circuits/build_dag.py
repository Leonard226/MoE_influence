"""Phase 1, Step 1 — minimal DAG builder for OLMoE on C4.

Computes the APS (Average Positive Score) edge weight only, on a small set of
prompts, as a sanity scan. Output saved to results/circuits/dag_C4_step1.pt.

The DAG has nodes = (layer, expert) pairs and directed cross-layer edges
(c, j) → (l, n) for c < l. APS on each edge is

    APS(c, j → l, n) = E_i [ max(S(g^{l,n}, e^{c,j}_{out,i}), 0) | (c,j) selected at i ]

where S(g^{l,n}, e^{c,j}_{out,i}) = g^{l,n} · LN_bar^l_i(r^{c,j}(i) · e^{c,j}_{out,i})
is the per-edge sub-score from the score decomposition (cf. MoEs/main.tex §2).

This script is modeled directly on experiments/variance/per_expert.py — same
forward-pass loop and reshapes — but instead of collapsing the per-receiver
score tensor via Var_n, we keep the receiver-expert dimension and accumulate
the positive part per (sender expert, receiver layer, receiver expert).

Step 2 will extend to ANS, mean signed score, and full 5000 prompts.
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
output_dir = os.path.join(config["result_path"], "circuits")
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

# Step 1: sanity scan on 50 prompts. Bump to 5000 in step 2.
N_PROMPTS = 50
BSZ = 50
MAX_TOKENS = 32

print(f"[Step 1] Building DAG on {N_PROMPTS} prompts (sanity scan).", flush=True)

# ---- Load model + tokenizer ----
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

# ---- Load C4 ----
print(f"Loading C4 ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = c4_dataset_helper(dataset_len=N_PROMPTS, seed=None, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# ---- Accumulators ----
# APS_accum[S, j, R, n] = Σ_{i : j ∈ top-K at S} max(S(g^{R,n}, e^{S,j}_{out,i}), 0)
APS_accum = torch.zeros(
    (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS), dtype=torch.float32, device=device
)
# count[S, j] = number of (token, top-k slot) events where expert j was selected at layer S.
count = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device)

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches (bsz={BSZ}, max_tokens={MAX_TOKENS}) ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(
        batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    _, hook_dict = model(input_ids=input_ids, attention_mask=attention_mask)

    after_res1   = hook_dict["hook_after_res1"]                # [bsz, L, n_tok, d_e]
    selected     = hook_dict["hook_selected_experts"]          # [bsz, L, n_tok, top_k]
    weighted_out = hook_dict["hook_expert_weighted_outputs"]   # [bsz, L, n_tok, top_k, d_e]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # 1 / RMS^R_i, per token, per receiver layer R.
    rms_sq  = after_res1.float().pow(2).mean(dim=-1) + EPS                # [bsz, L, n_tok]
    rms_inv = torch.rsqrt(rms_sq).permute(0, 2, 1).reshape(bt, N_LAYERS)  # [bt, L_recv]

    # Sender-side reshapes: [bt, S, k, ...]
    omega = (weighted_out.float()
             .permute(0, 2, 1, 3, 4)
             .reshape(bt, N_LAYERS, TOP_K, D_E))   # [bt, S, k, d_e]
    sel = (selected.long()
           .permute(0, 2, 1, 3)
           .reshape(bt, N_LAYERS, TOP_K))           # [bt, S, k]

    for S in range(N_LAYERS):
        sel_S = sel[:, S, :]                                    # [bt, top_k]
        count[S] += torch.bincount(sel_S.flatten(), minlength=N_EXPERTS)

        if S == N_LAYERS - 1:
            continue
        omega_S = omega[:, S, :, :]                             # [bt, top_k, d_e]

        for R in range(S + 1, N_LAYERS):
            # ln_bar^R(omega_S) = (omega_S ⊙ γ^R) / RMS^R_i
            ln_bar = (omega_S
                      * gamma_recv[R].view(1, 1, D_E)
                      * rms_inv[:, R].view(bt, 1, 1))            # [bt, k, d_e]
            # scores[bt, k, n] = g^{R,n} · ln_bar  — per-edge sub-score
            #   S(g^{R,n}, e^{S,j_k}_{out,i})  with j_k = sel_S[bt, k].
            scores = torch.einsum("ed,bkd->bke", G_recv[R], ln_bar)  # [bt, k, N_EXPERTS]

            # APS contribution = positive part. Accumulate over (sender expert, receiver expert).
            scores_pos = scores.clamp(min=0.0)                  # [bt, k, N_EXPERTS]
            APS_accum[S, :, R, :].index_add_(
                0, sel_S.flatten(), scores_pos.flatten(0, 1)
            )

            del ln_bar, scores, scores_pos

    del hook_dict, after_res1, selected, weighted_out
    del omega, sel, rms_sq, rms_inv
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    elapsed = time.time() - t_start
    print(f"  batch {bnum}/{n_batches}  elapsed={elapsed:.1f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# ---- Normalize: APS[S, j, R, n] = APS_accum / count[S, j] ----
count_safe = count.clamp(min=1).to(torch.float32)                      # [L, n_experts]
APS = APS_accum / count_safe.view(N_LAYERS, N_EXPERTS, 1, 1)
APS = APS.masked_fill((count == 0).view(N_LAYERS, N_EXPERTS, 1, 1), 0.0)

out_path = os.path.join(output_dir, "dag_C4_step1.pt")
torch.save({
    "APS": APS.cpu(),                          # [c, j, l, n]
    "count": count.cpu(),                      # [c, j]
    "n_prompts": N_PROMPTS,
    "max_tokens": MAX_TOKENS,
    "model": MODEL_ID,
    "step": "1 (sanity scan, APS only)",
}, out_path)
print(f"Saved {out_path}")

# ---- Quick sanity prints ----
print("\n==================== Step 1 sanity stats ====================\n")
print(f"APS shape: {tuple(APS.shape)}  (sender_layer × sender_expert × recv_layer × recv_expert)")
print(f"APS nonzero entries: {(APS != 0).sum().item()} / {APS.numel()}")
print(f"APS max:  {APS.max().item():.4e}")
nonzero_mean = APS[APS != 0].mean().item() if (APS != 0).any() else 0.0
print(f"APS mean (over nonzero): {nonzero_mean:.4e}")

# Top-10 edges globally.
print("\nTop-10 APS edges:")
flat = APS.reshape(-1)
top_vals, top_idx = torch.topk(flat, 10)
for v, i in zip(top_vals.tolist(), top_idx.tolist()):
    c = i // (N_EXPERTS * N_LAYERS * N_EXPERTS)
    rest = i % (N_EXPERTS * N_LAYERS * N_EXPERTS)
    j = rest // (N_LAYERS * N_EXPERTS)
    rest2 = rest % (N_LAYERS * N_EXPERTS)
    l = rest2 // N_EXPERTS
    n = rest2 % N_EXPERTS
    print(f"  M{c}E{j} → M{l}E{n}   APS = {v:.4e}")

# Spot-check the known M1E9 → M4E14 chain from prior paper.
print(f"\nSpot check  M1E9 → M4E14   APS = {APS[1, 9, 4, 14].item():.4e}")
print("  (should be among the larger entries if construction is correct)")

# Aggregate APS by sender layer — should reproduce M1, M4 stripe pattern from Fig 3d.
sender_layer_sum = APS.sum(dim=(1, 2, 3)).cpu().numpy()                # [L]
print("\nAPS aggregated by sender layer (should peak at M1 and M4):")
for c in range(N_LAYERS):
    bar = "#" * int(40 * sender_layer_sum[c] / max(sender_layer_sum.max(), 1e-30))
    print(f"  M{c:2d}  {sender_layer_sum[c]:.4e}  {bar}")
