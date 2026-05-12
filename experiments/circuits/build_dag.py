"""DAG builder for OLMoE on a chosen dataset, with multiple edge weights.

Computes five edge weights conditioned on sender selection:

    APS(c, j → l, n)   = E_i [ max(S(g^{l,n}, e^{c,j}_{out,i}), 0) | (c,j) selected ]
    ANS(c, j → l, n)   = E_i [ min(S(g^{l,n}, e^{c,j}_{out,i}), 0) | (c,j) selected ]
    AVG(c, j → l, n)  = E_i [     S(g^{l,n}, e^{c,j}_{out,i})     | (c,j) selected ]
    VAR(c, j → l, n) = Var_i [   S(g^{l,n}, e^{c,j}_{out,i})     | (c,j) selected ]
    AARV(c, j → l, n)  = E_i [ |rank_l^orig(n) − rank_l^pert(n)|   | (c,j) selected ]

where S(g^{l,n}, e^{c,j}_{out,i}) = g^{l,n} · LN_bar^l_i(r^{c,j}(i) · e^{c,j}_{out,i})
is the per-edge sub-score from the score decomposition (cf. MoEs/main.tex §2).
The first four are score-based; AARV is causal (the rank shift of receiver n
at layer l when (c,j)'s sub-score is removed via score subtraction, conditional
on sender (c,j) being selected at i).

Usage:
    python experiments/circuits/build_dag.py --dataset c4
    python experiments/circuits/build_dag.py --dataset math --n-prompts 5000

Output: {result_path}/circuits/dag_{dataset}.pt

Modeled on experiments/variance/per_expert.py — same forward-pass loop and
reshapes — but keeps the receiver-expert dimension instead of collapsing via Var_n.
"""
import argparse
import importlib
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

# Dataset registry: name -> (module_path, helper_function_name).
# All helpers must accept (dataset_len, seed, min_words) and return a list of strings.
DATASETS = {
    "c4":   ("dataset.c4_dataset",   "c4_dataset_helper"),
    "math": ("dataset.math_dataset", "open_r1_math_dataset_helper"),
    "code": ("dataset.code_dataset", "code_dataset_helper"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", choices=list(DATASETS), default="c4",
                    help="Which dataset to build the DAG on (default: c4).")
parser.add_argument("--n-prompts", type=int, default=5000,
                    help="Number of prompts to use (default: 5000).")
parser.add_argument("--seed", type=int, default=None,
                    help="Seed for dataset shuffling. None = sequential from "
                         "start. Different seeds yield (mostly) disjoint subsets, "
                         "useful for same-dataset baseline runs.")
args = parser.parse_args()

device = "cuda:0"
torch.set_default_device(device)
torch.set_grad_enabled(False)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS = 64
D_E = 2048
TOP_K = 8
EPS = 1e-5

N_PROMPTS = args.n_prompts
BSZ = 50
MAX_TOKENS = 32

print(f"Building DAG on dataset={args.dataset!r}, {N_PROMPTS} prompts.", flush=True)

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

# ---- Load dataset ----
mod_name, fn_name = DATASETS[args.dataset]
loader = getattr(importlib.import_module(mod_name), fn_name)
print(f"Loading dataset={args.dataset!r}, seed={args.seed!r} ({N_PROMPTS} prompts) ...",
      flush=True)
t0 = time.time()
prompts = loader(dataset_len=N_PROMPTS, seed=args.seed, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# ---- Accumulators ----
# APS_accum[S, j, R, n]   = Σ_{i : j ∈ top-K at S} max(S(g^{R,n}, e^{S,j}_{out,i}), 0)
# ANS_accum[S, j, R, n]   = Σ_{i : j ∈ top-K at S} min(S(g^{R,n}, e^{S,j}_{out,i}), 0)
# AVG_accum[S, j, R, n]  = Σ_{i : j ∈ top-K at S}     S(g^{R,n}, e^{S,j}_{out,i})
# sq_accum[S, j, R, n]    = Σ_{i : j ∈ top-K at S}     S(g^{R,n}, e^{S,j}_{out,i})^2
# (Var_i is computed at the end as E[S^2] - E[S]^2.)
SHAPE = (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS)
APS_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device)
ANS_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device)
AVG_accum = torch.zeros(SHAPE, dtype=torch.float32, device=device)
sq_accum   = torch.zeros(SHAPE, dtype=torch.float32, device=device)
# aarv_accum[S, j, R, n] = Σ_{i : j ∈ top-K at S} | rank_R^orig(n) - rank_R^pert(n) |
# where pert_score = orig_score - S(g^{R,n}, e^{S,j}_{out,i}).
aarv_accum = torch.zeros(SHAPE, dtype=torch.float32, device=device)
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
    after_norm2  = hook_dict["hook_after_norm2"]               # [bsz, L, n_tok, d_e]
    selected     = hook_dict["hook_selected_experts"]          # [bsz, L, n_tok, top_k]
    weighted_out = hook_dict["hook_expert_weighted_outputs"]   # [bsz, L, n_tok, top_k, d_e]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # 1 / RMS^R_i, per token, per receiver layer R.
    rms_sq  = after_res1.float().pow(2).mean(dim=-1) + EPS                # [bsz, L, n_tok]
    rms_inv = torch.rsqrt(rms_sq).permute(0, 2, 1).reshape(bt, N_LAYERS)  # [bt, L_recv]

    # Original assignment scores at every receiver layer (for AARV computation).
    after_norm2_r = (after_norm2.float()
                     .permute(0, 2, 1, 3)
                     .reshape(bt, N_LAYERS, D_E))                          # [bt, L, d_e]
    orig_score = torch.einsum("lnd,bld->bln", G_recv, after_norm2_r)       # [bt, L, N_EXPERTS]
    orig_sorted = torch.argsort(orig_score, dim=-1, descending=True)       # [bt, L, N_EXPERTS]
    orig_rank_of = torch.empty_like(orig_sorted)
    orig_rank_of.scatter_(-1, orig_sorted,
                          torch.arange(N_EXPERTS, device=device).expand_as(orig_sorted))
    # orig_rank_of[bt, l, n] = position of expert n in the layer-l ordering.

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

            # APS = positive part, ANS = negative part, AVG = signed, sq = squared.
            # Accumulate over (sender expert, receiver expert).
            scores_pos = scores.clamp(min=0.0)                  # [bt, k, N_EXPERTS]
            scores_neg = scores.clamp(max=0.0)                  # [bt, k, N_EXPERTS]
            scores_sq  = scores * scores                        # [bt, k, N_EXPERTS]
            sel_flat = sel_S.flatten()
            APS_accum[S, :, R, :].index_add_(0, sel_flat, scores_pos.flatten(0, 1))
            ANS_accum[S, :, R, :].index_add_(0, sel_flat, scores_neg.flatten(0, 1))
            AVG_accum[S, :, R, :].index_add_(0, sel_flat, scores.flatten(0, 1))
            sq_accum[S, :, R, :].index_add_(0, sel_flat, scores_sq.flatten(0, 1))

            # Per-edge AARV: for each slot k, ablate the sender's contribution
            # by subtracting `scores` from orig_score at layer R, re-rank, and
            # measure the per-receiver rank shift |orig_rank − pert_rank|.
            pert_score = orig_score[:, R, :].unsqueeze(1) - scores               # [bt, k, N_EXPERTS]
            pert_sorted = torch.argsort(pert_score, dim=-1, descending=True)     # [bt, k, N_EXPERTS]
            pert_rank_of = torch.empty_like(pert_sorted)
            pert_rank_of.scatter_(-1, pert_sorted,
                                  torch.arange(N_EXPERTS, device=device).expand_as(pert_sorted))
            orig_rank_R = orig_rank_of[:, R, :].unsqueeze(1).expand_as(pert_rank_of)  # [bt, k, N_EXPERTS]
            rank_shift = (pert_rank_of.float() - orig_rank_R.float()).abs()      # [bt, k, N_EXPERTS]
            aarv_accum[S, :, R, :].index_add_(0, sel_flat, rank_shift.flatten(0, 1))

            del ln_bar, scores, scores_pos, scores_neg, scores_sq
            del pert_score, pert_sorted, pert_rank_of, orig_rank_R, rank_shift

    del hook_dict, after_res1, after_norm2, selected, weighted_out
    del omega, sel, rms_sq, rms_inv, after_norm2_r
    del orig_score, orig_sorted, orig_rank_of
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed if elapsed > 0 else 0.0
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# ---- Normalize: weight[S, j, R, n] = accum / count[S, j] ----
count_safe = count.clamp(min=1).to(torch.float32)                      # [L, n_experts]
denom      = count_safe.view(N_LAYERS, N_EXPERTS, 1, 1)
zero_mask  = (count == 0).view(N_LAYERS, N_EXPERTS, 1, 1)

APS   = (APS_accum  / denom).masked_fill(zero_mask, 0.0)
ANS   = (ANS_accum  / denom).masked_fill(zero_mask, 0.0)
AVG  = (AVG_accum / denom).masked_fill(zero_mask, 0.0)
# Var_i[S | sender selected] = E[S^2] - (E[S])^2, computed pointwise per edge.
AVG_sq = (sq_accum / denom).masked_fill(zero_mask, 0.0)
VAR   = (AVG_sq - AVG * AVG).clamp(min=0.0)  # numerical safety
del AVG_sq
# Per-edge AARV: average rank shift over tokens where sender fired.
AARV  = (aarv_accum / denom).masked_fill(zero_mask, 0.0)

suffix = f"_s{args.seed}" if args.seed is not None else ""
out_path = os.path.join(output_dir, f"dag_{args.dataset}{suffix}.pt")
torch.save({
    "APS":   APS.cpu(),                         # [c, j, l, n]
    "ANS":   ANS.cpu(),                         # [c, j, l, n]
    "AVG":  AVG.cpu(),                          # [c, j, l, n]
    "VAR": VAR.cpu(),                           # [c, j, l, n] — Var_i[S | sender selected]
    "AARV":  AARV.cpu(),                        # [c, j, l, n] — AVG rank shift of receiver n
    "count": count.cpu(),                       # [c, j]
    "n_prompts": N_PROMPTS,
    "max_tokens": MAX_TOKENS,
    "model": MODEL_ID,
    "dataset": args.dataset,
    "seed": args.seed,
}, out_path)
print(f"Saved {out_path}")
