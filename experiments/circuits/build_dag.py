"""DAG builder for OLMoE on a chosen dataset, with multiple edge weights.

Computes four edge weights conditioned on sender selection:

    APS(c, j → l, n)   = E_i [ max(S(g^{l,n}, e^{c,j}_{out,i}), 0) | (c,j) selected ]
    ANS(c, j → l, n)   = E_i [ min(S(g^{l,n}, e^{c,j}_{out,i}), 0) | (c,j) selected ]
    mean(c, j → l, n)  = E_i [     S(g^{l,n}, e^{c,j}_{out,i})     | (c,j) selected ]
    V_tok(c, j → l, n) = Var_i [   S(g^{l,n}, e^{c,j}_{out,i})     | (c,j) selected ]

where S(g^{l,n}, e^{c,j}_{out,i}) = g^{l,n} · LN_bar^l_i(r^{c,j}(i) · e^{c,j}_{out,i})
is the per-edge sub-score from the score decomposition (cf. MoEs/main.tex §2).
V_tok captures per-edge consistency across tokens: low V_tok = stable, reliable
edge; high V_tok = oscillating contribution.

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
# mean_accum[S, j, R, n]  = Σ_{i : j ∈ top-K at S}     S(g^{R,n}, e^{S,j}_{out,i})
# sq_accum[S, j, R, n]    = Σ_{i : j ∈ top-K at S}     S(g^{R,n}, e^{S,j}_{out,i})^2
# (Var_i is computed at the end as E[S^2] - E[S]^2.)
SHAPE = (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS)
APS_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device)
ANS_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device)
mean_accum = torch.zeros(SHAPE, dtype=torch.float32, device=device)
sq_accum   = torch.zeros(SHAPE, dtype=torch.float32, device=device)
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

            # APS = positive part, ANS = negative part, mean = signed, sq = squared.
            # Accumulate over (sender expert, receiver expert).
            scores_pos = scores.clamp(min=0.0)                  # [bt, k, N_EXPERTS]
            scores_neg = scores.clamp(max=0.0)                  # [bt, k, N_EXPERTS]
            scores_sq  = scores * scores                        # [bt, k, N_EXPERTS]
            sel_flat = sel_S.flatten()
            APS_accum[S, :, R, :].index_add_(0, sel_flat, scores_pos.flatten(0, 1))
            ANS_accum[S, :, R, :].index_add_(0, sel_flat, scores_neg.flatten(0, 1))
            mean_accum[S, :, R, :].index_add_(0, sel_flat, scores.flatten(0, 1))
            sq_accum[S, :, R, :].index_add_(0, sel_flat, scores_sq.flatten(0, 1))

            del ln_bar, scores, scores_pos, scores_neg, scores_sq

    del hook_dict, after_res1, selected, weighted_out
    del omega, sel, rms_sq, rms_inv
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
mean  = (mean_accum / denom).masked_fill(zero_mask, 0.0)
# Var_i[S | sender selected] = E[S^2] - (E[S])^2, computed pointwise per edge.
mean_sq = (sq_accum / denom).masked_fill(zero_mask, 0.0)
V_tok   = (mean_sq - mean * mean).clamp(min=0.0)  # numerical safety
del mean_sq

suffix = f"_s{args.seed}" if args.seed is not None else ""
out_path = os.path.join(output_dir, f"dag_{args.dataset}{suffix}.pt")
torch.save({
    "APS":   APS.cpu(),                         # [c, j, l, n]
    "ANS":   ANS.cpu(),                         # [c, j, l, n]
    "mean":  mean.cpu(),                        # [c, j, l, n]
    "V_tok": V_tok.cpu(),                       # [c, j, l, n] — Var_i[S | sender selected]
    "count": count.cpu(),                       # [c, j]
    "n_prompts": N_PROMPTS,
    "max_tokens": MAX_TOKENS,
    "model": MODEL_ID,
    "dataset": args.dataset,
    "seed": args.seed,
}, out_path)
print(f"Saved {out_path}")

# ---- Quick sanity prints ----
print("\n==================== Step 2 sanity stats ====================\n")

def decode_flat(i):
    c = i // (N_EXPERTS * N_LAYERS * N_EXPERTS)
    rest = i % (N_EXPERTS * N_LAYERS * N_EXPERTS)
    j = rest // (N_LAYERS * N_EXPERTS)
    rest2 = rest % (N_LAYERS * N_EXPERTS)
    l = rest2 // N_EXPERTS
    n = rest2 % N_EXPERTS
    return c, j, l, n

for label, T in [("APS", APS), ("ANS", ANS), ("mean", mean)]:
    print(f"--- {label} ---")
    print(f"  shape: {tuple(T.shape)}")
    print(f"  nonzero entries: {(T != 0).sum().item()} / {T.numel()}")
    print(f"  max:  {T.max().item():+.4e}    min: {T.min().item():+.4e}")
    nz = T[T != 0]
    if nz.numel() > 0:
        print(f"  mean (over nonzero): {nz.mean().item():+.4e}")

# Top-10 edges by APS, ANS magnitude, mean magnitude.
def show_top10(T, label, key):
    print(f"\nTop-10 edges by {label} ({key}):")
    if key == "max":
        vals, idx = torch.topk(T.reshape(-1), 10)
    elif key == "min":
        vals, idx = torch.topk(-T.reshape(-1), 10)
        vals = -vals
    elif key == "abs":
        vals, idx = torch.topk(T.abs().reshape(-1), 10)
        vals = T.reshape(-1)[idx]
    for v, i in zip(vals.tolist(), idx.tolist()):
        c, j, l, n = decode_flat(i)
        print(f"  M{c}E{j} → M{l}E{n}   {label} = {v:+.4e}")

show_top10(APS,  "APS",  "max")
show_top10(ANS,  "ANS",  "min")
show_top10(mean, "mean", "abs")

# Spot-check the known M1E9 → M4E14 chain from prior paper.
print(f"\nSpot check  M1E9 → M4E14")
print(f"   APS   = {APS[1, 9, 4, 14].item():+.4e}")
print(f"   ANS   = {ANS[1, 9, 4, 14].item():+.4e}")
print(f"   mean  = {mean[1, 9, 4, 14].item():+.4e}")
print(f"   V_tok = {V_tok[1, 9, 4, 14].item():.4e}    (lower = more consistent edge)")

# Aggregate APS by sender layer — should reproduce M1, M4 stripe pattern from Fig 3d.
print("\nAPS aggregated by sender layer (should peak at M1 and M4):")
aps_by_sender = APS.sum(dim=(1, 2, 3)).cpu().numpy()                   # [L]
for c in range(N_LAYERS):
    bar = "#" * int(40 * aps_by_sender[c] / max(aps_by_sender.max(), 1e-30))
    print(f"  M{c:2d}  {aps_by_sender[c]:.4e}  {bar}")

print("\nANS aggregated by sender layer (inhibition; expect spread):")
ans_by_sender = (-ANS).sum(dim=(1, 2, 3)).cpu().numpy()                # [L], magnitude
for c in range(N_LAYERS):
    bar = "#" * int(40 * ans_by_sender[c] / max(ans_by_sender.max(), 1e-30))
    print(f"  M{c:2d}  {ans_by_sender[c]:.4e}  {bar}")
