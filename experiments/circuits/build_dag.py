"""DAG builder for a chosen MoE model + dataset, with multiple edge weights.

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
    python experiments/circuits/build_dag.py --model {olmoe,deepseek-v2-lite} --dataset {c4,math,code} --n-prompts 5000

Output: {result_path}/circuits/dag_{model}_{dataset}.pt

Modeled on experiments/variance/per_expert.py — same forward-pass loop and
reshapes — but keeps the receiver-expert dimension instead of collapsing via Var_n.
"""
import argparse
import importlib
import os
import sys
import time
from operator import attrgetter

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "circuits")
os.makedirs(output_dir, exist_ok=True)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from customized_models.modeling_deepseek_customized import DeepseekV2ForCausalLM
from customized_models.modeling_mixtral_customized import MixtralForCausalLM
from customized_models.modeling_qwen3_moe_customized import Qwen3MoeForCausalLM
from transformers import AutoTokenizer

# Dataset registry: name -> (module_path, helper_function_name).
# All helpers must accept (dataset_len, min_words) and return a list of strings.
DATASETS = {
    "c4":   ("dataset.c4_dataset",   "c4_dataset_helper"),
    "math": ("dataset.math_dataset", "open_r1_math_dataset_helper"),
    "code": ("dataset.code_dataset", "code_dataset_helper"),
}

# Model registry. `moe_layers` lists transformer-layer indices that have an MoE
# block; the DAG indexes its layers 0..len(moe_layers)-1 against this list.
MODELS = {
    "olmoe": {
        "id": "allenai/OLMoE-1B-7B-0924",
        "cls": OlmoeForCausalLM,
        "n_experts": 64,
        "top_k": 8,
        "d_e": 2048,
        "moe_layers": list(range(16)),
        "gate_path": "mlp.gate",
    },
    "deepseek-v2-lite": {
        "id": "deepseek-ai/DeepSeek-V2-Lite",
        "cls": DeepseekV2ForCausalLM,
        "n_experts": 64,
        "top_k": 6,
        "d_e": 2048,
        "moe_layers": list(range(1, 27)),  # layer 0 is dense
        "gate_path": "mlp.gate",
    },
    "mixtral-8x7b": {
        "id": "mistralai/Mixtral-8x7B-v0.1",
        "cls": MixtralForCausalLM,
        "n_experts": 8,
        "top_k": 2,
        "d_e": 4096,
        "moe_layers": list(range(32)),     # all layers are MoE
        "gate_path": "block_sparse_moe.gate",  # Mistral naming differs from OLMoE/DeepSeek
        "multi_gpu": True,                  # ~94GB bf16: needs sharding across GPUs
        "max_memory": {0: "20GiB", 1: "30GiB", 2: "30GiB", 3: "30GiB"},
    },
    "mixtral-8x22b": {
        "id": "mistralai/Mixtral-8x22B-v0.1",
        "cls": MixtralForCausalLM,         # same class as 8x7B; only config differs
        "n_experts": 8,
        "top_k": 2,
        "d_e": 6144,
        "moe_layers": list(range(56)),     # all layers are MoE
        "gate_path": "block_sparse_moe.gate",
        "multi_gpu": True,                  # ~282GB bf16: tight on 4x80GB
        "max_memory": {0: "60GiB", 1: "78GiB", 2: "78GiB", 3: "78GiB"},  # 294 GiB total
    },
    "qwen3-30b-a3b": {
        "id": "Qwen/Qwen3-30B-A3B",
        "cls": Qwen3MoeForCausalLM,
        "n_experts": 128,
        "top_k": 8,
        "d_e": 2048,
        "moe_layers": list(range(48)),     # all layers are MoE (mlp_only_layers=[])
        "gate_path": "mlp.gate",
        # 60 GB bf16 fits on a single 80GB A100; no multi_gpu flag needed.
    },
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=list(MODELS), default="olmoe", help="Which MoE model to build the DAG for (default: olmoe).")
parser.add_argument("--dataset", choices=list(DATASETS), default="c4", help="Which dataset to build the DAG on (default: c4).")
parser.add_argument("--n_prompts", type=int, default=5000, help="Number of prompts to use).")
parser.add_argument("--B", type=int, default=32, help="Batch size (lower if you OOM; default 32).")
args = parser.parse_args()

device = "cuda:0"
torch.set_grad_enabled(False)
# NOTE: torch.set_default_device(device) is deferred until AFTER from_pretrained;
# setting it before can pin the model skeleton to cuda:0 and break device_map="auto".

MODEL = MODELS[args.model]
MODEL_ID   = MODEL["id"]
MOE_LAYERS = MODEL["moe_layers"]
N_LAYERS   = len(MOE_LAYERS)
N_EXPERTS  = MODEL["n_experts"]
D_E        = MODEL["d_e"]
TOP_K      = MODEL["top_k"]
EPS = 1e-5

N_PROMPTS = args.n_prompts
BSZ = args.B
MAX_TOKENS = 32

print(f"Building DAG for model={args.model!r}, dataset={args.dataset!r}, {N_PROMPTS} prompts.", flush=True)

# ---- Load model + tokenizer ----
# For models too large for a single GPU (multi_gpu=True), use device_map="auto"
# so accelerate shards layers across visible GPUs. Hook tensors are pre-allocated
# on cuda:0 (via torch.set_default_device above); writes from off-device layers
# rely on implicit PyTorch cross-device copies.
print(f"Loading {MODEL_ID} ...", flush=True)
print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}", flush=True)
print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
t0 = time.time()
load_kwargs = dict(attn_implementation="eager", torch_dtype=torch.bfloat16)
if MODEL.get("multi_gpu", False):
    try:
        import accelerate
        from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
        import transformers
        print(f"  accelerate={accelerate.__version__}  transformers={transformers.__version__}", flush=True)
    except ImportError:
        raise RuntimeError("multi_gpu=True requires `accelerate`: pip install accelerate")

    # `from_pretrained(..., device_map=...)` silently fell back to CPU for the
    # customized model class. Workaround: plan with infer_auto_device_map, then
    # do the actual placement ourselves via dispatch_model.

    # Step 1: plan on a meta-device skeleton.
    print("  building empty model on meta device ...", flush=True)
    cfg = MODEL["cls"].config_class.from_pretrained(MODEL_ID)
    with init_empty_weights():
        empty_model = MODEL["cls"](cfg)
    no_split = empty_model._no_split_modules
    # Use per-model max_memory if declared; else fall back to a single-GPU budget.
    # GPU 0 typically gets a smaller share (it also hosts hook tensors).
    max_mem = MODEL.get("max_memory", {0: "75GiB"})
    computed_map = infer_auto_device_map(
        empty_model,
        max_memory=max_mem,
        no_split_module_classes=no_split,
        dtype=torch.bfloat16,
    )
    print(f"  computed device_map: {computed_map}", flush=True)
    del empty_model

    # Step 2: load weights normally (CPU), then physically dispatch ourselves.
    print("  loading checkpoint to CPU ...", flush=True)
    load_kwargs["low_cpu_mem_usage"] = True
    model = MODEL["cls"].from_pretrained(MODEL_ID, **load_kwargs).eval()
    print(f"  pre-dispatch first-param device = {next(model.parameters()).device}", flush=True)
    print("  dispatching to GPUs ...", flush=True)
    model = dispatch_model(model, device_map=computed_map)
    print(f"  hf_device_map = {getattr(model, 'hf_device_map', '<not present>')}", flush=True)
    print(f"  post-dispatch first-param device = {next(model.parameters()).device}", flush=True)
else:
    model = MODEL["cls"].from_pretrained(MODEL_ID, **load_kwargs).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# Now set the default device — our accumulators and the customized model's
# hook tensors (allocated inside its forward) will land on cuda:0.
torch.set_default_device(device)

# Load weights [L, n_experts, d_e]
gate_of = attrgetter(MODEL["gate_path"])   # e.g., "mlp.gate" or "block_sparse_moe.gate"
G_recv = torch.stack([gate_of(model.model.layers[R]).weight.detach().to(device, dtype=torch.float32) for R in MOE_LAYERS])
# [L, d_e]
gamma_recv = torch.stack([model.model.layers[R].post_attention_layernorm.weight.detach().to(device, dtype=torch.float32) for R in MOE_LAYERS])

# ---- Load dataset ----
mod_name, fn_name = DATASETS[args.dataset]
loader = getattr(importlib.import_module(mod_name), fn_name)
print(f"Loading dataset={args.dataset!r}  ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = loader(dataset_len=N_PROMPTS, min_words=MAX_TOKENS)
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
# aarv_accum[S, j, R, n] = Σ_{i : j ∈ top-K at S} | rank_R^orig(n) - rank_R^pert(n) | where pert_score = orig_score - S(g^{R,n}, e^{S,j}_{out,i}).
aarv_accum = torch.zeros(SHAPE, dtype=torch.float32, device=device)
# count[S, j] = number of (token, top-k slot) events where expert j was selected at layer S.
count = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device)

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches (batch_size={BSZ}, max_tokens={MAX_TOKENS}) ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    _, hook_dict = model(input_ids=input_ids, attention_mask=attention_mask)

    # Slice to MoE layers only. For models with dense layers (e.g., DeepSeek-V2-Lite
    # has a dense layer 0), the routing-related hook slots at non-MoE layers are
    # uninitialized memory and must not be read.
    after_res1   = hook_dict["hook_after_res1"][:, MOE_LAYERS, :, :]                # [bsz, L, n_tok, d_e]
    after_norm2  = hook_dict["hook_after_norm2"][:, MOE_LAYERS, :, :]               # [bsz, L, n_tok, d_e]
    selected     = hook_dict["hook_selected_experts"][:, MOE_LAYERS, :, :]          # [bsz, L, n_tok, top_k]
    weighted_out = hook_dict["hook_expert_weighted_outputs"][:, MOE_LAYERS, :, :, :]   # [bsz, L, n_tok, top_k, d_e]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # 1 / RMS^R_i, per token, per receiver layer R.
    rms_sq  = after_res1.float().pow(2).mean(dim=-1) + EPS                # [bsz, L, n_tok]
    rms_inv = torch.rsqrt(rms_sq).permute(0, 2, 1).reshape(bt, N_LAYERS)  # [bt, L_recv]

    # Original assignment scores at every receiver layer (for AARV computation).
    after_norm2_r = (after_norm2.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E))   # [bt, L, d_e]
    orig_score = torch.einsum("lnd,bld->bln", G_recv, after_norm2_r)       # [bt, L, N_EXPERTS]
    orig_sorted = torch.argsort(orig_score, dim=-1, descending=True)       # [bt, L, N_EXPERTS]
    orig_rank_of = torch.empty_like(orig_sorted)
    orig_rank_of.scatter_(-1, orig_sorted, torch.arange(N_EXPERTS, device=device).expand_as(orig_sorted))
    # orig_rank_of[bt, l, n] = position of expert n in the layer-l ordering.

    # Sender-side reshapes: [bt, S, k, ...]
    omega = (weighted_out.float().permute(0, 2, 1, 3, 4).reshape(bt, N_LAYERS, TOP_K, D_E))   # [bt, S, k, d_e]
    sel = (selected.long().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, TOP_K))                  # [bt, S, k]

    for S in range(N_LAYERS):
        sel_S = sel[:, S, :]                                    # [bt, top_k]
        count[S] += torch.bincount(sel_S.flatten(), minlength=N_EXPERTS)

        if S == N_LAYERS - 1:
            continue
        omega_S = omega[:, S, :, :]                             # [bt, top_k, d_e]

        for R in range(S + 1, N_LAYERS):
            # ln_bar^R(omega_S) = (omega_S ⊙ γ^R) / RMS^R_i
            ln_bar = omega_S * gamma_recv[R].view(1, 1, D_E) * rms_inv[:, R].view(bt, 1, 1)     # [bt, k, d_e]
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
            rank_shift = (pert_rank_of.float() - orig_rank_R.float()).abs()             # [bt, k, N_EXPERTS]
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

out_path = os.path.join(output_dir, f"dag_{args.model}_{args.dataset}.pt")
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
    "moe_layers": MOE_LAYERS,
    "dataset": args.dataset
}, out_path)
print(f"Saved {out_path}")
