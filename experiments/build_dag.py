"""DAG builder for a chosen MoE model + dataset.

Computes the influence-edge weight — softmax-mass perturbation under pairwise-isolated
ablation. With S(g^{l,n}, e^{c,j}_{out,i}) denoting the per-edge sub-score from the
score decomposition (cf. main.tex §2):

    W_softmax(c, j → l, n) = E_i [ | p_orig(n) − p_pert_{c→n}(n) | | (c,j) selected ]

Pairwise-isolated ablation:
    To compute the edge (c, j) → (l, n), we subtract sender (c, j)'s contribution
    ONLY from receiver n's score at layer l (NOT from other receivers' scores).
    The new top-K and renormalised softmax are then computed over this per-pair
    perturbed score vector. So for each (sender, receiver) pair we have its OWN
    perturbed network state — the edge weight is uncontaminated by the sender's
    side effects on other receivers at the same layer.

where p_orig(n)/p_pert_{c→n}(n) is expert n's routing weight at receiver layer l in
the original / pair-isolated-ablated routing scores (with p = 0 outside the selected
set), computed by the model's own p_fn (experiments/routing_variants.py) — renormalized
top-K softmax by default, or a model-specific variant (no renormalization, SparseMixer,
group-limited routing, scaled routing weight) per main.tex's "Architecture-specific
variations". p_orig and p_pert always use the same p_fn, so a model's routing rule is
encoded once. W_softmax is dimensionless in [0, 1] and cross-model comparable by
construction.

K is the model's top_k.

Also computed per sender expert / per vertex:
    n_tokens_selected[c, j]               = #tokens where (c,j) ∈ top-K at layer c
    top_weight/top_prompt/top_pos/top_token = K_TOP_TOKENS (=100) highest-routing-weight
        token events per sender (c, j); used downstream to inspect what kind of
        tokens "super experts" specialise in.
    act[l, n]                             = max over tokens routed to (l,n) of
        || down_proj_output(l, n) ||_∞    (Su et al., ICLR 2026: super-expert metric).

Usage:
    python experiments/build_dag.py --model {olmoe,deepseek-v2-lite,...} --dataset {c4,...} --n_prompts 500

Output: {result_path}/dags/{dataset}/dag_{model}_{dataset}.pt
"""
import argparse
import functools
import importlib
import os
import sys
import time
from operator import attrgetter

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = config["result_path"]
os.makedirs(output_dir, exist_ok=True)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from customized_models.modeling_deepseek_customized import DeepseekV2ForCausalLM
from customized_models.modeling_mixtral_customized import MixtralForCausalLM
from customized_models.modeling_qwen3_moe_customized import Qwen3MoeForCausalLM
from customized_models.modeling_phimoe_customized import PhiMoEForCausalLM
from transformers import AutoTokenizer

from experiments.routing_variants import (
    norm_denom_rmsnorm, norm_denom_layernorm,
    ln_bar_rmsnorm, ln_bar_layernorm,
    p_renorm_topk_softmax, p_no_renorm_softmax, p_sparsemixer,
)

# Dataset registry: name -> (module_path, helper_function_name).
# All helpers must accept (dataset_len, min_words) and return a list of strings.
DATASETS = {
    "c4":          ("dataset.c4_dataset",          "c4_dataset_helper"),
    "math":        ("dataset.math_dataset",        "open_r1_math_dataset_helper"),
    "code":        ("dataset.code_dataset",        "code_dataset_helper"),
    "wikitext2":   ("dataset.wikitext2_dataset",   "wikitext2_dataset_helper"),
    "gsm8k":       ("dataset.gsm8k_dataset",       "gsm8k_dataset_helper"),
    "humaneval":   ("dataset.humaneval_dataset",   "humaneval_dataset_helper"),
    "pile-arxiv":  ("dataset.pile_arxiv_dataset",  "pile_arxiv_dataset_helper"),
    "pile-github": ("dataset.pile_github_dataset", "pile_github_dataset_helper"),
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
        "experts_path": "mlp.experts",
        "down_proj_attr": "down_proj",
        "p_fn": p_no_renorm_softmax,       # norm_topk_prob=false
    },
    "deepseek-v2-lite": {
        "id": "deepseek-ai/DeepSeek-V2-Lite",
        "cls": DeepseekV2ForCausalLM,
        "n_experts": 64,
        "top_k": 6,
        "d_e": 2048,
        "moe_layers": list(range(1, 27)),  # layer 0 is dense
        "gate_path": "mlp.gate",
        "experts_path": "mlp.experts",
        "down_proj_attr": "down_proj",
        # norm_topk_prob=false, topk_method="greedy" (n_group=topk_group=1: no-op), routed_scaling_factor=1.0 (no-op).
        "p_fn": p_no_renorm_softmax,
        "n_shared_experts": 2,  # fused into one wider MLP; DAG gets one shared-expert vertex per layer
    },
    "mixtral-8x7b": {
        "id": "mistralai/Mixtral-8x7B-v0.1",
        "cls": MixtralForCausalLM,
        "n_experts": 8,
        "top_k": 2,
        "d_e": 4096,
        "moe_layers": list(range(32)),     # all layers are MoE
        "gate_path": "block_sparse_moe.gate",  # Mistral naming differs from OLMoE/DeepSeek
        "experts_path": "block_sparse_moe.experts",
        "down_proj_attr": "w2",            # Mixtral uses Megablocks naming (w1/w2/w3)
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
        "experts_path": "block_sparse_moe.experts",
        "down_proj_attr": "w2",
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
        "experts_path": "mlp.experts",
        "down_proj_attr": "down_proj",
        # 60GB bf16 nominally fits 1x80GB but hooks + activations push it OOM;
        # shard across 4 GPUs with plenty of headroom.
        "multi_gpu": True,
        "max_memory": {0: "15GiB", 1: "25GiB", 2: "25GiB", 3: "25GiB"},  # 90 GiB for 60GB model
    },
    "phi-3.5-moe": {
        "id": "microsoft/Phi-3.5-MoE-instruct",
        "cls": PhiMoEForCausalLM,
        "n_experts": 16,
        "top_k": 2,
        "d_e": 4096,
        "moe_layers": list(range(32)),     # all 32 layers are MoE
        "gate_path": "block_sparse_moe.gate",  # same naming as Mixtral
        "experts_path": "block_sparse_moe.experts",
        "down_proj_attr": "w2",            # Phi-3.5-MoE inherits Mixtral's w1/w2/w3 naming
        # 84GB bf16: doesn't fit 1x80GB cleanly; shard across 4 GPUs.
        "multi_gpu": True,
        "max_memory": {0: "20GiB", 1: "30GiB", 2: "30GiB", 3: "30GiB"},  # 110 GiB for 84GB model
        # post_attention_layernorm is nn.LayerNorm (with bias), not RMSNorm.
        "norm_denom_fn": norm_denom_layernorm,
        "ln_bar_fn": ln_bar_layernorm,
        "p_fn": functools.partial(p_sparsemixer, jitter_eps=0.01),  # router_jitter_noise
    }
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=list(MODELS), default="olmoe", help="Which MoE model to build the DAG for (default: olmoe).")
parser.add_argument("--dataset", choices=list(DATASETS), default="c4", help="Which dataset to build the DAG on (default: c4).")
parser.add_argument("--n_prompts", type=int, default=500, help="Number of prompts to use; capped to min(this, len(loaded_prompts)).")
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
K_TOP_TOKENS = 100  # per-sender buffer: keep top-100 routing-weight events per (c, j)

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
    cfg = MODEL["cls"].config_class.from_pretrained(MODEL_ID, trust_remote_code=True)
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# Now set the default device — our accumulators and the customized model's
# hook tensors (allocated inside its forward) will land on cuda:0.
torch.set_default_device(device)

# Load weights [L, n_experts, d_e]
# Most models expose decoder layers at `model.layers` and the post-attention norm at
# `post_attention_layernorm`; DBRX uses `transformer.blocks` and `norm_attn_norm.norm_2`.
layers_of = attrgetter(MODEL.get("layers_path", "model.layers"))
gate_of   = attrgetter(MODEL["gate_path"])
norm_of   = attrgetter(MODEL.get("norm_path", "post_attention_layernorm"))
layers    = layers_of(model)
G_recv     = torch.stack([gate_of(layers[R]).weight.detach().to(device, dtype=torch.float32) for R in MOE_LAYERS])
# [L, d_e]
gamma_recv = torch.stack([norm_of(layers[R]).weight.detach().to(device, dtype=torch.float32) for R in MOE_LAYERS])

# ---- Load dataset ----
mod_name, fn_name = DATASETS[args.dataset]
loader = getattr(importlib.import_module(mod_name), fn_name)
print(f"Loading dataset={args.dataset!r}  ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = loader(dataset_len=N_PROMPTS, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
# Cap N_PROMPTS to what the loader actually returned (e.g. HumanEval has only
# 164 prompts total, fewer after the min_words filter; without this cap the
# batching loop would slice past the list and pass [] to the tokenizer).
if len(prompts) < N_PROMPTS:
    print(f"  loader returned {len(prompts)} prompts (requested {N_PROMPTS}); capping N_PROMPTS.", flush=True)
    N_PROMPTS = len(prompts)

# ---- Accumulators ----
# wsm : softmax-mass perturbation statistic (pairwise-isolated ablation).
#       For edge (c,j) → (l,n):
#         - delta_{c,j,l,n} = p_orig(n) - p_pert_{c→n}(n)
#           where p_pert_{c→n} is the renorm(softmax_topK) of the
#           gating at receiver layer l AFTER subtracting only the
#           (c,j) contribution to n's score (other receivers'
#           scores unchanged); zero outside top-K.
#         - wsm = E_i[|delta|]
#       Influence-edge weight (W_softmax).
SHAPE = (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS)
wsm_accum        = torch.zeros(SHAPE, dtype=torch.float32, device=device)
# n_tokens_selected[S, j] = #tokens where expert j was in top-K at layer S.
n_tokens_selected = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device)
# act[L, N] = max over tokens routed to (L, N) of ||down_proj_output||_∞ (Su et al.,
# ICLR 2026). Updated incrementally via the forward-hook registered just below.
act_accum = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.float32, device=device)

# wsm_shared : same softmax-mass perturbation, but for shared-expert vertices
# (main.tex "Shared experts") — unconditional edges (l, s) -> (l', n), l' > l.
# n_tokens_total is the single running token count shared by every edge (no
# per-(sender) selection to condition on). None for models without shared experts.
N_SHARED = MODEL.get("n_shared_experts")
if N_SHARED is not None:
    wsm_shared_accum = torch.zeros((N_LAYERS, N_LAYERS, N_EXPERTS), dtype=torch.float32, device=device)
    n_tokens_total    = torch.zeros(1, dtype=torch.float32, device=device)
else:
    wsm_shared_accum = None
    n_tokens_total    = None

# ---- Register down_proj forward hooks for the Su et al. activation magnitude ----
# Each expert's down_proj receives a 2-D input [K_e, intermediate] and returns
# [K_e, hidden] where K_e = #tokens routed to that expert in the current batch.
# We collapse to a per-(L, N) scalar via .abs().amax() (over both K_e and hidden,
# i.e., L_∞ over channels and max over tokens routed). If no tokens routed in a
# batch, output.numel() == 0 and the hook skips. Hooks remain active for the
# lifetime of the process; act_accum is the running max across all batches.
experts_of_module = attrgetter(MODEL["experts_path"])
down_proj_attr    = MODEL["down_proj_attr"]
_down_proj_hooks  = []
def _make_down_proj_hook(L_idx, N_idx, target):
    def _hook(_module, _inp, output):
        if output.numel() == 0:
            return
        # multi_gpu=True models (Mixtral, Phi, Qwen3-30B, …) are sharded across
        # GPUs via accelerate; the hook fires on the expert's local device while
        # act_accum lives on cuda:0. Move the scalar to the accumulator's device
        # (and dtype) before the elementwise max.
        m = output.detach().abs().amax().to(device=target.device, dtype=target.dtype)
        target[L_idx, N_idx] = torch.maximum(target[L_idx, N_idx], m)
    return _hook
for _L_idx, _R in enumerate(MOE_LAYERS):
    _experts = experts_of_module(layers[_R])
    for _N_idx, _expert_mod in enumerate(_experts):
        _dp = getattr(_expert_mod, down_proj_attr)
        _h = _dp.register_forward_hook(_make_down_proj_hook(_L_idx, _N_idx, act_accum))
        _down_proj_hooks.append(_h)
print(f"Registered {len(_down_proj_hooks)} down_proj hooks for act feature "
      f"(experts_path='{MODEL['experts_path']}', down_proj_attr='{down_proj_attr}').", flush=True)

# Per-sender top-K-by-routing-weight token buffer. Empty slots have weight = -1
# (real routing weights live in [0, 1]). Layout: [S, j, slot].
TOPK_SHAPE = (N_LAYERS, N_EXPERTS, K_TOP_TOKENS)
top_weight = torch.full(TOPK_SHAPE, -1.0, dtype=torch.float32, device=device)
top_prompt = torch.zeros(TOPK_SHAPE, dtype=torch.int32, device=device)
top_pos    = torch.zeros(TOPK_SHAPE, dtype=torch.int16, device=device)
top_token  = torch.zeros(TOPK_SHAPE, dtype=torch.int32, device=device)

from experiments.wsm_core import accumulate_wsm

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
    shared_expert_output = (
        hook_dict["hook_shared_expert_output"][:, MOE_LAYERS, :, :]                # [bsz, L, n_tok, d_e]
        if N_SHARED is not None else None
    )

    accumulate_wsm(
        after_res1, after_norm2, selected, weighted_out,
        G_recv, gamma_recv, EPS, N_LAYERS, N_EXPERTS, TOP_K, D_E, device,
        input_ids, B,
        wsm_accum, n_tokens_selected,
        top_weight, top_prompt, top_pos, top_token, K_TOP_TOKENS,
        norm_denom_fn=MODEL.get("norm_denom_fn", norm_denom_rmsnorm),
        ln_bar_fn=MODEL.get("ln_bar_fn", ln_bar_rmsnorm),
        p_fn=MODEL.get("p_fn", p_renorm_topk_softmax),
        routing_scale=MODEL.get("routing_scale", 1.0),
        shared_expert_output=shared_expert_output,
        wsm_shared_accum=wsm_shared_accum,
        n_tokens_total=n_tokens_total,
    )

    del hook_dict, after_res1, after_norm2, selected, weighted_out, shared_expert_output
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed if elapsed > 0 else 0.0
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# ---- Normalize: weight[S, j, R, n] = accum / n_tokens_selected[S, j] ----
count_safe = n_tokens_selected.clamp(min=1).to(torch.float32)          # [L, n_experts]
denom      = count_safe.view(N_LAYERS, N_EXPERTS, 1, 1)
zero_mask  = (n_tokens_selected == 0).view(N_LAYERS, N_EXPERTS, 1, 1)

W_softmax = (wsm_accum / denom).masked_fill(zero_mask, 0.0)

# ---- Normalize shared-expert edges: weight[l, l', n] = accum / n_tokens_total ----
if N_SHARED is not None:
    W_softmax_shared = wsm_shared_accum / n_tokens_total.clamp(min=1).view(1, 1, 1)

# Tear down the activation-magnitude hooks before saving.
for _h in _down_proj_hooks:
    _h.remove()

dag_dir = os.path.join(output_dir, "dags", args.dataset)
os.makedirs(dag_dir, exist_ok=True)
out_path = os.path.join(dag_dir, f"dag_{args.model}_{args.dataset}.pt")
save_dict = {
    "W_softmax":         W_softmax.cpu(),                 # [c, j, l, n] — E_i[|p_orig − p_pert_{c→n}|]
    "act":               act_accum.cpu(),                 # [l, n]       — Su et al. activation magnitude
    "n_tokens_selected": n_tokens_selected.cpu(),         # [c, j]       — #tokens routed to (c, j)
    "top_weight":        top_weight.cpu(),                # [c, j, K_TOP_TOKENS] — empty slot = -1
    "top_prompt":        top_prompt.cpu(),                # [c, j, K_TOP_TOKENS] — global prompt idx
    "top_pos":           top_pos.cpu(),                   # [c, j, K_TOP_TOKENS] — position in prompt
    "top_token":         top_token.cpu(),                 # [c, j, K_TOP_TOKENS] — token id
    "k_top_tokens":      K_TOP_TOKENS,
    "n_prompts":         N_PROMPTS,
    "max_tokens":        MAX_TOKENS,
    "model":             MODEL_ID,
    "moe_layers":        MOE_LAYERS,
    "dataset":           args.dataset,
}
if N_SHARED is not None:
    # W_softmax_shared[l, l', n] — edge from layer l's shared-expert vertex to (l', n).
    save_dict["W_softmax_shared"] = W_softmax_shared.cpu()
torch.save(save_dict, out_path)
print(f"Saved {out_path}")