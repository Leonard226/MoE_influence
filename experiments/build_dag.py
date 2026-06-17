"""DAG builder for a chosen MoE model + dataset, with multiple edge weights.

Computes per-edge weights conditioned on sender selection. With S(g^{l,n}, e^{c,j}_{out,i})
denoting the per-edge sub-score from the score decomposition (cf. main.tex §2):

    APS(c, j → l, n)    = E_i  [ max(S, 0)                                       | (c,j) selected ]
    ANS(c, j → l, n)    = E_i  [ min(S, 0)                                       | (c,j) selected ]
    AARV(c, j → l, n)   = E_i  [ | rank_l^orig(n) − rank_l^pert(n) |             | (c,j) selected ]
    P_add(c, j → l, n)  = P_i  [ rank_l^orig(n) > K-1  ∧  rank_l^pert(n) ≤ K-1   | (c,j) selected ]
    P_rem(c, j → l, n)  = P_i  [ rank_l^orig(n) ≤ K-1  ∧  rank_l^pert(n) > K-1   | (c,j) selected ]

and the new softmax-mass perturbation family (PRIMARY edge weight for the influence DAG):

    W_softmax(c, j → l, n)        = E_i  [ | p_orig(n) − p_pert(n) |             | (c,j) selected ]
    W_softmax_var(c, j → l, n)    = Var_i[ | p_orig(n) − p_pert(n) |             | (c,j) selected ]
    W_softmax_signed(c, j → l, n) = E_i  [ p_pert(n) − p_orig(n)                 | (c,j) selected ]

where p_orig(n)/p_pert(n) is the renormalised top-K-softmax gating weight of expert n
at receiver layer l in the original / sender-ablated routing scores (with p=0 outside
the top-K). W_softmax is dimensionless in [0, 1] and cross-model comparable by
construction. W_softmax_signed > 0 ⇒ ablating (c,j) raises n's mass ⇒ (c,j) suppresses n.

For Phi-3.5-MoE the model actually uses `sparsemixer` rather than top-K softmax; we
override that with the cross-model unified renorm(softmax_topK) formula here so the
edge-weight definition is identical across all models.

rank_l^pert is the receiver-rank after ablating the sender's score contribution; K is the
model's top_k.

Also computed per sender expert / per vertex:
    n_tokens_selected[c, j]               = #tokens where (c,j) ∈ top-K at layer c
    top_weight/top_prompt/top_pos/top_token = K_TOP_TOKENS (=100) highest-routing-weight
        token events per sender (c, j); used downstream to inspect what kind of
        tokens "super experts" specialise in.
    act[l, n]                             = max over tokens routed to (l,n) of
        || down_proj_output(l, n) ||_∞    (Su et al., ICLR 2026: super-expert metric).

Usage:
    python experiments/build_dag.py --model {olmoe,deepseek-v2-lite,...} --dataset {c4,...} --n_prompts 500

Output: {result_path}/circuits/dag_{model}_{dataset}.pt
"""
import argparse
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
output_dir = os.path.join(config["result_path"], "circuits")
os.makedirs(output_dir, exist_ok=True)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from customized_models.modeling_deepseek_customized import DeepseekV2ForCausalLM
from customized_models.modeling_mixtral_customized import MixtralForCausalLM
from customized_models.modeling_qwen3_moe_customized import Qwen3MoeForCausalLM
from customized_models.modeling_phimoe_customized import PhiMoEForCausalLM
from customized_models.modeling_dbrx_customized import DbrxForCausalLM
from transformers import AutoTokenizer

# Dataset registry: name -> (module_path, helper_function_name).
# All helpers must accept (dataset_len, min_words) and return a list of strings.
DATASETS = {
    # Existing (5,000 prompts).
    "c4":          ("dataset.c4_dataset",          "c4_dataset_helper"),
    "math":        ("dataset.math_dataset",        "open_r1_math_dataset_helper"),
    "code":        ("dataset.code_dataset",        "code_dataset_helper"),
    # New (planned for 1,000 prompts; HumanEval caps at 164). Grouped by
    # category: natural language, math, code, scientific notation.
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
    "qwen3-235b-a22b": {
        # Same modeling class as Qwen3-30B-A3B; only config differs (94 layers, 4096 hidden).
        # 470GB bf16 doesn't fit 4x80GB single-node. int8 (~235GB) hit a loading-peak OOM
        # because bnb's loader holds fp16 on-device before quantizing. NF4 (4-bit double-quant)
        # halves the footprint to ~118GB so the loading peak fits per GPU. Router gate and
        # lm_head are kept in bf16 so the score-decomposition reads clean weights.
        "id": "Qwen/Qwen3-235B-A22B",
        "cls": Qwen3MoeForCausalLM,
        "n_experts": 128,
        "top_k": 8,
        "d_e": 4096,
        "moe_layers": list(range(94)),
        "gate_path": "mlp.gate",
        "experts_path": "mlp.experts",
        "down_proj_attr": "down_proj",
        "quantization": "nf4",
        "bnb_skip_modules": ["gate", "lm_head"],
        # Even distribution: accelerate's NF4 loading peaks at ~2.5× max_memory per GPU.
        # 30 GiB × 2.5 = 75 GiB peak, fits 80 GiB. Total 120 GiB > 118 GiB NF4 model.
        "max_memory": {0: "30GiB", 1: "30GiB", 2: "30GiB", 3: "30GiB"},
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
    },
    "dbrx": {
        "id": "alpindale/dbrx-instruct",
        "cls": DbrxForCausalLM,
        "n_experts": 16,
        "top_k": 4,
        "d_e": 6144,
        "moe_layers": list(range(40)),       # all 40 layers are MoE
        # DBRX has a different class hierarchy: blocks under transformer, FFN wraps router/experts,
        # and uses a NormAttentionNorm wrapper instead of separate input/post_attention_layernorm.
        "layers_path": "transformer.blocks",
        "gate_path": "ffn.router.layer",
        "norm_path": "norm_attn_norm.norm_2",
        # 264GB bf16: needs 4 GPUs single-node (same as Mixtral-8x22B).
        "multi_gpu": True,
        "max_memory": {0: "55GiB", 1: "70GiB", 2: "70GiB", 3: "70GiB"},  # 265 GiB for 264GB model
    },
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=list(MODELS), default="olmoe", help="Which MoE model to build the DAG for (default: olmoe).")
parser.add_argument("--dataset", choices=list(DATASETS), default="c4", help="Which dataset to build the DAG on (default: c4).")
parser.add_argument("--n_prompts", type=int, default=500, help="Number of prompts to use; capped to min(this, len(loaded_prompts)).")
parser.add_argument("--B", type=int, default=32, help="Batch size (lower if you OOM; default 32).")
parser.add_argument("--random-init", action="store_true",
                    help="Build the model with architecture-default init (Kaiming/scaled-normal/...) "
                         "instead of loading pretrained weights. Used for the null-baseline experiment. "
                         "Tokenizer is still loaded from MODEL_ID (we still tokenise real text).")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for --random-init weight initialisation. Recorded in the output "
                         "filename as dag_<model>_<dataset>_rand_s{seed}.pt. Ignored when --random-init "
                         "is not set.")
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

if args.random_init:
    # ------------------------------------------------------------------
    # Null-baseline path: build the model with the architecture's default
    # init scheme (no pretrained weights loaded). For multi_gpu=True models
    # we still shard across GPUs via accelerate's dispatch_model; for
    # single-GPU models we just instantiate on the target device.
    # ------------------------------------------------------------------
    print(f"  random-init mode: seed={args.seed}", flush=True)
    cfg = MODEL["cls"].config_class.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg.torch_dtype = torch.bfloat16

    # Seed RNG before any tensor allocation so the init is deterministic.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if MODEL.get("multi_gpu", False):
        try:
            from accelerate import dispatch_model, init_empty_weights, infer_auto_device_map
        except ImportError:
            raise RuntimeError("--random-init for multi_gpu models requires accelerate")

        # Step 1: plan device map on a meta-device skeleton.
        print("  planning device map on meta skeleton ...", flush=True)
        with init_empty_weights():
            empty_model = MODEL["cls"](cfg)
        no_split = empty_model._no_split_modules
        max_mem = MODEL.get("max_memory", {0: "75GiB"})
        computed_map = infer_auto_device_map(
            empty_model, max_memory=max_mem,
            no_split_module_classes=no_split, dtype=torch.bfloat16,
        )
        print(f"  computed device_map: {computed_map}", flush=True)
        del empty_model

        # Step 2: build the actual model on CPU directly in bf16. Setting the
        # default dtype before construction halves peak CPU RAM (no fp32
        # allocation followed by a bf16 cast) and shaves several minutes for
        # 47B-param models. PreTrainedModel.__init__ calls _init_weights under
        # the RNG we seeded above, so no second apply pass is needed.
        print(f"  materializing model on CPU directly in bf16 "
              f"(seed={args.seed}) ...", flush=True)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MODEL["cls"](cfg).eval()
        finally:
            torch.set_default_dtype(prev_dtype)

        print("  dispatching to GPUs ...", flush=True)
        model = dispatch_model(model, device_map=computed_map)
        print(f"  hf_device_map = {getattr(model, 'hf_device_map', '<not present>')}", flush=True)
    else:
        # Single-GPU random-init.
        print(f"  building model on {device} with architecture-default init "
              f"(seed={args.seed}) ...", flush=True)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MODEL["cls"](cfg).to(device).eval()
        finally:
            torch.set_default_dtype(prev_dtype)
elif MODEL.get("quantization") in ("int8", "nf4"):
    # Bitsandbytes quantization. HF handles device_map automatically; we skip the
    # manual init_empty_weights / dispatch_model dance. Router gate and lm_head are
    # in the skip list so the score-decomposition reads them in their native dtype.
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise RuntimeError("quantization requires bitsandbytes: pip install bitsandbytes")
    quant = MODEL["quantization"]
    skip_modules = MODEL.get("bnb_skip_modules", ["lm_head"])
    print(f"  loading with {quant} quantization (skip_modules={skip_modules})", flush=True)
    if quant == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip_modules,
        )
    else:  # nf4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_modules,  # name says int8 but applies to 4-bit too
        )
    load_kwargs.pop("torch_dtype", None)   # bnb manages dtype internally
    load_kwargs["quantization_config"] = bnb_config
    load_kwargs["device_map"] = "auto"
    if "max_memory" in MODEL:
        load_kwargs["max_memory"] = MODEL["max_memory"]
    model = MODEL["cls"].from_pretrained(MODEL_ID, **load_kwargs).eval()
    print(f"  hf_device_map = {getattr(model, 'hf_device_map', '<not present>')}", flush=True)
    print(f"  first-param device = {next(model.parameters()).device}", flush=True)
elif MODEL.get("multi_gpu", False):
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
# APS/ANS               : positive / negative parts of the per-edge sub-score.
# aarv_accum            : |rank shift| after sender-score ablation.
# padd/prem_accum       : indicator counts for the receiver crossing the top-K boundary.
# wsm/wsm_sq/wsm_signed : softmax-mass perturbation statistics. wsm = E_i[|delta|],
#                         wsm_sq = E_i[delta^2] (used to derive Var since |delta|^2 = delta^2),
#                         wsm_signed = E_i[p_pert - p_orig]. delta = p_orig(v) - p_pert(v),
#                         where p_*(v) is renorm(softmax_topK) of the gating at receiver
#                         layer R; zero outside top-K. PRIMARY influence-edge weight.
SHAPE = (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS)
APS_accum        = torch.zeros(SHAPE, dtype=torch.float32, device=device)
ANS_accum        = torch.zeros(SHAPE, dtype=torch.float32, device=device)
aarv_accum       = torch.zeros(SHAPE, dtype=torch.float32, device=device)
padd_accum       = torch.zeros(SHAPE, dtype=torch.float32, device=device)
prem_accum       = torch.zeros(SHAPE, dtype=torch.float32, device=device)
wsm_accum        = torch.zeros(SHAPE, dtype=torch.float32, device=device)
wsm_sq_accum     = torch.zeros(SHAPE, dtype=torch.float32, device=device)
wsm_signed_accum = torch.zeros(SHAPE, dtype=torch.float32, device=device)
# n_tokens_selected[S, j] = #tokens where expert j was in top-K at layer S.
n_tokens_selected = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device)
# act[L, N] = max over tokens routed to (L, N) of ||down_proj_output||_∞ (Su et al.,
# ICLR 2026). Updated incrementally via the forward-hook registered just below.
act_accum = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.float32, device=device)

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

from experiments.helper import update_topk_per_sender

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

    # Routing weights actually applied in the forward pass: softmax over all
    # experts, gather the top-K, then L1-renormalize (norm_topk_prob = true on
    # all seven models we analyze).
    all_softmax       = torch.softmax(orig_score, dim=-1)                                     # [bt, L, n_experts]
    selected_softmax  = torch.gather(all_softmax, dim=-1, index=sel)                          # [bt, L, top_k]
    routing_weight    = selected_softmax / selected_softmax.sum(dim=-1, keepdim=True)         # [bt, L, top_k]
    del all_softmax, selected_softmax

    # Per-event auxiliary indices (layer-independent), used by the top-K buffer update.
    event_bt   = torch.arange(bt, device=device).repeat_interleave(TOP_K)                     # [bt*top_k]
    event_bsz  = event_bt // n_tok
    event_pos  = (event_bt % n_tok).to(torch.int16)                                           # [bt*top_k]
    prompt_indices = torch.arange(B, B + bsz, dtype=torch.int32, device=device)               # [bsz]
    event_prompt   = prompt_indices[event_bsz]                                                # [bt*top_k] int32
    event_token    = input_ids.flatten()[event_bt].to(torch.int32)                            # [bt*top_k] int32

    for S in range(N_LAYERS):
        sel_S = sel[:, S, :]                                    # [bt, top_k]
        n_tokens_selected[S] += torch.bincount(sel_S.flatten(), minlength=N_EXPERTS)

        # Update top-K-by-routing-weight token buffer for sender (S, j).
        update_topk_per_sender(
            top_weight[S], top_prompt[S], top_pos[S], top_token[S],
            sel_S.flatten(), routing_weight[:, S, :].flatten(),
            event_prompt, event_pos, event_token,
            N_EXPERTS, K_TOP_TOKENS, max_per_j=bt,
        )

        if S == N_LAYERS - 1:
            continue
        omega_S = omega[:, S, :, :]                             # [bt, top_k, d_e]

        for R in range(S + 1, N_LAYERS):
            # ln_bar^R(omega_S) = (omega_S ⊙ γ^R) / RMS^R_i
            ln_bar = omega_S * gamma_recv[R].view(1, 1, D_E) * rms_inv[:, R].view(bt, 1, 1)     # [bt, k, d_e]
            # scores[bt, k, n] = g^{R,n} · ln_bar  — per-edge sub-score
            #   S(g^{R,n}, e^{S,j_k}_{out,i})  with j_k = sel_S[bt, k].
            scores = torch.einsum("ed,bkd->bke", G_recv[R], ln_bar)  # [bt, k, N_EXPERTS]

            # APS = positive part, ANS = negative part. Accumulate over (sender, receiver).
            scores_pos = scores.clamp(min=0.0)                  # [bt, k, N_EXPERTS]
            scores_neg = scores.clamp(max=0.0)                  # [bt, k, N_EXPERTS]
            sel_flat = sel_S.flatten()
            APS_accum[S, :, R, :].index_add_(0, sel_flat, scores_pos.flatten(0, 1))
            ANS_accum[S, :, R, :].index_add_(0, sel_flat, scores_neg.flatten(0, 1))

            # Causal-routing features: ablate sender's contribution at layer R,
            # re-rank receivers, derive AARV / P_add / P_rem from the shift.
            pert_score = orig_score[:, R, :].unsqueeze(1) - scores               # [bt, k, N_EXPERTS]
            pert_sorted = torch.argsort(pert_score, dim=-1, descending=True)     # [bt, k, N_EXPERTS]
            pert_rank_of = torch.empty_like(pert_sorted)
            pert_rank_of.scatter_(-1, pert_sorted,
                                  torch.arange(N_EXPERTS, device=device).expand_as(pert_sorted))
            orig_rank_R = orig_rank_of[:, R, :].unsqueeze(1).expand_as(pert_rank_of)   # [bt, k, N_EXPERTS]
            aarv = (orig_rank_R.float() - pert_rank_of.float()).abs()
            in_topk_orig = (orig_rank_R <= TOP_K - 1)
            in_topk_pert = (pert_rank_of <= TOP_K - 1)
            padd = (~in_topk_orig &  in_topk_pert).float()
            prem = ( in_topk_orig & ~in_topk_pert).float()

            aarv_accum[S, :, R, :].index_add_(0, sel_flat, aarv.flatten(0, 1))
            padd_accum[S, :, R, :].index_add_(0, sel_flat, padd.flatten(0, 1))
            prem_accum[S, :, R, :].index_add_(0, sel_flat, prem.flatten(0, 1))

            # ---- Softmax-mass perturbation (W_softmax family) ----
            # p_orig(v) at receiver R = renormalised top-K softmax of orig_score[R];
            # zero outside top-K. Already computed at line ~395 as `routing_weight`,
            # which is renorm(softmax_topK) for layer R; scatter back to N_EXPERTS.
            p_orig_R = torch.zeros((bt, N_EXPERTS), dtype=torch.float32, device=device)
            p_orig_R.scatter_(-1, sel[:, R, :], routing_weight[:, R, :])         # [bt, N_EXPERTS]
            p_orig_R = p_orig_R.unsqueeze(1).expand(bt, TOP_K, N_EXPERTS)        # broadcast over k

            # p_pert: cross-model unified renorm(softmax_topK) over the perturbed scores.
            # softmax over all N, then top-K from pert_sorted, then L1 renorm; zero elsewhere.
            pert_softmax_all = torch.softmax(pert_score, dim=-1)                 # [bt, k, N_EXPERTS]
            pert_topk_idx    = pert_sorted[:, :, :TOP_K]                         # [bt, k, TOP_K]
            pert_topk_vals   = torch.gather(pert_softmax_all, dim=-1, index=pert_topk_idx)  # [bt, k, TOP_K]
            pert_topk_vals   = pert_topk_vals / pert_topk_vals.sum(dim=-1, keepdim=True)
            p_pert = torch.zeros_like(pert_softmax_all)
            p_pert.scatter_(-1, pert_topk_idx, pert_topk_vals)                   # [bt, k, N_EXPERTS]
            del pert_softmax_all, pert_topk_idx, pert_topk_vals

            # delta = p_orig - p_pert in [-1, 1]. wsm = E[|delta|], wsm_sq = E[delta^2]
            # (which equals E[|delta|^2] — used for Var via E[delta^2] − E[|delta|]^2),
            # wsm_signed_accum += -delta = p_pert - p_orig so >0 means u suppresses v.
            delta = p_orig_R - p_pert                                            # [bt, k, N_EXPERTS]
            wsm_accum       [S, :, R, :].index_add_(0, sel_flat, delta.abs().flatten(0, 1))
            wsm_sq_accum    [S, :, R, :].index_add_(0, sel_flat, (delta * delta).flatten(0, 1))
            wsm_signed_accum[S, :, R, :].index_add_(0, sel_flat, (-delta).flatten(0, 1))

            del ln_bar, scores, scores_pos, scores_neg
            del pert_score, pert_sorted, pert_rank_of, orig_rank_R
            del aarv, in_topk_orig, in_topk_pert, padd, prem
            del p_orig_R, p_pert, delta

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

# ---- Normalize: weight[S, j, R, n] = accum / n_tokens_selected[S, j] ----
count_safe = n_tokens_selected.clamp(min=1).to(torch.float32)          # [L, n_experts]
denom      = count_safe.view(N_LAYERS, N_EXPERTS, 1, 1)
zero_mask  = (n_tokens_selected == 0).view(N_LAYERS, N_EXPERTS, 1, 1)

APS   = (APS_accum  / denom).masked_fill(zero_mask, 0.0)
ANS   = (ANS_accum  / denom).masked_fill(zero_mask, 0.0)
AARV  = (aarv_accum / denom).masked_fill(zero_mask, 0.0)
P_add = (padd_accum / denom).masked_fill(zero_mask, 0.0)
P_rem = (prem_accum / denom).masked_fill(zero_mask, 0.0)

# W_softmax family. Var[|delta|] = E[|delta|^2] − E[|delta|]^2 = E[delta^2] − E[|delta|]^2
# (legit since |delta|^2 = delta^2). The clamp(min=0) guards float32 catastrophic
# cancellation when both moments are tiny.
W_softmax        = (wsm_accum        / denom).masked_fill(zero_mask, 0.0)
W_softmax_E_sq   = (wsm_sq_accum     / denom).masked_fill(zero_mask, 0.0)
W_softmax_var    = (W_softmax_E_sq - W_softmax * W_softmax).clamp(min=0.0)
W_softmax_signed = (wsm_signed_accum / denom).masked_fill(zero_mask, 0.0)
del W_softmax_E_sq

# Tear down the activation-magnitude hooks before saving.
for _h in _down_proj_hooks:
    _h.remove()

if args.random_init:
    out_path = os.path.join(
        output_dir, f"dag_{args.model}_{args.dataset}_rand_s{args.seed}.pt"
    )
else:
    out_path = os.path.join(output_dir, f"dag_{args.model}_{args.dataset}.pt")
torch.save({
    "APS":               APS.cpu(),                       # [c, j, l, n]
    "ANS":               ANS.cpu(),                       # [c, j, l, n]
    "AARV":              AARV.cpu(),                      # [c, j, l, n] — mean |rank shift|
    "P_add":             P_add.cpu(),                     # [c, j, l, n] — receiver crosses INTO top-K
    "P_rem":             P_rem.cpu(),                     # [c, j, l, n] — receiver crosses OUT OF top-K
    "W_softmax":         W_softmax.cpu(),                 # [c, j, l, n] — E_i[|p_orig − p_pert|]      (PRIMARY)
    "W_softmax_var":     W_softmax_var.cpu(),             # [c, j, l, n] — Var_i[|p_orig − p_pert|]
    "W_softmax_signed":  W_softmax_signed.cpu(),          # [c, j, l, n] — E_i[p_pert − p_orig]; >0 ⇒ u suppresses v
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
    "random_init":       bool(args.random_init),
    "seed":              int(args.seed) if args.random_init else None,
}, out_path)
print(f"Saved {out_path}")