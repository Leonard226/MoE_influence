"""Multi-node DAG builder for Qwen3-235B-A22B (~470 GB bf16).

Single-node `build_dag.py` cannot fit Qwen3-235B-A22B on 4x80GB (even at NF4 it
is tight). This script partitions the 94 decoder layers across 8 ranks (2 nodes
x 4 GPUs = ~60 GiB / rank, bf16, no quantization) via manual NCCL pipeline.

Pipeline:
  Rank 0: embed -> layers[0:12]   -> send hidden_state to rank 1
  Rank k: recv  -> layers[k*12:(k+1)*12] -> send to rank k+1     (k=1..6)
  Rank 7: recv  -> layers[84:94]  -> final norm (discarded; we only need hooks)

Per-rank hook capture: each rank runs `Qwen3MoeDecoderLayer.forward()` directly
on its owned layers and collects the four hooks we need:
    hook_after_res1, hook_after_norm2, hook_selected_experts, hook_expert_weighted_outputs
Captured hook tensors are sent to rank 0 via dist.gather (per-rank tensors
shaped [n_owned, bsz, n_tok, ...]; rank 0 concatenates along layer axis).

Rank 0 owns the accumulators (APS, ANS, AVG, sq, aarv, count) and runs the
score-decomposition inner loop. The router gate and post-attention RMSNorm
weights from every layer are gathered to rank 0 once at startup (~5 GiB total).

Numerical semantics are identical to `build_dag.py` (modulo cross-rank dtype
casts and NCCL gather precision; both stay in float32 on rank 0 so it is
effectively bit-equal to single-node).

Launch:
    sbatch experiments/circuits/launch_multinode.sh
or directly:
    srun ... torchrun --nnodes 2 --nproc_per_node 4 \
        --rdzv_endpoint piora1:29500 build_dag_multinode.py \
        --dataset c4 --n_prompts 5000 --B 4
"""
import argparse
import importlib
import os
import sys
import time
from operator import attrgetter

import torch
import torch.distributed as dist
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "circuits")
os.makedirs(output_dir, exist_ok=True)

from customized_models.modeling_qwen3_moe_customized import (
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Dataset registry (same as build_dag.py).
# ---------------------------------------------------------------------------
DATASETS = {
    "c4":   ("dataset.c4_dataset",   "c4_dataset_helper"),
    "math": ("dataset.math_dataset", "open_r1_math_dataset_helper"),
    "code": ("dataset.code_dataset", "code_dataset_helper"),
}

# Model registry. Only the multi-node target lives here; single-node models
# remain in build_dag.py.
MODELS = {
    "qwen3-235b-a22b": {
        "id": "Qwen/Qwen3-235B-A22B",
        "cls": Qwen3MoeForCausalLM,
        "n_experts": 128,
        "top_k": 8,
        "d_e": 4096,
        "moe_layers": list(range(94)),
        "gate_path": "mlp.gate",
    },
}

# ---------------------------------------------------------------------------
# Distributed setup.
# ---------------------------------------------------------------------------
def init_dist():
    """Initialize NCCL process group from torchrun env vars.

    Falls back to single-process mode (RANK=0, WORLD_SIZE=1) for local debug.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def rprint(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Per-rank layer partition.
# ---------------------------------------------------------------------------
def partition_layers(n_total, world_size, rank):
    """Even split of [0..n_total) into world_size contiguous chunks.

    Returns (start, end) for this rank's slice (end exclusive).
    """
    base = n_total // world_size
    extra = n_total % world_size
    if rank < extra:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = extra * (base + 1) + (rank - extra) * base
        end = start + base
    return start, end


# ---------------------------------------------------------------------------
# Per-rank model loading.
# ---------------------------------------------------------------------------
def load_partitioned_model(model_id, rank, world_size, local_rank):
    """Instantiate full Qwen3MoeForCausalLM but only materialize this rank's
    decoder-layer slice (plus embed_tokens on rank 0, final norm on rank
    world_size-1, rotary_emb everywhere). Other modules stay on meta and are
    set to None.

    Strategy:
      1. Build empty (meta) skeleton via `init_empty_weights`.
      2. Decide a device_map that places owned modules on this rank's GPU and
         everything else on "meta".
      3. `load_checkpoint_and_dispatch` reads HF safetensors and only loads
         tensors whose target is a real device.
      4. After load, replace non-owned `model.layers[i]` with None so we can
         skip them in our custom forward.
    """
    import json
    import os
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cfg = Qwen3MoeForCausalLM.config_class.from_pretrained(model_id, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.torch_dtype = torch.bfloat16

    n_layers = cfg.num_hidden_layers
    start, end = partition_layers(n_layers, world_size, rank)
    owned = list(range(start, end))

    rprint(rank, f"[rank {rank}] owns decoder layers {start}..{end-1} ({end-start} layers)")

    with init_empty_weights():
        model = Qwen3MoeForCausalLM(cfg)

    gpu = f"cuda:{local_rank}"

    # Download once on rank 0 to populate the shared HF cache, then all ranks
    # read from cache to get the local path.
    if rank == 0:
        snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    if dist.is_initialized():
        dist.barrier()
    checkpoint_path = snapshot_download(
        model_id, allow_patterns=["*.safetensors", "*.json", "*.txt"]
    )

    # Decide which parameters this rank should materialize.
    # Everything not selected stays on meta (it'll never be called in forward).
    def is_owned(param_name: str) -> bool:
        if param_name.startswith("model.embed_tokens"):
            return rank == 0
        if param_name.startswith("model.norm"):
            return rank == world_size - 1
        if param_name.startswith("model.rotary_emb"):
            return True   # rotary_emb has no params, but keep for completeness
        if param_name.startswith("lm_head"):
            return False
        if param_name.startswith("model.layers."):
            layer_idx = int(param_name.split(".")[2])
            return layer_idx in owned
        return False

    # Read the safetensors index to map params -> shard files.
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # Single-file checkpoint (small models). Find the .safetensors file.
        st_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files in {checkpoint_path}")
        # All params live in the single shard.
        with safe_open(os.path.join(checkpoint_path, st_files[0]), framework="pt", device="cpu") as f:
            weight_map = {key: st_files[0] for key in f.keys()}

    # Group params by shard for efficient loading (open each file once).
    shards_to_load: dict[str, list[str]] = {}
    for param_name, shard_file in weight_map.items():
        if is_owned(param_name):
            shards_to_load.setdefault(shard_file, []).append(param_name)

    rprint(rank, f"[rank {rank}] loading {sum(len(v) for v in shards_to_load.values())} params from {len(shards_to_load)} shards")

    for shard_file, param_names in shards_to_load.items():
        full_path = os.path.join(checkpoint_path, shard_file)
        with safe_open(full_path, framework="pt", device=gpu) as f:
            for name in param_names:
                tensor = f.get_tensor(name)
                # set_module_tensor_to_device replaces the meta-device param with
                # this real tensor at the given location.
                set_module_tensor_to_device(model, name, gpu, value=tensor, dtype=torch.bfloat16)

    # Replace non-owned decoder layers with None so our forward skips them.
    inner = model.model
    new_layers = torch.nn.ModuleList()
    for i in range(n_layers):
        if i in owned:
            new_layers.append(inner.layers[i])
        else:
            new_layers.append(None)
    inner.layers = new_layers

    if rank != 0:
        inner.embed_tokens = None
    if rank != world_size - 1:
        inner.norm = None
    model.lm_head = None

    model.eval()
    torch.cuda.empty_cache()
    return model, cfg, owned


# ---------------------------------------------------------------------------
# Hook extraction from a single decoder layer's output.
# ---------------------------------------------------------------------------
def call_layer(decoder_layer, hidden_states, causal_mask, position_ids,
               position_embeddings):
    """Call a Qwen3MoeDecoderLayer and return (next_hidden, hook_tuple).

    hook_tuple = (after_res1, after_norm2, selected_experts, weighted_outputs)
    matching what the single-node script slices out of hook_dict.
    """
    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=position_embeddings,
        set_patch=None,
    )
    next_hidden = layer_outputs[0]
    my_hooks = layer_outputs[-1]
    # Index into the layer_hooks tuple defined in the customized model:
    #   layer_hooks = (hook_layer_input, hook_attn_output, hook_after_res1,
    #                  hook_after_norm2, hook_mlp_output, hook_layer_output)
    #   *attn_hooks  -> 7 entries
    #   *mlp_hooks   -> (router_logits, hook_selected_experts,
    #                    hook_expert_weighted_outputs, hook_routing_weights)
    # layer_hooks[2] = after_res1, layer_hooks[3] = after_norm2.
    # mlp_hooks starts at offset 6 + 7 = 13. selected_experts is mlp_hooks[1] = 14,
    # expert_weighted_outputs is mlp_hooks[2] = 15.
    after_res1 = my_hooks[2]
    after_norm2 = my_hooks[3]
    selected_experts = my_hooks[14]
    weighted_outputs = my_hooks[15]
    return next_hidden, (after_res1, after_norm2, selected_experts, weighted_outputs)


# ---------------------------------------------------------------------------
# Pipeline forward.
# ---------------------------------------------------------------------------
def pipeline_forward(model, cfg, owned, rank, world_size, local_rank,
                     input_ids, attention_mask):
    """Run a single forward pass across the pipeline.

    Inputs (input_ids, attention_mask) are already on this rank's GPU. Rank 0
    embeds; intermediate ranks recv/send hidden_state; rank world_size-1 runs
    the final RMSNorm (and discards the result).

    Returns the per-rank hook tensors as a dict with keys:
        after_res1:        [n_owned, bsz, n_tok, d_e]   (float32, on cuda)
        after_norm2:       [n_owned, bsz, n_tok, d_e]   (float32, on cuda)
        selected_experts:  [n_owned, bsz, n_tok, top_k] (long,    on cuda)
        weighted_outputs:  [n_owned, bsz, n_tok, top_k, d_e] (float32, on cuda)
    All ranks must call this in lock-step (no early return) to keep send/recv
    paired.
    """
    inner = model.model
    bsz, n_tok = input_ids.shape
    gpu = f"cuda:{local_rank}"

    # Position IDs & rotary embeddings — derive locally on every rank.
    # We need a "fake" hidden_states tensor for shape/dtype only when computing
    # rotary embeddings. Use embeddings on rank 0, else a zero-shaped tensor.
    if rank == 0:
        hidden_states = inner.embed_tokens(input_ids)
    else:
        # Receive shape via recv. We know it: [bsz, n_tok, hidden_size].
        hidden_states = torch.empty(
            (bsz, n_tok, cfg.hidden_size), dtype=torch.bfloat16, device=gpu,
        )
        dist.recv(hidden_states, src=rank - 1)

    position_ids = torch.arange(n_tok, device=gpu).unsqueeze(0).expand(bsz, -1)
    cache_position = torch.arange(n_tok, device=gpu)
    causal_mask = inner._update_causal_mask(
        attention_mask, hidden_states, cache_position,
        past_key_values=None, output_attentions=False,
    )
    position_embeddings = inner.rotary_emb(hidden_states, position_ids)

    # Run owned layers, collecting hooks.
    after_res1_chunks = []
    after_norm2_chunks = []
    selected_chunks = []
    weighted_chunks = []
    for i in owned:
        layer = inner.layers[i]
        hidden_states, hooks = call_layer(
            layer, hidden_states, causal_mask, position_ids, position_embeddings
        )
        ar, an, se, wo = hooks
        # Reshape weighted_outputs from [bsz*n_tok, top_k, hidden] -> [bsz, n_tok, top_k, hidden].
        # And selected_experts from [bsz*n_tok, top_k] -> [bsz, n_tok, top_k].
        d_e = cfg.hidden_size
        top_k = cfg.num_experts_per_tok
        wo = wo.reshape(bsz, n_tok, top_k, d_e)
        se = se.reshape(bsz, n_tok, top_k)
        # Cast to the dtypes rank 0 expects (float32 for activations, long for selection).
        after_res1_chunks.append(ar.detach().to(torch.float32))
        after_norm2_chunks.append(an.detach().to(torch.float32))
        selected_chunks.append(se.detach().to(torch.long))
        weighted_chunks.append(wo.detach().to(torch.float32))

    # Send to next rank.
    if rank < world_size - 1:
        # bf16 contiguous tensor.
        send_buf = hidden_states.contiguous().to(torch.bfloat16)
        dist.send(send_buf, dst=rank + 1)
    else:
        # Last rank — apply final norm and discard.
        _ = inner.norm(hidden_states)

    hooks = {
        "after_res1":       torch.stack(after_res1_chunks,  dim=0),  # [n_owned, bsz, n_tok, d_e]
        "after_norm2":      torch.stack(after_norm2_chunks, dim=0),
        "selected_experts": torch.stack(selected_chunks,    dim=0),
        "weighted_outputs": torch.stack(weighted_chunks,    dim=0),
    }
    return hooks


# ---------------------------------------------------------------------------
# Gather per-rank hook tensors to rank 0, ordered by layer index.
# ---------------------------------------------------------------------------
def gather_hooks(hooks, rank, world_size, n_total_layers, bsz, n_tok,
                 d_e, top_k, local_rank):
    """All ranks call this. Returns full hook tensors on rank 0 (None elsewhere).

    Uses point-to-point send/recv: each non-zero rank sends its 4 tensors to
    rank 0 in deterministic order; rank 0 assembles by stacking along layer
    axis.
    """
    if rank == 0:
        out = {
            "after_res1":       torch.empty((n_total_layers, bsz, n_tok, d_e), dtype=torch.float32, device=f"cuda:{local_rank}"),
            "after_norm2":      torch.empty((n_total_layers, bsz, n_tok, d_e), dtype=torch.float32, device=f"cuda:{local_rank}"),
            "selected_experts": torch.empty((n_total_layers, bsz, n_tok, top_k), dtype=torch.long, device=f"cuda:{local_rank}"),
            "weighted_outputs": torch.empty((n_total_layers, bsz, n_tok, top_k, d_e), dtype=torch.float32, device=f"cuda:{local_rank}"),
        }
        # Fill rank-0 slice.
        s0, e0 = partition_layers(n_total_layers, world_size, 0)
        out["after_res1"][s0:e0]       = hooks["after_res1"]
        out["after_norm2"][s0:e0]      = hooks["after_norm2"]
        out["selected_experts"][s0:e0] = hooks["selected_experts"]
        out["weighted_outputs"][s0:e0] = hooks["weighted_outputs"]
        # Recv from other ranks.
        for src in range(1, world_size):
            s, e = partition_layers(n_total_layers, world_size, src)
            dist.recv(out["after_res1"][s:e],       src=src)
            dist.recv(out["after_norm2"][s:e],      src=src)
            dist.recv(out["selected_experts"][s:e], src=src)
            dist.recv(out["weighted_outputs"][s:e], src=src)
        return out
    else:
        dist.send(hooks["after_res1"].contiguous(),       dst=0)
        dist.send(hooks["after_norm2"].contiguous(),      dst=0)
        dist.send(hooks["selected_experts"].contiguous(), dst=0)
        dist.send(hooks["weighted_outputs"].contiguous(), dst=0)
        return None


# ---------------------------------------------------------------------------
# Gather router gate weights and RMSNorm weights to rank 0 (once at startup).
# ---------------------------------------------------------------------------
def gather_layer_weights(model, owned, rank, world_size, n_total_layers, d_e,
                         n_experts, local_rank, gate_path, norm_path):
    """Collect G_recv[R] (router gate) and gamma_recv[R] (post_attn norm) from
    every layer onto rank 0 in float32.

    G_recv shape: [n_layers, n_experts, d_e]
    gamma_recv shape: [n_layers, d_e]
    """
    inner = model.model
    gpu = f"cuda:{local_rank}"
    gate_of = attrgetter(gate_path)
    norm_of = attrgetter(norm_path)

    # Local slices.
    if owned:
        local_G = torch.stack(
            [gate_of(inner.layers[R]).weight.detach().to(torch.float32) for R in owned]
        )  # [n_owned, n_experts, d_e]
        local_gamma = torch.stack(
            [norm_of(inner.layers[R]).weight.detach().to(torch.float32) for R in owned]
        )  # [n_owned, d_e]
    else:
        local_G = torch.empty((0, n_experts, d_e), dtype=torch.float32, device=gpu)
        local_gamma = torch.empty((0, d_e), dtype=torch.float32, device=gpu)

    if rank == 0:
        G_full = torch.empty((n_total_layers, n_experts, d_e), dtype=torch.float32, device=gpu)
        gamma_full = torch.empty((n_total_layers, d_e), dtype=torch.float32, device=gpu)
        s0, e0 = partition_layers(n_total_layers, world_size, 0)
        G_full[s0:e0] = local_G
        gamma_full[s0:e0] = local_gamma
        for src in range(1, world_size):
            s, e = partition_layers(n_total_layers, world_size, src)
            dist.recv(G_full[s:e], src=src)
            dist.recv(gamma_full[s:e], src=src)
        return G_full, gamma_full
    else:
        dist.send(local_G.contiguous(), dst=0)
        dist.send(local_gamma.contiguous(), dst=0)
        return None, None


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="qwen3-235b-a22b")
    parser.add_argument("--dataset", choices=list(DATASETS), default="c4")
    parser.add_argument("--n_prompts", type=int, default=5000)
    parser.add_argument("--B", type=int, default=4)
    args = parser.parse_args()

    rank, world_size, local_rank = init_dist()
    torch.set_grad_enabled(False)

    MODEL = MODELS[args.model]
    MODEL_ID = MODEL["id"]
    MOE_LAYERS = MODEL["moe_layers"]
    N_LAYERS = len(MOE_LAYERS)
    N_EXPERTS = MODEL["n_experts"]
    D_E = MODEL["d_e"]
    TOP_K = MODEL["top_k"]
    EPS = 1e-5

    N_PROMPTS = args.n_prompts
    BSZ = args.B
    MAX_TOKENS = 32

    rprint(rank, f"world_size={world_size}  rank={rank}  local_rank={local_rank}")
    rprint(rank, f"Building DAG for model={args.model!r}, dataset={args.dataset!r}, {N_PROMPTS} prompts.")

    # ---- Load model (per-rank slice) ----
    rprint(rank, f"Loading {MODEL_ID} ...")
    t0 = time.time()
    model, cfg, owned = load_partitioned_model(MODEL_ID, rank, world_size, local_rank)
    if dist.is_initialized():
        dist.barrier()
    rprint(rank, f"  loaded in {time.time() - t0:.1f}s")

    # Sanity: registry's N_LAYERS must equal config's num_hidden_layers (all MoE).
    assert N_LAYERS == cfg.num_hidden_layers, (
        f"MOE_LAYERS={N_LAYERS} but config has {cfg.num_hidden_layers}; "
        "Qwen3-235B-A22B is supposed to be all-MoE."
    )

    # ---- Gather G_recv, gamma_recv to rank 0 ----
    rprint(rank, "Gathering router gate + post-attn norm weights to rank 0 ...")
    t0 = time.time()
    G_recv, gamma_recv = gather_layer_weights(
        model, owned, rank, world_size, N_LAYERS, D_E, N_EXPERTS, local_rank,
        gate_path=MODEL["gate_path"], norm_path="post_attention_layernorm",
    )
    if dist.is_initialized():
        dist.barrier()
    rprint(rank, f"  gathered in {time.time() - t0:.1f}s")

    # ---- Tokenizer + dataset (rank 0 only) ----
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mod_name, fn_name = DATASETS[args.dataset]
        loader = getattr(importlib.import_module(mod_name), fn_name)
        print(f"Loading dataset={args.dataset!r}  ({N_PROMPTS} prompts) ...", flush=True)
        t0 = time.time()
        prompts = loader(dataset_len=N_PROMPTS, min_words=MAX_TOKENS)
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

        # ---- Accumulators ----
        device0 = f"cuda:{local_rank}"
        SHAPE = (N_LAYERS, N_EXPERTS, N_LAYERS, N_EXPERTS)
        APS_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device0)
        ANS_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device0)
        AVG_accum  = torch.zeros(SHAPE, dtype=torch.float32, device=device0)
        sq_accum   = torch.zeros(SHAPE, dtype=torch.float32, device=device0)
        aarv_accum = torch.zeros(SHAPE, dtype=torch.float32, device=device0)
        count = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device0)

    n_batches = (N_PROMPTS + BSZ - 1) // BSZ
    rprint(rank, f"Running {n_batches} batches (batch_size={BSZ}, max_tokens={MAX_TOKENS}) ...")
    t_start = time.time()

    if dist.is_initialized():
        dist.barrier()

    for batch_idx in range(n_batches):
        # --- Tokenize on rank 0, broadcast shape + tensors to all ranks ---
        if rank == 0:
            B = batch_idx * BSZ
            batch = prompts[B:B + BSZ]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=False, truncation=True,
                max_length=MAX_TOKENS,
            )
            input_ids = inputs["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = inputs["attention_mask"].to(f"cuda:{local_rank}")
            shape = torch.tensor(
                list(input_ids.shape), dtype=torch.long, device=f"cuda:{local_rank}"
            )
        else:
            shape = torch.zeros(2, dtype=torch.long, device=f"cuda:{local_rank}")

        if dist.is_initialized():
            dist.broadcast(shape, src=0)
            bsz_i, n_tok_i = int(shape[0].item()), int(shape[1].item())
            if rank != 0:
                input_ids = torch.zeros(
                    (bsz_i, n_tok_i), dtype=torch.long, device=f"cuda:{local_rank}"
                )
                attention_mask = torch.zeros(
                    (bsz_i, n_tok_i), dtype=torch.long, device=f"cuda:{local_rank}"
                )
            dist.broadcast(input_ids, src=0)
            dist.broadcast(attention_mask, src=0)
        else:
            bsz_i, n_tok_i = input_ids.shape

        # --- Pipeline forward + hook capture ---
        hooks = pipeline_forward(
            model, cfg, owned, rank, world_size, local_rank,
            input_ids, attention_mask,
        )

        # --- Gather hooks to rank 0 ---
        full_hooks = gather_hooks(
            hooks, rank, world_size, N_LAYERS, bsz_i, n_tok_i,
            D_E, TOP_K, local_rank,
        )

        # --- Rank 0 does the score-decomposition accumulator update ---
        if rank == 0:
            # full_hooks tensors are [n_layers, bsz, n_tok, ...]. We need
            # build_dag.py's [bsz, n_layers, n_tok, ...] layout — transpose
            # the first two dims.
            after_res1 = full_hooks["after_res1"].permute(1, 0, 2, 3)               # [bsz, L, n_tok, d_e]
            after_norm2 = full_hooks["after_norm2"].permute(1, 0, 2, 3)             # [bsz, L, n_tok, d_e]
            selected = full_hooks["selected_experts"].permute(1, 0, 2, 3)           # [bsz, L, n_tok, top_k]
            weighted_out = full_hooks["weighted_outputs"].permute(1, 0, 2, 3, 4)    # [bsz, L, n_tok, top_k, d_e]

            bsz, _, n_tok, _ = after_res1.shape
            bt = bsz * n_tok
            device0 = f"cuda:{local_rank}"

            # 1 / RMS^R_i per token per receiver layer R.
            rms_sq = after_res1.pow(2).mean(dim=-1) + EPS                          # [bsz, L, n_tok]
            rms_inv = torch.rsqrt(rms_sq).permute(0, 2, 1).reshape(bt, N_LAYERS)   # [bt, L]

            # Original assignment scores at every receiver layer (for AARV).
            after_norm2_r = after_norm2.permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)  # [bt, L, d_e]
            orig_score = torch.einsum("lnd,bld->bln", G_recv, after_norm2_r)            # [bt, L, n_experts]
            orig_sorted = torch.argsort(orig_score, dim=-1, descending=True)            # [bt, L, n_experts]
            orig_rank_of = torch.empty_like(orig_sorted)
            orig_rank_of.scatter_(
                -1, orig_sorted,
                torch.arange(N_EXPERTS, device=device0).expand_as(orig_sorted),
            )

            # Sender-side reshapes.
            omega = weighted_out.permute(0, 2, 1, 3, 4).reshape(bt, N_LAYERS, TOP_K, D_E)
            sel = selected.permute(0, 2, 1, 3).reshape(bt, N_LAYERS, TOP_K)

            for S in range(N_LAYERS):
                sel_S = sel[:, S, :]
                count[S] += torch.bincount(sel_S.flatten(), minlength=N_EXPERTS)

                if S == N_LAYERS - 1:
                    continue
                omega_S = omega[:, S, :, :]                                         # [bt, top_k, d_e]

                for R in range(S + 1, N_LAYERS):
                    ln_bar = omega_S * gamma_recv[R].view(1, 1, D_E) * rms_inv[:, R].view(bt, 1, 1)
                    scores = torch.einsum("ed,bkd->bke", G_recv[R], ln_bar)         # [bt, k, n_experts]

                    scores_pos = scores.clamp(min=0.0)
                    scores_neg = scores.clamp(max=0.0)
                    scores_sq = scores * scores
                    sel_flat = sel_S.flatten()
                    APS_accum[S, :, R, :].index_add_(0, sel_flat, scores_pos.flatten(0, 1))
                    ANS_accum[S, :, R, :].index_add_(0, sel_flat, scores_neg.flatten(0, 1))
                    AVG_accum[S, :, R, :].index_add_(0, sel_flat, scores.flatten(0, 1))
                    sq_accum[S, :, R, :].index_add_(0, sel_flat, scores_sq.flatten(0, 1))

                    pert_score = orig_score[:, R, :].unsqueeze(1) - scores         # [bt, k, n_experts]
                    pert_sorted = torch.argsort(pert_score, dim=-1, descending=True)
                    pert_rank_of = torch.empty_like(pert_sorted)
                    pert_rank_of.scatter_(
                        -1, pert_sorted,
                        torch.arange(N_EXPERTS, device=device0).expand_as(pert_sorted),
                    )
                    orig_rank_R = orig_rank_of[:, R, :].unsqueeze(1).expand_as(pert_rank_of)
                    rank_shift = (pert_rank_of.float() - orig_rank_R.float()).abs()
                    aarv_accum[S, :, R, :].index_add_(0, sel_flat, rank_shift.flatten(0, 1))

                    del ln_bar, scores, scores_pos, scores_neg, scores_sq
                    del pert_score, pert_sorted, pert_rank_of, orig_rank_R, rank_shift

            del full_hooks, after_res1, after_norm2, selected, weighted_out
            del omega, sel, rms_sq, rms_inv, after_norm2_r
            del orig_score, orig_sorted, orig_rank_of

        # Free per-rank hooks.
        del hooks
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        bnum = batch_idx + 1
        if rank == 0 and (bnum == 1 or bnum % 10 == 0 or bnum == n_batches):
            elapsed = time.time() - t_start
            rate = (bnum * BSZ) / elapsed if elapsed > 0 else 0.0
            eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
            print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
                  f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

    rprint(rank, f"\nDone in {time.time() - t_start:.1f}s.\n")

    # ---- Normalize + save (rank 0 only) ----
    if rank == 0:
        count_safe = count.clamp(min=1).to(torch.float32)
        denom = count_safe.view(N_LAYERS, N_EXPERTS, 1, 1)
        zero_mask = (count == 0).view(N_LAYERS, N_EXPERTS, 1, 1)

        APS = (APS_accum / denom).masked_fill(zero_mask, 0.0)
        ANS = (ANS_accum / denom).masked_fill(zero_mask, 0.0)
        AVG = (AVG_accum / denom).masked_fill(zero_mask, 0.0)
        AVG_sq = (sq_accum / denom).masked_fill(zero_mask, 0.0)
        VAR = (AVG_sq - AVG * AVG).clamp(min=0.0)
        del AVG_sq
        AARV = (aarv_accum / denom).masked_fill(zero_mask, 0.0)

        out_path = os.path.join(output_dir, f"dag_{args.model}_{args.dataset}.pt")
        torch.save({
            "APS":   APS.cpu(),
            "ANS":   ANS.cpu(),
            "AVG":   AVG.cpu(),
            "VAR":   VAR.cpu(),
            "AARV":  AARV.cpu(),
            "count": count.cpu(),
            "n_prompts": N_PROMPTS,
            "max_tokens": MAX_TOKENS,
            "model": MODEL_ID,
            "moe_layers": MOE_LAYERS,
            "dataset": args.dataset,
        }, out_path)
        print(f"Saved {out_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
