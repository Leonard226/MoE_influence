"""Multi-node expert ablation for the two models that exceed single-node
memory in bf16: Qwen3-235B-A22B (~470 GB) and DeepSeek-V2 (~472 GB).

Same experiment as experiments/ablate_experts.py (Su-mirrored down_proj-output
zeroing + c4 PPL), but the model is sharded pipeline-parallel across 2 nodes x
4 GPUs = 8 ranks, reusing the proven partition/forward machinery from
experiments/build_dag_multinode.py.

Differences from the single-node script:
  - Pipeline-parallel forward: rank 0 embeds, hidden states flow rank->rank via
    NCCL, the last rank runs norm + lm_head + per-sequence NLL.
  - lm_head is kept on the last rank (build_dag_multinode nulls it).
  - Ablation hooks (down_proj -> zeros) are registered on whichever rank owns
    the target expert's decoder layer.
  - Attention-sink / max|h0| / max|h|_all ARE collected here too (unlike
    earlier versions of this script): the customized decoder layers always
    populate layer_outputs[-1][11] with the real post-softmax attention
    weights [B, H, T, T], captured unconditionally inside self_attn.forward
    before any output_attentions gating -- no extra kwargs needed. Each rank
    contributes its owned layers' local sum/count (att_sink) and local max
    (max_h0: position 0 only; max_h_all: max_{l,b,t,d} |X^(l)_{b,t,d}|, any
    position); pipeline_forward_sink + eval_sink_pipeline combine them with a
    SUM/SUM/MAX/MAX reduce onto rank 0. Computed on a separate SHORT context
    (--sink-seq-len, default 256): materializing full eager-attention
    matrices at the main --seq-len 2048 is the same O(T^2) blowup that
    caused the single-node CUDA OOM earlier in this project.

Ablation operator and PPL protocol are byte-for-byte the single-node ones:
zero the expert's down_proj output; 256 random 2048-token c4-validation
windows; PPL = exp(mean per-sequence mean NLL). Results are written by rank 0
to {result_path}/ablation/{model}_c4.json in the SAME format as the
single-node runs (restart-safe; cached runs skipped) -- att_sink/max_h0/
max_h_all keys
included.

Launch (2 nodes x 4 GPUs), e.g. via a SLURM script mirroring
launch_multinode.sh but pointing at this file:
    srun ... torchrun --nnodes 2 --nproc_per_node 4 \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        experiments/ablate_experts_multinode.py --model qwen3-235b-a22b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from operator import attrgetter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
RESULTS = Path(_config["result_path"])

# Reuse the proven distributed primitives + model registry (customized model
# classes, per-model layer-kwargs / causal-mask fns, moe_layers, expert paths).
from experiments.build_dag_multinode import (  # noqa: E402
    MODELS, init_dist, rprint, partition_layers,
)

SEQLEN = 2048
N_SEQS = 256

# Ablation targets = same design as the single-node runs: singles (union of
# the two top-10 rankings) + the Su-criterion SE set + our out-top-3 + an
# FN-candidate set + a size-matched random set. Labels are model-absolute
# decoder-layer indices (LxEy = decoder layer x, expert y).
DEFAULT_TARGETS = {
    # Qwen3-235B. Singles = full union of top-10 act + top-10 out (18):
    #   both:     L3E120, L2E39
    #   out-only: L69E69, L70E113, L90E101, L50E113, L2E83, L49E69, L90E10, L90E14
    #   act-only: L3E22, L3E29, L3E0, L35E109, L3E55, L64E91, L3E39, L8E64
    # Sets test the hypotheses:
    #   SE set {L3E120,L2E39} (Su's criterion) -> predict catastrophic (H: SE causal)
    #   out-top-3 -> our influence set (H3)
    #   late high-out set -> high-out experts Su MISSES (H2 FN: L70/L90 excluded
    #     by the 0.75 layer filter, L69 by magnitude; L90E101 is act pctl 99.4,
    #     a filter-only FN) -> predict damage >> random
    "qwen3-235b-a22b": ("L3E120;L2E39;L69E69;L70E113;L90E101;L50E113;L2E83;"
                        "L49E69;L90E10;L90E14;"
                        "L3E22;L3E29;L3E0;L35E109;L3E55;L64E91;L3E39;L8E64;"
                        "L3E120+L2E39;"               # Su-criterion SE set (ours)
                        "L3E120+L2E39+L69E69;"        # out-top-3
                        "L69E69+L70E113+L90E101;"     # high-out, Su misses (FN)
                        "L11E64+L57E30+L83E19"),      # random-3
    # DeepSeek-V2. Singles = full union of top-10 act + top-10 out (18):
    #   both:     L18E96, L21E94
    #   act-only: L20E48, L43E57, L30E64, L22E121, L16E102, L27E114, L21E69, L10E29
    #   out-only: L1E119, L2E111, L5E34, L3E17, L3E64, L4E13, L5E88, L3E81
    # Sets:
    #   SE set {L18E96,L20E48,L21E94} (Su's 3 reproduced SEs, mid-net BOS chain)
    #   out-top-3 {L1E119,L18E96,L2E111}
    #   early-BOS chain {L1E119,L2E111,L3E17}: high-out, act pctl 0.7-72, the
    #     sharpest FN test (near-zero-act experts Su cannot see)
    "deepseek-v2": ("L18E96;L21E94;L20E48;L43E57;L30E64;L22E121;L16E102;"
                    "L27E114;L21E69;L10E29;L1E119;L2E111;L5E34;L3E17;L3E64;"
                    "L4E13;L5E88;L3E81;L45E24;"       # L45E24: top-1 by in(v)
                    "L18E96+L20E48+L21E94;"           # Su-criterion SE set (ours, 3)
                    "L1E119+L18E96+L2E111;"           # out-top-3
                    "L1E119+L2E111+L3E17;"            # early-BOS chain (FN)
                    "L14E7+L37E88+L52E140;"           # random-3
                    "L18E96+L20E48+L21E94+L1E119;"    # SE set + BOS hub
                    "L18E96+L20E48+L21E94+L1E119+L2E111"),  # SE set + 2 BOS
}


def parse_targets(spec: str) -> list[list[tuple[int, int]]]:
    """'L4E27;L1E18+L2E30' -> [[(4, 27)], [(1, 18), (2, 30)]]  (decoder-layer,
    expert)."""
    runs = []
    for run in spec.split(";"):
        group = []
        for lab in run.strip().split("+"):
            lab = lab.strip().upper()
            l_str, e_str = lab.lstrip("L").split("E")
            group.append((int(l_str), int(e_str)))
        runs.append(group)
    return runs


def label_of(group: list[tuple[int, int]]) -> str:
    return "+".join(f"L{l}E{e}" for l, e in group)


# ---------------------------------------------------------------------------
# Per-rank model loading -- adapted from build_dag_multinode.load_partitioned_
# model, but lm_head is OWNED by the last rank (kept, not nulled) so it can
# produce logits for the PPL loss.
# ---------------------------------------------------------------------------
def load_partitioned_model_ppl(model_cfg, rank, world_size, local_rank):
    import json as _json
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    cls = model_cfg["cls"]
    model_id = model_cfg["id"]
    cfg = cls.config_class.from_pretrained(model_id, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.torch_dtype = torch.bfloat16

    n_layers = cfg.num_hidden_layers
    start, end = partition_layers(n_layers, world_size, rank)
    owned = list(range(start, end))
    is_last = rank == world_size - 1
    rprint(rank, f"[rank {rank}] owns decoder layers {start}..{end-1}")

    with init_empty_weights():
        model = cls(cfg)
    gpu = f"cuda:{local_rank}"

    if rank == 0:
        snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    if dist.is_initialized():
        dist.barrier()
    checkpoint_path = snapshot_download(
        model_id, allow_patterns=["*.safetensors", "*.json", "*.txt"])

    def is_owned(param_name: str) -> bool:
        if param_name.startswith("model.embed_tokens"):
            return rank == 0
        if param_name.startswith("model.norm"):
            return is_last
        if param_name.startswith("model.rotary_emb"):
            return True
        if param_name.startswith("lm_head"):
            return is_last          # <-- kept on the last rank (vs. build_dag)
        if param_name.startswith("model.layers."):
            return int(param_name.split(".")[2]) in owned
        return False

    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = _json.load(f)["weight_map"]
    else:
        st_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
        with safe_open(os.path.join(checkpoint_path, st_files[0]),
                       framework="pt", device="cpu") as f:
            weight_map = {key: st_files[0] for key in f.keys()}

    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if is_owned(name):
            shards.setdefault(shard, []).append(name)
    # Some checkpoints tie lm_head to embed_tokens (no lm_head.weight in the
    # index). Handle after load if needed.
    for shard, names in shards.items():
        with safe_open(os.path.join(checkpoint_path, shard),
                       framework="pt", device=gpu) as f:
            for name in names:
                set_module_tensor_to_device(
                    model, name, gpu, value=f.get_tensor(name),
                    dtype=torch.bfloat16)

    inner = model.model
    new_layers = torch.nn.ModuleList()
    for i in range(n_layers):
        new_layers.append(inner.layers[i] if i in owned else None)
    inner.layers = new_layers
    if rank != 0:
        inner.embed_tokens = None
    if not is_last:
        inner.norm = None
        model.lm_head = None
    elif getattr(cfg, "tie_word_embeddings", False) and rank != 0:
        # Tied embeddings but embed_tokens lives on rank 0: materialize
        # lm_head.weight from the checkpoint explicitly.
        with safe_open(os.path.join(checkpoint_path,
                       weight_map.get("model.embed_tokens.weight",
                                      list(weight_map.values())[0])),
                       framework="pt", device=gpu) as f:
            if "model.embed_tokens.weight" in f.keys():
                set_module_tensor_to_device(
                    model, "lm_head.weight", gpu,
                    value=f.get_tensor("model.embed_tokens.weight"),
                    dtype=torch.bfloat16)

    model.eval()
    torch.cuda.empty_cache()
    return model, cfg, owned


# ---------------------------------------------------------------------------
# Pipeline forward that ends in norm + lm_head + per-sequence NLL on the last
# rank. Mirrors build_dag_multinode.pipeline_forward minus the hook capture.
# ---------------------------------------------------------------------------
def pipeline_forward_ppl(model, cfg, model_cfg, owned, rank, world_size,
                         local_rank, input_ids):
    inner = model.model
    bsz, n_tok = input_ids.shape
    gpu = f"cuda:{local_rank}"
    layer_kwargs_fn = model_cfg["layer_kwargs_fn"]
    causal_mask_fn = model_cfg["causal_mask_fn"]
    needs_pos_emb = model_cfg["needs_position_embeddings"]

    if rank == 0:
        hidden = inner.embed_tokens(input_ids)
    else:
        hidden = torch.empty((bsz, n_tok, cfg.hidden_size),
                             dtype=torch.bfloat16, device=gpu)
        dist.recv(hidden, src=rank - 1)

    position_ids = torch.arange(n_tok, device=gpu).unsqueeze(0).expand(bsz, -1)
    cache_position = torch.arange(n_tok, device=gpu)
    causal_mask = causal_mask_fn(inner, None, hidden, cache_position)
    position_embeddings = (inner.rotary_emb(hidden, position_ids)
                           if needs_pos_emb else None)

    kwargs = layer_kwargs_fn(causal_mask, position_ids, position_embeddings)
    for i in owned:
        hidden = inner.layers[i](hidden, **kwargs)[0]

    if rank < world_size - 1:
        dist.send(hidden.contiguous().to(torch.bfloat16), dst=rank + 1)
        return None  # non-final ranks contribute no loss

    # Last rank: norm -> lm_head -> per-sequence NLL.
    hidden = inner.norm(hidden)
    logits = model.lm_head(hidden)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)  # [B, T-1]
    return loss.mean(dim=1).tolist()  # per-sequence mean NLL


@torch.no_grad()
def pipeline_forward_sink(model, cfg, model_cfg, owned, rank, world_size,
                          local_rank, input_ids):
    """Single short-context forward pass, returning THIS RANK's local
    contribution to the attention-sink / max-activation statistics:
        (local_att_sum, local_att_count, local_max_h0, local_max_h_all)
    max_h0 is the peak |activation| at position 0 only (Su et al.'s sink
    position); max_h_all is the peak anywhere -- max_{l,b,t,d} |X^(l)_{b,t,d}|
    -- generalising max_h0 to all token positions, not just the BOS/sink
    slot. Unlike PPL (only the last rank has a loss), every rank owns layers
    that contribute here, so every rank returns real numbers -- the caller
    combines them with a SUM/SUM/MAX/MAX reduce onto rank 0.

    The customized decoder layers (customized_models/modeling_*) always
    populate layer_outputs[-1][11] with the real post-softmax attention
    weights [B, H, T, T] regardless of any output_attentions/use_cache
    kwarg -- the hook is captured unconditionally inside self_attn.forward,
    before any such gating (verified in both the Qwen3 and DeepSeek custom
    model files). The multinode loader forces _attn_implementation="eager",
    which is required for Qwen3's attention_interface to actually populate
    this hook (SDPA does not expose attention weights).

    Kept as a SEPARATE short-context (--sink-seq-len, default 256) pass
    from the main --seq-len 2048 PPL pass: materializing full [B,H,T,T]
    eager-attention matrices at T=2048 is the same O(T^2) blowup that
    caused the single-node CUDA OOM earlier in this project.
    """
    inner = model.model
    bsz, n_tok = input_ids.shape
    gpu = f"cuda:{local_rank}"
    layer_kwargs_fn = model_cfg["layer_kwargs_fn"]
    causal_mask_fn = model_cfg["causal_mask_fn"]
    needs_pos_emb = model_cfg["needs_position_embeddings"]

    local_max_h0 = 0.0
    local_max_h_all = 0.0

    if rank == 0:
        hidden = inner.embed_tokens(input_ids)
        # Include the embedding output itself (HF's hidden_states[0]) in the
        # max, matching the single-node eval_sink's convention of scanning
        # all L+1 residual-stream states, not just post-layer ones.
        local_max_h0 = max(local_max_h0, float(hidden[:, 0, :].abs().max()))
        local_max_h_all = max(local_max_h_all, float(hidden.abs().max()))
    else:
        hidden = torch.empty((bsz, n_tok, cfg.hidden_size),
                             dtype=torch.bfloat16, device=gpu)
        dist.recv(hidden, src=rank - 1)

    position_ids = torch.arange(n_tok, device=gpu).unsqueeze(0).expand(bsz, -1)
    cache_position = torch.arange(n_tok, device=gpu)
    causal_mask = causal_mask_fn(inner, None, hidden, cache_position)
    position_embeddings = (inner.rotary_emb(hidden, position_ids)
                           if needs_pos_emb else None)
    kwargs = layer_kwargs_fn(causal_mask, position_ids, position_embeddings)

    local_att_sum = 0.0
    local_att_count = 0
    for i in owned:
        layer_outputs = inner.layers[i](hidden, **kwargs)
        hidden = layer_outputs[0]
        attn_w = layer_outputs[-1][11]           # [B, H, T, T], post-softmax
        local_att_sum += float(attn_w[:, :, 1:, 0].mean())
        local_att_count += 1
        local_max_h0 = max(local_max_h0, float(hidden[:, 0, :].abs().max()))
        local_max_h_all = max(local_max_h_all, float(hidden.abs().max()))

    if rank < world_size - 1:
        dist.send(hidden.contiguous().to(torch.bfloat16), dst=rank + 1)
    # Last rank: no norm/lm_head needed -- sink stats only need the
    # per-layer hidden states and attention weights already captured above.

    return local_att_sum, local_att_count, local_max_h0, local_max_h_all


@torch.no_grad()
def eval_sink_pipeline(model, cfg, model_cfg, owned, rank, world_size,
                       local_rank, sink_ids):
    """Run the sink forward pass and reduce every rank's local contribution
    onto rank 0: att_sink = (sum of per-layer means) / (total layer count);
    max_h0/max_h_all = max over all layers/ranks. Returns the dict on rank 0,
    None elsewhere (mirrors sync_ppl's rank-0-only convention)."""
    gpu = f"cuda:{local_rank}"
    att_sum, att_count, max_h0, max_h_all = pipeline_forward_sink(
        model, cfg, model_cfg, owned, rank, world_size, local_rank,
        sink_ids.to(gpu))
    att_sum_t = torch.tensor([att_sum], dtype=torch.float64, device=gpu)
    att_count_t = torch.tensor([att_count], dtype=torch.float64, device=gpu)
    max_h0_t = torch.tensor([max_h0], dtype=torch.float64, device=gpu)
    max_h_all_t = torch.tensor([max_h_all], dtype=torch.float64, device=gpu)
    if dist.is_initialized():
        dist.reduce(att_sum_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(att_count_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(max_h0_t, dst=0, op=dist.ReduceOp.MAX)
        dist.reduce(max_h_all_t, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        return {"att_sink": float(att_sum_t.item() / att_count_t.item()),
                "max_h0": float(max_h0_t.item()),
                "max_h_all": float(max_h_all_t.item())}
    return None


def register_ablation_hooks(model, model_cfg, owned, group, local_rank):
    """Register down_proj-output-zeroing hooks for the members of `group`
    whose decoder layer this rank owns. Returns the handles to remove later."""
    handles = []
    experts_of = attrgetter(model_cfg["experts_path"])
    down_attr = model_cfg["down_proj_attr"]
    owned_set = set(owned)
    for (l, e) in group:
        if l not in owned_set:
            continue
        experts = experts_of(model.model.layers[l])
        down = getattr(experts[e], down_attr)

        def hook(_m, _inp, out):
            return torch.zeros_like(out)

        handles.append(down.register_forward_hook(hook))
    return handles


def su_c4_eval_windows(tok, n_seqs, seqlen):
    """Su et al.'s get_c4(eval_mode=True): 256 random seqlen-token windows from
    c4 validation docs longer than seqlen, random.seed(0). Same as the
    single-node script."""
    import datasets
    valdata = datasets.load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation")
    random.seed(0)
    windows = []
    while len(windows) < n_seqs:
        while True:
            i = random.randint(0, len(valdata) - 1)
            enc = tok(valdata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        windows.append(enc.input_ids[0, i:i + seqlen])
    return torch.stack(windows)  # [n_seqs, seqlen]


@torch.no_grad()
def eval_ppl_pipeline(model, cfg, model_cfg, owned, rank, world_size,
                      local_rank, windows, batch_size):
    """Run PPL over all windows. Returns ppl on the last rank, None elsewhere.
    windows is broadcast to every rank before this call."""
    gpu = f"cuda:{local_rank}"
    nlls: list[float] = []
    n = windows.shape[0]
    for i in range(0, n, batch_size):
        ids = windows[i:i + batch_size].to(gpu)
        out = pipeline_forward_ppl(model, cfg, model_cfg, owned, rank,
                                   world_size, local_rank, ids)
        if out is not None:
            nlls.extend(out)
    if rank == world_size - 1:
        return float(np.exp(np.mean(nlls)))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(DEFAULT_TARGETS))
    p.add_argument("--experts", default=None,
                   help="Ablation spec; default = curated targets for the model.")
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seqs", type=int, default=N_SEQS)
    p.add_argument("--seq-len", type=int, default=SEQLEN)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-sink-seqs", type=int, default=8)
    p.add_argument("--sink-seq-len", type=int, default=256,
                   help="Context length for the attention-sink metric (kept "
                        "short: eager attention is O(T^2), same reasoning as "
                        "the single-node script).")
    args = p.parse_args()

    rank, world_size, local_rank = init_dist()
    torch.set_grad_enabled(False)
    model_cfg = MODELS[args.model]
    n_experts = model_cfg["n_experts"]
    moe_layers = model_cfg["moe_layers"]
    min_layer = min(moe_layers)

    spec = args.experts or DEFAULT_TARGETS[args.model]
    runs = parse_targets(spec)

    # Random controls: only MoE layers, drawn identically on every rank.
    rng = np.random.default_rng(args.seed)
    named = {(l, e) for grp in runs for l, e in grp}
    controls: list[tuple[int, int]] = []
    while len(controls) < args.random_controls:
        l = int(rng.integers(min_layer, max(moe_layers) + 1))
        e = int(rng.integers(0, n_experts))
        if l in moe_layers and (l, e) not in named and (l, e) not in controls:
            controls.append((l, e))
    is_control = {label_of([c]) for c in controls}
    runs += [[c] for c in controls]

    rprint(rank, f"[{args.model}] world_size={world_size}; "
                 f"{len(runs)} ablation runs ({args.random_controls} controls)")

    # ---- Load partitioned model ----
    t0 = time.time()
    model, cfg, owned = load_partitioned_model_ppl(
        model_cfg, rank, world_size, local_rank)
    if dist.is_initialized():
        dist.barrier()
    rprint(rank, f"  loaded in {time.time() - t0:.1f}s")

    # ---- Build eval windows on rank 0, broadcast to all ranks ----
    gpu = f"cuda:{local_rank}"
    if rank == 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_cfg["id"], trust_remote_code=True)
        windows = su_c4_eval_windows(tok, args.n_seqs, args.seq_len).to(gpu)
        shape = torch.tensor(list(windows.shape), dtype=torch.long, device=gpu)
    else:
        shape = torch.zeros(2, dtype=torch.long, device=gpu)
    if dist.is_initialized():
        dist.broadcast(shape, src=0)
        if rank != 0:
            windows = torch.zeros((int(shape[0]), int(shape[1])),
                                  dtype=torch.long, device=gpu)
        dist.broadcast(windows, src=0)
    rprint(rank, f"  {windows.shape[0]} windows x {windows.shape[1]} tok")

    # sink_ids: no extra broadcast needed -- windows is already identical on
    # every rank, and n_sink_seqs/sink_seq_len are the same CLI args on every
    # rank (torchrun launches identical argv everywhere).
    sink_ids = windows[:args.n_sink_seqs, :args.sink_seq_len].contiguous()
    rprint(rank, f"  sink eval: {sink_ids.shape[0]} windows x "
                 f"{sink_ids.shape[1]} tok")

    out_path = RESULTS / "ablation" / f"{args.model}_c4.json"
    if rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if rank == 0 and out_path.exists():
        results = json.loads(out_path.read_text())

    def sync_ppl(ppl_last):
        """Move the PPL scalar from the last rank to rank 0."""
        if not dist.is_initialized():
            return ppl_last
        t = torch.tensor([ppl_last if rank == world_size - 1 else 0.0],
                         dtype=torch.float64, device=gpu)
        dist.broadcast(t, src=world_size - 1)
        return float(t.item())

    def have(label):
        flag = torch.tensor([1 if (rank == 0 and label in results) else 0],
                            dtype=torch.long, device=gpu)
        if dist.is_initialized():
            dist.broadcast(flag, src=0)
        return bool(flag.item())

    # ---- Baseline ----
    if not have("baseline"):
        ppl = eval_ppl_pipeline(model, cfg, model_cfg, owned, rank,
                                world_size, local_rank, windows, args.batch_size)
        ppl = sync_ppl(ppl if ppl is not None else 0.0)
        sink = eval_sink_pipeline(model, cfg, model_cfg, owned, rank,
                                  world_size, local_rank, sink_ids)
        if rank == 0:
            results["baseline"] = {"ppl": ppl, **sink}
            out_path.write_text(json.dumps(results, indent=2))
            print(f"baseline: ppl={ppl:.4f} att_sink={sink['att_sink']:.4f} "
                  f"max_h0={sink['max_h0']:.4g} max_h_all={sink['max_h_all']:.4g}",
                  flush=True)
    base = results.get("baseline", {}).get("ppl", float("nan")) if rank == 0 else 0.0

    # ---- Ablations ----
    for group in runs:
        lab = label_of(group)
        if have(lab):
            rprint(rank, f"{lab}: cached, skipping")
            continue
        handles = register_ablation_hooks(model, model_cfg, owned, group, local_rank)
        if dist.is_initialized():
            dist.barrier()
        ppl = eval_ppl_pipeline(model, cfg, model_cfg, owned, rank,
                                world_size, local_rank, windows, args.batch_size)
        ppl = sync_ppl(ppl if ppl is not None else 0.0)
        sink = eval_sink_pipeline(model, cfg, model_cfg, owned, rank,
                                  world_size, local_rank, sink_ids)
        for h in handles:
            h.remove()
        if rank == 0:
            results[lab] = {"ppl": ppl, **sink, "control": lab in is_control}
            out_path.write_text(json.dumps(results, indent=2))
            ratio = ppl / base if base else float("nan")
            print(f"{lab}{' [ctrl]' if lab in is_control else ''}: "
                  f"ppl={ppl:.4f} (x{ratio:.4f})  att_sink={sink['att_sink']:.4f} "
                  f"max_h0={sink['max_h0']:.4g} max_h_all={sink['max_h_all']:.4g}",
                  flush=True)

    rprint(rank, f"\nSaved: {out_path}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
