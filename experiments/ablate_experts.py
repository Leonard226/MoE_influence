"""Step B: ground-truth expert ablation, mirroring Su et al.'s procedure.

Ablation operator (theirs, reproduced exactly): the expert's down-projection
output is zeroed -- equivalent to their zeroing of the down_proj / w3 weight
tensors (Super-Experts-Profilling eval_utils.py, _load_layer_weight). The
router is untouched: the expert keeps its top-K slot and gating weight, but
contributes a zero vector. NO routing redistribution.

Evaluation protocol (theirs, reproduced exactly):
  - c4 validation (en/c4-validation.00000-of-00008.json.gz)
  - 256 random 2048-token windows: random doc with > seqlen tokens, random
    token offset (their get_c4 eval_mode, random.seed(0))
  - batch size 1; PPL = exp(mean over sequences of per-sequence mean NLL)

Reported per run: PPL (their metric). Additionally (our extension, clearly
separate): attention-sink share and position-0 massive-activation magnitude
on the first --n-sink-seqs windows.

Targets are single experts by default (expert sets via '+' are supported for
later, mirroring their --prune_experts semicolon lists).

Supported: olmoe, mixtral-*, phi-3.5-moe, qwen3-* (DeepSeek deferred).

Results merge into {result_path}/circuits/ablation_{model}_c4.json
(restart-safe; cached runs are skipped).

Usage (cluster, 1 GPU for olmoe / 4 for mixtral-8x7b):
    python experiments/ablate_experts.py --model olmoe
    python experiments/ablate_experts.py --model mixtral-8x7b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from operator import attrgetter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
CIRCUITS = Path(CFG["result_path"]) / "circuits"

# multi_gpu: model is too big for one GPU -> shard via device_map="auto".
# Single-GPU models are loaded and moved with .to("cuda") explicitly, which
# is deterministic (device_map="auto" can silently leave weights on CPU).
MODELS = {
    "olmoe": {"id": "allenai/OLMoE-1B-7B-0924", "n_experts": 64,
              "n_layers": 16, "experts_path": "mlp.experts",
              "down_attr": "down_proj", "num_dense": 0, "multi_gpu": False},
    "mixtral-8x7b": {"id": "mistralai/Mixtral-8x7B-v0.1", "n_experts": 8,
                     "n_layers": 32,
                     "experts_path": "block_sparse_moe.experts",
                     "down_attr": "w2", "num_dense": 0, "multi_gpu": True},
    "mixtral-8x22b": {"id": "mistralai/Mixtral-8x22B-v0.1", "n_experts": 8,
                      "n_layers": 56,
                      "experts_path": "block_sparse_moe.experts",
                      "down_attr": "w2", "num_dense": 0, "multi_gpu": True},
    "phi-3.5-moe": {"id": "microsoft/Phi-3.5-MoE-instruct", "n_experts": 16,
                    "n_layers": 32,
                    "experts_path": "block_sparse_moe.experts",
                    "down_attr": "w2", "num_dense": 0, "multi_gpu": True},
    "qwen3-30b-a3b": {"id": "Qwen/Qwen3-30B-A3B", "n_experts": 128,
                      "n_layers": 48, "experts_path": "mlp.experts",
                      "down_attr": "down_proj", "num_dense": 0,
                      "multi_gpu": False},
    # Qwen3-235B (~470 GB bf16) and DeepSeek-V2 (~472 GB) don't fit on one
    # 4x80 GB node in bf16. Load NF4-quantized (~118 GB), keeping gate +
    # lm_head in bf16 (as build_dag.py does) so routing decisions are clean.
    # Ablation dPPL is measured within the same quantized model, so it stays
    # a valid causal signal; only absolute PPL shifts vs the bf16 models.
    "qwen3-235b-a22b": {"id": "Qwen/Qwen3-235B-A22B", "n_experts": 128,
                        "n_layers": 94, "experts_path": "mlp.experts",
                        "down_attr": "down_proj", "num_dense": 0,
                        "multi_gpu": True, "quantization": "nf4",
                        "bnb_skip_modules": ["gate", "lm_head"]},
    # DeepSeek works with this ablation operator (down_proj output zeroing
    # never touches the custom MoEGate). Labels are model-absolute layer
    # indices, which equal the module-list indices, so num_dense=0 here;
    # layer 0 is dense (no experts) -> min_layer=1 for random controls.
    "deepseek-v2-lite": {"id": "deepseek-ai/DeepSeek-V2-Lite", "n_experts": 64,
                         "n_layers": 27, "experts_path": "mlp.experts",
                         "down_attr": "down_proj", "num_dense": 0,
                         "min_layer": 1, "multi_gpu": False,
                         "trust_remote_code": True, "attn_impl": "eager"},
    "deepseek-v2": {"id": "deepseek-ai/DeepSeek-V2", "n_experts": 160,
                    "n_layers": 60, "experts_path": "mlp.experts",
                    "down_attr": "down_proj", "num_dense": 0,
                    "min_layer": 1, "multi_gpu": True,
                    "trust_remote_code": True, "attn_impl": "eager",
                    "quantization": "nf4",
                    "bnb_skip_modules": ["gate", "lm_head"]},
}

# Ablation targets = UNION of the two top-10 rankings (tab:topk-match-c4):
# top-10 by act(v) [Su, layer-filtered] + top-10 by out(v) [ours, full net].
# This gives a causal dPPL for every cell of the baseline comparison table.
# Listed act-top-10 first (rank order), then the out-only additions; the
# script adds random controls. Single experts only ('+' joints excluded).
# (mixtral keeps L30E4 as a bonus FP exemplar: high act, excluded by Su's
#  layer filter, out percentile ~3.)
DEFAULT_TARGETS = {
    "olmoe": ("L1E9;L4E27;L1E18;L2E30;L9E8;L10E4;L11E56;L1E11;L6E18;L8E22;"
              "L4E14;L3E39;L0E6;L0E38;L6E4;L0E41;L0E36"),
    "mixtral-8x7b": ("L1E3;L1E4;L9E7;L19E6;L19E1;L23E3;L12E7;L18E5;L21E3;"
                     "L16E0;L17E0;L19E5;L19E2;L6E1;L6E6;L18E1;L30E4"),
    # Phi-3.5-MoE: singles (top-10 union) + sets. Su's code cannot run Phi,
    # so the "SE set" here is OUR application of their criterion (L3E7,
    # L5E10). The whitespace pair L0E6+L1E0 is the key remaining FN test:
    # newline/space experts at act pctl 17/59 -- the same archetype as
    # Mixtral's catastrophic L1E3.
    "phi-3.5-moe": ("L5E10;L3E7;L10E0;L12E13;L11E2;L23E3;L20E13;L20E15;L8E0;"
                    "L20E12;L27E9;L29E6;L3E3;L1E0;L28E12;L9E2;L0E6;"
                    "L3E7+L5E10;"              # Su-criterion SE set (ours) [H1/anchor]
                    "L3E7+L5E10+L10E0;"        # top-3 (act = out, degenerate) [H5]
                    "L5E10+L3E7+L10E0+L12E13;" # act-top-4 [H5: marginal pick]
                    "L3E7+L5E10+L10E0+L27E9;"  # out-top-4 [H3/H5: marginal pick]
                    "L0E6+L1E0;"               # whitespace pair, sub-threshold [H2]
                    "L5E10+L8E0+L9E2;"         # apostrophe set [H4 redundancy]
                    "L14E4+L26E11;"            # random-2
                    "L14E4+L26E11+L7E9"),      # random-3
    # Mixtral-8x22B: singles (top-10 union) + sets. Su reports no 8x22B SEs;
    # our Su-criterion reproduction gives {L0E2, L1E7}. L0E2 and L29E3 are
    # both newline experts (whitespace archetype).
    "mixtral-8x22b": ("L1E7;L0E2;L36E7;L37E7;L29E3;L3E2;L0E4;L28E7;L41E2;"
                      "L38E6;L22E7;L22E6;L9E1;L3E0;L28E3;"
                      "L0E2+L1E7;"             # Su-criterion SE set (ours) [H1/anchor]
                      "L0E2+L29E3;"            # newline pair [archetype]
                      "L0E2+L1E7+L29E3;"       # out-top-3 [H3]
                      "L1E7+L0E2+L36E7;"       # act-top-3 [H5: marginal pick]
                      "L22E6+L22E7;"           # function-word pair, act pctl ~30 [H2]
                      "L13E5+L44E2;"           # random-2
                      "L13E5+L44E2+L31E6"),    # random-3
    # Qwen3-30B: singles (top-10 union) + the criteria comparison at Su's
    # scale -- Su's published SE set vs our out-top-3 vs the all-FN hub set
    # vs a size-matched random-3.
    "qwen3-30b-a3b": ("L2E92;L1E68;L3E82;L3E107;L3E4;L2E46;L16E74;L20E77;"
                      "L12E24;L33E69;L21E69;L22E92;L0E106;L31E56;L24E111;"
                      "L1E68+L2E92+L3E82;"        # Su's SE set (README anchor)
                      "L2E92+L3E82+L21E69;"        # our out-top-3
                      "L21E69+L22E92+L33E69;"      # FN hubs (none SE-flagged)
                      "L9E45+L27E103+L41E17"),     # random-3
    # DeepSeek-V2-Lite: singles (top-10 union) + Su's paper SE pair, their
    # prune script's exact 4-expert set, our late-BOS pair (excluded by
    # their layer filter -> FN-via-filter test), size-matched randoms.
    "deepseek-v2-lite": ("L3E54;L4E38;L5E63;L2E3;L2E62;L19E57;L16E14;L19E47;"
                         "L19E33;L18E23;L25E11;L24E63;L11E31;L4E45;L4E16;"
                         "L11E49;"
                         "L3E54+L4E38;"                 # Su's paper SE set
                         "L3E54+L4E38+L2E3+L5E63;"      # their script's set
                         "L24E63+L25E11;"               # late BOS pair (FN)
                         "L7E22+L13E51;"                # random-2
                         "L7E22+L13E51+L9E30+L21E5"),   # random-4
    # Qwen3-235B (NF4): singles (top-10 union) + our Su-criterion SE set
    # {L3E120, L2E39} + out-top-3 + FN hub/late set + random. Both SEs are
    # early -> ablatable. Layer labels are model-absolute (num_dense 0).
    "qwen3-235b-a22b": ("L3E120;L2E39;L69E69;L70E113;L90E101;L50E113;L2E83;"
                        "L49E69;L90E10;L90E14;"
                        "L3E120+L2E39;"               # Su-criterion SE set (ours)
                        "L3E120+L2E39+L69E69;"        # out-top-3
                        "L69E69+L70E113+L90E101;"     # late high-out set (FN)
                        "L11E64+L57E30+L83E19"),      # random-3
    # DeepSeek-V2 (NF4): singles (top-10 union) + the BOS-chain SE set + our
    # out-top-3 + a mid/late set + random. Layer labels model-absolute
    # (num_dense 0, but layer 0 dense -> min_layer 1 for controls).
    "deepseek-v2": ("L18E96;L21E94;L20E48;L1E119;L2E111;L3E17;L5E34;L3E64;"
                    "L3E81;L4E13;"
                    "L18E96+L21E94;"                  # Su-criterion SE set (ours)
                    "L18E96+L21E94+L20E48;"           # SE top-3
                    "L1E119+L2E111+L3E17;"            # early-BOS chain (out-only)
                    "L14E7+L37E88+L52E140"),          # random-3
}

SEQLEN = 2048
N_SEQS = 256


def parse_targets(spec: str, num_dense: int) -> list[list[tuple[int, int]]]:
    """'L4E27;L1E18+L2E30' -> [[(4, 27)], [(1, 18), (2, 30)]]."""
    runs = []
    for run in spec.split(";"):
        group = []
        for lab in run.strip().split("+"):
            lab = lab.strip().upper()
            l_str, e_str = lab.lstrip("L").split("E")
            group.append((int(l_str) - num_dense, int(e_str)))
        runs.append(group)
    return runs


def label_of(group: list[tuple[int, int]], num_dense: int) -> str:
    return "+".join(f"L{l + num_dense}E{e}" for l, e in group)


class ZeroExpertOutput:
    """Context manager reproducing Su et al.'s pruning: forward hooks on the
    target experts' down-projection modules return zeros. Numerically
    identical to zeroing the down_proj/w3 weights (their implementation):
    the expert stays selected with its normal gating weight but contributes
    a zero vector."""

    def __init__(self, model, experts_path: str, down_attr: str,
                 group: list[tuple[int, int]]):
        self.model = model
        self.experts_path = experts_path
        self.down_attr = down_attr
        self.group = group
        self.handles = []

    def __enter__(self):
        layers = self.model.model.layers
        for l, e in self.group:
            experts = attrgetter(self.experts_path)(layers[l])
            down = getattr(experts[e], self.down_attr)

            def hook(_m, _inp, out):
                return torch.zeros_like(out)

            self.handles.append(down.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


def su_c4_eval_windows(tok, n_seqs: int, seqlen: int) -> list[torch.Tensor]:
    """Su et al.'s get_c4(eval_mode=True): 256 random seqlen-token windows
    from c4 validation docs longer than seqlen, random.seed(0)."""
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
    return windows


def su_wikitext2_eval_windows(tok, seqlen: int) -> list[torch.Tensor]:
    """Su et al.'s get_wikitext2(eval_mode=True) + chunking: tokenize
    '\n\n'.join(wikitext-2-raw-v1 test), split into contiguous seqlen
    chunks, truncate the tail. Deterministic."""
    import datasets
    testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1",
                                     split="test")
    enc = tok("\n\n".join(testdata["text"]), return_tensors="pt").input_ids[0]
    n = enc.numel() // seqlen
    return [enc[i * seqlen:(i + 1) * seqlen] for i in range(n)]


@torch.no_grad()
def eval_ppl(model, windows: list[torch.Tensor], batch_size: int = 1) -> float:
    """Su et al.'s PPL: exp(mean of per-sequence mean NLL). Batching is a
    pure speed optimisation -- the per-sequence NLL is computed explicitly,
    so any --batch-size gives numbers identical to their batch-1 loop."""
    from tqdm import tqdm
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nlls: list[float] = []
    for i in tqdm(range(0, len(windows), batch_size), desc="ppl", leave=False):
        ids = torch.stack(windows[i:i + batch_size]).to(model.device)
        logits = model(ids).logits
        shift_logits = logits[:, :-1, :].permute(0, 2, 1)
        shift_labels = ids[:, 1:]
        loss = loss_fct(shift_logits, shift_labels)      # [B, T-1]
        nlls.extend(loss.float().mean(dim=1).tolist())   # per-sequence mean
    return float(np.exp(np.mean(nlls)))


@torch.no_grad()
def eval_sink(model, sink_ids: torch.Tensor) -> dict:
    # output_attentions materialises [B, heads, T, T] per layer in eager
    # mode, which is O(T^2) and blows up at T=2048. Attention-to-token-0 is
    # fully visible in a short context, so sink_ids is pre-truncated by the
    # caller (--sink-seq-len).
    out = model(sink_ids.to(model.device), output_attentions=True,
                output_hidden_states=True)
    att = torch.stack([a[:, :, 1:, 0].mean() for a in out.attentions])
    h0 = max(float(h[:, 0, :].abs().max()) for h in out.hidden_states)
    return {"att_sink": float(att.mean()), "max_h0": h0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--experts", default=None,
                   help="Ablation spec 'L4E27;L4E14;...' (default = curated "
                        "candidates, see DEFAULT_TARGETS).")
    p.add_argument("--dataset", choices=["c4", "wikitext2"], default="c4",
                   help="Eval corpus. c4 = 256 random 2048-token validation "
                        "windows; wikitext2 = contiguous test-set chunks "
                        "(both exactly as in Su et al.'s data_utils).")
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seqs", type=int, default=N_SEQS)
    p.add_argument("--seq-len", type=int, default=SEQLEN)
    p.add_argument("--batch-size", type=int, default=8,
                   help="PPL eval batch size; numerically identical to Su's "
                        "batch-1 loop (per-sequence NLL computed explicitly).")
    p.add_argument("--n-sink-seqs", type=int, default=8)
    p.add_argument("--sink-seq-len", type=int, default=256,
                   help="Context length for the attention-sink metric "
                        "(output_attentions is O(T^2) in eager mode; "
                        "attention-to-token-0 is visible in a short window).")
    p.add_argument("--nf4-mem-frac", type=float, default=0.35,
                   help="NF4 models only: per-GPU memory budget as a fraction "
                        "of free memory. Loading peaks at ~2.6x this. Lower it "
                        "if loading OOMs; raise it if the model doesn't fit.")
    args = p.parse_args()

    cfg = MODELS[args.model]
    nd = cfg["num_dense"]
    spec = args.experts or DEFAULT_TARGETS.get(args.model)
    if spec is None:
        p.error(f"no default targets for {args.model}; pass --experts")
    runs = parse_targets(spec, nd)

    rng = np.random.default_rng(args.seed)
    named = {(l, e) for grp in runs for l, e in grp}
    controls: list[tuple[int, int]] = []
    while len(controls) < args.random_controls:
        l = int(rng.integers(cfg.get("min_layer", 0), cfg["n_layers"] - nd))
        e = int(rng.integers(0, cfg["n_experts"]))
        if (l, e) not in named and (l, e) not in controls:
            controls.append((l, e))
    runs += [[c] for c in controls]

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible -- device_map='auto' would fall "
                 "back to CPU and each PPL pass would take hours. Run on a "
                 "GPU node (or a torch build with CUDA).")

    n_gpu = torch.cuda.device_count()
    free_mem = {}
    for i in range(n_gpu):
        free, total = torch.cuda.mem_get_info(i)
        free_mem[i] = free
        print(f"GPU {i}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total",
              flush=True)
    print(f"total free GPU memory: {sum(free_mem.values()) / 1e9:.1f} GB",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    trc = cfg.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=trc)
    # sdpa for the (many) PPL forwards; the sink pass requests
    # output_attentions=True, for which transformers falls back to eager
    # attention on that call automatically.
    load_kwargs = dict(torch_dtype=torch.bfloat16,
                       attn_implementation=cfg.get("attn_impl", "sdpa"),
                       trust_remote_code=trc)
    if cfg.get("quantization") == "nf4":
        # NF4 4-bit (bitsandbytes). bnb manages device placement via
        # device_map="auto" reliably (unlike the bf16 dispatch path below).
        # Router gate + lm_head kept in bf16 so routing/logits are clean.
        from transformers import BitsAndBytesConfig
        skip = cfg.get("bnb_skip_modules", ["lm_head"])
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip)  # name says int8; applies to 4-bit too
        load_kwargs.pop("torch_dtype", None)
        # bitsandbytes' NF4 loader transiently holds ~2.6x the per-GPU budget
        # while quantizing (build_dag.py notes this). Per-GPU budget =
        # --nf4-mem-frac * free: it must be high enough that all GPUs together
        # exceed the ~118 GB model, yet low enough that the ~2.6x loading peak
        # stays under `free`. On 4x84 GB this window is narrow (~0.36); lower
        # the fraction if the peak OOMs, raise it if the model doesn't fit.
        frac = args.nf4_mem_frac
        max_memory = {i: int(free * frac) for i, free in free_mem.items()}
        budget_gb = sum(max_memory.values()) / 1e9
        print(f"NF4 per-GPU budget {max_memory[0]/1e9:.1f} GB "
              f"(loading peak ~{max_memory[0]*2.6/1e9:.0f} GB/GPU); "
              f"total {budget_gb:.0f} GB  (--nf4-mem-frac {frac})", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["id"], quantization_config=bnb, device_map="auto",
            max_memory=max_memory, **load_kwargs).eval()
    elif cfg["multi_gpu"]:
        # from_pretrained(device_map="auto") silently dumps these MoE model
        # classes entirely to CPU (observed for Mixtral; same issue noted in
        # build_dag.py). Working recipe: plan a device map on a meta skeleton,
        # load weights to CPU, then physically dispatch. Per-GPU budget = 90%
        # of free memory; no 'cpu' budget so an over-large model errors
        # clearly instead of offloading.
        from accelerate import (infer_auto_device_map, init_empty_weights,
                                 dispatch_model)
        from transformers import AutoConfig
        max_memory = {i: int(free * 0.9) for i, free in free_mem.items()}
        hf_cfg = AutoConfig.from_pretrained(cfg["id"], trust_remote_code=trc)
        with init_empty_weights():
            empty = AutoModelForCausalLM.from_config(
                hf_cfg, torch_dtype=torch.bfloat16)
        device_map = infer_auto_device_map(
            empty, max_memory=max_memory,
            no_split_module_classes=empty._no_split_modules,
            dtype=torch.bfloat16)
        del empty
        print(f"planned device_map spans devices "
              f"{sorted(set(device_map.values()))}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["id"], low_cpu_mem_usage=True, **load_kwargs).eval()
        model = dispatch_model(model, device_map=device_map)
    else:
        # Single GPU: explicit .to('cuda') is deterministic; device_map='auto'
        # can silently leave weights on CPU (~100x slower).
        model = AutoModelForCausalLM.from_pretrained(
            cfg["id"], **load_kwargs).to("cuda").eval()
    # Report the actual device spread across parameters (works whether or not
    # accelerate set hf_device_map). Any 'cpu' entry => partial offload =>
    # slow; abort so it doesn't crawl unnoticed. (bnb keeps a few meta/quant
    # bookkeeping params off-device, so ignore 'meta' and only fail on cpu.)
    devs = {str(p.device) for p in model.parameters()}
    print(f"parameter devices: {sorted(devs)}", flush=True)
    if any(d.startswith("cpu") for d in devs):
        sys.exit("Some parameters are on CPU (offloaded) -- would run at CPU "
                 "speed. Allocate more GPU memory / more GPUs, or set "
                 "CUDA_VISIBLE_DEVICES to the free GPUs.")

    if args.dataset == "c4":
        windows = su_c4_eval_windows(tok, args.n_seqs, args.seq_len)
    else:
        windows = su_wikitext2_eval_windows(tok, args.seq_len)
    sink_ids = torch.stack([w[:args.sink_seq_len]
                            for w in windows[:args.n_sink_seqs]])
    print(f"[{args.model}/{args.dataset}] {len(windows)} windows x "
          f"{args.seq_len} tok (Su et al. protocol); {len(runs)} ablation runs "
          f"({args.random_controls} random controls)")

    out_path = CIRCUITS / f"ablation_{args.model}_{args.dataset}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    if "baseline" not in results:
        print("baseline: computing PPL...", flush=True)
        ppl0 = eval_ppl(model, windows, args.batch_size)
        print(f"baseline: PPL={ppl0:.3f}; computing sink metrics...", flush=True)
        sink0 = eval_sink(model, sink_ids)
        results["baseline"] = {"ppl": ppl0, **sink0}
        out_path.write_text(json.dumps(results, indent=2))
    base = results["baseline"]
    print(f"baseline: ppl={base['ppl']:.3f} att_sink={base['att_sink']:.4f} "
          f"max_h0={base['max_h0']:.4g}")

    is_control = {label_of([c], nd) for c in controls}
    for group in runs:
        lab = label_of(group, nd)
        if lab in results:
            print(f"{lab}: cached, skipping")
            continue
        with ZeroExpertOutput(model, cfg["experts_path"], cfg["down_attr"],
                              group):
            ppl = eval_ppl(model, windows, args.batch_size)
            sink = eval_sink(model, sink_ids)
        results[lab] = {"ppl": ppl, **sink, "control": lab in is_control}
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{lab}{' [ctrl]' if lab in is_control else ''}: "
              f"ppl={ppl:.3f} (x{ppl / base['ppl']:.3f})  "
              f"att_sink={sink['att_sink']:.4f} (base {base['att_sink']:.4f})  "
              f"max_h0={sink['max_h0']:.4g} (base {base['max_h0']:.4g})")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
