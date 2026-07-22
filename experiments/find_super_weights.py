"""Iterative super-weight detection for target MoE experts.

Background: Yu et al.'s "super weights" are single scalars in an expert's
down_proj that create the massive/"super" activations Su et al. use to
define Super Experts. Their detection method is activation-guided: for a
token where down_proj's OUTPUT has an extreme outlier at channel j, and the
INPUT has an extreme outlier at channel k, the product X[k]*W[j,k] dominates
Y[j] = sum_k X[k]*W[j,k], so the super weight is simply W[j,k] -- a direct
lookup, no optimisation needed. They then zero that weight and repeat to
find additional ones (Phi-3-mini has six this way).

This script implements that FULL iterative method (not a same-channel
proxy): at each iteration, find the current peak-activation token among all
tokens routed to the expert, its dominant input channel k_it and output
channel j_it (independently re-derived each iteration -- NOT constrained to
share the previous iteration's input channel), record W[j_it, k_it] as a
candidate, zero it, and repeat. Successive candidates can therefore live in
completely different (input, output) channel pairs, unlike a same-column
proxy -- this is what actually answers "how many super weights does this
expert hold."

Efficiency note (exact, not an approximation): zeroing an entry of THIS
down_proj's own weight matrix cannot change THIS layer's down_proj INPUT --
that input is fixed by every computation upstream of this layer (routing
for this layer, and everything that produces its input, already happened
before down_proj runs). So X for every routed token is invariant across
iterations, and we only need ONE full-model forward pass (to cache X and Y
for every token routed to the expert); every refinement iteration afterward
is a cheap local matmul (X @ W_work.T) against a progressively-zeroed
in-memory copy of the weight matrix, not a fresh GPU forward pass. This
recovers ground truth, not a shortcut: for a fixed set of input tokens,
recomputing down_proj's own output with an edited version of its own
weights is mathematically identical to re-running the whole model and
reading the same layer back out.

Scope: this finds super weights WITHIN one expert's down_proj (does its own
peak output collapse as weights are removed). It does NOT track whether
removing them collapses activations model-wide across depth -- that is a
different, more expensive analysis (full forward re-runs, akin to the
att_sink/max_h_all metrics already added to the ablation scripts) and is
not implemented here.

For each candidate we report explained_frac = |X[k]*W[j,k]| / |Y[j]|: how
well the single-dominant-term approximation explains the actual output at
that iteration. Near 1.0 = a clean super-weight-driven spike; low = this
expert's current peak is not well-explained by any single weight -- the
loop naturally runs out of real candidates once explained_frac collapses,
which is itself evidence the expert has no (more) super weights.

Targets = the reproduced Super Expert sets + the influential-only ("blue")
experts identified as causally inert singles in tab:ablation-c4 -- the
direct falsification test: do influential-only experts lack a clean super
weight, unlike genuine Super Experts?

Calibration: c4 windows (same source as the ablation scripts), a modest
batch (default 32 x 2048) -- Yu et al. note their method needs just ONE
prompt, since super activations are stated to persist regardless of input.

Usage:
    python experiments/find_super_weights.py --model phi-3.5-moe
    python experiments/find_super_weights.py --model olmoe --experts L1E9,L4E14
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

# Single-node models only (mirrors ablate_experts.py's MODELS; multi-node
# DeepSeek-V2/Qwen3-235B deferred -- the iterative-loop successor would need
# this anyway, so no point building a one-off multi-node static pass now).
MODELS = {
    "olmoe": {"id": "allenai/OLMoE-1B-7B-0924", "experts_path": "mlp.experts",
              "down_attr": "down_proj", "multi_gpu": False},
    "mixtral-8x7b": {"id": "mistralai/Mixtral-8x7B-v0.1",
                     "experts_path": "block_sparse_moe.experts",
                     "down_attr": "w2", "multi_gpu": True},
    "mixtral-8x22b": {"id": "mistralai/Mixtral-8x22B-v0.1",
                      "experts_path": "block_sparse_moe.experts",
                      "down_attr": "w2", "multi_gpu": True},
    "phi-3.5-moe": {"id": "microsoft/Phi-3.5-MoE-instruct",
                    "experts_path": "block_sparse_moe.experts",
                    "down_attr": "w2", "multi_gpu": True},
    "qwen3-30b-a3b": {"id": "Qwen/Qwen3-30B-A3B", "experts_path": "mlp.experts",
                      "down_attr": "down_proj", "multi_gpu": False},
    "deepseek-v2-lite": {"id": "deepseek-ai/DeepSeek-V2-Lite",
                         "experts_path": "mlp.experts", "down_attr": "down_proj",
                         "multi_gpu": False, "trust_remote_code": True,
                         "attn_impl": "eager"},
}

# SE sets (reproduced, Su-exact criterion) + influential-only comparison
# experts (causally inert singles from tab:ablation-c4) per model.
DEFAULT_TARGETS = {
    "mixtral-8x7b": "L1E3;L17E0",
    "mixtral-8x22b": "L0E2;L1E7;L29E3;L36E7",
    "phi-3.5-moe": "L3E7;L5E10;L10E0",     # L10E0 is the red-row expert
    "olmoe": "L1E9;L1E18;L2E30;L4E27;L4E14",
    "deepseek-v2-lite": "L3E54;L4E38;L5E63;L25E11;L24E63",
    "qwen3-30b-a3b": "L1E68;L2E92;L3E82;L3E107;L21E69;L22E92;L33E69",
}


def su_c4_eval_windows(tok, n_seqs: int, seqlen: int) -> list[torch.Tensor]:
    """Same construction as ablate_experts.py's su_c4_eval_windows (c4
    validation, random.seed(0)), duplicated here to keep this script
    standalone and avoid touching the reproducibility-audited ablation
    scripts."""
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


def find_super_weights_iterative(model, experts_path: str, down_attr: str,
                                 layer: int, expert: int,
                                 windows: list[torch.Tensor], batch_size: int,
                                 max_iters: int) -> dict | None:
    """ONE full-model forward pass caches (X, Y) for every token routed to
    this expert; every subsequent iteration is a local matmul against a
    progressively-zeroed in-memory weight copy (see module docstring for why
    this is exact, not an approximation). Each iteration independently
    re-derives the peak token and its (input_channel, output_channel) --
    candidates are NOT constrained to share an input channel across
    iterations."""
    down = getattr(attrgetter(experts_path)(model.model.layers[layer])[expert],
                   down_attr)

    captured = {}
    chunks_x, chunks_y = [], []

    def pre_hook(_m, inp):
        captured["x"] = inp[0].detach()

    def fwd_hook(_m, _inp, out):
        x, y = captured.pop("x"), out.detach()
        if x.numel() == 0:
            return
        chunks_x.append(x.float().cpu())
        chunks_y.append(y.float().cpu())

    h1 = down.register_forward_pre_hook(pre_hook)
    h2 = down.register_forward_hook(fwd_hook)
    n_tokens_seen = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.stack(windows[i:i + batch_size]).to(model.device)
            model(batch)
            n_tokens_seen += batch.numel()
    h1.remove()
    h2.remove()

    if not chunks_x:
        return None  # no tokens ever routed to this expert in the calibration set

    # Use the device the expert's own weight already lives on -- for
    # multi-GPU dispatched models (Mixtral, Phi) this can differ from
    # model.device (the embedding's device), and the local recompute below
    # needs X and W_work co-located with each other.
    dev = down.weight.device
    X = torch.cat(chunks_x, dim=0).to(dev)           # [N_routed, D_in], INVARIANT across iterations
    Y0 = torch.cat(chunks_y, dim=0)                  # [N_routed, D_out], original (unperturbed) output
    n_routed = X.shape[0]
    W_work = down.weight.detach().float().clone().to(dev)  # [D_out, D_in], progressively zeroed

    candidates = []
    for it in range(max_iters):
        Y_it = X @ W_work.T                          # exact recompute, this expert's own weights only
        row_max, t_star = Y_it.abs().max(dim=-1)[0].max(dim=0)
        t_star = int(t_star)
        y_t, x_t = Y_it[t_star], X[t_star]
        k_star = int(x_t.abs().argmax())
        j_star = int(y_t.abs().argmax())
        w_val = float(W_work[j_star, k_star])
        predicted = float(x_t[k_star]) * w_val
        actual = float(y_t[j_star])
        explained = abs(predicted) / abs(actual) if actual != 0 else float("nan")
        candidates.append({
            "iter": it, "input_channel": k_star, "output_channel": j_star,
            "weight_value": w_val, "peak_activation": float(row_max),
            "explained_frac": explained, "token_index": t_star,
        })
        W_work[j_star, k_star] = 0.0                  # zero for the NEXT iteration

    return {
        "n_tokens_seen": n_tokens_seen, "n_tokens_routed": n_routed,
        "peak_activation_original": float(Y0.abs().max()),
        "candidates": candidates,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--experts", default=None,
                   help="Comma-separated 'LxEy' labels (default = curated "
                        "SE + influential-only targets, see DEFAULT_TARGETS).")
    p.add_argument("--max-iters", type=int, default=10,
                   help="Max zero-and-repeat iterations per expert (default "
                        "10; Yu et al. observed at most 6 super weights in "
                        "any single model). Iterations are cheap local "
                        "matmuls, not GPU forward passes, so this is "
                        "generous by design -- inspect the explained_frac "
                        "trajectory to see where real candidates run out.")
    p.add_argument("--n-seqs", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    cfg = MODELS[args.model]
    spec = args.experts or DEFAULT_TARGETS.get(args.model)
    if spec is None:
        p.error(f"no default targets for {args.model}; pass --experts")
    targets = []
    for lab in (spec.split(",") if "," in spec else spec.split(";")):
        lab = lab.strip().upper()
        l_str, e_str = lab.lstrip("L").split("E")
        targets.append((int(l_str), int(e_str), lab))

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible.")
    n_gpu = torch.cuda.device_count()
    free_mem = {}
    for i in range(n_gpu):
        free, total = torch.cuda.mem_get_info(i)
        free_mem[i] = free
        print(f"GPU {i}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total",
              flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    trc = cfg.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=trc)
    load_kwargs = dict(torch_dtype=torch.bfloat16,
                       attn_implementation=cfg.get("attn_impl", "sdpa"),
                       trust_remote_code=trc)
    if cfg["multi_gpu"]:
        from accelerate import (infer_auto_device_map, init_empty_weights,
                                 dispatch_model)
        from transformers import AutoConfig
        max_memory = {i: int(free * 0.9) for i, free in free_mem.items()}
        hf_cfg = AutoConfig.from_pretrained(cfg["id"], trust_remote_code=trc)
        with init_empty_weights():
            empty = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=torch.bfloat16)
        device_map = infer_auto_device_map(
            empty, max_memory=max_memory,
            no_split_module_classes=empty._no_split_modules, dtype=torch.bfloat16)
        del empty
        print(f"planned device_map spans devices "
              f"{sorted(set(device_map.values()))}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["id"], low_cpu_mem_usage=True, **load_kwargs).eval()
        model = dispatch_model(model, device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["id"], **load_kwargs).to("cuda").eval()
    devs = {str(pp.device) for pp in model.parameters()}
    print(f"parameter devices: {sorted(devs)}", flush=True)
    if any(d.startswith("cpu") for d in devs):
        sys.exit("Some parameters are on CPU (offloaded) -- aborting.")

    windows = su_c4_eval_windows(tok, args.n_seqs, args.seq_len)
    print(f"[{args.model}] calibration: {len(windows)} windows x "
          f"{args.seq_len} tok; {len(targets)} target experts", flush=True)

    out_path = CIRCUITS / f"super_weights_{args.model}_c4.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for layer, expert, lab in targets:
        if lab in results:
            print(f"{lab}: cached, skipping")
            continue
        rec = find_super_weights_iterative(
            model, cfg["experts_path"], cfg["down_attr"], layer, expert,
            windows, args.batch_size, args.max_iters)
        if rec is None:
            print(f"{lab}: no tokens routed in calibration set, skipping")
            continue
        results[lab] = rec
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{lab}: n_routed={rec['n_tokens_routed']} "
              f"peak_orig={rec['peak_activation_original']:.4g}", flush=True)
        for c in rec["candidates"]:
            print(f"       iter{c['iter']}: in_ch={c['input_channel']} "
                  f"out_ch={c['output_channel']} w={c['weight_value']:.4g} "
                  f"peak={c['peak_activation']:.4g} "
                  f"explained={c['explained_frac']:.3f}", flush=True)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
