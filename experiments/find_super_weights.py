"""Iterative super-weight detection, searched globally across every expert.

Background: Yu et al.'s "super weights" are single scalars in a down_proj
that create the massive/"super" activations Su et al. use to define Super
Experts. Their method: for a token where down_proj's OUTPUT has an extreme
outlier at channel j, and the INPUT has an extreme outlier at channel k, the
product X[k]*W[j,k] dominates Y[j] = sum_k X[k]*W[j,k], so the super weight
is simply W[j,k] -- a direct lookup. They apply this PER LAYER (one down_proj
per layer in a dense model): scan every layer's down_proj input/output for
the layer with the most extreme activation spike, read off its channel
indices, zero that weight, and repeat until the spikes are suppressed.

MoE models have one down_proj PER EXPERT per layer rather than one per
layer, so the faithful generalisation is to apply their exact per-unit
procedure to every EXPERT instead of every layer -- not a different
detection criterion, just the natural unit substitution. This script
therefore searches EVERY expert in EVERY MoE layer, not a curated shortlist:
restricting to Super Experts and out(v)-influential experts a priori would
beg the question of whether super weights exist elsewhere too (e.g. in some
other, so-far-uncharacterised kind of expert).

This is a genuine iterative method (not a same-channel proxy): at each
iteration, find the current peak-activation token among all tokens routed
to the expert, its dominant input channel k_it and output channel j_it
(independently re-derived each iteration -- NOT constrained to share the
previous iteration's input channel), record W[j_it, k_it] as a candidate,
zero it, and repeat. Successive candidates can therefore live in completely
different (input, output) channel pairs, unlike a same-column proxy.

Efficiency (exact, not an approximation, and what makes a global sweep
tractable): zeroing an entry of an expert's OWN down_proj weight matrix
cannot change that expert's own down_proj INPUT -- that input is fixed by
everything upstream (routing for this layer, and everything producing its
input, already happened before down_proj runs). So X for every routed token
is invariant across iterations, and a SINGLE full-model forward pass -- with
a pre/forward hook on every expert's down_proj in every MoE layer at once --
suffices to cache X and Y for every token routed to every expert, model-
wide. Hooks are read-only and add no compute, so hooking every expert
instead of a handful does not add GPU work; the only cost that scales with
the number of experts is the (cheap) local refinement afterward: each
iteration is a matmul against a progressively-zeroed in-memory copy of that
expert's weight matrix, not a fresh forward pass. This recovers ground
truth, not a shortcut: for a fixed set of input tokens, recomputing
down_proj's own output with an edited copy of its own weights is
mathematically identical to re-running the whole model and reading the same
layer back out.

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
which is itself evidence the expert has no (more) super weights. A
downstream consumer (e.g. a results table) defines "number of super
weights" as the count of leading iterations with explained_frac > 0.5; that
thresholding is NOT applied here, all `max_iters` candidates are always
recorded so the choice of threshold can be revisited without rerunning.

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
# DeepSeek-V2/Qwen3-235B deferred). n_experts/moe_layers match
# measure_routing_shift.py's registry (cross-checked against build_dag.py).
MODELS = {
    "olmoe": {"id": "allenai/OLMoE-1B-7B-0924", "experts_path": "mlp.experts",
              "down_attr": "down_proj", "multi_gpu": False,
              "n_experts": 64, "moe_layers": list(range(16))},
    "mixtral-8x7b": {"id": "mistralai/Mixtral-8x7B-v0.1",
                     "experts_path": "block_sparse_moe.experts",
                     "down_attr": "w2", "multi_gpu": True,
                     "n_experts": 8, "moe_layers": list(range(32))},
    "mixtral-8x22b": {"id": "mistralai/Mixtral-8x22B-v0.1",
                      "experts_path": "block_sparse_moe.experts",
                      "down_attr": "w2", "multi_gpu": True,
                      "n_experts": 8, "moe_layers": list(range(56))},
    "phi-3.5-moe": {"id": "microsoft/Phi-3.5-MoE-instruct",
                    "experts_path": "block_sparse_moe.experts",
                    "down_attr": "w2", "multi_gpu": True,
                    "n_experts": 16, "moe_layers": list(range(32))},
    "qwen3-30b-a3b": {"id": "Qwen/Qwen3-30B-A3B", "experts_path": "mlp.experts",
                      "down_attr": "down_proj", "multi_gpu": False,
                      "n_experts": 128, "moe_layers": list(range(48))},
    "deepseek-v2-lite": {"id": "deepseek-ai/DeepSeek-V2-Lite",
                         "experts_path": "mlp.experts", "down_attr": "down_proj",
                         "multi_gpu": False, "trust_remote_code": True,
                         "attn_impl": "eager",
                         "n_experts": 64, "moe_layers": list(range(1, 27))},
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


def down_module(model, experts_path: str, down_attr: str, layer: int, expert: int):
    return getattr(attrgetter(experts_path)(model.model.layers[layer])[expert],
                   down_attr)


def capture_all_routed(model, experts_path: str, down_attr: str,
                       moe_layers: list[int], n_experts: int,
                       windows: list[torch.Tensor], batch_size: int,
                       only: set[tuple[int, int]] | None = None
                       ) -> tuple[dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]], int]:
    """ONE full-model forward pass, with a pre/forward hook on EVERY expert's
    down_proj in EVERY MoE layer at once (or just `only`, if given, for fast
    debugging on a subset) -- caches (X, Y) for every token routed to every
    expert, model-wide, in a single pass. Returns {(layer, expert): (X, Y)}
    (X, Y concatenated over all routed tokens, on CPU) plus n_tokens_seen.
    Experts that received zero tokens in this calibration set are absent
    from the returned dict."""
    targets = only if only is not None else {
        (l, e) for l in moe_layers for e in range(n_experts)}

    captured_in: dict[tuple[int, int], torch.Tensor] = {}
    chunks_x: dict[tuple[int, int], list[torch.Tensor]] = {}
    chunks_y: dict[tuple[int, int], list[torch.Tensor]] = {}
    handles = []

    def make_hooks(key):
        def pre_hook(_m, inp):
            captured_in[key] = inp[0].detach()

        def fwd_hook(_m, _inp, out):
            x = captured_in.pop(key)
            y = out.detach()
            if x.numel() == 0:
                return
            chunks_x.setdefault(key, []).append(x.float().cpu())
            chunks_y.setdefault(key, []).append(y.float().cpu())
        return pre_hook, fwd_hook

    for (l, e) in targets:
        down = down_module(model, experts_path, down_attr, l, e)
        pre_hook, fwd_hook = make_hooks((l, e))
        handles.append(down.register_forward_pre_hook(pre_hook))
        handles.append(down.register_forward_hook(fwd_hook))

    n_tokens_seen = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.stack(windows[i:i + batch_size]).to(model.device)
            model(batch)
            n_tokens_seen += batch.numel()
    for h in handles:
        h.remove()

    routed = {key: (torch.cat(chunks_x[key], dim=0), torch.cat(chunks_y[key], dim=0))
              for key in chunks_x}
    return routed, n_tokens_seen


def refine_from_cache(X: torch.Tensor, Y0: torch.Tensor, weight: torch.Tensor,
                      dev: torch.device, n_tokens_seen: int,
                      max_iters: int) -> dict:
    """Local, no-further-forward-pass iterative refinement from already-
    cached (X, Y0) for one expert (see module docstring for exactness
    argument). Each iteration independently re-derives the peak token and
    its (input_channel, output_channel) -- candidates are NOT constrained to
    share an input channel across iterations."""
    X = X.to(dev)
    n_routed = X.shape[0]
    W_work = weight.detach().float().clone().to(dev)  # [D_out, D_in], progressively zeroed

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
                   help="Comma- or semicolon-separated 'LxEy' labels to "
                        "restrict the search to (for fast debugging). "
                        "Default: every expert in every MoE layer.")
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
    only = None
    if args.experts:
        only = set()
        for lab in (args.experts.split(",") if "," in args.experts else args.experts.split(";")):
            lab = lab.strip().upper()
            l_str, e_str = lab.lstrip("L").split("E")
            only.add((int(l_str), int(e_str)))

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
    print(f"[{args.model}] calibration: {len(windows)} windows x {args.seq_len} tok",
          flush=True)

    out_path = CIRCUITS / f"super_weights_{args.model}_c4.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    # Skip (layer, expert) pairs already cached under their "LxEy" label --
    # restart-safe, and lets a global sweep extend an existing curated-
    # targets run without recomputing anything.
    full_grid = {(l, e) for l in cfg["moe_layers"] for e in range(cfg["n_experts"])}
    already = {(l, e) for (l, e) in full_grid if f"L{l}E{e}" in results}
    pending = (only if only is not None else full_grid) - already
    print(f"[{args.model}] {len(already)} cached, {len(pending)} pending "
          f"(of {len(full_grid)} total experts)", flush=True)

    if pending:
        routed, n_tokens_seen = capture_all_routed(
            model, cfg["experts_path"], cfg["down_attr"], cfg["moe_layers"],
            cfg["n_experts"], windows, args.batch_size, only=pending)

        for (layer, expert), (X, Y0) in routed.items():
            lab = f"L{layer}E{expert}"
            down = down_module(model, cfg["experts_path"], cfg["down_attr"], layer, expert)
            rec = refine_from_cache(X, Y0, down.weight, down.weight.device,
                                    n_tokens_seen, args.max_iters)
            results[lab] = rec
            out_path.write_text(json.dumps(results, indent=2))
            print(f"{lab}: n_routed={rec['n_tokens_routed']} "
                  f"peak_orig={rec['peak_activation_original']:.4g}", flush=True)

        skipped = len(pending) - len(routed)
        if skipped:
            print(f"{skipped} of {len(pending)} pending experts got zero "
                  f"routed tokens in this calibration set (not written)",
                  flush=True)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
