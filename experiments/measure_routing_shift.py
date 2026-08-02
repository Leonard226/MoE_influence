"""Routing-reorganization metrics for existing ablations.

Tests the hypothesis that out(v) encodes routing CONTROL, not (only)
magnitude/sink importance: ablating a high-out expert should reorganize
which experts fire downstream, even when this barely moves PPL (the model
compensates for the lost COMPUTE, but the routing PROGRAM still changed).
ΔPPL alone cannot see this; this script gives the routing-side analogue.

Three metrics per (ablation, downstream MoE layer, token), each with the
convention "0 = no change, larger = more disruption":

  topk_displacement = 1 - |topK_orig ∩ topK_ablated| / K
      Simplest, most interpretable: fraction of top-K routing DECISIONS that
      flipped. In [0, 1].

  aarv (Average Absolute Rank Variation, from Li et al. ICLR'26's MoE
        cross-layer entanglement study -- applying THEIR own diagnostic to
        OUR causal ablations):
      (1/K) * sum_{e in topK_orig} |rank_orig(e) - rank_ablated(e)| / (N-1)
      Richer than displacement: captures how far a KEPT-or-dropped expert
      moved, not just set membership. Normalised by (N-1) for cross-model
      comparability.

  kl = KL(P_orig(routing) || P_ablated(routing))
      Catches continuous reweighting invisible to the two rank-based
      metrics above (e.g. an expert staying rank-1 while its probability
      mass shifts 0.9 -> 0.6).

Each metric is computed per downstream layer (giving a depth PROFILE, saved
in full -- routing effects are known to be depth-localised, not uniform:
see Li et al.'s M1/M4 "stripe" finding in OLMoE) and reduced to BOTH a mean-
and a peak-over-depth scalar (peak in case the effect is concentrated in a
few layers and would be diluted by averaging over dozens).

Router probabilities are captured via a plain forward hook on the gate
module (mlp.gate / block_sparse_moe.gate) -- a [tokens, N_experts] linear
output, no O(T^2) attention-style cost, works on stock HF models (no
customized_models fork needed, unlike the attention-sink work).

"Downstream" for an ablated set = MoE layers strictly after the DEEPEST
ablated layer (forward-influence-only, matching this project's W_softmax/
out(v) convention throughout).

Targets = every ablation already in ablation/{model}_c4.json (not a new
curated list): this gives a routing-shift number for every entry that
already has a ΔPPL, enabling a direct join/comparison table.

Calibration: a short c4 batch (default 16 x 256 tokens, same scale as the
ablation scripts' sink pass) -- kept small so baseline router probabilities
for every downstream layer fit comfortably in memory and can be computed
ONCE per model, then reused for every ablation's comparison.

Scope: single-node models only for this first pass (mirrors
find_super_weights.py); multi-node (DeepSeek-V2, Qwen3-235B) deferred.

Usage:
    python experiments/measure_routing_shift.py --model olmoe
    python experiments/measure_routing_shift.py --model phi-3.5-moe
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
RESULTS = Path(CFG["result_path"])

# Single-node models (matches find_super_weights.py / ablate_experts.py).
# gate_path/top_k/moe_layers/n_experts copied from build_dag.py's registry
# for consistency with the rest of this codebase. Ablation labels use the
# SAME layer-indexing convention as ablate_experts.py's MODELS (num_dense=0
# for every model here -- the label's integer already equals the module
# index directly, no dense-layer shift needed).
MODELS = {
    "olmoe": {"id": "allenai/OLMoE-1B-7B-0924", "n_experts": 64, "top_k": 8,
              "moe_layers": list(range(16)), "experts_path": "mlp.experts",
              "down_attr": "down_proj", "gate_path": "mlp.gate",
              "multi_gpu": False},
    "mixtral-8x7b": {"id": "mistralai/Mixtral-8x7B-v0.1", "n_experts": 8,
                     "top_k": 2, "moe_layers": list(range(32)),
                     "experts_path": "block_sparse_moe.experts",
                     "down_attr": "w2", "gate_path": "block_sparse_moe.gate",
                     "multi_gpu": True},
    "mixtral-8x22b": {"id": "mistralai/Mixtral-8x22B-v0.1", "n_experts": 8,
                      "top_k": 2, "moe_layers": list(range(56)),
                      "experts_path": "block_sparse_moe.experts",
                      "down_attr": "w2", "gate_path": "block_sparse_moe.gate",
                      "multi_gpu": True},
    "phi-3.5-moe": {"id": "microsoft/Phi-3.5-MoE-instruct", "n_experts": 16,
                    "top_k": 2, "moe_layers": list(range(32)),
                    "experts_path": "block_sparse_moe.experts",
                    "down_attr": "w2", "gate_path": "block_sparse_moe.gate",
                    "multi_gpu": True},
    "qwen3-30b-a3b": {"id": "Qwen/Qwen3-30B-A3B", "n_experts": 128,
                      "top_k": 8, "moe_layers": list(range(48)),
                      "experts_path": "mlp.experts", "down_attr": "down_proj",
                      "gate_path": "mlp.gate", "multi_gpu": False},
    "deepseek-v2-lite": {"id": "deepseek-ai/DeepSeek-V2-Lite", "n_experts": 64,
                         "top_k": 6, "moe_layers": list(range(1, 27)),
                         "experts_path": "mlp.experts", "down_attr": "down_proj",
                         "gate_path": "mlp.gate", "multi_gpu": False,
                         "trust_remote_code": True, "attn_impl": "eager"},
}


def su_c4_eval_windows(tok, n_seqs: int, seqlen: int) -> list[torch.Tensor]:
    """Same construction as ablate_experts.py's su_c4_eval_windows (c4
    validation, random.seed(0)), duplicated to keep this script standalone."""
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


def parse_label(lab: str) -> list[tuple[int, int]]:
    """'L0E2+L1E7' -> [(0, 2), (1, 7)]."""
    group = []
    for tok in lab.strip().upper().split("+"):
        l_str, e_str = tok.lstrip("L").split("E")
        group.append((int(l_str), int(e_str)))
    return group


class ZeroExpertOutput:
    """Su-mirrored ablation operator (identical semantics to
    ablate_experts.py's): zero the target experts' down_proj output. The
    router is untouched -- experts keep their top-K slot but contribute
    nothing."""

    def __init__(self, model, experts_path: str, down_attr: str,
                group: list[tuple[int, int]]):
        self.model = model
        self.experts_path = experts_path
        self.down_attr = down_attr
        self.group = group
        self.handles = []

    def __enter__(self):
        for (l, e) in self.group:
            experts = attrgetter(self.experts_path)(self.model.model.layers[l])
            down = getattr(experts[e], self.down_attr)

            def hook(_m, _inp, out):
                return torch.zeros_like(out)

            self.handles.append(down.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


@torch.no_grad()
def capture_router_probs(model, gate_path: str, layers: list[int],
                         windows: list[torch.Tensor], batch_size: int
                         ) -> dict[int, torch.Tensor]:
    """Hook every listed layer's gate module; return {layer: probs} where
    probs is [n_tokens, N_experts] softmax over the FULL expert set (not
    the sparse top-K-renormalised weights) -- the well-defined, strictly-
    positive distribution needed for KL and for ranking all N experts.

    Recomputed from the gate's own (weight, bias) applied to its INPUT via a
    pre-hook, rather than read off its output: a plain nn.Linear gate's
    forward() is literally F.linear(x, weight, bias), so this reproduces it
    exactly. DeepSeek's custom MoEGate module returns (topk_idx, topk_weight,
    aux_loss) -- no raw logits tensor -- but its forward computes internally
    `logits = F.linear(hidden_states, self.weight); scores = logits.softmax(-1)`
    before any topk/group-masking, so the same recompute is exact there too."""
    probs_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    def make_hook(l):
        def hook(m, inp):
            x = inp[0].detach().float()
            bias = m.bias.float() if getattr(m, "bias", None) is not None else None
            logits = torch.nn.functional.linear(x, m.weight.float(), bias)
            probs_by_layer[l].append(
                torch.softmax(logits, dim=-1).reshape(-1, logits.shape[-1]))
        return hook

    for l in layers:
        gate = attrgetter(gate_path)(model.model.layers[l])
        handles.append(gate.register_forward_pre_hook(make_hook(l)))

    for i in range(0, len(windows), batch_size):
        batch = torch.stack(windows[i:i + batch_size]).to(model.device)
        model(batch)

    for h in handles:
        h.remove()
    return {l: torch.cat(v, dim=0) for l, v in probs_by_layer.items()}


def compute_metrics(probs_base: dict[int, torch.Tensor],
                    probs_abl: dict[int, torch.Tensor],
                    top_k: int, n_experts: int) -> dict:
    """Per-layer topk_displacement / aarv / kl, reduced to mean + peak over
    depth. Tensor ops stay on-device; only final scalars are pulled to CPU."""
    profile = {"topk_displacement": {}, "aarv": {}, "kl": {}}
    for l in sorted(probs_abl):
        pb, pa = probs_base[l], probs_abl[l]            # [n_tok, N], same device

        topk_b_idx = pb.topk(top_k, dim=-1).indices      # [n_tok, K], sorted desc
        topk_a_idx = pa.topk(top_k, dim=-1).indices

        mask_b = torch.zeros_like(pb, dtype=torch.bool).scatter_(1, topk_b_idx, True)
        mask_a = torch.zeros_like(pa, dtype=torch.bool).scatter_(1, topk_a_idx, True)
        overlap = (mask_b & mask_a).sum(dim=-1).float() / top_k
        displacement = float((1.0 - overlap).mean())

        # topk_b_idx is sorted by descending pb score, so its OWN ranks in
        # pb are exactly [0, 1, ..., K-1] -- no need to compute rank_b.
        rank_a = pa.argsort(dim=-1, descending=True).argsort(dim=-1)     # [n_tok, N]
        ranks_a_at_topk_b = torch.gather(rank_a, 1, topk_b_idx).float()  # [n_tok, K]
        orig_ranks = torch.arange(top_k, device=pb.device).float().unsqueeze(0)
        aarv = float(((orig_ranks - ranks_a_at_topk_b).abs().mean(dim=-1)
                     / (n_experts - 1)).mean())

        eps = 1e-12
        kl = float((pb * (torch.log(pb.clamp_min(eps))
                          - torch.log(pa.clamp_min(eps)))).sum(dim=-1).mean())

        profile["topk_displacement"][l] = displacement
        profile["aarv"][l] = aarv
        profile["kl"][l] = kl

    out = {}
    for metric, per_layer in profile.items():
        vals = list(per_layer.values())
        out[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{metric}_peak"] = float(np.max(vals)) if vals else float("nan")
        out[f"{metric}_profile"] = per_layer
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--top-k", type=int, default=None,
                   help="Override the model's own routing top-K (default: "
                        "use the model's actual operational top-K).")
    p.add_argument("--n-seqs", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    cfg = MODELS[args.model]
    top_k = args.top_k or cfg["top_k"]

    ablation_path = RESULTS / "ablation" / f"{args.model}_c4.json"
    if not ablation_path.exists():
        sys.exit(f"Missing {ablation_path}; run ablate_experts.py --model "
                 f"{args.model} first (this script joins against its ΔPPL).")
    ablation_results = json.loads(ablation_path.read_text())
    labels = [k for k in ablation_results if k != "baseline"]
    base_ppl = ablation_results["baseline"]["ppl"]

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
          f"{args.seq_len} tok; top_k={top_k}; {len(labels)} ablations "
          f"to process", flush=True)

    out_path = RESULTS / "routing_shift" / f"{args.model}_c4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    print("computing baseline routing snapshot...", flush=True)
    probs_base = capture_router_probs(model, cfg["gate_path"], cfg["moe_layers"],
                                      windows, args.batch_size)

    for lab in labels:
        if lab in results:
            print(f"{lab}: cached, skipping")
            continue
        group = parse_label(lab)
        ablated_layers = {l for l, e in group}
        downstream = [l for l in cfg["moe_layers"] if l > max(ablated_layers)]
        if not downstream:
            print(f"{lab}: no downstream MoE layers, skipping")
            continue

        with ZeroExpertOutput(model, cfg["experts_path"], cfg["down_attr"], group):
            probs_abl = capture_router_probs(model, cfg["gate_path"], downstream,
                                             windows, args.batch_size)

        metrics = compute_metrics({l: probs_base[l] for l in downstream},
                                  probs_abl, top_k, cfg["n_experts"])
        metrics["is_control"] = bool(ablation_results[lab].get("control", False))
        metrics["dppl_pct"] = 100.0 * (ablation_results[lab]["ppl"] / base_ppl - 1.0)
        metrics["n_downstream_layers"] = len(downstream)
        results[lab] = metrics
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{lab}: dPPL={metrics['dppl_pct']:+7.2f}%  "
              f"disp(mean/peak)={metrics['topk_displacement_mean']:.3f}/"
              f"{metrics['topk_displacement_peak']:.3f}  "
              f"AARV(mean/peak)={metrics['aarv_mean']:.3f}/{metrics['aarv_peak']:.3f}  "
              f"KL(mean/peak)={metrics['kl_mean']:.3f}/{metrics['kl_peak']:.3f}",
              flush=True)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()